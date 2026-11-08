"""Unified Scikit-Learn style estimator (SeqSM) for Markov chains, state machines, and schedule generation."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta
from typing import Any, Literal, Self, overload

import numpy as np
import polars as pl
from numpy.typing import DTypeLike, NDArray

from seqsm.encoding import StateEncoder
from seqsm.machine import StateMachine
from seqsm.markov import KOrderMarkovChain
from seqsm.motifs import detect_period, score_anomalies
from seqsm.periodic import PhaseConditionedMarkovChain, select_regimes
from seqsm.roles import SignalRoles, infer_roles
from seqsm.schedule import ScheduleGenerator
from seqsm.timeslots import (
    RegimeSpec,
    TimeConfig,
    datetime_us,
    preset_regimes,
    resample_to_slots,
    resolve_regimes,
    to_datetime64,
)
from seqsm.timeutil import duration_us, parse_duration
from seqsm.types import ModeType, PredictionMethod, ScheduleConstraints, ScheduleMethod


class SeqSM:
    """Unified estimator for K-order Markov chains, state machines, and schedule generation."""

    def __init__(
        self,
        k: int = 1,
        *,
        mode: ModeType = "clock",
        smoothing: float = 0.0,
        feature_cols: Sequence[str] | None = None,
        episode_col: str | None = None,
        timestamp_col: str | None = None,
        max_time_gap: timedelta | float | None = None,
        min_prob: float = 0.0,
        min_count: int = 1,
        seed: int | None = 42,
        dtype: DTypeLike = np.float64,
        state_cols: Sequence[str] | None = None,
        context_cols: Sequence[str] | None = None,
        time_config: TimeConfig | Mapping[str, Any] | None = None,
        auto_roles: bool | Mapping[str, Any] = False,
    ) -> None:
        """Creates the estimator.

        Signal roles: ``state_cols`` (alias ``feature_cols``) define the atomic state; ``context_cols`` are
        drivers that condition the kernel without being states; ``time_config`` makes the timestamp a
        transition signal through regime-conditioned kernels; ``auto_roles=True`` infers the roles that
        are not given (see ``seqsm.roles.infer_roles``; pass a dict to forward its options, e.g.
        ``{"min_predictability": 0.05}``).
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")
        if state_cols is not None and feature_cols is not None and tuple(state_cols) != tuple(feature_cols):
            raise ValueError("state_cols and feature_cols are aliases; pass only one (or identical values).")
        if state_cols is not None:
            feature_cols = state_cols
        if isinstance(time_config, Mapping):
            time_config = TimeConfig(**dict(time_config))
        self.k: int = k
        self.mode: ModeType = mode
        self.smoothing: float = smoothing
        self.feature_cols: tuple[str, ...] | None = tuple(feature_cols) if feature_cols is not None else None
        self.episode_col: str | None = episode_col
        self.timestamp_col: str | None = timestamp_col
        self.max_time_gap: timedelta | float | None = max_time_gap
        self.min_prob: float = min_prob
        self.min_count: int = min_count
        self.seed: int | None = seed
        self.dtype: DTypeLike = dtype
        self._rng: np.random.Generator = np.random.default_rng(seed)

        self.context_cols: tuple[str, ...] | None = tuple(context_cols) if context_cols is not None else None
        self.time_config: TimeConfig | None = time_config
        self.auto_roles: bool | Mapping[str, Any] = auto_roles

        self._encoder: StateEncoder | None = None
        self._chain: KOrderMarkovChain | None = None
        self._phase: PhaseConditionedMarkovChain | None = None
        self.signal_roles: SignalRoles | None = None
        self.regime_report: pl.DataFrame | None = None
        self._eff_episode_col: str | None = episode_col
        self._eff_timestamp_col: str | None = timestamp_col
        self._eff_context_cols: tuple[str, ...] = ()
        self._regime_spec: RegimeSpec | None = None
        self._n_time_regimes: int = 1
        self._context_ids: dict[tuple[Any, ...], int] = {}
        self._step: timedelta | None = None
        self._regime_meta: dict[str, Any] = {}

    @property
    def markov_chain(self) -> KOrderMarkovChain:
        """Returns the fitted KOrderMarkovChain instance."""
        if self._chain is None:
            raise ValueError("SeqSM has not been fitted.")
        return self._chain

    @property
    def state_cols(self) -> tuple[str, ...] | None:
        """Returns the declared state columns (alias of ``feature_cols``)."""
        return self.feature_cols

    @property
    def phase_chain(self) -> PhaseConditionedMarkovChain | None:
        """Returns the regime-conditioned chain, or None when no time/context signal was configured."""
        return self._phase

    @property
    def n_regimes(self) -> int:
        """Returns the number of kernel regimes (1 for a homogeneous model)."""
        return self._phase.n_regimes if self._phase is not None else 1

    @property
    def state_encoder(self) -> StateEncoder:
        """Returns the fitted StateEncoder instance."""
        if self._encoder is None:
            raise ValueError("SeqSM has not been fitted.")
        return self._encoder

    @property
    def stationary_distribution(self) -> pl.DataFrame:
        """Returns the marginal stationary distribution over atomic states as a Polars DataFrame."""
        if self._chain is None or self._encoder is None:
            raise ValueError("SeqSM has not been fitted.")
        mu = self._chain.marginal_stationary_distribution()
        vocab = self._encoder.vocabulary
        return vocab.with_columns(pl.Series("stationary_prob", mu))

    def _resolve_episodes(self, df: pl.DataFrame) -> NDArray[np.int32] | None:
        """Resolves episode boundary IDs based on episode_col or timestamp_col + max_time_gap."""
        if self._eff_episode_col is not None and self._eff_episode_col in df.columns:
            ep_series = df[self._eff_episode_col]
            if ep_series.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                return ep_series.to_numpy().astype(np.int32)
            dense_ep = ep_series.to_physical() if ep_series.dtype == pl.Categorical else ep_series.rank("dense") - 1
            return dense_ep.to_numpy().astype(np.int32)

        if (
            self._eff_timestamp_col is not None
            and self.max_time_gap is not None
            and self._eff_timestamp_col in df.columns
        ):
            ts = df[self._eff_timestamp_col]
            diffs = ts.diff()
            if isinstance(self.max_time_gap, (timedelta, str)):
                gap_delta = parse_duration(self.max_time_gap)
                if ts.dtype in (pl.Datetime, pl.Date, pl.Duration):
                    gap_us = int(gap_delta.total_seconds() * 1_000_000)
                    is_break = diffs.dt.total_microseconds() > gap_us
                else:
                    is_break = diffs > gap_delta.total_seconds()
            else:
                num_gap = float(self.max_time_gap)
                if ts.dtype in (pl.Datetime, pl.Date, pl.Duration):
                    gap_us = int(num_gap * 1_000_000)
                    is_break = diffs.dt.total_microseconds() > gap_us
                else:
                    is_break = diffs > num_gap

            is_break_clean = is_break.fill_null(False)
            ep_ids = is_break_clean.cum_sum().to_numpy().astype(np.int32)
            return ep_ids

        return None

    def fit(self, df: pl.DataFrame | pl.LazyFrame) -> Self:
        """Fits state encoder and K-order Markov chain from aligned feature table.

        With ``time_config`` and/or ``context_cols`` the pooled chain is complemented by one kernel per
        regime (``phase_chain``); everything that does not name a regime keeps using the pooled chain.
        """
        frame = df.collect() if isinstance(df, pl.LazyFrame) else df

        state_cols = self.feature_cols
        context_cols = self.context_cols
        self._eff_episode_col = self.episode_col
        self._eff_timestamp_col = self.timestamp_col
        self.signal_roles = None
        if self.auto_roles:
            opts = dict(self.auto_roles) if isinstance(self.auto_roles, Mapping) else {}
            roles = infer_roles(frame, episode_col=self.episode_col, timestamp_col=self.timestamp_col, **opts)
            self.signal_roles = roles
            if state_cols is None:
                state_cols = roles.state
            if context_cols is None:
                context_cols = roles.context
            self._eff_episode_col = self.episode_col or roles.episode
            self._eff_timestamp_col = self.timestamp_col or roles.time
        ctx_cols = tuple(context_cols) if context_cols else ()
        self._eff_context_cols = ctx_cols
        if state_cols is None:
            excluded = {c for c in (self._eff_episode_col, self._eff_timestamp_col, *ctx_cols) if c is not None}
            state_cols = tuple(c for c in frame.columns if c not in excluded)
        cols_to_use = tuple(state_cols)
        overlap = set(cols_to_use) & set(ctx_cols)
        if overlap:
            raise ValueError(f"Columns {sorted(overlap)} cannot be both state and context columns.")

        ep_ids = self._resolve_episodes(frame)
        tc = self.time_config
        ts_col = self._eff_timestamp_col
        self._step = None
        if tc is not None:
            if ts_col is None or ts_col not in frame.columns:
                raise ValueError("time_config requires a timestamp column (timestamp_col or auto_roles).")
            if tc.slot_duration is not None:
                keep = list(dict.fromkeys([*cols_to_use, *ctx_cols]))
                frame, ep_ids_slot = resample_to_slots(frame, ts_col, tc.slot_duration, keep, ep_ids, tc.fill)
                ep_ids = ep_ids_slot if ep_ids is not None or len(np.unique(ep_ids_slot)) > 1 else None
                self._step = parse_duration(tc.slot_duration)
            else:
                ts_us = datetime_us(frame[ts_col]).astype(np.int64)
                if len(ts_us) > 1:
                    self._step = timedelta(microseconds=int(np.median(np.diff(ts_us))))

        self._encoder = StateEncoder(feature_cols=cols_to_use)
        self._encoder.fit(frame)
        states = self._encoder.transform(frame)

        self._phase = None
        self._regime_spec = None
        self._n_time_regimes = 1
        self._context_ids = {}
        self._regime_meta = {}
        self.regime_report = None
        if tc is not None or ctx_cols:
            regimes = self._fit_regimes(frame, states, ep_ids, ctx_cols)
            self._phase = PhaseConditionedMarkovChain(
                k=self.k,
                n_regimes=self._n_time_regimes * max(1, len(self._context_ids)),
                mode=self.mode,
                smoothing=self.smoothing,
                min_regime_count=tc.min_regime_count if tc is not None else 1,
                dtype=self.dtype,
            ).fit_from_arrays(states, regimes, episode_ids=ep_ids, n_states=self._encoder.n_states)
            self._chain = self._phase.pooled
            return self

        self._chain = KOrderMarkovChain(
            k=self.k,
            mode=self.mode,
            smoothing=self.smoothing,
            dtype=self.dtype,
        )
        self._chain.fit_from_arrays(states, episode_ids=ep_ids, n_states=self._encoder.n_states)
        return self

    def _fit_regimes(
        self,
        frame: pl.DataFrame,
        states: NDArray[np.int32],
        ep_ids: NDArray[np.int32] | None,
        ctx_cols: tuple[str, ...],
    ) -> NDArray[np.int32]:
        """Resolves the regime definition and returns the regime id of every row."""
        tc = self.time_config
        time_reg = np.zeros(len(frame), dtype=np.int32)
        if tc is not None:
            ts_col = self._eff_timestamp_col
            assert ts_col is not None
            ts_us = datetime_us(frame[ts_col])
            period: str | timedelta | None = None
            if tc.regimes == "auto":
                spec, self.regime_report = select_regimes(
                    states,
                    ts_us,
                    tc.candidates,
                    k=self.k,
                    mode=self.mode,
                    episode_ids=ep_ids,
                    criterion=tc.criterion,
                    folds=tc.cv_folds,
                    smoothing=max(self.smoothing, 0.1),
                    min_regime_count=tc.min_regime_count,
                )
            elif tc.regimes == "phase":
                period = tc.period
                if period is None or period == "auto":
                    lag = detect_period(states)
                    if lag <= 1 or self._step is None:
                        raise ValueError(
                            "period='auto' found no cycle longer than one slot (needs a slot grid); pass an explicit period."
                        )
                    period = self._step * lag
                spec = resolve_regimes(tc, period)
            else:
                spec = resolve_regimes(tc)
            self._regime_spec = spec
            self._n_time_regimes = spec.n_regimes
            time_reg = spec(ts_us)
            if callable(tc.regimes):
                self._regime_meta = {"kind": "callable"}
            elif tc.regimes == "phase":
                assert period is not None
                self._regime_meta = {"kind": "phase", "period_us": duration_us(period), "n_regimes": spec.n_regimes}
            else:
                self._regime_meta = {"kind": "preset", "name": spec.name}
        else:
            self._regime_spec = preset_regimes("none")
            self._regime_meta = {"kind": "preset", "name": "none"}

        ctx_ids = np.zeros(len(frame), dtype=np.int32)
        if ctx_cols:
            vocab = frame.select(list(ctx_cols)).unique().sort(list(ctx_cols)).with_row_index("__ctx__")
            self._context_ids = {row[:-1]: int(row[-1]) for row in vocab.select([*ctx_cols, "__ctx__"]).iter_rows()}
            joined = frame.select(list(ctx_cols)).join(vocab, on=list(ctx_cols), how="left")
            ctx_ids = joined["__ctx__"].to_numpy().astype(np.int32)
        return (ctx_ids * self._n_time_regimes + time_reg).astype(np.int32)

    def _context_id(self, context: Mapping[str, Any] | None) -> int:
        """Maps a context dict to its fitted context id (0 when the model has no context columns)."""
        if not self._eff_context_cols:
            if context:
                raise ValueError("This model has no context columns.")
            return 0
        if context is None:
            raise ValueError(f"context with keys {list(self._eff_context_cols)} is required.")
        try:
            key = tuple(context[c] for c in self._eff_context_cols)
        except KeyError as exc:
            raise ValueError(f"context is missing column {exc.args[0]!r}.") from exc
        cid = self._context_ids.get(key)
        if cid is None:
            raise ValueError(f"Unseen context {dict(zip(self._eff_context_cols, key, strict=True))}.")
        return cid

    def regime_of(self, at: datetime | None = None, context: Mapping[str, Any] | None = None) -> int:
        """Returns the kernel regime id for a timestamp and/or context (0 for a homogeneous model)."""
        if self._phase is None or self._regime_spec is None:
            raise ValueError("Model has no time_config or context_cols; there are no regimes.")
        spec = self._regime_spec
        if spec.n_regimes == 1:
            tr = 0
        elif at is None:
            raise ValueError("at (a datetime) is required: the model's regimes depend on time.")
        else:
            tr = int(spec(np.array([to_datetime64(at)]))[0])
        return self._context_id(context) * self._n_time_regimes + tr

    def _regime_sequence(self, at: datetime | None, context: Mapping[str, Any] | None, steps: int) -> list[int] | None:
        """Regimes of the next ``steps`` predicted slots, or None to use the pooled kernel."""
        if at is None and context is None:
            return None
        if self._phase is None:
            raise ValueError("at/context given, but the model was fitted without time_config or context_cols.")
        if steps > 1 and at is not None and self._step is None:
            raise ValueError("Multi-step regime prediction needs a known step; fit with a slot_duration.")
        out: list[int] = []
        for h in range(steps):
            t = at + h * self._step if at is not None and self._step is not None else at
            out.append(self.regime_of(t, context))
        return out

    def partial_fit(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        decay: float = 1.0,
    ) -> Self:
        """Sequential partial fitting update with concept-drift count decay gamma in (0, 1]."""
        if self._encoder is None or self._chain is None:
            return self.fit(df)
        if self._phase is not None:
            raise NotImplementedError("partial_fit is not supported with time_config or context_cols; refit instead.")

        frame = df.collect() if isinstance(df, pl.LazyFrame) else df
        self._encoder.partial_fit(frame)
        states = self._encoder.transform(frame)
        ep_ids = self._resolve_episodes(frame)

        self._chain.partial_fit_from_arrays(
            states,
            episode_ids=ep_ids,
            n_states=self._encoder.n_states,
            decay=decay,
        )
        return self

    @overload
    def predict_proba(
        self,
        history: pl.DataFrame | Sequence[int] | None = None,
        steps: int = 1,
        *,
        return_diagnostics: Literal[True],
        at: datetime | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]: ...

    @overload
    def predict_proba(
        self,
        history: pl.DataFrame | Sequence[int] | None,
        steps: int,
        return_diagnostics: Literal[True],
        *,
        at: datetime | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]: ...

    @overload
    def predict_proba(
        self,
        history: pl.DataFrame | Sequence[int] | None = None,
        steps: int = 1,
        return_diagnostics: Literal[False] = False,
        *,
        at: datetime | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> NDArray[np.float64]: ...

    def predict_proba(
        self,
        history: pl.DataFrame | Sequence[int] | None = None,
        steps: int = 1,
        return_diagnostics: bool = False,
        *,
        at: datetime | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> NDArray[np.float64] | tuple[NDArray[np.float64], dict[str, Any]]:
        """Returns next-state probability distribution vector of shape (|S|,) or (steps, |S|).

        Args:
            history: Optional observation history as a Polars DataFrame or sequence of state IDs.
            steps: Forecast horizon in time steps (default 1).
            return_diagnostics: If True, returns a tuple of (probabilities, diagnostics_dict).
            at: Timestamp of the first predicted step; selects the time-regime kernel of a model fitted
                with ``time_config`` (later steps advance by the slot duration).
            context: Values of the ``context_cols`` that condition the kernel.
                Without ``at``/``context`` the pooled kernel is used.

        Returns:
            Probability vector/matrix, or (probs, diagnostics) if return_diagnostics is True.
        """
        if self._chain is None or self._encoder is None:
            raise ValueError("SeqSM has not been fitted.")

        regimes = self._regime_sequence(at, context, steps)
        mu = self._chain.marginal_stationary_distribution()

        if history is None:
            probs = mu if steps == 1 else np.tile(mu, (steps, 1))
            if return_diagnostics:
                return probs, {"fallback": True, "reason": "none_history", "window": None, "window_id": -1}
            return probs

        if isinstance(history, pl.DataFrame):
            hist_states = self._encoder.transform(history)
        else:
            hist_states = np.asarray(history, dtype=np.int32)
            if len(hist_states) > 0 and (np.any(hist_states < 0) or np.any(hist_states >= self._encoder.n_states)):
                raise ValueError(f"history contains state IDs outside valid range [0, {self._encoder.n_states - 1}].")

        if len(hist_states) < self.k:
            probs = mu if steps == 1 else np.tile(mu, (steps, 1))
            if return_diagnostics:
                win_partial = tuple(int(x) for x in hist_states) if len(hist_states) > 0 else None
                return probs, {"fallback": True, "reason": "short_history", "window": win_partial, "window_id": -1}
            return probs

        win = tuple(int(x) for x in hist_states[-self.k :])
        u = self._chain.encoder.encode_window(win)
        if u < 0:
            probs = mu if steps == 1 else np.tile(mu, (steps, 1))
            if return_diagnostics:
                return probs, {"fallback": True, "reason": "unseen_window", "window": win, "window_id": -1}
            return probs

        if regimes is not None and self._phase is not None:
            if steps == 1:
                probs = self._phase.predict_proba_window(u, regimes[0])
            else:
                probs = self._phase.multi_step_distribution(u, regimes)
        elif steps == 1:
            probs = self._chain.predict_proba_window(u)
        else:
            probs = self._chain.multi_step_distribution(u, steps=steps)

        if return_diagnostics:
            return probs, {"fallback": False, "reason": None, "window": win, "window_id": u}
        return probs

    def predict(
        self,
        history: pl.DataFrame | Sequence[int] | None = None,
        steps: int = 1,
        method: PredictionMethod = "sample",
        *,
        at: datetime | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> pl.DataFrame:
        """Predicts the next step(s) as a decoded Polars DataFrame (``at``/``context``: see predict_proba)."""
        if self._chain is None or self._encoder is None:
            raise ValueError("SeqSM has not been fitted.")

        if method == "viterbi" and (at is not None or context is not None):
            raise ValueError("method='viterbi' decodes with the pooled kernel; omit at/context.")
        if method == "viterbi" and isinstance(self._chain, KOrderMarkovChain):
            hist_ids: Sequence[int] = ()
            if history is not None:
                if isinstance(history, pl.DataFrame):
                    hist_ids = [int(x) for x in self._encoder.transform(history)]
                else:
                    hist_ids = [int(x) for x in history]
            _, state_path = self._chain.decode_viterbi(hist_ids, steps)
            return self._encoder.inverse_transform(state_path)

        probs = self.predict_proba(history=history, steps=steps, at=at, context=context)
        rng = self._rng if self.seed is not None else np.random.default_rng()

        if probs.ndim == 1:
            if method in ("map", "viterbi"):
                chosen_states = [int(np.argmax(probs))]
            else:
                chosen_states = [int(rng.choice(len(probs), p=probs))]
        else:
            chosen_states = []
            for h in range(len(probs)):
                row = probs[h]
                if method in ("map", "viterbi"):
                    chosen_states.append(int(np.argmax(row)))
                else:
                    chosen_states.append(int(rng.choice(len(row), p=row)))

        return self._encoder.inverse_transform(chosen_states)

    def generate_schedule(
        self,
        window: int | timedelta | tuple[datetime, datetime],
        *,
        init_history: pl.DataFrame | Sequence[int] | None = None,
        method: ScheduleMethod = "sample",
        time_unit: timedelta | str | None = None,
        start_time: datetime | None = None,
        constraints: ScheduleConstraints | None = None,
        n_schedules: int = 1,
        seed: int | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> pl.DataFrame:
        """
        Generates realistic synthetic schedules over the specified time window following arc transition probabilities.

        With a time/context model every transition uses the kernel of the regime of its slot
        (``start_time`` + step * ``time_unit``; ``time_unit`` defaults to the fitted slot duration, else 1m);
        ``context`` fixes the context column values for the whole rollout.

        Returns:
            pl.DataFrame with columns:
                ['schedule_id', 'time_step', 'timestamp', 'state_id', <feature_cols...>,
                 'dwell_duration', 'transition_prob', 'arc_valid']
        """
        if self._chain is None or self._encoder is None:
            raise ValueError("SeqSM has not been fitted.")

        # Resolve time_unit delta
        step_delta = parse_duration(time_unit) if time_unit is not None else (self._step or parse_duration("1m"))

        # Resolve total step count
        base_time = start_time if start_time is not None else datetime(2026, 1, 1, 0, 0, 0)
        if isinstance(window, int):
            n_steps = window
        elif isinstance(window, timedelta):
            n_steps = max(1, int(window.total_seconds() / step_delta.total_seconds()))
        elif isinstance(window, tuple):
            base_time = window[0]
            diff = window[1] - window[0]
            n_steps = max(1, int(diff.total_seconds() / step_delta.total_seconds()))
        else:
            n_steps = 60

        # Encode init_history if provided
        init_arr: NDArray[np.int32] | None = None
        if init_history is not None:
            if isinstance(init_history, pl.DataFrame):
                init_arr = self._encoder.transform(init_history)
            else:
                init_arr = np.asarray(init_history, dtype=np.int32)
                if len(init_arr) > 0 and (np.any(init_arr < 0) or np.any(init_arr >= self._encoder.n_states)):
                    raise ValueError(
                        f"init_history contains state IDs outside valid range [0, {self._encoder.n_states - 1}]."
                    )

        gen = ScheduleGenerator(
            self._chain,
            self._encoder,
            min_prob=self.min_prob,
            min_count=self.min_count,
            regime_chains=self._phase.regime_chains() if self._phase is not None else None,
        )
        regime_at = self._make_regime_at(base_time, step_delta, context)

        all_schedule_ids: list[int] = []
        all_time_steps: list[int] = []
        all_timestamps: list[datetime] = []
        all_state_ids: list[int] = []
        all_dwells: list[int] = []
        all_probs: list[float] = []
        all_valids: list[bool] = []
        all_violated: list[bool] = []

        base_seed = seed if seed is not None else (self.seed if self.seed is not None else 42)
        ss = np.random.SeedSequence(base_seed)
        child_seeds = [int(s.generate_state(1)[0]) for s in ss.spawn(n_schedules)]

        for sched_id in range(n_schedules):
            traj = gen.generate(
                n_steps=n_steps,
                init_history=init_arr,
                method=method,
                constraints=constraints,
                seed=child_seeds[sched_id],
                regime_at=regime_at,
            )

            curr_t = 0
            for idx, sid in enumerate(traj.state_ids):
                dwell = traj.dwell_durations[idx]
                prob = traj.transition_probs[idx]
                valid = traj.arc_valid[idx]
                violated = traj.constraint_violated[idx] if idx < len(traj.constraint_violated) else False

                all_schedule_ids.append(sched_id)
                all_time_steps.append(curr_t)
                all_timestamps.append(base_time + curr_t * step_delta)
                all_state_ids.append(sid)
                all_dwells.append(dwell)
                all_probs.append(prob)
                all_valids.append(valid)
                all_violated.append(violated)

                curr_t += dwell
                if curr_t >= n_steps:
                    break

        decoded_features = self._encoder.inverse_transform(all_state_ids)
        base_df = pl.DataFrame(
            {
                "schedule_id": np.array(all_schedule_ids, dtype=np.int32),
                "time_step": np.array(all_time_steps, dtype=np.int32),
                "timestamp": all_timestamps,
                "state_id": np.array(all_state_ids, dtype=np.int32),
                "dwell_duration": np.array(all_dwells, dtype=np.int32),
                "transition_prob": np.array(all_probs, dtype=np.float64),
                "arc_valid": all_valids,
                "constraint_violated": all_violated,
            }
        )

        # Combine schedule metadata with feature columns
        return pl.concat([base_df, decoded_features], how="horizontal")

    def _make_regime_at(
        self, base_time: datetime, step_delta: timedelta, context: Mapping[str, Any] | None
    ) -> Callable[[int], int] | None:
        """Builds the slot-index -> regime map for schedule rollouts (None for a homogeneous model)."""
        if self._phase is None or self._regime_spec is None:
            if context:
                raise ValueError("context given, but the model has no context columns.")
            return None
        ctx_id = self._context_id(context)
        spec = self._regime_spec
        n_time = self._n_time_regimes
        if spec.n_regimes == 1:
            return lambda _step: ctx_id * n_time
        base_np = to_datetime64(base_time)
        step_us = round(step_delta.total_seconds() * 1_000_000)

        def regime_at(step: int) -> int:
            ts = base_np + np.timedelta64(step * step_us, "us")
            return ctx_id * n_time + int(spec(np.array([ts]))[0])

        return regime_at

    def detect_period(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        timestamp_col: str | None = None,
    ) -> int:
        """Detects dominant stream cycle lag in discrete steps via Categorical Match Autocorrelation (CMA)."""
        ts_col = timestamp_col if timestamp_col is not None else self.timestamp_col
        if self._encoder is None:
            raise ValueError("SeqSM has not been fitted.")

        frame = df.collect() if isinstance(df, pl.LazyFrame) else df
        if ts_col is not None and ts_col in frame.columns:
            frame = frame.sort(ts_col)

        states = self._encoder.transform(frame)
        return detect_period(states)

    def score_anomalies(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        window_size: int = 5,
        threshold_quantile: float = 0.99,
        mode: Literal["two_factor", "nll"] = "two_factor",
    ) -> pl.DataFrame:
        """Scores transition and dwell anomalies on an input sequence."""
        if self._chain is None or self._encoder is None:
            raise ValueError("SeqSM has not been fitted.")
        states = self._encoder.transform(df)
        return score_anomalies(
            self._chain,
            states,
            window_size=window_size,
            threshold_quantile=threshold_quantile,
            mode=mode,
        )

    def to_state_machine(
        self,
        min_prob: float | None = None,
        min_count: int | None = None,
        top_k_edges: int | None = None,
        renormalize: bool = True,
        mode: Literal["atomic", "debruijn"] = "atomic",
        regime: int | None = None,
    ) -> StateMachine:
        """Constructs a pruned StateMachine graph representation (``regime``: one regime's kernel)."""
        if self._chain is None or self._encoder is None:
            raise ValueError("SeqSM has not been fitted.")
        p = min_prob if min_prob is not None else self.min_prob
        c = min_count if min_count is not None else self.min_count
        chain = self._chain
        if regime is not None:
            if self._phase is None:
                raise ValueError("regime requires a model fitted with time_config or context_cols.")
            chain = self._phase.chain_for(regime)
        return StateMachine.from_markov(
            chain,
            self._encoder,
            min_prob=p,
            min_count=c,
            top_k_edges=top_k_edges,
            renormalize=renormalize,
            mode=mode,
        )

    def plot_state_machine(
        self,
        ax: Any = None,
        layout: Literal["circular", "random"] = "circular",
        min_prob: float | None = None,
        figsize: tuple[float, float] = (10, 8),
        node_color: str = "#0284C7",
        edge_color: str = "#475569",
        show_edge_labels: bool = True,
        show_diffs: bool = False,
        title: str | None = None,
    ) -> Any:
        """Convenience method to construct and render the state machine onto Matplotlib."""
        fsm = self.to_state_machine(min_prob=min_prob)
        return fsm.plot(
            ax=ax,
            layout=layout,
            min_prob=min_prob,
            figsize=figsize,
            node_color=node_color,
            edge_color=edge_color,
            show_edge_labels=show_edge_labels,
            show_diffs=show_diffs,
            title=title,
        )

    def compare_similarity(self, other: SeqSM) -> dict[str, float]:
        """Calculates multi-aspect similarity metrics against another fitted SeqSM estimator."""
        fsm_self = self.to_state_machine()
        fsm_other = other.to_state_machine()
        return fsm_self.compare_similarity(fsm_other)

    def __repr__(self) -> str:
        fitted = self._chain is not None and self._encoder is not None
        n_states = self._encoder.n_states if self._encoder is not None else 0
        return (
            f"SeqSM(k={self.k}, mode={self.mode!r}, min_prob={self.min_prob}, fitted={fitted}, "
            f"n_states={n_states}, n_regimes={self.n_regimes})"
        )

    def __str__(self) -> str:
        if self._chain is None or self._encoder is None:
            return f"SeqSM Estimator (k={self.k}, mode='{self.mode}', not fitted)"
        fsm = self.to_state_machine()
        return f"SeqSM Estimator (k={self.k}, mode='{self.mode}'):\n{fsm}"


__all__ = ["SeqSM", "parse_duration"]
