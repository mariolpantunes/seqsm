"""Unified Scikit-Learn style estimator (SeqSM) for Markov chains, state machines, and schedule generation."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Self

import numpy as np
import polars as pl
from numpy.typing import DTypeLike, NDArray

from seqsm.encoding import StateEncoder
from seqsm.machine import StateMachine
from seqsm.markov import KOrderMarkovChain
from seqsm.motifs import score_anomalies
from seqsm.schedule import ScheduleGenerator
from seqsm.types import ModeType, PredictionMethod, ScheduleConstraints, ScheduleMethod


def parse_duration(duration: str | timedelta) -> timedelta:
    """Parses a duration string (e.g. '1d', '12h', '30m', '15s', '500ms') into a timedelta."""
    if isinstance(duration, timedelta):
        return duration
    if not isinstance(duration, str):
        raise TypeError(f"Expected str or timedelta, got {type(duration).__name__}")

    s = duration.strip().lower()
    if not s:
        raise ValueError("Duration string cannot be empty.")

    match = re.fullmatch(r"^(\d+(?:\.\d+)?)\s*(ms|us|µs|s|m|h|d|w)$", s)
    if not match:
        raise ValueError(
            f"Invalid duration format: '{duration}'. Expected format like '1d', '12h', '30m', '15s', '500ms'."
        )

    val_str, unit = match.groups()
    val = float(val_str)
    if unit in ("us", "µs"):
        return timedelta(microseconds=val)
    if unit == "ms":
        return timedelta(milliseconds=val)
    if unit == "s":
        return timedelta(seconds=val)
    if unit == "m":
        return timedelta(minutes=val)
    if unit == "h":
        return timedelta(hours=val)
    if unit == "d":
        return timedelta(days=val)
    if unit == "w":
        return timedelta(weeks=val)
    raise ValueError(f"Unknown duration unit: '{unit}' in '{duration}'")


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
    ) -> None:
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}.")
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

        self._encoder: StateEncoder | None = None
        self._chain: KOrderMarkovChain | None = None

    @property
    def markov_chain(self) -> KOrderMarkovChain:
        """Returns the fitted KOrderMarkovChain instance."""
        if self._chain is None:
            raise ValueError("SeqSM has not been fitted.")
        return self._chain

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
        if self.episode_col is not None and self.episode_col in df.columns:
            ep_series = df[self.episode_col]
            if ep_series.dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
                return ep_series.to_numpy().astype(np.int32)
            dense_ep = ep_series.to_physical() if ep_series.dtype == pl.Categorical else ep_series.rank("dense") - 1
            return dense_ep.to_numpy().astype(np.int32)

        if self.timestamp_col is not None and self.max_time_gap is not None and self.timestamp_col in df.columns:
            ts = df[self.timestamp_col]
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
        """Fits state encoder and K-order Markov chain from aligned feature table."""
        frame = df.collect() if isinstance(df, pl.LazyFrame) else df

        # Inferred feature columns: all columns minus episode/timestamp
        cols_to_use = self.feature_cols
        if cols_to_use is None:
            excluded = {col for col in (self.episode_col, self.timestamp_col) if col is not None}
            cols_to_use = tuple(c for c in frame.columns if c not in excluded)

        self._encoder = StateEncoder(feature_cols=cols_to_use)
        self._encoder.fit(frame)
        states = self._encoder.transform(frame)

        ep_ids = self._resolve_episodes(frame)
        self._chain = KOrderMarkovChain(
            k=self.k,
            mode=self.mode,
            smoothing=self.smoothing,
            dtype=self.dtype,
        )
        self._chain.fit_from_arrays(states, episode_ids=ep_ids, n_states=self._encoder.n_states)
        return self

    def partial_fit(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        decay: float = 1.0,
    ) -> Self:
        """Sequential partial fitting update with concept-drift count decay gamma in (0, 1]."""
        if self._encoder is None or self._chain is None:
            return self.fit(df)

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

    def predict_proba(
        self,
        history: pl.DataFrame | Sequence[int] | None = None,
        steps: int = 1,
    ) -> NDArray[np.float64]:
        """Returns next-state probability distribution vector of shape (|S|,) or (steps, |S|)."""
        if self._chain is None or self._encoder is None:
            raise ValueError("SeqSM has not been fitted.")

        mu = self._chain.marginal_stationary_distribution()

        if history is None:
            if steps == 1:
                return mu
            return np.tile(mu, (steps, 1))

        if isinstance(history, pl.DataFrame):
            hist_states = self._encoder.transform(history)
        else:
            hist_states = np.asarray(history, dtype=np.int32)

        if len(hist_states) < self.k:
            if steps == 1:
                return mu
            return np.tile(mu, (steps, 1))

        win = tuple(int(x) for x in hist_states[-self.k :])
        u = self._chain.encoder.encode_window(win)
        if u < 0:
            if steps == 1:
                return mu
            return np.tile(mu, (steps, 1))

        if steps == 1:
            return self._chain.predict_proba_window(u)
        return self._chain.multi_step_distribution(u, steps=steps)

    def predict(
        self,
        history: pl.DataFrame | Sequence[int] | None = None,
        steps: int = 1,
        method: PredictionMethod = "sample",
    ) -> pl.DataFrame:
        """Predicts the next step(s) as a decoded Polars DataFrame."""
        if self._chain is None or self._encoder is None:
            raise ValueError("SeqSM has not been fitted.")

        probs = self.predict_proba(history=history, steps=steps)
        rng = np.random.default_rng(self.seed)

        if probs.ndim == 1:
            if method == "map":
                chosen_states = [int(np.argmax(probs))]
            else:
                chosen_states = [int(rng.choice(len(probs), p=probs))]
        else:
            chosen_states = []
            for h in range(len(probs)):
                row = probs[h]
                if method == "map":
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
        time_unit: timedelta | str = "1m",
        start_time: datetime | None = None,
        constraints: ScheduleConstraints | None = None,
        n_schedules: int = 1,
        seed: int | None = None,
    ) -> pl.DataFrame:
        """
        Generates realistic synthetic schedules over the specified time window following arc transition probabilities.

        Returns:
            pl.DataFrame with columns:
                ['schedule_id', 'time_step', 'timestamp', 'state_id', <feature_cols...>,
                 'dwell_duration', 'transition_prob', 'arc_valid']
        """
        if self._chain is None or self._encoder is None:
            raise ValueError("SeqSM has not been fitted.")

        # Resolve time_unit delta
        step_delta = parse_duration(time_unit)

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

        gen = ScheduleGenerator(
            self._chain,
            self._encoder,
            min_prob=self.min_prob,
            min_count=self.min_count,
        )

        all_schedule_ids: list[int] = []
        all_time_steps: list[int] = []
        all_timestamps: list[datetime] = []
        all_state_ids: list[int] = []
        all_dwells: list[int] = []
        all_probs: list[float] = []
        all_valids: list[bool] = []

        eff_seed = seed if seed is not None else (self.seed if self.seed is not None else 42)

        for sched_id in range(n_schedules):
            traj = gen.generate(
                n_steps=n_steps,
                init_history=init_arr,
                method=method,
                constraints=constraints,
                seed=eff_seed + sched_id,
            )

            curr_t = 0
            for idx, sid in enumerate(traj.state_ids):
                dwell = traj.dwell_durations[idx]
                prob = traj.transition_probs[idx]
                valid = traj.arc_valid[idx]

                all_schedule_ids.append(sched_id)
                all_time_steps.append(curr_t)
                all_timestamps.append(base_time + curr_t * step_delta)
                all_state_ids.append(sid)
                all_dwells.append(dwell)
                all_probs.append(prob)
                all_valids.append(valid)

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
            }
        )

        # Combine schedule metadata with feature columns
        return pl.concat([base_df, decoded_features], how="horizontal")

    def score_anomalies(
        self,
        df: pl.DataFrame | pl.LazyFrame,
        window_size: int = 5,
        threshold_quantile: float = 0.99,
    ) -> pl.DataFrame:
        """Scores transition anomalies on an input sequence."""
        if self._chain is None or self._encoder is None:
            raise ValueError("SeqSM has not been fitted.")
        states = self._encoder.transform(df)
        return score_anomalies(
            self._chain,
            states,
            window_size=window_size,
            threshold_quantile=threshold_quantile,
        )

    def to_state_machine(
        self,
        min_prob: float | None = None,
        min_count: int | None = None,
        top_k_edges: int | None = None,
    ) -> StateMachine:
        """Constructs a pruned StateMachine graph representation."""
        if self._chain is None or self._encoder is None:
            raise ValueError("SeqSM has not been fitted.")
        p = min_prob if min_prob is not None else self.min_prob
        c = min_count if min_count is not None else self.min_count
        return StateMachine.from_markov(
            self._chain,
            self._encoder,
            min_prob=p,
            min_count=c,
            top_k_edges=top_k_edges,
        )


__all__ = ["SeqSM"]
