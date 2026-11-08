"""Phase (regime) conditioned K-order Markov chains: time as a transition signal.

``P^{(K, r)}_{u, v} = P(s_{t+1} = v | window_t = u, regime(t+1) = r)``: one count matrix per regime over a
shared K-window vocabulary, with a pooled fallback for rows a regime has not attested (SPEC section 2.10).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Self

import numpy as np
import polars as pl
from numpy.typing import DTypeLike, NDArray

from seqsm.encoding import KWindowEncoder
from seqsm.markov import KOrderMarkovChain, dwell_distribution_from_samples
from seqsm.timeslots import RegimeSpec, preset_regimes
from seqsm.types import DwellDistribution, ModeType


def collapse_with_regimes(
    states: NDArray[np.int32],
    regimes: NDArray[np.int32],
    episodes: NDArray[np.int32] | None,
    mode: ModeType,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32] | None, NDArray[np.int32]]:
    """Returns (states, regimes, episodes, dwells) as seen by the chain.

    In ``event`` mode runs of equal states (within one episode) collapse to one event whose regime is the
    regime at the run start; ``dwells`` holds the run lengths. In ``clock`` mode nothing collapses.
    """
    states = np.asarray(states, dtype=np.int32)
    regimes = np.asarray(regimes, dtype=np.int32)
    if len(regimes) != len(states):
        raise ValueError(f"regimes length ({len(regimes)}) must match states length ({len(states)}).")
    eps = np.asarray(episodes, dtype=np.int32) if episodes is not None and len(episodes) > 0 else None
    if mode != "event" or len(states) == 0:
        return states, regimes, eps, np.ones(len(states), dtype=np.int32)

    change = np.empty(len(states), dtype=bool)
    change[0] = True
    change[1:] = states[1:] != states[:-1]
    if eps is not None:
        change[1:] |= eps[1:] != eps[:-1]
    starts = np.flatnonzero(change)
    dwells = np.diff(np.append(starts, len(states))).astype(np.int32)
    return states[starts], regimes[starts], (eps[starts] if eps is not None else None), dwells


def _transition_regimes(
    states: NDArray[np.int32],
    regimes: NDArray[np.int32],
    episodes: NDArray[np.int32] | None,
    k: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], int]:
    """Returns (source window ids, targets, target regimes, n_windows) on already collapsed arrays."""
    enc = KWindowEncoder(k)
    src_u, tgt_v, t_idx = enc.fit_transform(states, episodes)
    return src_u, tgt_v, regimes[t_idx + 1] if len(t_idx) else np.empty(0, dtype=np.int32), enc.n_windows


def _window_ids(encoder: KWindowEncoder, windows: NDArray[np.int32], n_states: int) -> NDArray[np.int64]:
    """Vectorized K-window lookup (-1 for windows the encoder never saw)."""
    n_win, k = windows.shape
    if encoder.n_windows == 0 or n_win == 0:
        return np.full(n_win, -1, dtype=np.int64)
    base = n_states + 1
    if float(base) ** k >= 2.0**62:
        return np.array([encoder.encode_window(tuple(int(x) for x in w)) for w in windows], dtype=np.int64)
    powers = base ** np.arange(k - 1, -1, -1, dtype=np.int64)
    known = encoder.k_tuples.astype(np.int64) @ powers
    order = np.argsort(known)
    keys = windows.astype(np.int64) @ powers
    pos = np.searchsorted(known[order], keys)
    pos = np.minimum(pos, len(known) - 1)
    hit = known[order][pos] == keys
    return np.where(hit, order[pos], -1).astype(np.int64)


class PhaseConditionedMarkovChain:
    """K-order Markov chain whose transition kernel depends on a discrete regime (phase class)."""

    def __init__(
        self,
        k: int = 1,
        n_regimes: int = 1,
        *,
        mode: ModeType = "clock",
        smoothing: float = 0.0,
        min_regime_count: int = 1,
        dtype: DTypeLike = np.float64,
    ) -> None:
        if n_regimes < 1:
            raise ValueError(f"n_regimes must be >= 1, got {n_regimes}.")
        if min_regime_count < 1:
            raise ValueError(f"min_regime_count must be >= 1, got {min_regime_count}.")
        self._k: int = k
        self._n_regimes: int = n_regimes
        self._mode: ModeType = mode
        self._smoothing: float = smoothing
        self._min_regime_count: int = min_regime_count
        self._dtype: DTypeLike = dtype
        self._pooled: KOrderMarkovChain | None = None
        self._regime_counts: NDArray[np.float64] | None = None
        self._regime_dwell_samples: dict[int, dict[int, list[int]]] = {}
        self._chains: dict[int, KOrderMarkovChain] = {}

    @property
    def k(self) -> int:
        """Returns the history order."""
        return self._k

    @property
    def n_regimes(self) -> int:
        """Returns the number of regimes R."""
        return self._n_regimes

    @property
    def min_regime_count(self) -> int:
        """Returns the minimum transitions a (regime, window) row needs before it overrides the pooled row."""
        return self._min_regime_count

    @property
    def pooled(self) -> KOrderMarkovChain:
        """Returns the regime-agnostic chain (sum over regimes)."""
        if self._pooled is None:
            raise ValueError("PhaseConditionedMarkovChain has not been fitted.")
        return self._pooled

    @property
    def regime_counts(self) -> NDArray[np.float64]:
        """Returns raw per-regime counts of shape (R, N_K, |S|)."""
        if self._regime_counts is None:
            raise ValueError("PhaseConditionedMarkovChain has not been fitted.")
        return self._regime_counts

    @property
    def regime_dwell_samples(self) -> dict[int, dict[int, list[int]]]:
        """Returns observed dwells as {regime: {state: samples}} (event mode only)."""
        return self._regime_dwell_samples

    def fit_from_arrays(
        self,
        states: NDArray[np.int32],
        regimes: NDArray[np.int32],
        episode_ids: NDArray[np.int32] | None = None,
        n_states: int | None = None,
    ) -> Self:
        """Fits the pooled chain and one count matrix per regime on the same window vocabulary."""
        reg = np.asarray(regimes, dtype=np.int32)
        if len(reg) and (reg.min() < 0 or reg.max() >= self._n_regimes):
            raise ValueError(f"regimes must lie in [0, {self._n_regimes - 1}].")
        pooled = KOrderMarkovChain(
            self._k, mode=self._mode, smoothing=self._smoothing, dtype=self._dtype
        ).fit_from_arrays(states, episode_ids=episode_ids, n_states=n_states)

        eff_s, eff_r, eff_e, eff_d = collapse_with_regimes(states, reg, episode_ids, self._mode)
        src_u, tgt_v, tgt_r, n_win = _transition_regimes(eff_s, eff_r, eff_e, self._k)
        counts = np.zeros((self._n_regimes, pooled.n_windows, pooled.n_states), dtype=np.float64)
        if len(src_u) and n_win == pooled.n_windows:
            np.add.at(counts, (tgt_r, src_u, tgt_v), 1.0)
        self._pooled = pooled
        self._regime_counts = counts
        self._chains = {}

        self._regime_dwell_samples = {}
        if self._mode == "event":
            for s, r, d in zip(eff_s.tolist(), eff_r.tolist(), eff_d.tolist(), strict=True):
                if s >= 0:
                    self._regime_dwell_samples.setdefault(r, {}).setdefault(s, []).append(d)
        return self

    def set_fitted(
        self,
        pooled: KOrderMarkovChain,
        regime_counts: NDArray[np.float64],
        dwell_samples: dict[int, dict[int, list[int]]] | None = None,
    ) -> Self:
        """Restores a fitted state (used by deserialization)."""
        if regime_counts.shape[0] != self._n_regimes:
            raise ValueError(f"regime_counts has {regime_counts.shape[0]} regimes, expected {self._n_regimes}.")
        self._pooled = pooled
        self._regime_counts = np.asarray(regime_counts, dtype=np.float64)
        self._regime_dwell_samples = dwell_samples or {}
        self._chains = {}
        return self

    def _check_regime(self, regime: int) -> int:
        r = int(regime)
        if not 0 <= r < self._n_regimes:
            raise ValueError(f"regime {r} outside [0, {self._n_regimes - 1}].")
        return r

    def regime_support(self, regime: int) -> NDArray[np.bool_]:
        """Returns a boolean mask over windows whose row is regime-specific (not the pooled fallback)."""
        r = self._check_regime(regime)
        return np.sum(self.regime_counts[r], axis=1) >= self._min_regime_count

    def chain_for(self, regime: int) -> KOrderMarkovChain:
        """Returns the chain of one regime (pooled rows where the regime lacks support)."""
        r = self._check_regime(regime)
        cached = self._chains.get(r)
        if cached is not None:
            return cached
        pooled = self.pooled
        support = self.regime_support(r)
        counts = np.where(support[:, None], self.regime_counts[r], pooled.count_matrix)
        dwells: dict[int, DwellDistribution] = {}
        if self._mode == "event":
            local = self._regime_dwell_samples.get(r, {})
            for sid, dist in pooled.dwell_stats.items():
                samples = local.get(sid, [])
                dwells[sid] = (
                    dwell_distribution_from_samples(sid, samples) if len(samples) >= self._min_regime_count else dist
                )
        chain = pooled.with_counts(counts, dwells if self._mode == "event" else None)
        self._chains[r] = chain
        return chain

    def regime_chains(self) -> list[KOrderMarkovChain]:
        """Returns the chains of all regimes in id order."""
        return [self.chain_for(r) for r in range(self._n_regimes)]

    def predict_proba_window(self, source_id: int, regime: int) -> NDArray[np.float64]:
        """Returns the next-state distribution of window ``source_id`` under ``regime``."""
        return self.chain_for(regime).predict_proba_window(source_id)

    def multi_step_distribution(self, source_id: int, regimes: Sequence[int]) -> NDArray[np.float64]:
        """Propagates a window distribution through an inhomogeneous kernel sequence.

        ``regimes[h]`` is the regime of the h-th predicted step. Returns shape (len(regimes), |S|).
        """
        steps = len(regimes)
        if steps < 1:
            raise ValueError("regimes must contain at least one entry.")
        n_windows = self.pooled.n_windows
        cur = np.zeros(n_windows, dtype=np.float64)
        if 0 <= source_id < n_windows:
            cur[source_id] = 1.0
        elif n_windows > 0:
            cur.fill(1.0 / n_windows)
        out = np.zeros((steps, self.pooled.n_states), dtype=np.float64)
        for h, r in enumerate(regimes):
            chain = self.chain_for(r)
            out[h] = cur @ chain.transition_matrix
            if h + 1 < steps:
                cur = cur @ np.asarray(chain.companion_matrix(dense=True))
        return out

    def nll_sum(
        self,
        states: NDArray[np.int32],
        regimes: NDArray[np.int32] | None = None,
        episode_ids: NDArray[np.int32] | None = None,
        floor: float = 1e-6,
    ) -> tuple[float, int]:
        """Returns (summed NLL in nats, evaluated transitions); windows never seen in fit are skipped.

        With ``regimes=None`` the pooled (regime-agnostic) kernel scores the sequence.
        """
        pooled = self.pooled
        arr = np.asarray(states, dtype=np.int32)
        reg = np.asarray(regimes, dtype=np.int32) if regimes is not None else np.zeros(len(arr), dtype=np.int32)
        eps = np.asarray(episode_ids, dtype=np.int32) if episode_ids is not None else None
        eff_s, eff_r, eff_e, _ = collapse_with_regimes(arr, reg, eps, self._mode)
        enc = KWindowEncoder(self._k)
        _, tgt_v, t_idx = enc.fit_transform(eff_s, eff_e)
        if len(tgt_v) == 0:
            return 0.0, 0
        offsets = np.arange(-self._k + 1, 1)
        windows = eff_s[t_idx[:, None] + offsets[None, :]]
        src = _window_ids(pooled.encoder, windows, pooled.n_states)
        ok = src >= 0
        if not ok.any():
            return 0.0, 0
        src, tgt = src[ok], tgt_v[ok]
        if regimes is None:
            probs = pooled.transition_matrix[src, tgt]
        else:
            probs = np.empty(len(src), dtype=np.float64)
            tr = eff_r[t_idx[ok] + 1]
            for r in np.unique(tr):
                m = tr == r
                probs[m] = self.chain_for(int(r)).transition_matrix[src[m], tgt[m]]
        return float(-np.sum(np.log(np.maximum(probs, floor)))), len(src)

    def average_nll(
        self,
        states: NDArray[np.int32],
        regimes: NDArray[np.int32] | None = None,
        episode_ids: NDArray[np.int32] | None = None,
        floor: float = 1e-6,
    ) -> float:
        """Mean negative log-likelihood (nats) of a held-out sequence (NaN if nothing can be scored)."""
        total, count = self.nll_sum(states, regimes, episode_ids, floor)
        return total / count if count else float("nan")

    def __repr__(self) -> str:
        fitted = self._pooled is not None
        return f"PhaseConditionedMarkovChain(k={self._k}, n_regimes={self._n_regimes}, mode={self._mode!r}, fitted={fitted})"


def regime_description_length(
    states: NDArray[np.int32],
    regimes: NDArray[np.int32],
    n_regimes: int,
    *,
    k: int = 1,
    mode: ModeType = "clock",
    episode_ids: NDArray[np.int32] | None = None,
) -> tuple[float, float, int, int]:
    """Returns (BIC, log-likelihood, free parameters, transitions) of the regime-conditioned ML chain."""
    eff_s, eff_r, eff_e, _ = collapse_with_regimes(states, regimes, episode_ids, mode)
    src_u, tgt_v, tgt_r, n_win = _transition_regimes(eff_s, eff_r, eff_e, k)
    n = len(src_u)
    if n == 0:
        return float("inf"), 0.0, 0, 0
    n_states = int(max(eff_s.max(), 0)) + 1
    row_key = tgt_r.astype(np.int64) * max(n_win, 1) + src_u.astype(np.int64)
    cell_key = row_key * n_states + tgt_v.astype(np.int64)
    cells, cell_counts = np.unique(cell_key, return_counts=True)
    rows_of_cells = cells // n_states
    rows, inverse = np.unique(rows_of_cells, return_inverse=True)
    row_totals = np.bincount(inverse, weights=cell_counts).astype(np.float64)
    loglik = float(np.sum(cell_counts * np.log(cell_counts / row_totals[inverse])))
    n_params = int(len(cells) - len(rows))
    bic = -2.0 * loglik + n_params * float(np.log(n))
    return bic, loglik, n_params, n


def _fold_ids(episodes: NDArray[np.int32] | None, n: int, folds: int) -> NDArray[np.int32]:
    """Contiguous fold ids: whole episodes per fold when several exist, else row blocks."""
    if episodes is not None and len(np.unique(episodes)) > 1:
        uniq = np.unique(episodes)
        chunks = np.array_split(uniq, min(folds, len(uniq)))
        lookup = {int(e): i for i, ch in enumerate(chunks) for e in ch}
        return np.array([lookup[int(e)] for e in episodes], dtype=np.int32)
    return np.minimum(np.arange(n) * folds // max(n, 1), folds - 1).astype(np.int32)


def cross_validated_nll(
    states: NDArray[np.int32],
    regimes: NDArray[np.int32],
    n_regimes: int,
    *,
    k: int = 1,
    mode: ModeType = "clock",
    smoothing: float = 0.1,
    min_regime_count: int = 1,
    episode_ids: NDArray[np.int32] | None = None,
    folds: int = 4,
) -> float:
    """Blocked K-fold held-out NLL (nats per transition) of a regime definition.

    Folds are contiguous (whole episodes, or row blocks for a single episode), so a regime is judged on
    unseen days rather than on neighbours of its own training rows.
    """
    n = len(states)
    fold = _fold_ids(episode_ids, n, folds)
    eps = np.asarray(episode_ids, dtype=np.int32) if episode_ids is not None else np.zeros(n, dtype=np.int32)
    n_states = int(states.max()) + 1 if n else 0
    total, count = 0.0, 0
    for f in np.unique(fold):
        train, test = fold != f, fold == f
        t_states = states[train]
        change = np.zeros(len(t_states), dtype=bool)
        if len(t_states) > 1:
            change[1:] = (fold[train][1:] != fold[train][:-1]) | (eps[train][1:] != eps[train][:-1])
        t_eps = np.cumsum(change).astype(np.int32)
        model = PhaseConditionedMarkovChain(
            k, n_regimes, mode=mode, smoothing=smoothing, min_regime_count=min_regime_count
        ).fit_from_arrays(t_states, regimes[train], episode_ids=t_eps, n_states=n_states)
        tot, cnt = model.nll_sum(states[test], regimes[test], eps[test])
        total += tot
        count += cnt
    return total / count if count else float("inf")


def select_regimes(
    states: NDArray[np.int32],
    timestamps_us: NDArray[Any],
    candidates: Sequence[str | RegimeSpec],
    *,
    k: int = 1,
    mode: ModeType = "clock",
    episode_ids: NDArray[np.int32] | None = None,
    criterion: Literal["bic", "cv"] = "bic",
    folds: int = 4,
    smoothing: float = 0.1,
    min_regime_count: int = 1,
) -> tuple[RegimeSpec, pl.DataFrame]:
    """Picks the regime definition with the best criterion.

    ``"bic"`` is the minimum description length of the maximum-likelihood chain; ``"cv"`` is the blocked
    cross-validated held-out NLL, which is more conservative on few days of data.
    ``"none"`` (one regime) is always scored, so a time signal is only adopted when it pays for itself.
    Returns the winning spec and a table sorted by the criterion.
    """
    if criterion not in ("bic", "cv"):
        raise ValueError(f"criterion must be 'bic' or 'cv', got {criterion!r}.")
    specs: list[RegimeSpec] = [preset_regimes("none")]
    for cand in candidates:
        spec = preset_regimes(cand) if isinstance(cand, str) else cand
        if all(spec.name != other.name for other in specs):
            specs.append(spec)
    rows: list[dict[str, object]] = []
    for spec in specs:
        reg = spec(timestamps_us)
        bic, ll, n_params, n = regime_description_length(
            states, reg, spec.n_regimes, k=k, mode=mode, episode_ids=episode_ids
        )
        cv = (
            cross_validated_nll(
                states,
                reg,
                spec.n_regimes,
                k=k,
                mode=mode,
                smoothing=smoothing,
                min_regime_count=min_regime_count,
                episode_ids=episode_ids,
                folds=folds,
            )
            if criterion == "cv"
            else float("nan")
        )
        rows.append(
            {
                "regime": spec.name,
                "n_regimes": spec.n_regimes,
                "bic": bic,
                "cv_nll": cv,
                "loglik": ll,
                "n_params": n_params,
                "n": n,
            }
        )
    table = pl.DataFrame(rows).sort("cv_nll" if criterion == "cv" else "bic", "n_params")
    best_name = str(table["regime"][0])
    best = next(sp for sp in specs if sp.name == best_name)
    return best, table


__all__ = [
    "PhaseConditionedMarkovChain",
    "collapse_with_regimes",
    "cross_validated_nll",
    "regime_description_length",
    "select_regimes",
]
