"""K-order and Variable-order Markov Chain implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np
from numpy.typing import DTypeLike, NDArray

from seqsm.encoding import KWindowEncoder
from seqsm.kernels import (
    compute_transition_probs_njit,
    count_transitions_njit,
    rle_collapse_njit,
)
from seqsm.linalg import (
    companion_matrix_power,
    conditional_entropy_rate,
    marginal_stationary_distribution,
    model_perplexity,
    stationary_distribution_power_iteration,
)
from seqsm.types import BackoffMethod, DwellDistribution, ModeType


class KOrderMarkovChain:
    """K-order Markov Chain over aligned discrete sequences with De Bruijn companion operator."""

    def __init__(
        self,
        k: int = 1,
        *,
        mode: ModeType = "clock",
        smoothing: float = 0.0,
        sparse_threshold: int = 8192,
        dtype: DTypeLike = np.float64,
    ) -> None:
        if k < 1:
            raise ValueError(f"Markov history order k must be >= 1, got {k}.")
        if smoothing < 0.0:
            raise ValueError(f"Smoothing parameter alpha must be >= 0.0, got {smoothing}.")

        self._k: int = k
        self._mode: ModeType = mode
        self._smoothing: float = smoothing
        self._sparse_threshold: int = sparse_threshold
        self._dtype: DTypeLike = dtype

        self._encoder: KWindowEncoder = KWindowEncoder(k=k)
        self._n_states: int = 0
        self._count_matrix: NDArray[np.int64] | None = None
        self._transition_matrix: NDArray[np.float64] | None = None
        self._companion_matrix: NDArray[np.float64] | None = None
        self._dwell_stats: dict[int, DwellDistribution] = {}

    @property
    def k(self) -> int:
        """Returns the Markov conditioning history order."""
        return self._k

    @property
    def mode(self) -> ModeType:
        """Returns the execution mode ('clock' or 'event')."""
        return self._mode

    @property
    def n_states(self) -> int:
        """Returns total number of atomic states."""
        return self._n_states

    @property
    def n_windows(self) -> int:
        """Returns number of observed K-history windows."""
        return self._encoder.n_windows

    @property
    def encoder(self) -> KWindowEncoder:
        """Returns the underlying KWindowEncoder."""
        return self._encoder

    @property
    def count_matrix(self) -> NDArray[np.int64]:
        """Returns empirical transition count matrix C of shape (N_K, |S|)."""
        if self._count_matrix is None:
            raise ValueError("KOrderMarkovChain has not been fitted.")
        return self._count_matrix

    @property
    def transition_matrix(self) -> NDArray[np.float64]:
        """Returns row-stochastic transition probability matrix P of shape (N_K, |S|)."""
        if self._transition_matrix is None:
            raise ValueError("KOrderMarkovChain has not been fitted.")
        return self._transition_matrix

    @property
    def dwell_stats(self) -> dict[int, DwellDistribution]:
        """Returns empirical state dwell distributions."""
        return self._dwell_stats

    def companion_matrix(self, dense: bool = True) -> NDArray[np.float64]:
        """Returns square De Bruijn companion transition matrix P_tilde of shape (N_K, N_K)."""
        _ = dense
        if self._companion_matrix is None:
            raise ValueError("KOrderMarkovChain has not been fitted.")
        return self._companion_matrix

    def fit_from_arrays(
        self,
        states: NDArray[np.int32],
        episode_ids: NDArray[np.int32] | None = None,
        dwell_times: NDArray[np.int32] | None = None,
        n_states: int | None = None,
    ) -> Self:
        """Fits Markov transition counts and companion operators from state and episode arrays."""
        states_arr = np.asarray(states, dtype=np.int32)
        episodes_arr = (
            np.asarray(episode_ids, dtype=np.int32) if episode_ids is not None else np.empty(0, dtype=np.int32)
        )

        # In event mode, collapse identical consecutive states
        if self._mode == "event":
            if dwell_times is None:
                eff_states, eff_dwells = rle_collapse_njit(states_arr, episodes_arr)
            else:
                eff_states = states_arr
                eff_dwells = np.asarray(dwell_times, dtype=np.int32)
            eff_episodes = np.empty(0, dtype=np.int32)  # Boundaries already respected in collapse
        else:
            eff_states = states_arr
            eff_episodes = episodes_arr
            eff_dwells = np.empty(0, dtype=np.int32)

        # Infer state space cardinality
        if n_states is not None:
            self._n_states = n_states
        else:
            valid_states = eff_states[eff_states >= 0]
            self._n_states = int(np.max(valid_states) + 1) if len(valid_states) > 0 else 0

        # Fit K-window encoder and extract transitions
        src_u, tgt_v, _ = self._encoder.fit_transform(
            eff_states,
            eff_episodes if len(eff_episodes) > 0 else None,
        )

        n_windows = self._encoder.n_windows
        self._count_matrix = count_transitions_njit(src_u, tgt_v, n_windows, self._n_states)
        self._transition_matrix = compute_transition_probs_njit(self._count_matrix, self._smoothing)

        # Build square companion matrix (N_K x N_K)
        self._companion_matrix = np.zeros((n_windows, n_windows), dtype=np.float64)
        for u in range(n_windows):
            win_u = self._encoder.decode_window(u)
            for v in range(self._n_states):
                p_uv = self._transition_matrix[u, v]
                if p_uv <= 0.0:
                    continue
                succ_tuple = (*win_u[1:], v)
                u_prime = self._encoder.encode_window(succ_tuple)
                if u_prime >= 0:
                    self._companion_matrix[u, u_prime] += p_uv
                else:
                    self._companion_matrix[u, u] += p_uv

            # Maintain row-stochasticity
            row_sum = np.sum(self._companion_matrix[u, :])
            if row_sum > 0.0:
                self._companion_matrix[u, :] /= row_sum
            elif n_windows > 0:
                self._companion_matrix[u, :] = 1.0 / n_windows

        # Compute dwell stats if in event mode
        if self._mode == "event" and len(eff_dwells) > 0:
            self._dwell_stats.clear()
            dwell_buckets: dict[int, list[int]] = {}
            for s, d in zip(eff_states, eff_dwells, strict=False):
                if s >= 0:
                    dwell_buckets.setdefault(int(s), []).append(int(d))

            for sid, dwells in dwell_buckets.items():
                dwell_arr = np.array(dwells, dtype=np.float64)
                self._dwell_stats[sid] = DwellDistribution(
                    state_id=sid,
                    count=len(dwells),
                    mean=float(np.mean(dwell_arr)),
                    std=float(np.std(dwell_arr)),
                    median=float(np.median(dwell_arr)),
                    min_dwell=int(np.min(dwell_arr)),
                    max_dwell=int(np.max(dwell_arr)),
                    samples=tuple(dwells),
                )

        return self

    def predict_proba_window(self, source_id: int) -> NDArray[np.float64]:
        """Returns the next atomic state probability distribution of shape (|S|,) given window u."""
        if self._transition_matrix is None:
            raise ValueError("KOrderMarkovChain has not been fitted.")
        if 0 <= source_id < self._encoder.n_windows:
            return self._transition_matrix[source_id]
        # Unknown window: fallback to marginal or uniform
        if self._n_states > 0:
            return np.full(self._n_states, 1.0 / self._n_states, dtype=np.float64)
        return np.empty(0, dtype=np.float64)

    def multi_step_distribution(self, source_id: int, steps: int) -> NDArray[np.float64]:
        """
        Computes probability distribution over atomic states after `steps` transitions.

        Uses OpenBLAS matrix power of companion matrix P_tilde^(steps-1) * P.
        """
        if self._companion_matrix is None or self._transition_matrix is None:
            raise ValueError("KOrderMarkovChain has not been fitted.")
        if steps < 1:
            raise ValueError(f"Steps must be >= 1, got {steps}.")
        if steps == 1:
            return self.predict_proba_window(source_id)

        n_windows = self._encoder.n_windows
        e_u = np.zeros(n_windows, dtype=np.float64)
        if 0 <= source_id < n_windows:
            e_u[source_id] = 1.0
        else:
            e_u.fill(1.0 / n_windows)

        comp_pow = companion_matrix_power(self._companion_matrix, steps - 1)
        win_dist = e_u @ comp_pow
        state_dist = win_dist @ self._transition_matrix
        return state_dist

    def stationary_distribution(self, tol: float = 1e-10, max_iter: int = 10_000) -> NDArray[np.float64]:
        """Computes stationary distribution pi over K-windows."""
        if self._companion_matrix is None:
            raise ValueError("KOrderMarkovChain has not been fitted.")
        return stationary_distribution_power_iteration(self._companion_matrix, tol=tol, max_iter=max_iter)

    def marginal_stationary_distribution(self) -> NDArray[np.float64]:
        """Computes marginal stationary distribution mu over atomic states."""
        pi = self.stationary_distribution()
        return marginal_stationary_distribution(pi, self._encoder.k_tuples, self._n_states)

    def entropy_rate(self) -> float:
        """Computes conditional entropy rate H(S_{t+1} | W_t) in bits."""
        pi = self.stationary_distribution()
        if self._transition_matrix is None:
            raise ValueError("KOrderMarkovChain has not been fitted.")
        return conditional_entropy_rate(pi, self._transition_matrix)

    def perplexity(self) -> float:
        """Computes perplexity = 2^(H(S))."""
        return model_perplexity(self.entropy_rate())


class VariableOrderMarkovChain:
    """Variable-order Markov chain combining models of order 1..max_k via backoff or interpolation."""

    def __init__(
        self,
        max_k: int = 2,
        *,
        mode: ModeType = "clock",
        backoff: BackoffMethod = "witten_bell",
        weights: Sequence[float] | None = None,
        smoothing: float = 1e-6,
    ) -> None:
        if max_k < 1:
            raise ValueError(f"max_k must be >= 1, got {max_k}.")
        self._max_k: int = max_k
        self._mode: ModeType = mode
        self._backoff: BackoffMethod = backoff
        self._smoothing: float = smoothing
        self._weights: tuple[float, ...] | None = tuple(weights) if weights is not None else None

        self._chains: list[KOrderMarkovChain] = [
            KOrderMarkovChain(k=order, mode=mode, smoothing=smoothing) for order in range(1, max_k + 1)
        ]

    @property
    def max_k(self) -> int:
        """Returns the maximum Markov order."""
        return self._max_k

    @property
    def chains(self) -> list[KOrderMarkovChain]:
        """Returns list of fitted chains ordered from k=1 to max_k."""
        return self._chains

    def fit_from_arrays(
        self,
        states: NDArray[np.int32],
        episode_ids: NDArray[np.int32] | None = None,
        dwell_times: NDArray[np.int32] | None = None,
        n_states: int | None = None,
    ) -> Self:
        """Fits all sub-chains of orders 1..max_k."""
        for chain in self._chains:
            chain.fit_from_arrays(states, episode_ids, dwell_times, n_states=n_states)
        return self

    def predict_proba(self, history: Sequence[int] | NDArray[np.int32]) -> NDArray[np.float64]:
        """Combines predictions across orders according to the backoff / interpolation policy."""
        hist_arr = tuple(int(x) for x in history)
        n_states = self._chains[0].n_states

        if self._backoff == "linear":
            weights = self._weights
            if weights is None:
                weights = tuple(1.0 / self._max_k for _ in range(self._max_k))
            total_prob = np.zeros(n_states, dtype=np.float64)
            for idx, chain in enumerate(self._chains):
                order = chain.k
                if len(hist_arr) >= order:
                    win = hist_arr[-order:]
                    u = chain.encoder.encode_window(win)
                    p = chain.predict_proba_window(u)
                else:
                    p = chain.marginal_stationary_distribution()
                total_prob += weights[idx] * p
            norm = np.sum(total_prob)
            return total_prob / norm if norm > 0.0 else total_prob

        # Witten-Bell or tuned backoff: try highest order first
        for chain in reversed(self._chains):
            order = chain.k
            if len(hist_arr) >= order:
                win = hist_arr[-order:]
                u = chain.encoder.encode_window(win)
                if u >= 0:
                    counts = chain.count_matrix[u]
                    if np.sum(counts) > 0:
                        return chain.predict_proba_window(u)

        # Fallback to order 1 marginal
        return self._chains[0].marginal_stationary_distribution()


__all__ = ["KOrderMarkovChain", "VariableOrderMarkovChain"]
