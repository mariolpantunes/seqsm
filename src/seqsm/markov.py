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
    conditional_entropy_rate,
    marginal_stationary_distribution,
    model_perplexity,
    stationary_distribution_power_iteration,
)
from seqsm.types import BackoffMethod, CSRMatrix, DwellDistribution, ModeType


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
        self._count_matrix: NDArray[np.float64] | None = None
        self._transition_matrix: NDArray[np.float64] | None = None
        self._companion_matrix: NDArray[np.float64] | None = None
        self._sparse_companion: CSRMatrix | None = None
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
    def count_matrix(self) -> NDArray[np.float64]:
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

    def companion_matrix(self, dense: bool = True) -> NDArray[np.float64] | CSRMatrix:
        """Returns square De Bruijn companion transition matrix P_tilde of shape (N_K, N_K)."""
        if self._companion_matrix is None and self._sparse_companion is None:
            raise ValueError("KOrderMarkovChain has not been fitted.")
        if dense:
            if self._companion_matrix is not None:
                return self._companion_matrix
            assert self._sparse_companion is not None
            return self._sparse_companion.to_dense()
        if self._sparse_companion is not None:
            return self._sparse_companion
        assert self._companion_matrix is not None
        nrows, ncols = self._companion_matrix.shape
        indptr = np.empty(nrows + 1, dtype=np.int32)
        indptr[0] = 0
        indices_list: list[int] = []
        data_list: list[float] = []
        for r in range(nrows):
            nonzero_cols = np.where(self._companion_matrix[r, :] > 0.0)[0]
            indices_list.extend(nonzero_cols)
            data_list.extend(self._companion_matrix[r, nonzero_cols])
            indptr[r + 1] = len(indices_list)
        return CSRMatrix(
            data=np.asarray(data_list, dtype=np.float64),
            indices=np.asarray(indices_list, dtype=np.int32),
            indptr=indptr,
            shape=(nrows, ncols),
        )

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
                eff_states, eff_dwells, eff_episodes = rle_collapse_njit(states_arr, episodes_arr)
            else:
                eff_states = states_arr
                eff_dwells = np.asarray(dwell_times, dtype=np.int32)
                eff_episodes = episodes_arr
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
        counts_arr = count_transitions_njit(src_u, tgt_v, n_windows, self._n_states).astype(np.float64)
        self._count_matrix = counts_arr
        self._transition_matrix = compute_transition_probs_njit(counts_arr, self._smoothing)
        self._build_companion_matrix(src_u)

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

    def _build_companion_matrix(self, src_u: NDArray[np.int32]) -> None:
        """Constructs companion transition operator (dense array or CSRMatrix)."""
        n_windows = self._encoder.n_windows
        assert self._transition_matrix is not None
        if len(src_u) > 0:
            win_counts = np.bincount(src_u, minlength=n_windows).astype(np.float64)
            win_sum = float(np.sum(win_counts))
            win_marginal = (
                win_counts / win_sum if win_sum > 0.0 else np.full(n_windows, 1.0 / max(1, n_windows), dtype=np.float64)
            )
        else:
            win_marginal = np.full(n_windows, 1.0 / max(1, n_windows), dtype=np.float64)

        if n_windows >= self._sparse_threshold:
            csr_indptr = [0]
            csr_indices: list[int] = []
            csr_data: list[float] = []
            for u in range(n_windows):
                win_u = self._encoder.decode_window(u)
                attested: list[tuple[int, float]] = []
                for v in range(self._n_states):
                    p_uv = float(self._transition_matrix[u, v])
                    if p_uv <= 0.0:
                        continue
                    succ_tuple = (*win_u[1:], v)
                    u_prime = self._encoder.encode_window(succ_tuple)
                    if u_prime >= 0:
                        attested.append((u_prime, p_uv))

                if attested:
                    denom = sum(p for _, p in attested)
                    attested.sort(key=lambda x: x[0])
                    for u_prime, p in attested:
                        csr_indices.append(u_prime)
                        csr_data.append(p / denom)
                elif n_windows > 0:
                    for w in range(n_windows):
                        if win_marginal[w] > 0.0:
                            csr_indices.append(w)
                            csr_data.append(float(win_marginal[w]))
                csr_indptr.append(len(csr_indices))

            self._sparse_companion = CSRMatrix(
                data=np.asarray(csr_data, dtype=np.float64),
                indices=np.asarray(csr_indices, dtype=np.int32),
                indptr=np.asarray(csr_indptr, dtype=np.int32),
                shape=(n_windows, n_windows),
            )
            self._companion_matrix = None
        else:
            self._companion_matrix = np.zeros((n_windows, n_windows), dtype=np.float64)
            for u in range(n_windows):
                win_u = self._encoder.decode_window(u)
                attested_dense: list[tuple[int, float]] = []
                for v in range(self._n_states):
                    p_uv = float(self._transition_matrix[u, v])
                    if p_uv <= 0.0:
                        continue
                    succ_tuple = (*win_u[1:], v)
                    u_prime = self._encoder.encode_window(succ_tuple)
                    if u_prime >= 0:
                        attested_dense.append((u_prime, p_uv))

                if attested_dense:
                    denom = sum(p for _, p in attested_dense)
                    for u_prime, p in attested_dense:
                        self._companion_matrix[u, u_prime] += p / denom
                elif n_windows > 0:
                    self._companion_matrix[u, :] = win_marginal

            self._sparse_companion = None

    def partial_fit_from_arrays(
        self,
        states: NDArray[np.int32],
        episode_ids: NDArray[np.int32] | None = None,
        dwell_times: NDArray[np.int32] | None = None,
        n_states: int | None = None,
        decay: float = 1.0,
    ) -> Self:
        """Incrementally fits Markov transition counts with concept-drift count decay gamma in (0, 1]."""
        if self._count_matrix is None or self._transition_matrix is None:
            return self.fit_from_arrays(states, episode_ids=episode_ids, dwell_times=dwell_times, n_states=n_states)

        states_arr = np.asarray(states, dtype=np.int32)
        episodes_arr = (
            np.asarray(episode_ids, dtype=np.int32) if episode_ids is not None else np.empty(0, dtype=np.int32)
        )

        if self._mode == "event":
            if dwell_times is None:
                eff_states, eff_dwells, eff_episodes = rle_collapse_njit(states_arr, episodes_arr)
            else:
                eff_states = states_arr
                eff_dwells = np.asarray(dwell_times, dtype=np.int32)
                eff_episodes = episodes_arr
        else:
            eff_states = states_arr
            eff_episodes = episodes_arr
            eff_dwells = np.empty(0, dtype=np.int32)

        if n_states is not None:
            self._n_states = max(self._n_states, n_states)
        else:
            valid_states = eff_states[eff_states >= 0]
            if len(valid_states) > 0:
                self._n_states = max(self._n_states, int(np.max(valid_states) + 1))

        old_n_windows = self._encoder.n_windows
        old_n_states = self._count_matrix.shape[1]

        src_u, tgt_v, _ = self._encoder.partial_fit_transform(
            eff_states,
            eff_episodes if len(eff_episodes) > 0 else None,
        )

        new_n_windows = self._encoder.n_windows
        new_n_states = self._n_states

        new_counts = np.zeros((new_n_windows, new_n_states), dtype=np.float64)
        new_counts[:old_n_windows, :old_n_states] = float(decay) * self._count_matrix

        delta_counts = count_transitions_njit(src_u, tgt_v, new_n_windows, new_n_states)
        new_counts += delta_counts.astype(np.float64)

        self._count_matrix = new_counts
        self._transition_matrix = compute_transition_probs_njit(self._count_matrix, self._smoothing)
        self._build_companion_matrix(src_u)

        if self._mode == "event" and len(eff_dwells) > 0:
            dwell_buckets: dict[int, list[int]] = {sid: list(dist.samples) for sid, dist in self._dwell_stats.items()}
            for s, d in zip(eff_states, eff_dwells, strict=False):
                if s >= 0:
                    dwell_buckets.setdefault(int(s), []).append(int(d))

            self._dwell_stats.clear()
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
        if self._n_states > 0:
            return np.full(self._n_states, 1.0 / self._n_states, dtype=np.float64)
        return np.empty(0, dtype=np.float64)

    def multi_step_distribution(self, source_id: int, steps: int) -> NDArray[np.float64]:
        """
        Computes probability distribution over atomic states for steps transitions.

        Returns array of shape (|S|,) when steps == 1, or (steps, |S|) when steps > 1.
        """
        if (self._companion_matrix is None and self._sparse_companion is None) or self._transition_matrix is None:
            raise ValueError("KOrderMarkovChain has not been fitted.")
        if steps < 1:
            raise ValueError(f"Steps must be >= 1, got {steps}.")
        if steps == 1:
            return self.predict_proba_window(source_id)

        n_windows = self._encoder.n_windows
        e_u = np.zeros(n_windows, dtype=np.float64)
        if 0 <= source_id < n_windows:
            e_u[source_id] = 1.0
        elif n_windows > 0:
            e_u.fill(1.0 / n_windows)

        dense_comp = (
            self._companion_matrix if self._companion_matrix is not None else self._sparse_companion.to_dense()  # type: ignore[union-attr]
        )

        out = np.zeros((steps, self._n_states), dtype=np.float64)
        out[0] = e_u @ self._transition_matrix
        cur_win = e_u
        for h in range(1, steps):
            cur_win = cur_win @ dense_comp
            out[h] = cur_win @ self._transition_matrix

        return out

    def stationary_distribution(self, tol: float = 1e-10, max_iter: int = 10_000) -> NDArray[np.float64]:
        """Computes stationary distribution pi over K-windows."""
        mat = self._companion_matrix if self._companion_matrix is not None else self._sparse_companion
        if mat is None:
            raise ValueError("KOrderMarkovChain has not been fitted.")
        return stationary_distribution_power_iteration(mat, tol=tol, max_iter=max_iter)

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
        if any(x < 0 or x >= n_states for x in hist_arr):
            raise ValueError(f"history contains state IDs outside valid range [0, {n_states - 1}].")

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
