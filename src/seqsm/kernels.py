"""Numba-accelerated kernels for transition counting, valid-arc rollouts, and Viterbi path decoding."""

from __future__ import annotations

import numba as nb
import numpy as np
from numpy.typing import NDArray


@nb.njit(nb.int64[:, :](nb.int32[:], nb.int32[:], nb.int64, nb.int64), nogil=True, cache=True)
def count_transitions_njit(
    sources: NDArray[np.int32],
    targets: NDArray[np.int32],
    n_sources: int,
    n_targets: int,
) -> NDArray[np.int64]:
    """Accumulates empirical transition counts from source windows to target states."""
    counts = np.zeros((n_sources, n_targets), dtype=np.int64)
    n = len(sources)
    for i in range(n):
        u = sources[i]
        v = targets[i]
        if u >= 0 and v >= 0 and u < n_sources and v < n_targets:
            counts[u, v] += 1
    return counts


@nb.njit(nb.float64[:, :](nb.int64[:, :], nb.float64), nogil=True, cache=True)
def compute_transition_probs_njit(
    counts: NDArray[np.int64],
    alpha: float,
) -> NDArray[np.float64]:
    """Computes row-stochastic transition probabilities with optional Lidstone smoothing."""
    n_sources, n_targets = counts.shape
    probs = np.zeros((n_sources, n_targets), dtype=np.float64)
    for u in range(n_sources):
        row_sum = 0
        for v in range(n_targets):
            row_sum += counts[u, v]
        denom = float(row_sum) + alpha * float(n_targets)
        if denom > 0.0:
            for v in range(n_targets):
                probs[u, v] = (float(counts[u, v]) + alpha) / denom
        else:
            uniform = 1.0 / float(n_targets)
            for v in range(n_targets):
                probs[u, v] = uniform
    return probs


@nb.njit(nb.types.Tuple((nb.int32[:], nb.int32[:]))(nb.int32[:], nb.int32[:]), nogil=True, cache=True)
def rle_collapse_njit(
    states: NDArray[np.int32],
    episodes: NDArray[np.int32],
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Collapses consecutive identical states into distinct event states and dwell counts."""
    n = len(states)
    if n == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    # First pass: count runs
    n_runs = 1
    for i in range(1, n):
        if states[i] != states[i - 1] or (len(episodes) > 0 and episodes[i] != episodes[i - 1]):
            n_runs += 1

    out_states = np.empty(n_runs, dtype=np.int32)
    out_dwells = np.empty(n_runs, dtype=np.int32)

    run_idx = 0
    cur_state = states[0]
    cur_ep = episodes[0] if len(episodes) > 0 else 0
    cur_count = 1

    for i in range(1, n):
        ep = episodes[i] if len(episodes) > 0 else 0
        if states[i] != cur_state or ep != cur_ep:
            out_states[run_idx] = cur_state
            out_dwells[run_idx] = cur_count
            run_idx += 1
            cur_state = states[i]
            cur_ep = ep
            cur_count = 1
        else:
            cur_count += 1

    out_states[run_idx] = cur_state
    out_dwells[run_idx] = cur_count
    return out_states, out_dwells


@nb.njit(
    nb.types.Tuple((nb.int32[:], nb.int32[:], nb.float64[:], nb.float64[:]))(
        nb.float64[:, :], nb.int64[:, :], nb.float64, nb.int64
    ),
    nogil=True,
    cache=True,
)
def build_valid_arc_csr_njit(
    probs: NDArray[np.float64],
    counts: NDArray[np.int64],
    min_prob: float,
    min_count: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float64], NDArray[np.float64]]:
    """Builds a CSR representation of valid-arc transition probabilities and cumulative distribution."""
    n_sources, n_targets = probs.shape
    indptr = np.zeros(n_sources + 1, dtype=np.int32)

    # First pass: count valid arcs per source
    total_arcs = 0
    for u in range(n_sources):
        valid_in_row = 0
        for v in range(n_targets):
            if probs[u, v] >= min_prob and counts[u, v] >= min_count:
                valid_in_row += 1
        if valid_in_row == 0:
            valid_in_row = 1  # Fallback to argmax arc
        indptr[u + 1] = indptr[u] + valid_in_row
        total_arcs += valid_in_row

    indices = np.zeros(total_arcs, dtype=np.int32)
    filtered_probs = np.zeros(total_arcs, dtype=np.float64)
    cumprobs = np.zeros(total_arcs, dtype=np.float64)

    arc_idx = 0
    for u in range(n_sources):
        row_start = arc_idx
        p_sum = 0.0
        n_added = 0
        for v in range(n_targets):
            if probs[u, v] >= min_prob and counts[u, v] >= min_count:
                indices[arc_idx] = v
                filtered_probs[arc_idx] = probs[u, v]
                p_sum += probs[u, v]
                arc_idx += 1
                n_added += 1

        if n_added == 0:
            # Fallback: argmax unpruned
            best_v = 0
            best_p = probs[u, 0]
            for v in range(1, n_targets):
                if probs[u, v] > best_p:
                    best_p = probs[u, v]
                    best_v = v
            indices[arc_idx] = best_v
            filtered_probs[arc_idx] = 1.0
            cumprobs[arc_idx] = 1.0
            arc_idx += 1
        else:
            # Renormalize
            inv_sum = 1.0 / p_sum if p_sum > 0.0 else 1.0
            running_cdf = 0.0
            for j in range(row_start, arc_idx):
                filtered_probs[j] *= inv_sum
                running_cdf += filtered_probs[j]
                cumprobs[j] = running_cdf
            cumprobs[arc_idx - 1] = 1.0

    return indptr, indices, filtered_probs, cumprobs


@nb.njit(nb.int32[:](nb.float64[:, :], nb.float64[:], nb.int64), nogil=True, cache=True)
def viterbi_decode_path_njit(
    log_companion: NDArray[np.float64],
    init_log_probs: NDArray[np.float64],
    horizon: int,
) -> NDArray[np.int32]:
    """Dynamic programming Viterbi path decoding over square companion transition log-probabilities."""
    n_states = len(init_log_probs)
    dp = np.copy(init_log_probs)
    backpointers = np.zeros((horizon, n_states), dtype=np.int32)

    for h in range(1, horizon):
        next_dp = np.full(n_states, -1e30, dtype=np.float64)
        for u_prev in range(n_states):
            prev_val = dp[u_prev]
            if prev_val <= -1e25:
                continue
            for u_next in range(n_states):
                score = prev_val + log_companion[u_prev, u_next]
                if score > next_dp[u_next]:
                    next_dp[u_next] = score
                    backpointers[h, u_next] = u_prev
        dp = next_dp

    best_final = 0
    best_val = dp[0]
    for i in range(1, n_states):
        if dp[i] > best_val:
            best_val = dp[i]
            best_final = i

    path = np.zeros(horizon, dtype=np.int32)
    path[horizon - 1] = best_final
    curr = best_final
    for h in range(horizon - 1, 0, -1):
        curr = backpointers[h, curr]
        path[h - 1] = curr

    return path
