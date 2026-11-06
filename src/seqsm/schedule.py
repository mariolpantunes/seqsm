"""Schedule generation engine executing valid-arc rollouts in clock and event modes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from seqsm.encoding import StateEncoder
from seqsm.kernels import build_valid_arc_csr_njit
from seqsm.markov import KOrderMarkovChain
from seqsm.types import ScheduleConstraints, ScheduleMethod, ScheduleTrajectory


class ScheduleGenerator:
    """Core generator executing valid-arc schedule rollouts in clock and semi-Markov event modes."""

    def __init__(
        self,
        chain: KOrderMarkovChain,
        encoder: StateEncoder,
        min_prob: float = 0.0,
        min_count: int = 1,
    ) -> None:
        self._chain: KOrderMarkovChain = chain
        self._encoder: StateEncoder = encoder
        self._min_prob: float = min_prob
        self._min_count: int = min_count

        # Build valid-arc CSR structure
        probs = chain.transition_matrix
        counts = chain.count_matrix
        self._indptr, self._indices, self._filtered_probs, self._cumprobs = build_valid_arc_csr_njit(
            probs, counts, min_prob, min_count
        )

    def generate(
        self,
        n_steps: int,
        init_history: NDArray[np.int32] | None = None,
        method: ScheduleMethod = "sample",
        constraints: ScheduleConstraints | None = None,
        seed: int = 42,
    ) -> ScheduleTrajectory:
        """
        Executes a valid-arc schedule rollout for n_steps.

        In clock mode, yields exactly n_steps transitions.
        In event mode, simulates events with empirical dwell sampling until sum(dwells) >= n_steps.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}.")

        rng = np.random.default_rng(seed)
        k_enc = self._chain.encoder
        k = self._chain.k

        # 1. Resolve initial state / history
        if init_history is not None:
            hist_arr = np.asarray(init_history, dtype=np.int32)
            if len(hist_arr) < k:
                raise ValueError(f"init_history must contain at least k={k} states, got {len(hist_arr)}.")
            win = tuple(int(x) for x in hist_arr[-k:])
            curr_u = max(k_enc.encode_window(win), 0)
            curr_window = list(win)
        else:
            # Sample initial window from stationary distribution
            pi = self._chain.stationary_distribution()
            if method == "map":
                curr_u = int(np.argmax(pi))
            else:
                curr_u = int(rng.choice(len(pi), p=pi))
            curr_window = list(k_enc.decode_window(curr_u))

        out_states: list[int] = []
        out_dwells: list[int] = []
        out_probs: list[float] = []
        out_valids: list[bool] = []

        total_steps = 0
        dwell_stats = self._chain.dwell_stats

        while total_steps < n_steps:
            # Candidate arcs for curr_u
            start_arc = self._indptr[curr_u]
            end_arc = self._indptr[curr_u + 1]
            n_arcs = end_arc - start_arc

            if n_arcs == 0:
                # No arcs: fallback to unpruned argmax or reset
                next_v = int(np.argmax(self._chain.transition_matrix[curr_u]))
                next_p = float(self._chain.transition_matrix[curr_u, next_v])
                is_valid = False
            elif method == "map":
                # Argmax of valid arcs
                best_rel = int(np.argmax(self._filtered_probs[start_arc:end_arc]))
                next_v = int(self._indices[start_arc + best_rel])
                next_p = float(self._filtered_probs[start_arc + best_rel])
                is_valid = True
            else:
                # Sample from CDF
                rand_val = rng.uniform(0.0, 1.0)
                sub_cum = self._cumprobs[start_arc:end_arc]
                rel_idx = int(np.searchsorted(sub_cum, rand_val))
                if rel_idx >= n_arcs:
                    rel_idx = n_arcs - 1
                next_v = int(self._indices[start_arc + rel_idx])
                next_p = float(self._filtered_probs[start_arc + rel_idx])
                is_valid = True

            # Dwell duration resolution
            if self._chain.mode == "event":
                # Draw dwell time from empirical distribution
                if next_v in dwell_stats and len(dwell_stats[next_v].samples) > 0:
                    samples = dwell_stats[next_v].samples
                    dwell = int(rng.choice(samples))
                else:
                    dwell = 1
                # Enforce bounds
                dwell = max(1, dwell)
                total_steps += dwell
            else:
                dwell = 1
                total_steps += 1

            out_states.append(next_v)
            out_dwells.append(dwell)
            out_probs.append(next_p)
            out_valids.append(is_valid)

            # Update window
            curr_window.append(next_v)
            if len(curr_window) > k:
                curr_window.pop(0)

            # Look up next_u
            next_u = k_enc.encode_window(tuple(curr_window))
            if next_u >= 0:
                curr_u = next_u
            else:
                # If unseen transition occurs, find best matching window
                curr_u = 0

        return ScheduleTrajectory(
            state_ids=tuple(out_states),
            dwell_durations=tuple(out_dwells),
            transition_probs=tuple(out_probs),
            arc_valid=tuple(out_valids),
        )


__all__ = ["ScheduleGenerator"]
