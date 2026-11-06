"""Schedule generation engine executing valid-arc rollouts in clock and event modes."""

from __future__ import annotations

from typing import Any

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

    def _resolve_state(self, state: int | dict[str, Any] | None) -> int | None:
        """Resolves an integer state ID from int or feature dictionary."""
        if state is None:
            return None
        if isinstance(state, (int, np.integer)):
            return int(state)
        if isinstance(state, dict):
            return self._encoder.dict_to_state(state)
        return None

    def _hierarchical_backoff(self, target_window: list[int], pi: NDArray[np.float64]) -> int:
        """Finds closest attested K-window via hierarchical suffix backoff (K -> K-1 -> 1 -> mu)."""
        k_enc = self._chain.encoder
        k = self._chain.k
        if len(target_window) >= k:
            win_k = tuple(int(x) for x in target_window[-k:])
            u = k_enc.encode_window(win_k)
            if u >= 0:
                return u

        k_tuples = k_enc.k_tuples
        n_win = k_enc.n_windows
        for suf_len in range(min(k - 1, len(target_window)), 0, -1):
            suf = tuple(int(x) for x in target_window[-suf_len:])
            matching = [idx for idx in range(n_win) if tuple(k_tuples[idx, -suf_len:]) == suf]
            if matching:
                return max(matching, key=lambda idx: float(pi[idx]))

        if n_win > 0:
            return int(np.argmax(pi))
        return 0

    def generate(
        self,
        n_steps: int,
        init_history: NDArray[np.int32] | None = None,
        method: ScheduleMethod = "sample",
        constraints: ScheduleConstraints | None = None,
        seed: int = 42,
    ) -> ScheduleTrajectory:
        """
        Executes a valid-arc schedule rollout for n_steps with dynamic constraint satisfaction.

        In clock mode, yields exactly n_steps transitions.
        In event mode, simulates events with empirical dwell sampling clamped to n_steps.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}.")

        rng = np.random.default_rng(seed)
        k_enc = self._chain.encoder
        k = self._chain.k
        pi = self._chain.stationary_distribution()

        # Resolve constraints
        forbidden: set[int] = set()
        max_dwell: dict[int, int] = {}
        term_state: int | None = None
        if constraints is not None:
            for fs in constraints.forbidden_states:
                sid = self._resolve_state(fs)
                if sid is not None and sid >= 0:
                    forbidden.add(sid)
            if constraints.max_total_dwell is not None:
                max_dwell = dict(constraints.max_total_dwell)
            term_state = self._resolve_state(constraints.terminal_state)

        # 1. Resolve initial state / history
        if init_history is not None:
            hist_arr = np.asarray(init_history, dtype=np.int32)
            if len(hist_arr) < k:
                raise ValueError(f"init_history must contain at least k={k} states, got {len(hist_arr)}.")
            curr_window = [int(x) for x in hist_arr[-k:]]
            curr_u = self._hierarchical_backoff(curr_window, pi)
        elif constraints is not None and constraints.initial_state is not None:
            init_s = self._resolve_state(constraints.initial_state)
            matching = [idx for idx in range(k_enc.n_windows) if k_enc.decode_window(idx)[-1] == init_s]
            if matching:
                curr_u = max(matching, key=lambda idx: float(pi[idx]))
            else:
                curr_u = int(np.argmax(pi))
            curr_window = list(k_enc.decode_window(curr_u))
        else:
            if method == "map":
                curr_u = int(np.argmax(pi))
            else:
                curr_u = int(rng.choice(len(pi), p=pi))
            curr_window = list(k_enc.decode_window(curr_u))

        state_dwell_spent: dict[int, int] = {}
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

            cands: list[tuple[int, float]] = []
            if n_arcs > 0:
                for idx in range(start_arc, end_arc):
                    v = int(self._indices[idx])
                    p = float(self._filtered_probs[idx])
                    if v in forbidden:
                        continue
                    if v in max_dwell and state_dwell_spent.get(v, 0) >= max_dwell[v]:
                        continue
                    cands.append((v, p))

            is_valid = True
            rem_steps = n_steps - total_steps
            if term_state is not None and rem_steps == 1 and any(v == term_state for v, _ in cands):
                next_v = term_state
                next_p = next(p for v, p in cands if v == term_state)
            elif cands:
                denom = sum(p for _, p in cands)
                norm_probs = [p / denom for _, p in cands]
                if method == "map":
                    best_idx = int(np.argmax(norm_probs))
                    next_v, next_p = cands[best_idx]
                else:
                    chosen_idx = int(rng.choice(len(cands), p=norm_probs))
                    next_v, next_p = cands[chosen_idx]
            elif n_arcs > 0:
                # Relax constraints if all filtered out
                if method == "map":
                    best_rel = int(np.argmax(self._filtered_probs[start_arc:end_arc]))
                    next_v = int(self._indices[start_arc + best_rel])
                    next_p = float(self._filtered_probs[start_arc + best_rel])
                else:
                    sub_cum = self._cumprobs[start_arc:end_arc]
                    rand_val = rng.uniform(0.0, 1.0)
                    rel_idx = int(np.searchsorted(sub_cum, rand_val))
                    rel_idx = min(rel_idx, n_arcs - 1)
                    next_v = int(self._indices[start_arc + rel_idx])
                    next_p = float(self._filtered_probs[start_arc + rel_idx])
            else:
                next_v = int(np.argmax(self._chain.transition_matrix[curr_u]))
                next_p = float(self._chain.transition_matrix[curr_u, next_v])
                is_valid = False

            # Dwell duration resolution & clamping (H5 & M5)
            rem = n_steps - total_steps
            if self._chain.mode == "event":
                if next_v in dwell_stats and len(dwell_stats[next_v].samples) > 0:
                    samples = dwell_stats[next_v].samples
                    dwell = int(rng.choice(samples))
                else:
                    dwell = 1
                dwell = max(1, min(dwell, rem))
                total_steps += dwell
            else:
                dwell = 1
                total_steps += 1

            state_dwell_spent[next_v] = state_dwell_spent.get(next_v, 0) + dwell
            out_states.append(next_v)
            out_dwells.append(dwell)
            out_probs.append(next_p)
            out_valids.append(is_valid)

            # Update window & hierarchical context backoff (C4)
            curr_window.append(next_v)
            curr_u = self._hierarchical_backoff(curr_window, pi)

        return ScheduleTrajectory(
            state_ids=tuple(out_states),
            dwell_durations=tuple(out_dwells),
            transition_probs=tuple(out_probs),
            arc_valid=tuple(out_valids),
        )


__all__ = ["ScheduleGenerator"]
