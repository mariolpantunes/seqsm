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

        # Precompute suffix mapping for O(1) hierarchical context backoff
        self._suffix_map: dict[tuple[int, ...], list[int]] = {}
        k_enc = chain.encoder
        k = chain.k
        for u in range(k_enc.n_windows):
            win = k_enc.decode_window(u)
            for suf_len in range(1, k):
                suf = win[-suf_len:]
                self._suffix_map.setdefault(suf, []).append(u)

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

        for suf_len in range(min(k - 1, len(target_window)), 0, -1):
            suf = tuple(int(x) for x in target_window[-suf_len:])
            matching = self._suffix_map.get(suf)
            if matching:
                return max(matching, key=lambda idx: float(pi[idx]))

        if k_enc.n_windows > 0:
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
        mandatory: list[tuple[int, int, int]] = []
        if constraints is not None:
            for fs in constraints.forbidden_states:
                sid = self._resolve_state(fs)
                if sid is not None and sid >= 0:
                    forbidden.add(sid)
            if constraints.max_total_dwell is not None:
                max_dwell = dict(constraints.max_total_dwell)
            term_state = self._resolve_state(constraints.terminal_state)
            for ms, time_spec in constraints.mandatory_states:
                sid = self._resolve_state(ms)
                if sid is not None and sid >= 0:
                    if isinstance(time_spec, (int, np.integer)):
                        mandatory.append((sid, int(time_spec), int(time_spec)))
                    elif isinstance(time_spec, (tuple, list)) and len(time_spec) == 2:
                        mandatory.append((sid, int(time_spec[0]), int(time_spec[1])))
        mandatory_satisfied = [False] * len(mandatory)

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
        out_violated: list[bool] = []

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
            constraint_violated = False
            rem_steps = n_steps - total_steps

            # Check urgent mandatory checkpoints (due or overdue)
            urgent_mandatory_set = {
                mandatory[m_idx][0]
                for m_idx, (_, _, t_max) in enumerate(mandatory)
                if not mandatory_satisfied[m_idx] and total_steps >= t_max
            }

            # 1. Enforce terminal state if at final step
            if term_state is not None and rem_steps == 1 and any(v == term_state for v, _ in cands):
                next_v = term_state
                next_p = next(p for v, p in cands if v == term_state)
            # 2. Enforce urgent mandatory checkpoint
            elif urgent_mandatory_set and any(v in urgent_mandatory_set for v, _ in cands):
                urgent_cands = [item for item in cands if item[0] in urgent_mandatory_set]
                denom = sum(p for _, p in urgent_cands)
                norm_probs = [p / denom for _, p in urgent_cands]
                if method in ("map", "constrained"):
                    best_idx = int(np.argmax(norm_probs))
                    next_v, next_p = urgent_cands[best_idx]
                else:
                    chosen_idx = int(rng.choice(len(urgent_cands), p=norm_probs))
                    next_v, next_p = urgent_cands[chosen_idx]
            elif cands:
                # In constrained mode, prioritize active mandatory state visits
                active_mandatory = {
                    ms_sid
                    for m_idx, (ms_sid, t_min, t_max) in enumerate(mandatory)
                    if not mandatory_satisfied[m_idx] and t_min <= total_steps <= t_max
                }
                preferred_cands = [item for item in cands if item[0] in active_mandatory]
                target_cands = preferred_cands if (method == "constrained" and preferred_cands) else cands

                denom = sum(p for _, p in target_cands)
                norm_probs = [p / denom for _, p in target_cands]
                if method in ("map", "constrained"):
                    best_idx = int(np.argmax(norm_probs))
                    next_v, next_p = target_cands[best_idx]
                else:
                    chosen_idx = int(rng.choice(len(target_cands), p=norm_probs))
                    next_v, next_p = target_cands[chosen_idx]
            elif n_arcs > 0:
                # Relax constraints if all filtered out
                constraint_violated = True
                is_valid = False
                if method in ("map", "constrained"):
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
                constraint_violated = True

            # Dwell duration resolution & clamping (H5, M5, N1)
            rem = n_steps - total_steps
            if self._chain.mode == "event":
                if next_v in dwell_stats and len(dwell_stats[next_v].samples) > 0:
                    samples = dwell_stats[next_v].samples
                    dwell = int(rng.choice(samples))
                else:
                    dwell = 1
                dwell = max(1, min(dwell, rem))

                # Clamping in constrained mode to avoid leaping over upcoming mandatory deadlines
                if method == "constrained":
                    pending_deadlines = [
                        t_max
                        for m_idx, (_, _, t_max) in enumerate(mandatory)
                        if not mandatory_satisfied[m_idx] and t_max > total_steps
                    ]
                    if pending_deadlines:
                        earliest_deadline = min(pending_deadlines)
                        time_to_earliest = earliest_deadline - total_steps
                        if time_to_earliest >= 1 and not any(
                            mandatory[m_idx][0] == next_v
                            for m_idx in range(len(mandatory))
                            if not mandatory_satisfied[m_idx] and mandatory[m_idx][2] == earliest_deadline
                        ):
                            dwell = max(1, min(dwell, time_to_earliest))
            else:
                dwell = 1

            # Mark satisfied mandatory checkpoints and flag missed deadlines
            for m_idx, (ms_sid, t_min, t_max) in enumerate(mandatory):
                if not mandatory_satisfied[m_idx]:
                    if next_v == ms_sid:
                        if total_steps <= t_max and (total_steps + dwell) > t_min:
                            mandatory_satisfied[m_idx] = True
                        elif total_steps > t_max:
                            constraint_violated = True
                            mandatory_satisfied[m_idx] = True
                    else:
                        if total_steps >= t_max or (total_steps + dwell) > t_max:
                            constraint_violated = True

            total_steps += dwell
            state_dwell_spent[next_v] = state_dwell_spent.get(next_v, 0) + dwell
            out_states.append(next_v)
            out_dwells.append(dwell)
            out_probs.append(next_p)
            out_valids.append(is_valid)
            out_violated.append(constraint_violated)

            # Update window & hierarchical context backoff (C4)
            curr_window.append(next_v)
            curr_u = self._hierarchical_backoff(curr_window, pi)

        unmet_mandatory = tuple(
            ms_sid for m_idx, (ms_sid, _, _) in enumerate(mandatory) if not mandatory_satisfied[m_idx]
        )
        if unmet_mandatory and out_violated:
            out_violated[-1] = True

        return ScheduleTrajectory(
            state_ids=tuple(out_states),
            dwell_durations=tuple(out_dwells),
            transition_probs=tuple(out_probs),
            arc_valid=tuple(out_valids),
            constraint_violated=tuple(out_violated),
            unmet_mandatory=unmet_mandatory,
        )


__all__ = ["ScheduleGenerator"]
