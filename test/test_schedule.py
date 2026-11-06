"""Unit tests for ScheduleGenerator in clock and event modes."""

from __future__ import annotations

import unittest

import numpy as np
import polars as pl

from seqsm.encoding import StateEncoder
from seqsm.markov import KOrderMarkovChain
from seqsm.schedule import ScheduleGenerator
from seqsm.types import ScheduleConstraints


class TestSchedule(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pl.DataFrame(
            {
                "room": ["kitchen", "living", "bedroom", "kitchen", "living"],
                "light": ["on", "dim", "off", "on", "dim"],
            }
        )
        self.encoder = StateEncoder()
        self.encoder.fit(self.df)
        self.states = self.encoder.transform(self.df)

    def test_clock_mode_generation(self) -> None:
        chain = KOrderMarkovChain(k=1, mode="clock")
        chain.fit_from_arrays(self.states)

        gen = ScheduleGenerator(chain, self.encoder)
        traj = gen.generate(n_steps=5, method="sample", seed=123)

        self.assertEqual(len(traj.state_ids), 5)
        self.assertEqual(len(traj.dwell_durations), 5)
        self.assertTrue(all(d == 1 for d in traj.dwell_durations))
        self.assertEqual(len(traj.transition_probs), 5)
        self.assertEqual(len(traj.arc_valid), 5)

    def test_event_mode_generation_with_dwell(self) -> None:
        chain = KOrderMarkovChain(k=1, mode="event")
        # Repeating pattern with runs: 0, 0, 1, 1, 1, 2, 2
        seq = np.array([0, 0, 1, 1, 1, 2, 2, 0, 0], dtype=np.int32)
        chain.fit_from_arrays(seq, n_states=3)

        gen = ScheduleGenerator(chain, self.encoder)
        traj = gen.generate(n_steps=10, method="map", seed=42)

        self.assertGreater(len(traj.state_ids), 0)
        # Total dwell must equal n_steps due to interval clamping (M5)
        self.assertEqual(sum(traj.dwell_durations), 10)

    def test_hierarchical_context_backoff_unseen_history_c4(self) -> None:
        chain = KOrderMarkovChain(k=2, mode="clock")
        # Attested windows: (0, 1), (1, 2), (2, 0)
        chain.fit_from_arrays(np.array([0, 1, 2, 0, 1, 2], dtype=np.int32))

        gen = ScheduleGenerator(chain, self.encoder)
        # Unseen window (99, 1): suffix of length 1 is 1 -> backs off to window ending in 1 -> (0, 1)
        unseen_hist = np.array([99, 1], dtype=np.int32)
        traj = gen.generate(n_steps=4, init_history=unseen_hist, method="map")
        self.assertEqual(len(traj.state_ids), 4)
        # From (0, 1), next should be 2
        self.assertEqual(traj.state_ids[0], 2)

    def test_schedule_constraints_satisfaction_c5(self) -> None:
        chain = KOrderMarkovChain(k=1, mode="clock")
        # Subcycle between 1 and 2 allows avoiding state 0
        seq = np.array([0, 1, 2, 1, 2, 1, 2, 0, 1, 2], dtype=np.int32)
        chain.fit_from_arrays(seq)

        gen = ScheduleGenerator(chain, self.encoder)

        # 1. Forbidden state constraint: forbid state 0
        constraints_forbid = ScheduleConstraints(forbidden_states=[0])
        # Start at history [1] so it can cycle between 1 and 2
        traj_forbid = gen.generate(n_steps=10, init_history=np.array([1]), constraints=constraints_forbid, seed=42)
        self.assertNotIn(0, traj_forbid.state_ids)

        # 2. Terminal state constraint: steer to state 2
        constraints_term = ScheduleConstraints(terminal_state=2)
        traj_term = gen.generate(n_steps=5, init_history=np.array([1]), constraints=constraints_term, seed=42)
        self.assertEqual(traj_term.state_ids[-1], 2)

        # 3. Max total dwell constraint: state 0 max dwell = 1
        constraints_max = ScheduleConstraints(max_total_dwell={0: 1})
        traj_max = gen.generate(n_steps=10, init_history=np.array([0]), constraints=constraints_max, seed=42)
        count_0 = sum(d for s, d in zip(traj_max.state_ids, traj_max.dwell_durations, strict=False) if s == 0)
        self.assertLessEqual(count_0, 1)

    def test_mandatory_states_and_constrained_method_r1(self) -> None:
        chain = KOrderMarkovChain(k=1, mode="clock")
        seq = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int32)
        chain.fit_from_arrays(seq)
        gen = ScheduleGenerator(chain, self.encoder)

        # Mandatory state 2 at step 2
        constraints = ScheduleConstraints(mandatory_states=[(2, 2)])
        traj = gen.generate(n_steps=5, constraints=constraints, method="constrained", seed=42)
        self.assertEqual(traj.state_ids[2], 2)
        self.assertFalse(any(traj.constraint_violated))

        # Mandatory state with interval (1, 3)
        constraints_range = ScheduleConstraints(mandatory_states=[(2, (1, 3))])
        traj_range = gen.generate(n_steps=5, constraints=constraints_range, method="constrained", seed=42)
        self.assertIn(2, traj_range.state_ids[1:4])

    def test_constraint_relaxation_flag_r5(self) -> None:
        chain = KOrderMarkovChain(k=1, mode="clock")
        seq = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
        chain.fit_from_arrays(seq)
        gen = ScheduleGenerator(chain, self.encoder)

        # Forbid all states (0, 1, 2) forcing relaxation
        constraints_impossible = ScheduleConstraints(forbidden_states=[0, 1, 2])
        traj = gen.generate(n_steps=4, constraints=constraints_impossible, method="sample", seed=42)
        self.assertTrue(any(traj.constraint_violated))
        self.assertFalse(all(traj.arc_valid))

    def test_initial_state_constraint(self) -> None:
        chain = KOrderMarkovChain(k=1, mode="clock")
        chain.fit_from_arrays(np.array([0, 1, 2, 0, 1, 2], dtype=np.int32))
        gen = ScheduleGenerator(chain, self.encoder)
        constraints = ScheduleConstraints(initial_state=2)
        traj = gen.generate(n_steps=3, constraints=constraints, method="map")
        self.assertEqual(len(traj.state_ids), 3)

    def test_backoff_no_matching_suffix(self) -> None:
        chain = KOrderMarkovChain(k=2, mode="clock")
        chain.fit_from_arrays(np.array([0, 1, 2, 0, 1, 2], dtype=np.int32))
        gen = ScheduleGenerator(chain, self.encoder)
        # History [99, 98] has no matching suffix among windows (0, 1), (1, 2), (2, 0)
        traj = gen.generate(n_steps=3, init_history=np.array([99, 98], dtype=np.int32), method="map")
        self.assertEqual(len(traj.state_ids), 3)

    def test_init_history_provided(self) -> None:
        chain = KOrderMarkovChain(k=2, mode="clock")
        chain.fit_from_arrays(np.array([0, 1, 2, 0, 1, 2], dtype=np.int32))

        gen = ScheduleGenerator(chain, self.encoder)
        init_hist = np.array([0, 1], dtype=np.int32)
        traj = gen.generate(n_steps=3, init_history=init_hist, method="map")

        self.assertEqual(len(traj.state_ids), 3)
        # For history (0, 1), next in sequence was 2
        self.assertEqual(traj.state_ids[0], 2)

    def test_invalid_n_steps_raises(self) -> None:
        chain = KOrderMarkovChain(k=1)
        chain.fit_from_arrays(self.states)
        gen = ScheduleGenerator(chain, self.encoder)
        with self.assertRaises(ValueError):
            gen.generate(n_steps=0)

    def test_short_init_history_raises(self) -> None:
        chain = KOrderMarkovChain(k=2)
        chain.fit_from_arrays(np.array([0, 1, 2, 0, 1, 2], dtype=np.int32))
        gen = ScheduleGenerator(chain, self.encoder)
        with self.assertRaises(ValueError):
            gen.generate(n_steps=3, init_history=np.array([0], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
