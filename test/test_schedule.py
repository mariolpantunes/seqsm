"""Unit tests for ScheduleGenerator in clock and event modes."""

from __future__ import annotations

import unittest

import numpy as np
import polars as pl

from seqsm.encoding import StateEncoder
from seqsm.markov import KOrderMarkovChain
from seqsm.schedule import ScheduleGenerator


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
        # Total dwell must reach at least n_steps
        self.assertGreaterEqual(sum(traj.dwell_durations), 10)

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
