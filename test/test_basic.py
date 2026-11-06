"""Basic initialization and types test."""

import unittest

import seqsm
from seqsm.types import DwellDistribution, RoutineCycle, ScheduleConstraints, ScheduleTrajectory


class TestBasic(unittest.TestCase):
    def test_import(self) -> None:
        self.assertEqual(seqsm.__version__, "0.1.0")

    def test_types_instantiation(self) -> None:
        dist = DwellDistribution(
            state_id=1, count=10, mean=5.0, std=1.0, median=5.0, min_dwell=2, max_dwell=8, samples=(2, 5, 8)
        )
        self.assertEqual(dist.state_id, 1)

        constraints = ScheduleConstraints(initial_state=0)
        self.assertEqual(constraints.initial_state, 0)

        traj = ScheduleTrajectory(
            state_ids=(0, 1), dwell_durations=(1, 2), transition_probs=(0.8, 0.9), arc_valid=(True, True)
        )
        self.assertEqual(len(traj.state_ids), 2)

        cycle = RoutineCycle(state_sequence=(0, 1, 0), frequency=5, period=2.0, persistence=0.95)
        self.assertEqual(cycle.frequency, 5)


if __name__ == "__main__":
    unittest.main()
