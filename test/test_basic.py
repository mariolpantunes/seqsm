"""Basic initialization and types test."""

import unittest

import numpy as np

import seqsm
from seqsm.types import CSRMatrix, DwellDistribution, RoutineCycle, ScheduleConstraints, ScheduleTrajectory


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

    def test_csrmatrix_dot_vector_left_r7(self) -> None:
        csr = CSRMatrix(
            data=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            indices=np.array([0, 1, 0], dtype=np.int32),
            indptr=np.array([0, 2, 3], dtype=np.int32),
            shape=(2, 2),
        )
        vec = np.array([10.0, 20.0], dtype=np.float64)
        res = csr.dot_vector_left(vec)
        # col 0: 10*1.0 + 20*3.0 = 70.0; col 1: 10*2.0 = 20.0
        np.testing.assert_allclose(res, [70.0, 20.0])

        # Test empty CSR
        empty_csr = CSRMatrix(
            data=np.empty(0, dtype=np.float64),
            indices=np.empty(0, dtype=np.int32),
            indptr=np.array([0, 0], dtype=np.int32),
            shape=(1, 2),
        )
        res_empty = empty_csr.dot_vector_left(np.array([1.0]))
        np.testing.assert_allclose(res_empty, [0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
