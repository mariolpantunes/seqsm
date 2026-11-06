"""Unit tests for OpenBLAS linear algebra, stationary distribution, and entropy operations."""

from __future__ import annotations

import unittest

import numpy as np

from seqsm.linalg import (
    companion_matrix_power,
    conditional_entropy_rate,
    jensen_shannon_divergence,
    marginal_stationary_distribution,
    model_perplexity,
    stationary_distribution_power_iteration,
)


class TestLinalg(unittest.TestCase):
    def test_companion_matrix_power(self) -> None:
        mat = np.array([[0.5, 0.5], [0.2, 0.8]], dtype=np.float64)

        # 0 steps: identity
        eye = companion_matrix_power(mat, 0)
        np.testing.assert_array_equal(eye, np.eye(2))

        # Negative steps raises
        with self.assertRaises(ValueError):
            companion_matrix_power(mat, -1)

        # 2 steps
        p2 = companion_matrix_power(mat, 2)
        np.testing.assert_allclose(p2, mat @ mat)

    def test_stationary_distribution_known_analytic(self) -> None:
        # P = [[0.8, 0.2], [0.4, 0.6]]
        # Stationary pi = [2/3, 1/3]
        P = np.array([[0.8, 0.2], [0.4, 0.6]], dtype=np.float64)
        pi = stationary_distribution_power_iteration(P)

        expected = np.array([2.0 / 3.0, 1.0 / 3.0], dtype=np.float64)
        np.testing.assert_allclose(pi, expected, atol=1e-8)
        np.testing.assert_allclose(pi @ P, pi, atol=1e-8)
        self.assertAlmostEqual(float(np.sum(pi)), 1.0)

    def test_stationary_distribution_edge_cases(self) -> None:
        # Empty
        empty = np.empty((0, 0), dtype=np.float64)
        self.assertEqual(len(stationary_distribution_power_iteration(empty)), 0)

        # Single element
        single = np.array([[1.0]], dtype=np.float64)
        np.testing.assert_array_equal(stationary_distribution_power_iteration(single), [1.0])

        # Cyclic 2-state: eigenvalue fallback
        cyclic = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        pi_cyc = stationary_distribution_power_iteration(cyclic)
        self.assertAlmostEqual(float(np.sum(pi_cyc)), 1.0)
        np.testing.assert_allclose(pi_cyc, [0.5, 0.5], atol=1e-6)

    def test_marginal_stationary_distribution(self) -> None:
        # K=2 windows: (0, 0), (0, 1), (1, 0)
        k_tuples = np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int32)
        pi_windows = np.array([0.2, 0.3, 0.5], dtype=np.float64)

        # Marginal for state 0 = pi[(0, 0)] + pi[(1, 0)] = 0.2 + 0.5 = 0.7
        # Marginal for state 1 = pi[(0, 1)] = 0.3
        marginal = marginal_stationary_distribution(pi_windows, k_tuples, n_states=2)
        np.testing.assert_allclose(marginal, [0.7, 0.3])
        self.assertAlmostEqual(float(np.sum(marginal)), 1.0)

    def test_entropy_rate_and_perplexity(self) -> None:
        # Deterministic transition: H = 0, Perplexity = 1
        det_trans = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        pi_det = np.array([0.5, 0.5], dtype=np.float64)
        h_det = conditional_entropy_rate(pi_det, det_trans)
        self.assertAlmostEqual(h_det, 0.0)
        self.assertAlmostEqual(model_perplexity(h_det), 1.0)

        # Uniform 2-state: H = 1.0 bit, Perplexity = 2.0
        unif_trans = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float64)
        pi_unif = np.array([0.5, 0.5], dtype=np.float64)
        h_unif = conditional_entropy_rate(pi_unif, unif_trans)
        self.assertAlmostEqual(h_unif, 1.0)
        self.assertAlmostEqual(model_perplexity(h_unif), 2.0)

    def test_jensen_shannon_divergence(self) -> None:
        p = np.array([0.5, 0.5], dtype=np.float64)
        # Identical: JSD = 0
        self.assertAlmostEqual(jensen_shannon_divergence(p, p), 0.0)

        # Disjoint: JSD = 1.0
        p1 = np.array([1.0, 0.0], dtype=np.float64)
        p2 = np.array([0.0, 1.0], dtype=np.float64)
        self.assertAlmostEqual(jensen_shannon_divergence(p1, p2), 1.0)


if __name__ == "__main__":
    unittest.main()
