"""Unit tests for KOrderMarkovChain and VariableOrderMarkovChain."""

from __future__ import annotations

import unittest

import numpy as np

from seqsm.markov import KOrderMarkovChain, VariableOrderMarkovChain
from seqsm.types import CSRMatrix


class TestMarkovChain(unittest.TestCase):
    def test_invalid_parameters_raise(self) -> None:
        with self.assertRaises(ValueError):
            KOrderMarkovChain(k=0)
        with self.assertRaises(ValueError):
            KOrderMarkovChain(smoothing=-0.1)

    def test_unfitted_access_raises(self) -> None:
        chain = KOrderMarkovChain()
        with self.assertRaises(ValueError):
            _ = chain.count_matrix
        with self.assertRaises(ValueError):
            _ = chain.transition_matrix
        with self.assertRaises(ValueError):
            _ = chain.companion_matrix()
        with self.assertRaises(ValueError):
            chain.predict_proba_window(0)
        with self.assertRaises(ValueError):
            chain.stationary_distribution()

    def test_fit_clock_k1(self) -> None:
        chain = KOrderMarkovChain(k=1, mode="clock", smoothing=0.0)
        # Sequence: 0 -> 1 -> 0 -> 1 -> 0
        states = np.array([0, 1, 0, 1, 0], dtype=np.int32)
        chain.fit_from_arrays(states)

        self.assertEqual(chain.k, 1)
        self.assertEqual(chain.n_states, 2)
        self.assertEqual(chain.n_windows, 2)

        # Transition matrix should be [[0, 1], [1, 0]]
        P = chain.transition_matrix
        np.testing.assert_allclose(P[0], [0.0, 1.0])
        np.testing.assert_allclose(P[1], [1.0, 0.0])

        # Companion matrix equals P for K=1
        comp = chain.companion_matrix()
        np.testing.assert_allclose(comp, P)

    def test_fit_clock_k2(self) -> None:
        chain = KOrderMarkovChain(k=2, mode="clock", smoothing=0.0)
        # Sequence: 0 -> 1 -> 2 -> 0 -> 1 -> 2
        states = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
        chain.fit_from_arrays(states)

        self.assertEqual(chain.k, 2)
        self.assertEqual(chain.n_states, 3)

        # Row-stochasticity check
        P = chain.transition_matrix
        np.testing.assert_allclose(np.sum(P, axis=1), np.ones(chain.n_windows))

        comp = chain.companion_matrix()
        np.testing.assert_allclose(np.sum(comp, axis=1), np.ones(chain.n_windows))

    def test_fit_event_mode_with_dwell(self) -> None:
        chain = KOrderMarkovChain(k=1, mode="event")
        # Run sequence: 0, 0, 0, 1, 1, 0, 0
        states = np.array([0, 0, 0, 1, 1, 0, 0], dtype=np.int32)
        chain.fit_from_arrays(states)

        self.assertEqual(chain.mode, "event")
        dwells = chain.dwell_stats
        self.assertIn(0, dwells)
        self.assertIn(1, dwells)
        # State 0 was visited twice: lengths 3 and 2
        self.assertEqual(dwells[0].count, 2)
        self.assertAlmostEqual(dwells[0].mean, 2.5)
        # State 1 was visited once: length 2
        self.assertEqual(dwells[1].count, 1)
        self.assertEqual(dwells[1].mean, 2.0)

    def test_multi_step_prediction(self) -> None:
        chain = KOrderMarkovChain(k=1, mode="clock")
        states = np.array([0, 1, 0, 1, 0], dtype=np.int32)
        chain.fit_from_arrays(states)

        # From window 0 (state 0): step 1 should predict state 1
        p1 = chain.multi_step_distribution(0, steps=1)
        np.testing.assert_allclose(p1, [0.0, 1.0])

        # Step 2 should cycle back to state 0
        p2 = chain.multi_step_distribution(0, steps=2)
        self.assertEqual(p2.shape, (2, 2))
        np.testing.assert_allclose(p2[0], [0.0, 1.0])
        np.testing.assert_allclose(p2[1], [1.0, 0.0])

        with self.assertRaises(ValueError):
            chain.multi_step_distribution(0, steps=0)

    def test_stationary_and_entropy(self) -> None:
        chain = KOrderMarkovChain(k=1, smoothing=0.1)
        states = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.int32)
        chain.fit_from_arrays(states)

        pi = chain.stationary_distribution()
        self.assertEqual(len(pi), chain.n_windows)
        self.assertAlmostEqual(float(np.sum(pi)), 1.0)

        mu = chain.marginal_stationary_distribution()
        self.assertEqual(len(mu), chain.n_states)
        self.assertAlmostEqual(float(np.sum(mu)), 1.0)

        h = chain.entropy_rate()
        self.assertGreaterEqual(h, 0.0)
        self.assertGreaterEqual(chain.perplexity(), 1.0)

    def test_companion_attested_successor_renormalization_and_simulation_c1(self) -> None:
        # K=2 chain with smoothing to induce non-zero probabilities for unobserved transitions
        chain = KOrderMarkovChain(k=2, smoothing=0.1)
        # Sequence: 0 -> 1 -> 2 -> 0 -> 1 -> 2
        states = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
        chain.fit_from_arrays(states)

        comp = chain.companion_matrix()
        # Row-stochasticity check: every row must sum to 1.0
        np.testing.assert_allclose(comp.sum(axis=1), np.ones(chain.n_windows), atol=1e-12)

        # Multi-step distribution vs brute force 2-step
        dist_2step = chain.multi_step_distribution(source_id=0, steps=2)
        self.assertEqual(dist_2step.shape, (2, chain.n_states))
        np.testing.assert_allclose(dist_2step.sum(axis=1), [1.0, 1.0], atol=1e-12)
        p_tilde = chain.companion_matrix()
        p_rect = chain.transition_matrix
        brute_2step = p_tilde[0, :] @ p_rect
        np.testing.assert_allclose(dist_2step[1], brute_2step, atol=1e-12)

    def test_event_mode_preserves_episode_boundaries_c6(self) -> None:
        chain = KOrderMarkovChain(k=2, mode="event", smoothing=0.0)
        # Episode 0: state 0 (runs 3x), state 1 (runs 2x)
        # Episode 1: state 2 (runs 2x), state 0 (runs 3x)
        states = np.array([0, 0, 0, 1, 1, 2, 2, 0, 0, 0], dtype=np.int32)
        episodes = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int32)
        chain.fit_from_arrays(states, episode_ids=episodes)

        # In event mode, collapsed sequence is:
        # Ep 0: 0, 1
        # Ep 1: 2, 0
        # With K=2, within Ep 0: (0, 1) has NO successor inside Ep 0
        # Within Ep 1: (2, 0) has NO successor inside Ep 1
        u_01 = chain.encoder.encode_window((0, 1))
        if u_01 >= 0:
            count_to_2 = chain.count_matrix[u_01, 2]
            self.assertEqual(count_to_2, 0)

    def test_sparse_companion_csr_h7(self) -> None:
        # Fit with low sparse_threshold to trigger direct CSR construction
        chain = KOrderMarkovChain(k=1, smoothing=0.0, sparse_threshold=2)
        states = np.array([0, 1, 0, 1, 0], dtype=np.int32)
        chain.fit_from_arrays(states)

        dense_comp = chain.companion_matrix(dense=True)
        self.assertIsInstance(dense_comp, np.ndarray)
        self.assertEqual(dense_comp.shape, (2, 2))

        sparse_comp = chain.companion_matrix(dense=False)
        self.assertIsInstance(sparse_comp, CSRMatrix)
        np.testing.assert_allclose(sparse_comp.to_dense(), dense_comp)

        pi_sparse = chain.stationary_distribution()
        np.testing.assert_allclose(pi_sparse, [0.5, 0.5], atol=1e-6)

    def test_multi_step_distribution_vs_one_step_composition(self) -> None:
        chain = KOrderMarkovChain(k=1, smoothing=0.0)
        states = np.array([0, 1, 2, 0, 1, 2], dtype=np.int32)
        chain.fit_from_arrays(states)

        d1 = chain.multi_step_distribution(0, 1)
        p1 = chain.predict_proba_window(0)
        np.testing.assert_allclose(d1, p1)

        d3 = chain.multi_step_distribution(0, 3)
        self.assertEqual(d3.shape, (3, chain.n_states))
        np.testing.assert_allclose(d3.sum(axis=1), [1.0, 1.0, 1.0])


class TestVariableOrderMarkovChain(unittest.TestCase):
    def test_invalid_max_k_raises(self) -> None:
        with self.assertRaises(ValueError):
            VariableOrderMarkovChain(max_k=0)

    def test_fit_and_predict_backoff(self) -> None:
        vom = VariableOrderMarkovChain(max_k=2, backoff="witten_bell")
        states = np.array([0, 1, 0, 1, 0, 1, 2], dtype=np.int32)
        vom.fit_from_arrays(states)

        self.assertEqual(len(vom.chains), 2)
        self.assertEqual(vom.max_k, 2)

        # Seen 2-window (0, 1) should use order 2
        p_seen = vom.predict_proba([0, 1])
        self.assertAlmostEqual(float(np.sum(p_seen)), 1.0)

        # Unseen 2-window (2, 2) should back off to order 1
        p_unseen = vom.predict_proba([2, 2])
        self.assertAlmostEqual(float(np.sum(p_unseen)), 1.0)

    def test_predict_linear_interpolation(self) -> None:
        vom = VariableOrderMarkovChain(max_k=2, backoff="linear", weights=[0.4, 0.6])
        states = np.array([0, 1, 0, 1, 2], dtype=np.int32)
        vom.fit_from_arrays(states)

        p = vom.predict_proba([0, 1])
        self.assertAlmostEqual(float(np.sum(p)), 1.0)


if __name__ == "__main__":
    unittest.main()
