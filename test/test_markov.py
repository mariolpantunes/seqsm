"""Unit tests for KOrderMarkovChain and VariableOrderMarkovChain."""

from __future__ import annotations

import unittest

import numpy as np

from seqsm.markov import KOrderMarkovChain, VariableOrderMarkovChain


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
        np.testing.assert_allclose(p2, [1.0, 0.0])

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
