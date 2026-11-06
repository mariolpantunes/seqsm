"""Unit tests for sequence motifs and anomaly scoring."""

from __future__ import annotations

import unittest

import numpy as np

from seqsm.markov import KOrderMarkovChain, VariableOrderMarkovChain
from seqsm.motifs import compare_chains, detect_period, score_anomalies


class TestMotifs(unittest.TestCase):
    def setUp(self) -> None:
        self.chain = KOrderMarkovChain(k=1, smoothing=0.01)
        # Regular sequence: alternating 0 and 1
        seq = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
        self.chain.fit_from_arrays(seq)

    def test_score_anomalies_empty(self) -> None:
        res = score_anomalies(self.chain, np.empty(0, dtype=np.int32))
        self.assertEqual(len(res), 0)
        self.assertIn("is_anomaly", res.columns)

    def test_score_anomalies_flags_rare_transitions(self) -> None:
        # Regular: 0, 1, 0, 1; Anomalous: sudden transition to 99
        test_seq = np.array([0, 1, 0, 1, 99, 99, 0, 1], dtype=np.int32)
        scores = score_anomalies(self.chain, test_seq, window_size=2, threshold_quantile=0.8)

        self.assertEqual(len(scores), len(test_seq))
        self.assertIn("rolling_nll", scores.columns)
        # Index 4 (transition to unseen 99) should have high NLL
        self.assertGreater(scores["nll_trans"][4], 10.0)
        self.assertGreater(scores["nll"][4], 9.0)

    def test_score_anomalies_with_vom(self) -> None:
        vom = VariableOrderMarkovChain(max_k=2)
        vom.fit_from_arrays(np.array([0, 1, 0, 1, 0, 1], dtype=np.int32))
        test_seq = np.array([0, 1, 0, 1], dtype=np.int32)
        scores = score_anomalies(vom, test_seq)
        self.assertEqual(len(scores), 4)

    def test_compare_chains(self) -> None:
        chain1 = KOrderMarkovChain(k=1)
        chain1.fit_from_arrays(np.array([0, 1, 0, 1], dtype=np.int32))

        chain2 = KOrderMarkovChain(k=1)
        chain2.fit_from_arrays(np.array([0, 1, 0, 1], dtype=np.int32))

        comp = compare_chains(chain1, chain2)
        self.assertIn("jsd_total", comp.columns)
        self.assertAlmostEqual(comp["jsd_total"][0], 0.0)

    def test_detect_period_cma(self) -> None:
        # Period 4 repeating pattern
        p4_seq = np.tile(np.array([0, 1, 2, 3], dtype=np.int32), 25)
        period4 = detect_period(p4_seq)
        self.assertEqual(period4, 4)

        # Period 2 repeating pattern
        p2_seq = np.tile(np.array([0, 1], dtype=np.int32), 30)
        period2 = detect_period(p2_seq)
        self.assertEqual(period2, 2)

        # Short sequence edge cases
        self.assertEqual(detect_period(np.array([0, 1], dtype=np.int32)), 1)
        self.assertEqual(detect_period(np.empty(0, dtype=np.int32)), 1)

    def test_joint_semi_markov_nll_two_factor(self) -> None:
        seq = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        dwells = np.array([5, 5, 5, 5, 5, 5], dtype=np.int32)
        chain_event = KOrderMarkovChain(k=1, mode="event", smoothing=0.01)
        chain_event.fit_from_arrays(seq, dwell_times=dwells)

        res = score_anomalies(chain_event, seq, dwell_times=dwells, mode="two_factor")
        self.assertEqual(len(res), len(seq))
        self.assertIn("nll_trans", res.columns)
        self.assertIn("nll_dwell", res.columns)
        self.assertIn("rolling_nll", res.columns)

    def test_detect_period_ignores_negative_states_r12(self) -> None:
        # Periodic 0, 1 with interleaved -1
        seq = np.array([-1, -1, -1, -1, -1], dtype=np.int32)
        # All -1 states should not crash and should return 1
        self.assertEqual(detect_period(seq), 1)

    def test_score_anomalies_custom_parameters_r11(self) -> None:
        seq = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        res_nll = score_anomalies(self.chain, seq, mode="nll", prob_floor=1e-6)
        self.assertEqual(len(res_nll), len(seq))
        res_weighted = score_anomalies(
            self.chain,
            seq,
            mode="two_factor",
            weight_trans=0.7,
            weight_dwell=0.3,
            penalty_unseen_trans=25.0,
        )
        self.assertEqual(len(res_weighted), len(seq))


if __name__ == "__main__":
    unittest.main()
