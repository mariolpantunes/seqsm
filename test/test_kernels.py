"""Unit tests for Numba-accelerated kernels."""

from __future__ import annotations

import unittest

import numpy as np

from seqsm.kernels import (
    build_valid_arc_csr_njit,
    cma_detect_period_njit,
    compute_transition_probs_njit,
    count_transitions_njit,
    rle_collapse_njit,
    viterbi_decode_csr_njit,
    viterbi_decode_path_njit,
)


class TestKernels(unittest.TestCase):
    def test_count_transitions_njit(self) -> None:
        sources = np.array([0, 0, 1, 1, 2, -1, 5], dtype=np.int32)
        targets = np.array([0, 1, 0, 1, 0, 1, 1], dtype=np.int32)
        n_sources = 3
        n_targets = 2

        counts = count_transitions_njit(sources, targets, n_sources, n_targets)
        self.assertEqual(counts.shape, (3, 2))
        self.assertEqual(counts[0, 0], 1)
        self.assertEqual(counts[0, 1], 1)
        self.assertEqual(counts[1, 0], 1)
        self.assertEqual(counts[1, 1], 1)
        self.assertEqual(counts[2, 0], 1)
        self.assertEqual(counts[2, 1], 0)

    def test_compute_transition_probs_row_stochastic(self) -> None:
        counts = np.array([[10, 0], [0, 0]], dtype=np.int64)

        # No smoothing
        probs = compute_transition_probs_njit(counts, 0.0)
        np.testing.assert_allclose(probs[0], [1.0, 0.0])
        # Zero-count row defaults to uniform
        np.testing.assert_allclose(probs[1], [0.5, 0.5])

        # Lidstone smoothing
        probs_smooth = compute_transition_probs_njit(counts, 1.0)
        self.assertAlmostEqual(probs_smooth[0, 0], 11.0 / 12.0)
        self.assertAlmostEqual(probs_smooth[0, 1], 1.0 / 12.0)
        np.testing.assert_allclose(probs_smooth.sum(axis=1), [1.0, 1.0])

    def test_rle_collapse_njit(self) -> None:
        # Empty
        s_empty = np.empty(0, dtype=np.int32)
        e_empty = np.empty(0, dtype=np.int32)
        states, dwells, eps_out = rle_collapse_njit(s_empty, e_empty)
        self.assertEqual(len(states), 0)
        self.assertEqual(len(dwells), 0)
        self.assertEqual(len(eps_out), 0)

        # Sequence with runs and episode break
        # Episode 0: state 1 (x3), state 2 (x1)
        # Episode 1: state 2 (x2) -> episode break should split state 2
        seq = np.array([1, 1, 1, 2, 2, 2], dtype=np.int32)
        eps = np.array([0, 0, 0, 0, 1, 1], dtype=np.int32)

        out_s, out_d, out_e = rle_collapse_njit(seq, eps)
        self.assertEqual(len(out_s), 3)
        np.testing.assert_array_equal(out_s, [1, 2, 2])
        np.testing.assert_array_equal(out_d, [3, 1, 2])
        np.testing.assert_array_equal(out_e, [0, 0, 1])

    def test_build_valid_arc_csr_njit(self) -> None:
        probs = np.array([[0.8, 0.2], [0.05, 0.95]], dtype=np.float64)
        counts = np.array([[80, 20], [5, 95]], dtype=np.int64)

        # Filter out transitions with count < 10 or prob < 0.1
        indptr, indices, filtered_probs, cumprobs = build_valid_arc_csr_njit(probs, counts, 0.1, 10)
        self.assertEqual(len(indptr), 3)

        # Row 0 has 2 valid arcs: 0 and 1
        self.assertEqual(indptr[1] - indptr[0], 2)
        np.testing.assert_allclose(filtered_probs[indptr[0] : indptr[1]], [0.8, 0.2])

        # Row 1 has only 1 valid arc: 1 (prob 0.95, count 95). Renormalized to 1.0!
        self.assertEqual(indptr[2] - indptr[1], 1)
        self.assertEqual(indices[indptr[1]], 1)
        self.assertAlmostEqual(filtered_probs[indptr[1]], 1.0)
        self.assertAlmostEqual(cumprobs[indptr[1]], 1.0)

    def test_build_valid_arc_fallback_isolated(self) -> None:
        # All transitions below threshold -> falls back to argmax
        probs = np.array([[0.3, 0.7]], dtype=np.float64)
        counts = np.array([[1, 2]], dtype=np.int64)

        indptr, indices, filtered_probs, cumprobs = build_valid_arc_csr_njit(probs, counts, 0.9, 10)
        self.assertEqual(indptr[1], 1)
        self.assertEqual(indices[0], 1)  # argmax is 1
        self.assertEqual(filtered_probs[0], 1.0)
        self.assertEqual(cumprobs[0], 1.0)

    def test_viterbi_decode_path_njit(self) -> None:
        # 2 states. Deterministic transition 0 -> 1 and 1 -> 0
        log_comp = np.array([[-1e30, 0.0], [0.0, -1e30]], dtype=np.float64)
        init_log = np.array([0.0, -1e30], dtype=np.float64)  # start at 0
        horizon = 4

        path = viterbi_decode_path_njit(log_comp, init_log, horizon)
        np.testing.assert_array_equal(path, [0, 1, 0, 1])

    def test_viterbi_brute_force_reference_h3(self) -> None:
        import itertools

        rng = np.random.default_rng(12345)
        for n_states in [3, 4]:
            for horizon in [3, 4, 5]:
                log_comp = rng.uniform(-5.0, 0.0, size=(n_states, n_states))
                init_log = rng.uniform(-5.0, 0.0, size=n_states)

                viterbi_path = viterbi_decode_path_njit(log_comp, init_log, horizon)
                viterbi_score = init_log[viterbi_path[0]] + sum(
                    log_comp[viterbi_path[h - 1], viterbi_path[h]] for h in range(1, horizon)
                )

                best_score = -np.inf
                best_path: tuple[int, ...] = ()
                for cand_path in itertools.product(range(n_states), repeat=horizon):
                    score = init_log[cand_path[0]] + sum(
                        log_comp[cand_path[h - 1], cand_path[h]] for h in range(1, horizon)
                    )
                    if score > best_score:
                        best_score = score
                        best_path = cand_path

                self.assertAlmostEqual(viterbi_score, best_score, places=9)
                np.testing.assert_array_equal(viterbi_path, np.array(best_path, dtype=np.int32))

    def test_viterbi_sparse_csr_matches_brute_force_n3(self) -> None:
        import itertools

        rng = np.random.default_rng(54321)
        for n_states in [3, 4]:
            for horizon in [3, 4, 5]:
                dense_mat = rng.uniform(-5.0, 0.0, size=(n_states, n_states))
                for r in range(n_states):
                    dense_mat[r, int(rng.integers(0, n_states))] = -1e20

                indptr = np.zeros(n_states + 1, dtype=np.int32)
                indices_list: list[int] = []
                data_list: list[float] = []
                for r in range(n_states):
                    for c in range(n_states):
                        val = dense_mat[r, c]
                        if val > -1e10:
                            indices_list.append(c)
                            data_list.append(val)
                    indptr[r + 1] = len(indices_list)

                csr_indptr = indptr
                csr_indices = np.array(indices_list, dtype=np.int32)
                csr_log_data = np.array(data_list, dtype=np.float64)

                init_log = rng.uniform(-5.0, 0.0, size=n_states)

                csr_path = viterbi_decode_csr_njit(csr_indptr, csr_indices, csr_log_data, init_log, horizon)
                csr_score = init_log[csr_path[0]] + sum(
                    dense_mat[csr_path[h - 1], csr_path[h]] for h in range(1, horizon)
                )

                best_score = -np.inf
                best_path: tuple[int, ...] = ()
                for cand_path in itertools.product(range(n_states), repeat=horizon):
                    score = init_log[cand_path[0]] + sum(
                        dense_mat[cand_path[h - 1], cand_path[h]] for h in range(1, horizon)
                    )
                    if score > best_score:
                        best_score = score
                        best_path = cand_path

                self.assertAlmostEqual(csr_score, best_score, places=9)
                np.testing.assert_array_equal(csr_path, np.array(best_path, dtype=np.int32))

    def test_cma_detect_period_njit(self) -> None:
        p4_seq = np.tile(np.array([0, 1, 2, 3], dtype=np.int32), 15)
        p_base = 0.25
        denom = 0.75
        tau, rho = cma_detect_period_njit(p4_seq, 20, p_base, denom, 0.05)
        self.assertEqual(tau, 4)
        self.assertAlmostEqual(rho, 1.0)


if __name__ == "__main__":
    unittest.main()
