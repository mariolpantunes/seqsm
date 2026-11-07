"""Comprehensive unit tests for StateEncoder and KWindowEncoder."""

from __future__ import annotations

import unittest

import numpy as np
import polars as pl

from seqsm.encoding import KWindowEncoder, StateEncoder


class TestStateEncoder(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pl.DataFrame(
            {
                "motion": [0, 1, 1, 0, 1],
                "light": ["off", "on", "on", "off", "dim"],
                "presence": [True, True, True, False, True],
            }
        )

    def test_fit_and_vocabulary(self) -> None:
        encoder = StateEncoder()
        encoder.fit(self.df)
        self.assertEqual(encoder.n_states, 4)  # 4 unique states
        vocab = encoder.vocabulary
        self.assertEqual(len(vocab), 4)
        self.assertIn("state_id", vocab.columns)

    def test_unfitted_access_raises(self) -> None:
        encoder = StateEncoder()
        with self.assertRaises(ValueError):
            _ = encoder.n_states
        with self.assertRaises(ValueError):
            _ = encoder.vocabulary
        with self.assertRaises(ValueError):
            _ = encoder.feature_cols
        with self.assertRaises(ValueError):
            encoder.transform(self.df)

    def test_explicit_feature_cols_and_missing_error(self) -> None:
        encoder = StateEncoder(feature_cols=["motion", "light"])
        encoder.fit(self.df)
        self.assertEqual(encoder.feature_cols, ("motion", "light"))

        with self.assertRaises(ValueError):
            bad_encoder = StateEncoder(feature_cols=["motion", "non_existent"])
            bad_encoder.fit(self.df)

    def test_transform_and_unseen_state(self) -> None:
        encoder = StateEncoder()
        encoder.fit(self.df)

        test_df = pl.DataFrame(
            {
                "motion": [0, 1, 0],
                "light": ["off", "dim", "neon"],  # "neon" is unseen
                "presence": [False, True, True],
            }
        )
        states = encoder.transform(test_df)
        self.assertEqual(len(states), 3)
        self.assertEqual(states[2], -1)  # unseen state maps to -1
        self.assertGreaterEqual(states[0], 0)
        self.assertGreaterEqual(states[1], 0)

    def test_inverse_transform(self) -> None:
        encoder = StateEncoder(feature_cols=["motion", "light"])
        states = encoder.fit_transform(self.df)

        inv_df = encoder.inverse_transform(states)
        self.assertEqual(len(inv_df), len(self.df))
        self.assertEqual(inv_df["motion"].to_list(), self.df["motion"].to_list())
        self.assertEqual(inv_df["light"].to_list(), self.df["light"].to_list())

        # Unknown ID maps to None
        inv_unk = encoder.inverse_transform([-1, 999])
        self.assertIsNone(inv_unk["motion"][0])
        self.assertIsNone(inv_unk["light"][1])

    def test_state_to_dict_and_diff(self) -> None:
        encoder = StateEncoder()
        encoder.fit(self.df)
        s0_dict = encoder.state_to_dict(0)
        self.assertIn("motion", s0_dict)

        with self.assertRaises(KeyError):
            encoder.state_to_dict(999)

        diff = encoder.state_diff(0, 1)
        self.assertIsInstance(diff, dict)

    def test_lazyframe_support(self) -> None:
        encoder = StateEncoder()
        lazy_df = self.df.lazy()
        encoder.fit(lazy_df)
        states = encoder.transform(lazy_df)
        self.assertEqual(len(states), len(self.df))

    def test_state_encoder_transform_preserves_left_row_order_n8(self) -> None:
        # Create varied rows with non-monotonic category occurrences
        raw_data = {
            "motion": [0, 1, 0, 1, 1, 0, 0, 1],
            "light": ["dim", "on", "off", "on", "dim", "off", "on", "dim"],
            "presence": [False, True, True, True, False, False, True, False],
        }
        df = pl.DataFrame(raw_data)
        encoder = StateEncoder()
        states = encoder.fit_transform(df)

        # Invariant 1: transformed array length strictly equals input DataFrame height
        self.assertEqual(len(states), len(df))

        # Invariant 2: inverse transformation preserves exact row sequence and values
        reconstructed = encoder.inverse_transform(states)
        self.assertTrue(df.equals(reconstructed))

        # Invariant 3: arbitrary shuffled order is strictly preserved
        permutation = [7, 2, 5, 0, 3, 1, 6, 4]
        shuffled_df = df[permutation]
        shuffled_states = encoder.transform(shuffled_df)
        np.testing.assert_array_equal(shuffled_states, states[permutation])


class TestKWindowEncoder(unittest.TestCase):
    def test_invalid_k_raises(self) -> None:
        with self.assertRaises(ValueError):
            KWindowEncoder(k=0)

    def test_fit_transform_k1(self) -> None:
        encoder = KWindowEncoder(k=1)
        states = np.array([0, 1, 2, 1, 0], dtype=np.int32)
        sources_u, targets_v, times = encoder.fit_transform(states)

        self.assertEqual(len(sources_u), 4)
        self.assertEqual(len(targets_v), 4)
        np.testing.assert_array_equal(targets_v, np.array([1, 2, 1, 0], dtype=np.int32))
        np.testing.assert_array_equal(times, np.array([0, 1, 2, 3], dtype=np.int32))

    def test_fit_transform_k2_with_episodes(self) -> None:
        encoder = KWindowEncoder(k=2)
        # Episode 0: [0, 1, 2], Episode 1: [3, 4]
        states = np.array([0, 1, 2, 3, 4], dtype=np.int32)
        episodes = np.array([0, 0, 0, 1, 1], dtype=np.int32)

        sources_u, targets_v, _ = encoder.fit_transform(states, episodes)
        # Only (0, 1) -> 2 in episode 0 is valid for k=2 transition!
        self.assertEqual(len(sources_u), 1)
        self.assertEqual(targets_v[0], 2)
        self.assertEqual(encoder.decode_window(sources_u[0]), (0, 1))

    def test_unknown_state_filtering(self) -> None:
        encoder = KWindowEncoder(k=2)
        states = np.array([0, -1, 1, 2], dtype=np.int32)
        sources_u, targets_v, _ = encoder.fit_transform(states)
        self.assertEqual(len(sources_u), 0)

    def test_encode_and_decode_round_trip(self) -> None:
        encoder = KWindowEncoder(k=2)
        states = np.array([1, 2, 3, 1, 2, 4], dtype=np.int32)
        encoder.fit_transform(states)

        idx = encoder.encode_window((1, 2))
        self.assertGreaterEqual(idx, 0)
        self.assertEqual(encoder.decode_window(idx), (1, 2))

        # Unseen window
        self.assertEqual(encoder.encode_window((9, 9)), -1)

        # Invalid window length raises
        with self.assertRaises(ValueError):
            encoder.encode_window([1])

        with self.assertRaises(KeyError):
            encoder.decode_window(999)

    def test_k_tuples_property(self) -> None:
        encoder = KWindowEncoder(k=2)
        states = np.array([0, 1, 2], dtype=np.int32)
        encoder.fit_transform(states)
        tuples = encoder.k_tuples
        self.assertEqual(tuples.shape, (1, 2))
        np.testing.assert_array_equal(tuples[0], [0, 1])

    def test_k_window_sliding_view_vectorized_equivalence_n4(self) -> None:
        # Compare vectorized KWindowEncoder output against ground-truth loop extraction
        def loop_reference(
            k: int, states: np.ndarray, episodes: np.ndarray | None = None
        ) -> tuple[list[tuple[int, ...]], list[int], list[int]]:
            ref_windows: list[tuple[int, ...]] = []
            ref_targets: list[int] = []
            ref_times: list[int] = []
            n = len(states)
            for t in range(k - 1, n - 1):
                target = int(states[t + 1])
                if target < 0:
                    continue
                if episodes is not None and episodes[t + 1] != episodes[t]:
                    continue
                start = t - k + 1
                win = tuple(int(states[i]) for i in range(start, t + 1))
                if any(s < 0 for s in win):
                    continue
                if episodes is not None:
                    first_ep = episodes[start]
                    if any(episodes[i] != first_ep for i in range(start + 1, t + 1)):
                        continue
                ref_windows.append(win)
                ref_targets.append(target)
                ref_times.append(t)
            return ref_windows, ref_targets, ref_times

        rng = np.random.default_rng(42)
        for k in (1, 2, 3):
            for with_episodes in (False, True):
                raw_states = rng.integers(-1, 5, size=60, dtype=np.int32)
                raw_episodes = np.repeat(np.arange(6, dtype=np.int32), 10) if with_episodes else None

                encoder = KWindowEncoder(k=k)
                u_vec, v_vec, t_vec = encoder.fit_transform(raw_states, episode_ids=raw_episodes)
                ref_wins, ref_tgts, ref_ts = loop_reference(k, raw_states, raw_episodes)

                self.assertEqual(len(v_vec), len(ref_tgts))
                np.testing.assert_array_equal(v_vec, np.array(ref_tgts, dtype=np.int32))
                np.testing.assert_array_equal(t_vec, np.array(ref_ts, dtype=np.int32))
                for idx, win in enumerate(ref_wins):
                    self.assertEqual(encoder.decode_window(u_vec[idx]), win)

                # Test empty / short sequence handling
                empty_u, empty_v, empty_t = encoder.fit_transform(np.empty(0, dtype=np.int32))
                self.assertEqual(len(empty_u), 0)
                self.assertEqual(len(empty_v), 0)
                self.assertEqual(len(empty_t), 0)


if __name__ == "__main__":
    unittest.main()
