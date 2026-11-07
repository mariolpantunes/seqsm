"""Unit tests for top-level SeqSM estimator."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from seqsm.estimator import SeqSM, parse_duration


class TestEstimator(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pl.DataFrame(
            {
                "timestamp": [
                    datetime(2026, 1, 1, 8, 0),
                    datetime(2026, 1, 1, 8, 5),
                    datetime(2026, 1, 1, 8, 10),
                    datetime(2026, 1, 1, 8, 15),
                    datetime(2026, 1, 1, 8, 20),
                ],
                "motion": [0, 1, 1, 0, 1],
                "light": ["off", "on", "on", "off", "dim"],
                "episode": [0, 0, 0, 1, 1],
            }
        )

    def test_unfitted_access_raises(self) -> None:
        model = SeqSM()
        with self.assertRaises(ValueError):
            _ = model.markov_chain
        with self.assertRaises(ValueError):
            _ = model.state_encoder
        with self.assertRaises(ValueError):
            _ = model.stationary_distribution
        with self.assertRaises(ValueError):
            model.predict()
        with self.assertRaises(ValueError):
            model.predict_proba()
        with self.assertRaises(ValueError):
            model.generate_schedule(window=10)

    def test_fit_and_feature_inference(self) -> None:
        model = SeqSM(k=1, episode_col="episode", timestamp_col="timestamp")
        model.fit(self.df)

        # Inferred feature columns should be motion and light only
        self.assertEqual(model.state_encoder.feature_cols, ("motion", "light"))
        self.assertGreater(model.state_encoder.n_states, 1)

        stat_df = model.stationary_distribution
        self.assertIn("stationary_prob", stat_df.columns)
        self.assertIn("motion", stat_df.columns)

    def test_fit_with_timestamp_gap_episodes(self) -> None:
        model = SeqSM(
            k=1,
            timestamp_col="timestamp",
            max_time_gap=timedelta(minutes=7),
            feature_cols=["motion", "light"],
        )
        model.fit(self.df)
        self.assertEqual(model.markov_chain.k, 1)

    def test_predict_proba_bounds_checking_m7(self) -> None:
        model = SeqSM(k=1, feature_cols=["motion", "light"])
        model.fit(self.df)
        n_states = model.state_encoder.n_states

        with self.assertRaises(ValueError):
            model.predict_proba(history=[-1])
        with self.assertRaises(ValueError):
            model.predict_proba(history=[n_states])
        with self.assertRaises(ValueError):
            model.predict(history=[-1])
        with self.assertRaises(ValueError):
            model.predict(history=[n_states + 5])
        with self.assertRaises(ValueError):
            model.generate_schedule(window=5, init_history=[-1])
        with self.assertRaises(ValueError):
            model.generate_schedule(window=5, init_history=[n_states])

    def test_predict_and_proba(self) -> None:
        model = SeqSM(k=1, feature_cols=["motion", "light"], smoothing=0.1)
        model.fit(self.df)

        pred_df = model.predict(steps=1, method="map")
        self.assertEqual(len(pred_df), 1)
        self.assertIn("motion", pred_df.columns)
        self.assertIn("light", pred_df.columns)

        proba = model.predict_proba(steps=1)
        self.assertEqual(len(proba), model.state_encoder.n_states)

        # Multi-step proba and history as DataFrame
        proba_df_hist = model.predict_proba(history=self.df.head(2), steps=2)
        self.assertEqual(proba_df_hist.shape, (2, model.state_encoder.n_states))

        # History with unseen sequence falls back to marginal
        unseen_df = pl.DataFrame({"motion": [99], "light": ["unknown"]})
        proba_unseen = model.predict_proba(history=unseen_df)
        self.assertEqual(len(proba_unseen), model.state_encoder.n_states)

        # Predict with sample method
        pred_sample = model.predict(steps=1, method="sample")
        self.assertEqual(len(pred_sample), 1)

    def test_partial_fit_incremental_and_decay_c2(self) -> None:
        # Fitting episode 0 then episode 1 with decay=1.0 matches full concatenated fit
        df_full = pl.DataFrame(
            {
                "motion": [0, 1, 0, 1, 1, 0],
                "light": ["off", "on", "off", "on", "on", "off"],
                "episode": [0, 0, 0, 1, 1, 1],
            }
        )
        ep0 = df_full.filter(pl.col("episode") == 0)
        ep1 = df_full.filter(pl.col("episode") == 1)

        m_full = SeqSM(k=1, episode_col="episode", feature_cols=["motion", "light"], smoothing=0.1)
        m_full.fit(df_full)

        m_inc = SeqSM(k=1, episode_col="episode", feature_cols=["motion", "light"], smoothing=0.1)
        m_inc.fit(ep0)
        m_inc.partial_fit(ep1, decay=1.0)

        # Count matrix and transition matrix must match full fit
        np.testing.assert_allclose(m_inc.markov_chain.count_matrix, m_full.markov_chain.count_matrix)
        np.testing.assert_allclose(m_inc.markov_chain.transition_matrix, m_full.markov_chain.transition_matrix)

        # Test decay < 1.0 discounts prior counts
        m_decay = SeqSM(k=1, episode_col="episode", feature_cols=["motion", "light"], smoothing=0.0)
        m_decay.fit(ep0)
        m_decay.partial_fit(ep1, decay=0.5)
        c_decay = m_decay.markov_chain.count_matrix
        self.assertLess(float(c_decay.sum()), float(m_full.markov_chain.count_matrix.sum()))

    def test_predict_multi_step_decoding_c3(self) -> None:
        model = SeqSM(k=1, feature_cols=["motion", "light"], smoothing=0.1)
        model.fit(self.df)

        # 1. Multi-step without history
        pred_3_map = model.predict(steps=3, method="map")
        self.assertEqual(len(pred_3_map), 3)
        self.assertIn("motion", pred_3_map.columns)
        self.assertIn("light", pred_3_map.columns)

        pred_3_sample = model.predict(steps=3, method="sample")
        self.assertEqual(len(pred_3_sample), 3)

        # 2. Multi-step with DataFrame history and integer sequence history
        pred_5_map = model.predict(history=self.df.head(2), steps=5, method="map")
        self.assertEqual(len(pred_5_map), 5)

        pred_5_sample = model.predict(history=[0, 1], steps=5, method="sample")
        self.assertEqual(len(pred_5_sample), 5)

        # Verify all decoded rows belong to known state vocabulary
        known_tuples = set(model.state_encoder.vocabulary.select(["motion", "light"]).iter_rows())
        for row in pred_5_map.iter_rows():
            self.assertIn(row, known_tuples)
        for row in pred_5_sample.iter_rows():
            self.assertIn(row, known_tuples)

        # 3. Verify MAP decodes exact argmax of each step
        probs_5 = model.predict_proba(history=self.df.head(2), steps=5)
        expected_sids = [int(np.argmax(probs_5[h])) for h in range(5)]
        expected_df = model.state_encoder.inverse_transform(expected_sids)
        self.assertTrue(pred_5_map.equals(expected_df))

    def test_duration_parsing_units_and_errors_h6(self) -> None:
        self.assertEqual(parse_duration("1d"), timedelta(days=1))
        self.assertEqual(parse_duration("12h"), timedelta(hours=12))
        self.assertEqual(parse_duration("30m"), timedelta(minutes=30))
        self.assertEqual(parse_duration("15s"), timedelta(seconds=15))
        self.assertEqual(parse_duration("500ms"), timedelta(milliseconds=500))
        self.assertEqual(parse_duration("10us"), timedelta(microseconds=10))
        self.assertEqual(parse_duration("2w"), timedelta(weeks=2))
        self.assertEqual(parse_duration(timedelta(minutes=5)), timedelta(minutes=5))

        with self.assertRaises(ValueError):
            parse_duration("invalid_duration")
        with self.assertRaises(ValueError):
            parse_duration("")
        with self.assertRaises(TypeError):
            parse_duration(123)  # type: ignore[arg-type]

    def test_resolve_episodes_string_and_numeric_h4(self) -> None:
        # String episode labels
        df_str = pl.DataFrame(
            {
                "motion": [0, 1, 0, 1],
                "ep": ["morning", "morning", "evening", "evening"],
            }
        )
        model = SeqSM(k=1, episode_col="ep", feature_cols=["motion"])
        model.fit(df_str)
        self.assertEqual(model.markov_chain.k, 1)

        # Max time gap as string duration
        df_gap = pl.DataFrame(
            {
                "timestamp": [datetime(2026, 1, 1, 8, 0), datetime(2026, 1, 1, 9, 0)],
                "motion": [0, 1],
            }
        )
        model_gap = SeqSM(k=1, timestamp_col="timestamp", max_time_gap="30m", feature_cols=["motion"])
        model_gap.fit(df_gap)
        self.assertEqual(model_gap.markov_chain.k, 1)

    def test_generate_schedule_with_init_history(self) -> None:
        model = SeqSM(k=1, feature_cols=["motion", "light"])
        model.fit(self.df)

        sched_df_hist = model.generate_schedule(window=5, init_history=self.df.head(2))
        self.assertGreater(len(sched_df_hist), 0)

        sched_arr_hist = model.generate_schedule(window=5, init_history=[0, 1])
        self.assertGreater(len(sched_arr_hist), 0)

        # Partial fit check
        model.partial_fit(self.df)

    def test_generate_schedule_modes(self) -> None:
        model = SeqSM(k=1, feature_cols=["motion", "light"])
        model.fit(self.df)

        # 1. Integer step window
        sched_int = model.generate_schedule(window=10, n_schedules=2, seed=42)
        self.assertIn("schedule_id", sched_int.columns)
        self.assertIn("time_step", sched_int.columns)
        self.assertIn("timestamp", sched_int.columns)
        self.assertIn("motion", sched_int.columns)
        self.assertIn("light", sched_int.columns)
        self.assertEqual(len(sched_int["schedule_id"].unique()), 2)

        # 2. Timedelta window
        sched_td = model.generate_schedule(
            window=timedelta(minutes=15),
            time_unit="5m",
            start_time=datetime(2026, 1, 1, 9, 0),
        )
        self.assertGreater(len(sched_td), 0)

        # 3. Tuple window (start, end)
        sched_tuple = model.generate_schedule(
            window=(datetime(2026, 1, 1, 10, 0), datetime(2026, 1, 1, 10, 30)),
            time_unit="1m",
        )
        self.assertGreater(len(sched_tuple), 0)

    def test_score_anomalies_and_to_state_machine(self) -> None:
        model = SeqSM(k=1, feature_cols=["motion", "light"])
        model.fit(self.df)

        anomalies = model.score_anomalies(self.df)
        self.assertEqual(len(anomalies), len(self.df))

        fsm = model.to_state_machine(min_prob=0.0)
        edges = fsm.edges_dataframe()
        self.assertGreater(len(edges), 0)

    def test_detect_period(self) -> None:
        model = SeqSM(k=1, feature_cols=["motion", "light"])
        model.fit(self.df)
        period = model.detect_period(self.df)
        self.assertGreaterEqual(period, 1)

    def test_predict_method_viterbi_r4(self) -> None:
        model = SeqSM(k=1, feature_cols=["motion", "light"])
        model.fit(self.df)
        pred_vit = model.predict(steps=3, method="viterbi")
        self.assertEqual(len(pred_vit), 3)
        self.assertIn("motion", pred_vit.columns)
        self.assertIn("light", pred_vit.columns)

    def test_parse_duration_zero_and_compound_r16(self) -> None:
        from seqsm.estimator import parse_duration

        with self.assertRaises(ValueError) as ctx:
            parse_duration("0s")
        self.assertIn("Duration must be strictly positive", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            parse_duration("1h30m")
        self.assertIn("compound strings", str(ctx2.exception))

    def test_generate_schedule_constraint_violated_column_r5(self) -> None:
        model = SeqSM(k=1, feature_cols=["motion", "light"])
        model.fit(self.df)
        sched = model.generate_schedule(window=5, seed=123)
        self.assertIn("constraint_violated", sched.columns)
        self.assertIn("arc_valid", sched.columns)

    def test_detect_period_with_timestamp_col_r12(self) -> None:
        df_ts = self.df.with_columns(
            pl.datetime_range(datetime(2026, 1, 1), datetime(2026, 1, 1, 4), interval="1h", eager=True).alias("ts")
        )
        model = SeqSM(k=1, feature_cols=["motion", "light"], timestamp_col="ts")
        model.fit(df_ts)
        period = model.detect_period(df_ts, timestamp_col="ts")
        self.assertGreaterEqual(period, 1)

    def test_predict_proba_diagnostics_n7(self) -> None:
        model = SeqSM(k=2, feature_cols=["motion", "light"])
        model.fit(self.df)

        # 1. history is None
        probs, diag = model.predict_proba(history=None, return_diagnostics=True)
        self.assertTrue(diag["fallback"])
        self.assertEqual(diag["reason"], "none_history")
        self.assertIsNone(diag["window"])
        self.assertEqual(diag["window_id"], -1)
        self.assertEqual(len(probs), model.state_encoder.n_states)

        # 2. short history (< k)
        probs_short, diag_short = model.predict_proba(history=[0], return_diagnostics=True)
        self.assertTrue(diag_short["fallback"])
        self.assertEqual(diag_short["reason"], "short_history")
        self.assertEqual(diag_short["window"], (0,))
        self.assertEqual(diag_short["window_id"], -1)

        # 3. unseen window
        all_states = list(range(model.state_encoder.n_states))
        unseen_win = None
        for s1 in all_states:
            for s2 in all_states:
                if model._chain.encoder.encode_window((s1, s2)) < 0:  # type: ignore[union-attr]
                    unseen_win = (s1, s2)
                    break
            if unseen_win is not None:
                break
        self.assertIsNotNone(unseen_win)
        assert unseen_win is not None
        probs_unseen, diag_unseen = model.predict_proba(history=list(unseen_win), return_diagnostics=True)
        self.assertTrue(diag_unseen["fallback"])
        self.assertEqual(diag_unseen["reason"], "unseen_window")
        self.assertEqual(diag_unseen["window"], unseen_win)
        self.assertEqual(diag_unseen["window_id"], -1)

        # 4. attested window
        attested_win = model._chain.encoder.decode_window(0)  # type: ignore[union-attr]
        probs_attested, diag_attested = model.predict_proba(history=list(attested_win), return_diagnostics=True)
        self.assertFalse(diag_attested["fallback"])
        self.assertIsNone(diag_attested["reason"])
        self.assertEqual(diag_attested["window"], attested_win)
        self.assertEqual(diag_attested["window_id"], 0)

        # 5. return_diagnostics=False returns pure array
        p_raw = model.predict_proba(history=list(attested_win), return_diagnostics=False)
        self.assertIsInstance(p_raw, np.ndarray)
        np.testing.assert_array_equal(p_raw, probs_attested)


if __name__ == "__main__":
    unittest.main()
