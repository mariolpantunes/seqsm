"""Unit tests for top-level SeqSM estimator."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import polars as pl

from seqsm.estimator import SeqSM


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
        self.assertEqual(len(proba_df_hist), model.state_encoder.n_states)

        # History with unseen sequence falls back to marginal
        unseen_df = pl.DataFrame({"motion": [99], "light": ["unknown"]})
        proba_unseen = model.predict_proba(history=unseen_df)
        self.assertEqual(len(proba_unseen), model.state_encoder.n_states)

        # Predict with sample method
        pred_sample = model.predict(steps=1, method="sample")
        self.assertEqual(len(pred_sample), 1)

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


if __name__ == "__main__":
    unittest.main()
