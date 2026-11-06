"""Unit tests for model serialization and Parquet export."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import polars as pl

from seqsm.estimator import SeqSM
from seqsm.io import export_schedule_parquet, load_model, save_model


class TestIO(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pl.DataFrame(
            {
                "sensor": ["A", "B", "A", "B"],
                "value": [1, 2, 1, 2],
            }
        )
        self.model = SeqSM(k=1)
        self.model.fit(self.df)

    def test_save_and_load_model_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "subfolder" / "model.pkl"
            save_model(self.model, model_path)
            self.assertTrue(model_path.exists())

            loaded = load_model(model_path)
            self.assertEqual(loaded.k, self.model.k)
            self.assertEqual(loaded.state_encoder.n_states, self.model.state_encoder.n_states)

            # Test predictions match
            p1 = self.model.predict_proba()
            p2 = loaded.predict_proba()
            self.assertEqual(list(p1), list(p2))

    def test_export_schedule_parquet(self) -> None:
        sched = self.model.generate_schedule(window=5, seed=1)
        with tempfile.TemporaryDirectory() as tmp_dir:
            pq_path = Path(tmp_dir) / "sched.parquet"
            export_schedule_parquet(sched, pq_path)
            self.assertTrue(pq_path.exists())

            read_df = pl.read_parquet(pq_path)
            self.assertEqual(len(read_df), len(sched))
            self.assertEqual(read_df.columns, sched.columns)


if __name__ == "__main__":
    unittest.main()
