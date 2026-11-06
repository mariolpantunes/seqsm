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

    def test_save_model_type_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_path = Path(tmp_dir) / "bad.pkl"
            with self.assertRaises(TypeError):
                save_model("not_a_model", bad_path)  # type: ignore[arg-type]

    def test_load_model_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_model("/non/existent/path/model.pkl")

    def test_load_model_corrupted_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            corrupt_path = Path(tmp_dir) / "corrupt.pkl"
            with open(corrupt_path, "wb") as f:
                f.write(b"this is garbage non-pickle data 1234567890")
            with self.assertRaises(ValueError):
                load_model(corrupt_path)

    def test_load_model_truncated_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            trunc_path = Path(tmp_dir) / "trunc.pkl"
            save_model(self.model, trunc_path)
            # Truncate to first 20 bytes
            with open(trunc_path, "r+b") as f:
                f.truncate(20)
            with self.assertRaises(ValueError):
                load_model(trunc_path)

    def test_load_model_version_mismatch(self) -> None:
        import pickle

        with tempfile.TemporaryDirectory() as tmp_dir:
            mismatch_path = Path(tmp_dir) / "mismatch.pkl"
            payload = {
                "format": "seqsm",
                "version": "99.99.99",
                "model": self.model,
            }
            with open(mismatch_path, "wb") as f:
                pickle.dump(payload, f)

            with self.assertRaises(ValueError) as ctx:
                load_model(mismatch_path)
            self.assertIn("Version mismatch", str(ctx.exception))

    def test_load_model_invalid_format(self) -> None:
        import pickle

        with tempfile.TemporaryDirectory() as tmp_dir:
            invalid_path = Path(tmp_dir) / "invalid.pkl"
            # Bare pickle or missing format key
            with open(invalid_path, "wb") as f:
                pickle.dump({"random": 123}, f)

            with self.assertRaises(ValueError):
                load_model(invalid_path)

    def test_load_model_non_model_payload_type_error(self) -> None:
        import pickle

        with tempfile.TemporaryDirectory() as tmp_dir:
            invalid_path = Path(tmp_dir) / "invalid_type.pkl"
            with open(invalid_path, "wb") as f:
                pickle.dump({"format": "seqsm", "version": "0.1.0", "model": 123}, f)

            with self.assertRaises(TypeError):
                load_model(invalid_path)

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
