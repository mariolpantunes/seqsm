"""Model serialization, checkpointing, and Parquet export utilities."""

from __future__ import annotations

import pickle
from pathlib import Path

import polars as pl

from seqsm.estimator import SeqSM


def save_model(model: SeqSM, filepath: str | Path) -> None:
    """Serializes a fitted SeqSM model using pickle."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath: str | Path) -> SeqSM:
    """Deserializes a SeqSM model from disk."""
    path = Path(filepath)
    with open(path, "rb") as f:
        model: SeqSM = pickle.load(f)
    return model


def export_schedule_parquet(df: pl.DataFrame, filepath: str | Path) -> None:
    """Exports a generated schedule DataFrame to Parquet format."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


__all__ = ["export_schedule_parquet", "load_model", "save_model"]
