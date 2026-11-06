"""Model serialization, checkpointing, and Parquet export utilities."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import polars as pl

import seqsm
from seqsm.estimator import SeqSM


def save_model(model: SeqSM, filepath: str | Path) -> None:
    """Serializes a fitted SeqSM model using pickle with version metadata."""
    if not isinstance(model, SeqSM):
        raise TypeError(f"Expected model to be an instance of SeqSM, got {type(model).__name__}")
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "format": "seqsm",
        "version": seqsm.__version__,
        "model": model,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(filepath: str | Path) -> SeqSM:
    """Deserializes a SeqSM model from disk, validating version compatibility."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    try:
        with open(path, "rb") as f:
            payload = pickle.load(f)
    except Exception as exc:
        raise ValueError(f"Corrupted or invalid SeqSM model file: {exc}") from exc

    if not isinstance(payload, dict) or payload.get("format") != "seqsm":
        raise ValueError("Invalid SeqSM model file format.")

    file_version = payload.get("version")
    if file_version != seqsm.__version__:
        raise ValueError(
            f"Version mismatch: model was saved with seqsm {file_version}, current version is {seqsm.__version__}"
        )

    model = payload.get("model")
    if not isinstance(model, SeqSM):
        raise TypeError("Payload does not contain a valid SeqSM model instance.")
    return model


def export_schedule_parquet(df: pl.DataFrame, filepath: str | Path) -> None:
    """Exports a generated schedule DataFrame to Parquet format."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


__all__ = ["export_schedule_parquet", "load_model", "save_model"]
