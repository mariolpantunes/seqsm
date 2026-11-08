"""Model serialization, checkpointing, and Parquet export utilities."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import seqsm
from seqsm.encoding import StateEncoder
from seqsm.estimator import SeqSM
from seqsm.markov import KOrderMarkovChain
from seqsm.periodic import PhaseConditionedMarkovChain
from seqsm.timeslots import TimeConfig, phase_regimes, preset_regimes
from seqsm.timeutil import duration_us
from seqsm.types import CSRMatrix, DwellDistribution


def save_model(model: SeqSM, filepath: str | Path) -> None:
    """Serializes a fitted SeqSM model using a safe, single-file compressed NPZ container."""
    if not isinstance(model, SeqSM):
        raise TypeError(f"Expected model to be an instance of SeqSM, got {type(model).__name__}")
    if model._chain is None or model._encoder is None:
        raise ValueError("SeqSM model must be fitted before saving.")

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    max_time_gap_val: float | None = None
    max_time_gap_is_td = False
    if isinstance(model.max_time_gap, timedelta):
        max_time_gap_val = model.max_time_gap.total_seconds()
        max_time_gap_is_td = True
    elif isinstance(model.max_time_gap, (int, float)):
        max_time_gap_val = float(model.max_time_gap)

    vocab_records = model._encoder.vocabulary.to_dicts()
    dwell_stats_dict = {
        str(sid): {
            "state_id": d.state_id,
            "count": d.count,
            "mean": d.mean,
            "std": d.std,
            "median": d.median,
            "min_dwell": d.min_dwell,
            "max_dwell": d.max_dwell,
            "samples": list(d.samples),
        }
        for sid, d in model._chain.dwell_stats.items()
    }

    saved_feature_cols = list(model._encoder.feature_cols)

    metadata: dict[str, Any] = {
        "format": "seqsm",
        "version": seqsm.__version__,
        "estimator_params": {
            "k": model.k,
            "mode": model.mode,
            "smoothing": model.smoothing,
            "feature_cols": saved_feature_cols,
            "episode_col": model.episode_col,
            "timestamp_col": model.timestamp_col,
            "max_time_gap_val": max_time_gap_val,
            "max_time_gap_is_td": max_time_gap_is_td,
            "min_prob": model.min_prob,
            "min_count": model.min_count,
            "seed": model.seed,
            "dtype": str(np.dtype(model.dtype)),
        },
        "chain_params": {
            "k": model._chain.k,
            "mode": model._chain.mode,
            "smoothing": model._chain._smoothing,
            "sparse_threshold": model._chain._sparse_threshold,
            "dtype": str(np.dtype(model._chain._dtype)),
            "n_states": model._chain.n_states,
        },
        "state_vocabulary": vocab_records,
        "dwell_stats": dwell_stats_dict,
    }

    meta_bytes = np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8)

    save_dict: dict[str, Any] = {
        "metadata_bytes": meta_bytes,
        "k_tuples": np.asarray(model._chain.encoder.k_tuples, dtype=np.int32),
        "count_matrix": np.asarray(model._chain.count_matrix, dtype=np.float64),
        "transition_matrix": np.asarray(model._chain.transition_matrix, dtype=np.float64),
    }

    if model._chain._companion_matrix is not None:
        save_dict["companion_matrix"] = np.asarray(model._chain._companion_matrix, dtype=np.float64)
        save_dict["has_companion_matrix"] = np.array([True])
    else:
        save_dict["has_companion_matrix"] = np.array([False])

    if model._chain._sparse_companion is not None:
        save_dict["csr_data"] = np.asarray(model._chain._sparse_companion.data, dtype=np.float64)
        save_dict["csr_indices"] = np.asarray(model._chain._sparse_companion.indices, dtype=np.int32)
        save_dict["csr_indptr"] = np.asarray(model._chain._sparse_companion.indptr, dtype=np.int32)
        save_dict["csr_shape"] = np.array(model._chain._sparse_companion.shape, dtype=np.int32)
        save_dict["has_sparse_companion"] = np.array([True])
    else:
        save_dict["has_sparse_companion"] = np.array([False])

    if model._chain._cached_pi is not None:
        save_dict["cached_pi"] = np.asarray(model._chain._cached_pi, dtype=np.float64)
        save_dict["has_cached_pi"] = np.array([True])
    else:
        save_dict["has_cached_pi"] = np.array([False])

    if model._chain._cached_mu is not None:
        save_dict["cached_mu"] = np.asarray(model._chain._cached_mu, dtype=np.float64)
        save_dict["has_cached_mu"] = np.array([True])
    else:
        save_dict["has_cached_mu"] = np.array([False])

    phase = model._phase
    if phase is not None:
        if model._regime_meta.get("kind") == "callable":
            raise ValueError("Models with a callable regimes function cannot be serialized; use a preset or 'phase'.")
        tc = model.time_config
        ctx_vocab: list[list[Any]] = []
        for key, cid in model._context_ids.items():
            if not all(isinstance(v, (str, int, float, bool)) or v is None for v in key):
                raise ValueError("Context values must be str/int/float/bool to be serialized.")
            ctx_vocab.append([*key, cid])
        metadata["phase"] = {
            "n_regimes": phase.n_regimes,
            "min_regime_count": phase.min_regime_count,
            "n_time_regimes": model._n_time_regimes,
            "regime_meta": model._regime_meta,
            "step_us": int(model._step.total_seconds() * 1_000_000) if model._step is not None else None,
            "context_cols": list(model._eff_context_cols),
            "context_vocab": ctx_vocab,
            "timestamp_col": model._eff_timestamp_col,
            "episode_col": model._eff_episode_col,
            "has_time_config": tc is not None,
            "slot_duration_us": duration_us(tc.slot_duration) if tc is not None and tc.slot_duration else None,
            "fill": tc.fill if tc is not None else "ffill",
            "dwell_samples": {
                str(r): {str(s): list(v) for s, v in per.items()} for r, per in phase.regime_dwell_samples.items()
            },
        }
        save_dict["phase_counts"] = np.asarray(phase.regime_counts, dtype=np.float64)
        save_dict["metadata_bytes"] = np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8)

    with open(path, "wb") as f:
        np.savez_compressed(f, **save_dict)


def _restore_phase(model: SeqSM, chain: KOrderMarkovChain, meta: dict[str, Any], counts: Any) -> None:
    """Rebuilds the regime-conditioned chain, regime function and context vocabulary of a saved model."""
    rmeta = meta["regime_meta"]
    if rmeta["kind"] == "phase":
        spec = phase_regimes(timedelta(microseconds=int(rmeta["period_us"])), int(rmeta["n_regimes"]))
    else:
        spec = preset_regimes(str(rmeta["name"]))
    model._regime_spec = spec
    model._regime_meta = dict(rmeta)
    model._n_time_regimes = int(meta["n_time_regimes"])
    model._step = timedelta(microseconds=int(meta["step_us"])) if meta["step_us"] is not None else None
    model._eff_context_cols = tuple(meta["context_cols"])
    model.context_cols = model._eff_context_cols or None
    model._context_ids = {tuple(row[:-1]): int(row[-1]) for row in meta["context_vocab"]}
    model._eff_timestamp_col = meta["timestamp_col"]
    model._eff_episode_col = meta["episode_col"]
    if meta["has_time_config"]:
        regimes = "phase" if rmeta["kind"] == "phase" else str(rmeta["name"])
        slot_us = meta["slot_duration_us"]
        model.time_config = TimeConfig(
            slot_duration=timedelta(microseconds=int(slot_us)) if slot_us is not None else None,
            regimes=regimes,
            period=timedelta(microseconds=int(rmeta["period_us"])) if rmeta["kind"] == "phase" else None,
            n_regimes=int(rmeta["n_regimes"]) if rmeta["kind"] == "phase" else None,
            fill=meta["fill"],
            min_regime_count=int(meta["min_regime_count"]),
        )
    dwell = {int(r): {int(s): [int(x) for x in v] for s, v in per.items()} for r, per in meta["dwell_samples"].items()}
    model._phase = PhaseConditionedMarkovChain(
        k=chain.k,
        n_regimes=int(meta["n_regimes"]),
        mode=chain.mode,
        smoothing=chain._smoothing,
        min_regime_count=int(meta["min_regime_count"]),
        dtype=chain._dtype,
    ).set_fitted(chain, counts, dwell)


def load_model(filepath: str | Path) -> SeqSM:
    """Deserializes a SeqSM model from disk safely using NPZ without pickle execution."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    try:
        with open(path, "rb") as f:
            loader = None
            try:
                try:
                    loader = np.load(f, allow_pickle=False)
                except Exception as exc:
                    raise ValueError(f"Corrupted or invalid SeqSM model file: {exc}") from exc

                if "metadata_bytes" not in loader:
                    raise ValueError("Invalid SeqSM model file format.")

                meta_raw = loader["metadata_bytes"].tobytes().decode("utf-8")
                payload = json.loads(meta_raw)

                if not isinstance(payload, dict) or payload.get("format") != "seqsm":
                    raise ValueError("Invalid SeqSM model file format.")

                file_version = str(payload.get("version", ""))
                curr_version = str(seqsm.__version__)
                file_major_minor = file_version.split(".")[:2]
                curr_major_minor = curr_version.split(".")[:2]
                if file_major_minor != curr_major_minor:
                    raise ValueError(
                        f"Version mismatch: model was saved with seqsm {file_version}, current version is {seqsm.__version__}. "
                        "Major and minor versions must match for model compatibility."
                    )

                est_params = payload.get("estimator_params")
                if not isinstance(est_params, dict):
                    raise TypeError("Payload does not contain valid estimator parameters.")

                max_tg: timedelta | float | None = None
                if est_params.get("max_time_gap_val") is not None:
                    if est_params.get("max_time_gap_is_td", False):
                        max_tg = timedelta(seconds=float(est_params["max_time_gap_val"]))
                    else:
                        max_tg = float(est_params["max_time_gap_val"])

                model = SeqSM(
                    k=int(est_params["k"]),
                    mode=est_params["mode"],
                    smoothing=float(est_params["smoothing"]),
                    feature_cols=est_params["feature_cols"],
                    episode_col=est_params["episode_col"],
                    timestamp_col=est_params["timestamp_col"],
                    max_time_gap=max_tg,
                    min_prob=float(est_params["min_prob"]),
                    min_count=int(est_params["min_count"]),
                    seed=est_params["seed"],
                    dtype=np.dtype(est_params["dtype"]),
                )

                vocab_data = payload.get("state_vocabulary")
                if not isinstance(vocab_data, list):
                    raise TypeError("Invalid vocabulary data in model file.")
                vocab_df = pl.DataFrame(vocab_data)
                enc = StateEncoder(feature_cols=model.feature_cols)
                enc._vocabulary = vocab_df
                for row in vocab_df.iter_rows(named=True):
                    sid = int(row["state_id"])
                    tup = tuple(row[col] for col in enc.feature_cols)
                    enc._tuple_to_id[tup] = sid
                    enc._id_to_tuple[sid] = tup
                model._encoder = enc

                chain_params = payload.get("chain_params")
                if not isinstance(chain_params, dict):
                    raise TypeError("Payload does not contain valid chain parameters.")

                chain = KOrderMarkovChain(
                    k=int(chain_params["k"]),
                    mode=chain_params["mode"],
                    smoothing=float(chain_params["smoothing"]),
                    sparse_threshold=int(chain_params["sparse_threshold"]),
                    dtype=np.dtype(chain_params["dtype"]),
                )
                chain._n_states = int(chain_params["n_states"])

                k_tuples = np.array(loader["k_tuples"], dtype=np.int32)
                k_enc = chain._encoder
                for idx in range(len(k_tuples)):
                    tup = tuple(int(x) for x in k_tuples[idx])
                    k_enc._id_to_window[idx] = tup
                    k_enc._window_to_id[tup] = idx
                k_enc._cached_k_tuples = k_tuples

                chain._count_matrix = np.array(loader["count_matrix"], dtype=np.float64)
                chain._transition_matrix = np.array(loader["transition_matrix"], dtype=np.float64)

                if bool(loader["has_companion_matrix"][0]):
                    chain._companion_matrix = np.array(loader["companion_matrix"], dtype=np.float64)

                if bool(loader["has_sparse_companion"][0]):
                    chain._sparse_companion = CSRMatrix(
                        data=np.array(loader["csr_data"], dtype=np.float64),
                        indices=np.array(loader["csr_indices"], dtype=np.int32),
                        indptr=np.array(loader["csr_indptr"], dtype=np.int32),
                        shape=(int(loader["csr_shape"][0]), int(loader["csr_shape"][1])),
                    )

                if bool(loader["has_cached_pi"][0]):
                    chain._cached_pi = np.array(loader["cached_pi"], dtype=np.float64)

                if bool(loader["has_cached_mu"][0]):
                    chain._cached_mu = np.array(loader["cached_mu"], dtype=np.float64)

                dwell_data = payload.get("dwell_stats", {})
                if isinstance(dwell_data, dict):
                    chain._dwell_stats = {
                        int(sid): DwellDistribution(
                            state_id=int(d["state_id"]),
                            count=int(d["count"]),
                            mean=float(d["mean"]),
                            std=float(d["std"]),
                            median=float(d["median"]),
                            min_dwell=int(d["min_dwell"]),
                            max_dwell=int(d["max_dwell"]),
                            samples=tuple(int(x) for x in d["samples"]),
                        )
                        for sid, d in dwell_data.items()
                    }

                model._chain = chain
                phase_meta = payload.get("phase")
                if isinstance(phase_meta, dict):
                    _restore_phase(model, chain, phase_meta, np.array(loader["phase_counts"], dtype=np.float64))
                return model
            finally:
                if loader is not None:
                    loader.close()
    except FileNotFoundError:
        raise
    except (TypeError, ValueError) as exc:
        if "pickled" in str(exc) or "Cannot load" in str(exc):
            raise ValueError(f"Corrupted or invalid SeqSM model file: {exc}") from exc
        raise
    except Exception as exc:
        raise ValueError(f"Corrupted or invalid SeqSM model file: {exc}") from exc


def export_schedule_parquet(df: pl.DataFrame, filepath: str | Path) -> None:
    """Exports a generated schedule DataFrame to Parquet format."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


__all__ = ["export_schedule_parquet", "load_model", "save_model"]
