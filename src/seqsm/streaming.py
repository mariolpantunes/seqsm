"""Online/streaming wrapper around SeqSM for live sensor sources and viewers."""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any, Literal

import numpy as np
import polars as pl

from seqsm.estimator import SeqSM


class StreamingSeqSM:
    """Incremental state-machine builder for single-event live data streams.

    Maintains a rolling buffer of the last ``k`` discretized observations so that each
    incoming sensor reading produces a valid ``(k + 1)``-step transition for
    :meth:`SeqSM.partial_fit`.
    """

    def __init__(
        self,
        k: int = 1,
        mode: Literal["clock", "event"] = "clock",
        smoothing: float = 1e-3,
        decay: float = 0.98,
        bins: Mapping[str, Sequence[float]] | None = None,
    ) -> None:
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = k
        self.decay = decay
        self.bins = {col: list(edges) for col, edges in (bins or {}).items()}
        self.model = SeqSM(k=k, mode=mode, smoothing=smoothing)
        self._buffer: deque[dict[str, Any]] = deque(maxlen=k + 1)
        self._fitted = False
        self.n_events = 0

    def _discretize(self, record: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for col, val in record.items():
            if col in self.bins and isinstance(val, (int, float)):
                edges = self.bins[col]
                idx = bisect_right(edges, float(val)) - 1
                idx = max(0, min(idx, len(edges) - 2))
                out[col] = f"{edges[idx]:g}-{edges[idx + 1]:g}"
            else:
                out[col] = val
        return out

    def update(self, record: Mapping[str, Any], top_k: int = 5) -> dict[str, Any]:
        """Ingest one sensor reading, update the state machine, and predict the next state."""
        discrete = self._discretize(record)
        self._buffer.append(discrete)
        self.n_events += 1

        if len(self._buffer) >= self.k + 1:
            window_df = pl.DataFrame(list(self._buffer))
            if not self._fitted:
                self.model.fit(window_df)
                self._fitted = True
            else:
                self.model.partial_fit(window_df, decay=self.decay)

        predictions = self.predict_next(top_k=top_k) if self._fitted else []
        graph = self.state_machine_json() if self._fitted else {"nodes": [], "edges": []}
        return {
            "event": self.n_events,
            "current": discrete,
            "predictions": predictions,
            "graph": graph,
        }

    def predict_next(self, steps: int = 1, top_k: int = 5) -> list[dict[str, Any]]:
        """Return the top-k most likely next states from the current rolling history."""
        if not self._fitted or len(self._buffer) < self.k:
            return []
        hist_df = pl.DataFrame(list(self._buffer)[-self.k :])
        probs = self.model.predict_proba(hist_df, steps=steps)
        encoder = self.model._encoder
        if encoder is None or probs.size == 0:
            return []

        top_indices = np.argsort(probs)[::-1][:top_k]
        decoded = encoder.inverse_transform(top_indices.tolist())
        rows = decoded.to_dicts()
        return [
            {"state": rows[i], "probability": float(probs[int(idx)])}
            for i, idx in enumerate(top_indices)
            if float(probs[int(idx)]) > 0.0
        ]

    def state_machine_json(self) -> dict[str, Any]:
        """Return JSON-serializable nodes and edges for live web viewers."""
        if not self._fitted:
            return {"nodes": [], "edges": []}
        plot_data = self.model.to_state_machine().to_plot_data()
        return {
            "nodes": [asdict(n) for n in plot_data.nodes],
            "edges": [asdict(e) for e in plot_data.edges],
        }
