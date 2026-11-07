"""Core types and dataclasses for seqsm."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import DTypeLike, NDArray

# Model operation modes
ModeType = Literal["clock", "event"]
PredictionMethod = Literal["sample", "map", "viterbi"]
ScheduleMethod = Literal["sample", "map", "constrained"]
BackoffMethod = Literal["witten_bell", "linear", "tuned"]


@dataclass(frozen=True)
class CSRMatrix:
    """Compressed Sparse Row (CSR) matrix representation.

    Note:
        Row pointers (`indptr`) and column indices (`indices`) are explicitly stored as
        32-bit signed integers (`np.int32`) to optimize cache locality and ensure direct
        compatibility with Numba JIT kernels. Consequently, the maximum number of non-zero
        elements (nnz) and maximum dimension size are bounded by 2^31 - 1 (~2.14 billion entries).
        For models exceeding this threshold, 64-bit sparse structures would be required.
    """

    data: NDArray[np.float64]
    indices: NDArray[np.int32]
    indptr: NDArray[np.int32]
    shape: tuple[int, int]

    def to_dense(self) -> NDArray[np.float64]:
        out = np.zeros(self.shape, dtype=np.float64)
        for i in range(self.shape[0]):
            start, end = int(self.indptr[i]), int(self.indptr[i + 1])
            out[i, self.indices[start:end]] = self.data[start:end]
        return out

    def dot_vector_left(self, vec: NDArray[np.float64]) -> NDArray[np.float64]:
        """Computes vector-matrix product: vec @ CSR of shape (shape[1],)."""
        if len(self.data) == 0:
            return np.zeros(self.shape[1], dtype=np.float64)
        row_counts = np.diff(self.indptr)
        row_indices = np.repeat(np.arange(self.shape[0], dtype=np.int32), row_counts)
        weights = vec[row_indices] * self.data
        return np.bincount(self.indices, weights=weights, minlength=self.shape[1]).astype(np.float64)


@dataclass(frozen=True)
class DwellDistribution:
    """Empirical dwell-time distribution for a given state."""

    state_id: int
    count: int
    mean: float
    std: float
    median: float
    min_dwell: int
    max_dwell: int
    samples: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ScheduleConstraints:
    """Constraints for guided schedule generation."""

    initial_state: int | dict[str, Any] | None = None
    terminal_state: int | dict[str, Any] | None = None
    mandatory_states: Sequence[tuple[int | dict[str, Any], int | tuple[int, int]]] = field(default_factory=tuple)
    forbidden_states: Sequence[int | dict[str, Any]] = field(default_factory=tuple)
    max_total_dwell: dict[int, int] | None = None


@dataclass(frozen=True)
class ScheduleTrajectory:
    """Raw trajectory output from schedule rollouts."""

    state_ids: tuple[int, ...]
    dwell_durations: tuple[int, ...]
    transition_probs: tuple[float, ...]
    arc_valid: tuple[bool, ...]
    constraint_violated: tuple[bool, ...] = field(default_factory=tuple)
    unmet_mandatory: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RoutineCycle:
    """Extracted behavioral routine cycle."""

    state_sequence: tuple[int, ...]
    frequency: int
    period: float
    persistence: float


__all__ = [
    "BackoffMethod",
    "CSRMatrix",
    "DTypeLike",
    "DwellDistribution",
    "ModeType",
    "NDArray",
    "PredictionMethod",
    "RoutineCycle",
    "ScheduleConstraints",
    "ScheduleMethod",
    "ScheduleTrajectory",
]
