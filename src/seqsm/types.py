"""Core types and dataclasses for seqsm."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from numpy.typing import DTypeLike, NDArray

# Model operation modes
ModeType = Literal["clock", "event"]
PredictionMethod = Literal["sample", "map"]
ScheduleMethod = Literal["sample", "map", "constrained"]
BackoffMethod = Literal["witten_bell", "linear", "tuned"]


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


@dataclass(frozen=True)
class RoutineCycle:
    """Extracted behavioral routine cycle."""

    state_sequence: tuple[int, ...]
    frequency: int
    period: float
    persistence: float


__all__ = [
    "BackoffMethod",
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
