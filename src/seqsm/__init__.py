"""seqsm: Sequential State Machines and K-order Markov chains."""

from seqsm.encoding import KWindowEncoder, StateEncoder
from seqsm.types import (
    DwellDistribution,
    RoutineCycle,
    ScheduleConstraints,
    ScheduleTrajectory,
)

__version__ = "0.1.0"

__all__ = [
    "DwellDistribution",
    "KWindowEncoder",
    "RoutineCycle",
    "ScheduleConstraints",
    "ScheduleTrajectory",
    "StateEncoder",
]
