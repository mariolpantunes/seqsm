"""seqsm: Sequential State Machines and K-order Markov chains."""

from seqsm.encoding import KWindowEncoder, StateEncoder
from seqsm.estimator import SeqSM
from seqsm.io import export_schedule_parquet, load_model, save_model
from seqsm.machine import StateMachine
from seqsm.markov import KOrderMarkovChain, VariableOrderMarkovChain
from seqsm.motifs import compare_chains, score_anomalies
from seqsm.schedule import ScheduleGenerator
from seqsm.types import (
    DwellDistribution,
    PlotEdge,
    PlotNode,
    RoutineCycle,
    ScheduleConstraints,
    ScheduleTrajectory,
    StatePlotData,
)

__version__ = "0.1.0"

__all__ = [
    "DwellDistribution",
    "KOrderMarkovChain",
    "KWindowEncoder",
    "PlotEdge",
    "PlotNode",
    "RoutineCycle",
    "ScheduleConstraints",
    "ScheduleGenerator",
    "ScheduleTrajectory",
    "SeqSM",
    "StateEncoder",
    "StateMachine",
    "StatePlotData",
    "VariableOrderMarkovChain",
    "compare_chains",
    "export_schedule_parquet",
    "load_model",
    "save_model",
    "score_anomalies",
]
