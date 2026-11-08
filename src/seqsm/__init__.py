"""seqsm: Sequential State Machines and K-order Markov chains."""

from seqsm.encoding import KWindowEncoder, StateEncoder
from seqsm.estimator import SeqSM
from seqsm.io import export_schedule_parquet, load_model, save_model
from seqsm.machine import StateMachine
from seqsm.markov import KOrderMarkovChain, VariableOrderMarkovChain
from seqsm.motifs import compare_chains, score_anomalies
from seqsm.periodic import PhaseConditionedMarkovChain, select_regimes
from seqsm.roles import ColumnProfile, SignalRoles, discretize_columns, infer_roles
from seqsm.schedule import ScheduleGenerator
from seqsm.timeslots import RegimeSpec, TimeConfig, resample_to_slots
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
    "ColumnProfile",
    "DwellDistribution",
    "KOrderMarkovChain",
    "KWindowEncoder",
    "PhaseConditionedMarkovChain",
    "PlotEdge",
    "PlotNode",
    "RegimeSpec",
    "RoutineCycle",
    "ScheduleConstraints",
    "ScheduleGenerator",
    "ScheduleTrajectory",
    "SeqSM",
    "SignalRoles",
    "StateEncoder",
    "StateMachine",
    "StatePlotData",
    "TimeConfig",
    "VariableOrderMarkovChain",
    "compare_chains",
    "discretize_columns",
    "export_schedule_parquet",
    "infer_roles",
    "load_model",
    "resample_to_slots",
    "save_model",
    "score_anomalies",
    "select_regimes",
]
