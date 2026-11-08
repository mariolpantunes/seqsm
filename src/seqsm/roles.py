"""Automatic signal roles: which columns are states, context (drivers), time, episode, or ignored.

Heuristics (all reported per column in ``SignalRoles.profiles``):

1. Datetime/Date columns are the time axis; monotone numeric columns named like a clock are too.
2. A column named like a run id whose values form contiguous non-decreasing runs is the episode column.
3. Constant, identifier-like (almost all unique) and high-cardinality columns are ignored; continuous
   floats are ignored with a hint to ``discretize_columns`` first.
4. Calendar / exogenous names (hour, weekday, month, holiday, ...) become context: they condition the
   kernel but produce no edges.
5. Remaining low-cardinality columns are states. Two states that determine each other (normalized
   mutual information >= ``redundancy_nmi``) are merged by dropping the later one.
6. Optionally columns with next-step predictability below ``min_predictability`` (noise) are demoted, and
   the joint state space is capped at ``max_joint_states`` by dropping the least predictable columns.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

Role = Literal["state", "context", "time", "episode", "ignore"]

_CONTEXT_NAME = re.compile(
    r"(^|_)(hour|minute|weekday|dow|day_of_week|dayofweek|month|season|holiday|weekend|day_type|daytype|"
    r"time_of_day|tod)($|_)",
    re.IGNORECASE,
)
_EPISODE_NAME = re.compile(r"(^|_)(episode|session|run|trip|sequence|seq|day_idx|day)(_?id|_?idx)?($|_)", re.IGNORECASE)
_TIME_NAME = re.compile(r"(^|_)(time|timestamp|ts|datetime|date)($|_)", re.IGNORECASE)


@dataclass(frozen=True)
class ColumnProfile:
    """Role decision and evidence for one column."""

    name: str
    dtype: str
    n_unique: int
    unique_ratio: float
    role: Role
    reason: str
    predictability: float | None = None
    time_dependence: float | None = None


@dataclass(frozen=True)
class SignalRoles:
    """Partition of a table's columns into signal roles."""

    state: tuple[str, ...]
    context: tuple[str, ...]
    time: str | None
    episode: str | None
    ignored: tuple[str, ...]
    profiles: tuple[ColumnProfile, ...]

    def table(self) -> pl.DataFrame:
        """Returns the per-column evidence as a Polars DataFrame."""
        return pl.DataFrame(
            {
                "column": [p.name for p in self.profiles],
                "dtype": [p.dtype for p in self.profiles],
                "n_unique": [p.n_unique for p in self.profiles],
                "unique_ratio": [p.unique_ratio for p in self.profiles],
                "role": [p.role for p in self.profiles],
                "predictability": [p.predictability for p in self.profiles],
                "time_dependence": [p.time_dependence for p in self.profiles],
                "reason": [p.reason for p in self.profiles],
            }
        )


def _codes(series: pl.Series) -> NDArray[np.int64]:
    """Dense integer codes of a column (nulls form their own code)."""
    arr = series.cast(pl.String).fill_null("<null>").to_numpy()
    _, inv = np.unique(arr, return_inverse=True)
    return np.asarray(inv, dtype=np.int64).ravel()


def _entropy(codes: NDArray[np.int64]) -> float:
    if len(codes) == 0:
        return 0.0
    p = np.bincount(codes).astype(np.float64)
    p = p[p > 0] / len(codes)
    return float(-np.sum(p * np.log2(p)))


def mutual_information(a: NDArray[np.int64], b: NDArray[np.int64]) -> tuple[float, float, float]:
    """Returns (I(a;b), H(a), H(b)) in bits for two aligned integer code arrays."""
    if len(a) != len(b):
        raise ValueError("a and b must have the same length.")
    if len(a) == 0:
        return 0.0, 0.0, 0.0
    h_a, h_b = _entropy(a), _entropy(b)
    joint = a * (int(b.max()) + 1) + b
    h_ab = _entropy(np.unique(joint, return_inverse=True)[1].astype(np.int64).ravel())
    return max(0.0, h_a + h_b - h_ab), h_a, h_b


def _nmi(a: NDArray[np.int64], b: NDArray[np.int64]) -> float:
    mi, h_a, h_b = mutual_information(a, b)
    denom = min(h_a, h_b)
    return mi / denom if denom > 0 else 0.0


def next_step_predictability(codes: NDArray[np.int64], episodes: NDArray[np.int32] | None = None) -> float:
    """Normalized I(x_t; x_{t+1}) / H(x_{t+1}) in [0, 1] within episodes: how much the past explains the next step."""
    if len(codes) < 3:
        return 0.0
    a, b = codes[:-1], codes[1:]
    if episodes is not None:
        keep = episodes[:-1] == episodes[1:]
        a, b = a[keep], b[keep]
    if len(a) == 0:
        return 0.0
    mi, _, h_b = mutual_information(a, b)
    return mi / h_b if h_b > 0 else 0.0


def _contiguous_runs(values: NDArray[np.int64]) -> bool:
    """True if values are non-decreasing (each id forms one contiguous block)."""
    return bool(len(values) > 1 and np.all(np.diff(values) >= 0) and values[0] != values[-1])


def _hour_codes(ts: pl.Series) -> NDArray[np.int64] | None:
    if ts.dtype in (pl.Datetime, pl.Date):
        return ts.dt.hour().to_numpy().astype(np.int64) if ts.dtype != pl.Date else None
    return None


def discretize_columns(
    df: pl.DataFrame,
    columns: Sequence[str],
    n_bins: int | Literal["auto"] = "auto",
    method: Literal["quantile", "uniform"] = "quantile",
) -> pl.DataFrame:
    """Replaces continuous columns by integer bin ids so they can serve as state columns.

    ``n_bins="auto"`` uses the Freedman-Diaconis rule clipped to [2, 12]; ``quantile`` bins hold equal counts.
    """
    out = df
    for col in columns:
        x = df[col].cast(pl.Float64)
        arr = x.drop_nulls().to_numpy()
        if len(arr) == 0:
            raise ValueError(f"Column {col!r} has no values to discretize.")
        if n_bins == "auto":
            q75, q25 = np.percentile(arr, [75, 25])
            width = 2.0 * (q75 - q25) / (len(arr) ** (1.0 / 3.0))
            span = float(arr.max() - arr.min())
            nb = int(np.clip(np.ceil(span / width), 2, 12)) if width > 0 else 2
        else:
            nb = int(n_bins)
        if nb < 2:
            raise ValueError(f"n_bins must be >= 2, got {nb}.")
        if method == "quantile":
            edges = np.unique(np.quantile(arr, np.linspace(0, 1, nb + 1)[1:-1]))
        else:
            edges = np.linspace(float(arr.min()), float(arr.max()), nb + 1)[1:-1]
        ids = np.searchsorted(edges, x.fill_null(float(np.median(arr))).to_numpy(), side="right").astype(np.int32)
        out = out.with_columns(pl.Series(col, ids))
    return out


def infer_roles(
    df: pl.DataFrame,
    *,
    episode_col: str | None = None,
    timestamp_col: str | None = None,
    max_state_cardinality: int = 20,
    redundancy_nmi: float = 0.99,
    min_predictability: float = 0.0,
    max_joint_states: int | None = 500,
    context_hints: re.Pattern[str] | None = None,
) -> SignalRoles:
    """Infers the role of every column of ``df`` with transparent heuristics.

    Explicit ``episode_col`` / ``timestamp_col`` always win. See the module docstring for the rules.
    """
    n = len(df)
    ctx_pat = context_hints if context_hints is not None else _CONTEXT_NAME
    roles: dict[str, tuple[Role, str]] = {}

    time_col = timestamp_col
    if time_col is None:
        for c in df.columns:
            if df[c].dtype in (pl.Datetime, pl.Date):
                time_col = c
                break
        if time_col is None:
            for c in df.columns:
                s = df[c]
                if s.dtype.is_numeric() and _TIME_NAME.search(c) and s.n_unique() == n and s.is_sorted():
                    time_col = c
                    break
    if time_col is not None and time_col in df.columns:
        roles[time_col] = ("time", "explicit" if timestamp_col else "datetime/clock column")

    ep_col = episode_col
    if ep_col is None:
        for c in df.columns:
            if c in roles or not _EPISODE_NAME.search(c):
                continue
            s = df[c]
            if (
                s.dtype.is_integer()
                and _contiguous_runs(s.to_numpy().astype(np.int64))
                and s.n_unique() < max(2, n // 4)
            ):
                ep_col = c
                break
    if ep_col is not None and ep_col in df.columns:
        roles[ep_col] = ("episode", "explicit" if episode_col else "run-id name with contiguous non-decreasing blocks")

    ep_arr: NDArray[np.int32] | None = None
    if ep_col is not None and ep_col in df.columns:
        ep_arr = _codes(df[ep_col]).astype(np.int32)
    hour = _hour_codes(df[time_col]) if time_col is not None and time_col in df.columns else None

    stats: dict[str, tuple[int, float, float | None, float | None]] = {}
    candidates: list[str] = []
    for c in df.columns:
        s = df[c]
        n_u = int(s.n_unique())
        ratio = n_u / n if n else 0.0
        pred: float | None = None
        tdep: float | None = None
        if c in roles:
            stats[c] = (n_u, ratio, None, None)
            continue
        if n_u <= 1:
            roles[c] = ("ignore", "constant")
        elif ctx_pat.search(c) and n_u <= max(max_state_cardinality, 366):
            roles[c] = ("context", "calendar/exogenous name hint")
        elif s.dtype.is_float() and n_u > max_state_cardinality:
            roles[c] = ("ignore", "continuous: run discretize_columns() first")
        elif n_u > max_state_cardinality and ratio > 0.5:
            roles[c] = ("ignore", "identifier-like (almost every row unique)")
        elif n_u > max_state_cardinality:
            roles[c] = ("ignore", f"high cardinality ({n_u} > {max_state_cardinality})")
        else:
            codes = _codes(s)
            pred = next_step_predictability(codes, ep_arr)
            if hour is not None:
                tdep = _nmi(codes, hour)
            if min_predictability > 0.0 and pred < min_predictability:
                roles[c] = ("ignore", f"noise: next-step predictability {pred:.3f} < {min_predictability}")
            else:
                roles[c] = ("state", "low-cardinality discrete column")
                candidates.append(c)
        stats[c] = (n_u, ratio, pred, tdep)

    # Redundancy: drop a state that is functionally equivalent to an earlier one
    kept: list[str] = []
    for c in candidates:
        c_codes = _codes(df[c])
        dup = next((o for o in kept if _nmi(c_codes, _codes(df[o])) >= redundancy_nmi), None)
        if dup is not None:
            roles[c] = ("ignore", f"redundant with {dup} (NMI >= {redundancy_nmi})")
        else:
            kept.append(c)

    # Joint state-space cap: drop the least predictable columns until the observed joint space fits
    if max_joint_states is not None:
        while len(kept) > 1 and df.select(kept).n_unique() > max_joint_states:
            worst = min(kept, key=lambda col: stats[col][2] or 0.0)
            kept.remove(worst)
            roles[worst] = ("ignore", f"joint state space above {max_joint_states}: least predictable column dropped")

    profiles = tuple(
        ColumnProfile(
            name=c,
            dtype=str(df[c].dtype),
            n_unique=stats[c][0],
            unique_ratio=stats[c][1],
            role=roles[c][0],
            reason=roles[c][1],
            predictability=stats[c][2],
            time_dependence=stats[c][3],
        )
        for c in df.columns
    )
    return SignalRoles(
        state=tuple(c for c in df.columns if roles[c][0] == "state"),
        context=tuple(c for c in df.columns if roles[c][0] == "context"),
        time=time_col if time_col in df.columns else None,
        episode=ep_col if ep_col in df.columns else None,
        ignored=tuple(c for c in df.columns if roles[c][0] == "ignore"),
        profiles=profiles,
    )


__all__ = [
    "ColumnProfile",
    "Role",
    "SignalRoles",
    "discretize_columns",
    "infer_roles",
    "mutual_information",
    "next_step_predictability",
]
