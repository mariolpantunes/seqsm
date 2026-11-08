"""Time as a signal: slot quantization and regime (phase-class) functions.

A *regime* is a coarse class of the calendar or cyclic phase (e.g. ``day-part``) that
conditions the transition kernel: ``P(s' | window, regime)`` (SPEC section 2.10).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from seqsm.timeutil import duration_us, parse_duration

RegimeFn = Callable[[NDArray[Any]], NDArray[np.int32]]

_US_PER_HOUR = 3_600_000_000
_US_PER_DAY = 86_400_000_000

PRESETS: tuple[str, ...] = (
    "none",
    "day-part",
    "hour",
    "weekday",
    "weekend",
    "weekend-day-part",
    "phase",
)

DEFAULT_CANDIDATES: tuple[str, ...] = ("none", "weekend", "day-part", "weekend-day-part", "hour")


@dataclass(frozen=True)
class TimeConfig:
    """Declares how timestamps act as a transition signal.

    Attributes:
        slot_duration: Quantization step (e.g. '15m'). Rows are aggregated per slot (last value
            wins) and gaps inside an episode are filled, giving a uniform clock grid.
            None keeps the rows as they are.
        regimes: A preset name (see ``PRESETS``), ``"auto"`` (BIC selection among ``candidates``),
            or a callable mapping ``datetime64[us]`` arrays to integer regime ids in [0, n_regimes).
        period: Cycle length for ``regimes="phase"`` (e.g. '1d' for a datetime column, or an int number of
            ticks for an integer time column), or ``"auto"`` to detect it with categorical match
            autocorrelation on the slot sequence.
        n_regimes: Number of equal-width phase bins (``phase``) or the id count of a callable.
        fill: ``"ffill"`` repeats the previous slot value into empty slots; ``"skip"`` leaves them out
            (the chain then sees consecutive observed slots).
        candidates: Presets scored when ``regimes="auto"``.
        min_regime_count: A (regime, window) row with fewer observed transitions falls back to the
            pooled row (attested-successor renormalization, decision C1).
        criterion: How ``regimes="auto"`` scores candidates: ``"cv"`` (blocked cross-validated held-out
            NLL, robust on few days) or ``"bic"`` (minimum description length).
        cv_folds: Number of contiguous folds for ``criterion="cv"``.
    """

    slot_duration: str | timedelta | None = None
    regimes: str | RegimeFn = "day-part"
    period: str | timedelta | int | None = None
    n_regimes: int | None = None
    fill: Literal["ffill", "skip"] = "ffill"
    candidates: tuple[str, ...] = DEFAULT_CANDIDATES
    min_regime_count: int = 1
    criterion: Literal["bic", "cv"] = "cv"
    cv_folds: int = 4

    def __post_init__(self) -> None:
        if self.criterion not in ("bic", "cv"):
            raise ValueError(f"criterion must be 'bic' or 'cv', got {self.criterion!r}.")
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {self.cv_folds}.")
        if self.min_regime_count < 1:
            raise ValueError(f"min_regime_count must be >= 1, got {self.min_regime_count}.")
        if self.fill not in ("ffill", "skip"):
            raise ValueError(f"fill must be 'ffill' or 'skip', got {self.fill!r}.")
        if self.slot_duration is not None:
            duration_us(self.slot_duration)
        if isinstance(self.period, int) and (isinstance(self.period, bool) or self.period < 2):
            raise ValueError(f"An integer period must be >= 2 ticks, got {self.period!r}.")
        if isinstance(self.regimes, str) and self.regimes != "auto" and self.regimes not in PRESETS:
            raise ValueError(f"Unknown regimes preset {self.regimes!r}; expected one of {PRESETS} or 'auto'.")
        if callable(self.regimes) and (self.n_regimes is None or self.n_regimes < 1):
            raise ValueError("A callable regimes function requires n_regimes >= 1.")
        if self.n_regimes is not None and self.n_regimes < 1:
            raise ValueError(f"n_regimes must be >= 1, got {self.n_regimes}.")
        for name in self.candidates:
            if name not in PRESETS or name == "phase":
                raise ValueError(f"Unsupported auto candidate {name!r}; choose from {PRESETS[:-1]}.")


@dataclass(frozen=True)
class RegimeSpec:
    """A resolved regime function with its id count."""

    name: str
    n_regimes: int
    fn: RegimeFn

    def __call__(self, ts_us: NDArray[Any]) -> NDArray[np.int32]:
        out = np.asarray(self.fn(ts_us), dtype=np.int32)
        if out.size and (out.min() < 0 or out.max() >= self.n_regimes):
            raise ValueError(f"Regime function {self.name!r} returned ids outside [0, {self.n_regimes - 1}].")
        return out


def _us_since_epoch(ts_us: NDArray[np.datetime64]) -> NDArray[np.int64]:
    return np.asarray(ts_us, dtype="datetime64[us]").astype(np.int64)


def _hour_of_day(ts_us: NDArray[np.datetime64]) -> NDArray[np.int64]:
    return (_us_since_epoch(ts_us) % _US_PER_DAY) // _US_PER_HOUR


def _weekday(ts_us: NDArray[np.datetime64]) -> NDArray[np.int64]:
    days = _us_since_epoch(ts_us) // _US_PER_DAY
    return (days + 3) % 7  # 1970-01-01 was a Thursday; Monday == 0


def _day_part(ts_us: NDArray[np.datetime64]) -> NDArray[np.int64]:
    return _hour_of_day(ts_us) // 6  # night, morning, afternoon, evening


def phase_regimes(period: str | timedelta, n_regimes: int) -> RegimeSpec:
    """Equal-width bins of ``t mod period`` (epoch-anchored, so fit and rollout agree)."""
    p_us = duration_us(period)
    if n_regimes < 1:
        raise ValueError(f"n_regimes must be >= 1, got {n_regimes}.")

    def fn(ts_us: NDArray[np.datetime64]) -> NDArray[np.int32]:
        phase = _us_since_epoch(ts_us) % p_us
        return np.minimum((phase * n_regimes) // p_us, n_regimes - 1).astype(np.int32)

    return RegimeSpec(f"phase({parse_duration(period)}/{n_regimes})", n_regimes, fn)


def tick_phase_regimes(period: int, n_regimes: int | None = None) -> RegimeSpec:
    """Equal-width bins of ``tick mod period`` for an integer (already discrete) time column.

    ``n_regimes`` defaults to ``period`` (one regime per phase value, e.g. 96 quarter-hour slots of a day).
    """
    if period < 2:
        raise ValueError(f"period must be >= 2 ticks, got {period}.")
    n = period if n_regimes is None else n_regimes
    if not 1 <= n <= period:
        raise ValueError(f"n_regimes must lie in [1, {period}], got {n}.")

    def fn(ticks: NDArray[np.int64]) -> NDArray[np.int32]:
        return (((np.asarray(ticks, dtype=np.int64) % period) * n) // period).astype(np.int32)

    return RegimeSpec(f"ticks({period}/{n})", n, fn)


def discrete_candidates(period: int) -> list[RegimeSpec]:
    """Phase-bin candidates (2, 3, 4, 6, 8, 12, 24 bins and one per tick) for ``regimes='auto'`` on tick columns."""
    sizes = sorted({n for n in (2, 3, 4, 6, 8, 12, 24) if n < period} | {period})
    return [tick_phase_regimes(period, n) for n in sizes]


def time_array(series: pl.Series) -> tuple[NDArray[Any], bool]:
    """Returns (time values, is_discrete). Integer columns are discrete ticks; others become ``datetime64[us]``."""
    if series.dtype.is_integer():
        return series.to_numpy().astype(np.int64), True
    return datetime_us(series), False


def preset_regimes(name: str) -> RegimeSpec:
    """Returns one of the calendar presets (everything but 'phase' and 'auto')."""
    if name == "none":
        return RegimeSpec("none", 1, lambda ts: np.zeros(len(ts), dtype=np.int32))
    if name == "day-part":
        return RegimeSpec("day-part", 4, lambda ts: _day_part(ts).astype(np.int32))
    if name == "hour":
        return RegimeSpec("hour", 24, lambda ts: _hour_of_day(ts).astype(np.int32))
    if name == "weekday":
        return RegimeSpec("weekday", 7, lambda ts: _weekday(ts).astype(np.int32))
    if name == "weekend":
        return RegimeSpec("weekend", 2, lambda ts: (_weekday(ts) >= 5).astype(np.int32))
    if name == "weekend-day-part":
        return RegimeSpec(
            "weekend-day-part",
            8,
            lambda ts: ((_weekday(ts) >= 5).astype(np.int64) * 4 + _day_part(ts)).astype(np.int32),
        )
    raise ValueError(f"Unknown regimes preset {name!r}.")


def resolve_regimes(config: TimeConfig, period: str | timedelta | int | None = None) -> RegimeSpec:
    """Turns a TimeConfig into a RegimeSpec ('auto' must be resolved by the caller first)."""
    reg = config.regimes
    if callable(reg):
        assert config.n_regimes is not None
        return RegimeSpec(getattr(reg, "__name__", "custom"), config.n_regimes, reg)
    if reg == "auto":
        raise ValueError("regimes='auto' must be resolved by select_regimes before calling resolve_regimes.")
    if reg == "phase":
        per = period if period is not None else config.period
        if per is None or per == "auto":
            raise ValueError("regimes='phase' needs an explicit period (or period='auto' resolved by the estimator).")
        if isinstance(per, int):
            return tick_phase_regimes(per, config.n_regimes)
        n = config.n_regimes if config.n_regimes is not None else 4
        return phase_regimes(per, n)
    return preset_regimes(reg)


def datetime_us(series: pl.Series) -> NDArray[np.datetime64]:
    """Converts a Polars Datetime/Date column (or numeric epoch seconds) to ``datetime64[us]``."""
    if series.dtype == pl.Date:
        return series.cast(pl.Datetime("us")).to_numpy().astype("datetime64[us]")
    if series.dtype == pl.Datetime:
        return series.dt.cast_time_unit("us").to_numpy().astype("datetime64[us]")
    if series.dtype.is_numeric():
        return (series.cast(pl.Float64).to_numpy() * 1_000_000.0).astype(np.int64).astype("datetime64[us]")
    raise TypeError(f"Time column must be Datetime, Date or numeric seconds, got {series.dtype}.")


def to_datetime64(value: datetime) -> np.datetime64:
    """Converts a Python datetime into ``datetime64[us]``."""
    return np.datetime64(value, "us")


def resample_to_slots(
    frame: pl.DataFrame,
    timestamp_col: str,
    slot_duration: str | timedelta,
    value_cols: Sequence[str],
    episode_ids: NDArray[np.int32] | None = None,
    fill: Literal["ffill", "skip"] = "ffill",
) -> tuple[pl.DataFrame, NDArray[np.int32]]:
    """Quantizes rows to ``slot = floor(t / dt)`` and returns (slot frame, episode ids).

    Rows sharing a slot are aggregated by their last value. With ``fill="ffill"`` empty slots inside an
    episode repeat the previous value, so the output is an equally spaced clock grid.
    The returned frame holds ``timestamp_col`` (slot start, ``Datetime[us]``) and ``value_cols``.
    """
    slot_us = duration_us(slot_duration)
    t_us = datetime_us(frame[timestamp_col]).astype(np.int64)
    slots = t_us // slot_us
    ep = episode_ids.astype(np.int64) if episode_ids is not None else np.zeros(len(frame), dtype=np.int64)

    work = frame.select(list(value_cols)).with_columns(
        pl.Series("__ep__", ep),
        pl.Series("__slot__", slots),
    )
    work = work.sort(["__ep__", "__slot__"], maintain_order=True)
    agg = work.group_by(["__ep__", "__slot__"], maintain_order=True).agg([pl.col(c).last() for c in value_cols])

    if fill == "ffill":
        bounds = agg.group_by("__ep__", maintain_order=True).agg(
            pl.col("__slot__").min().alias("lo"), pl.col("__slot__").max().alias("hi")
        )
        grid = (
            bounds.with_columns(pl.int_ranges(pl.col("lo"), pl.col("hi") + 1).alias("__slot__"))
            .select("__ep__", "__slot__")
            .explode("__slot__")
        )
        agg = (
            grid.join(agg, on=["__ep__", "__slot__"], how="left")
            .sort(["__ep__", "__slot__"], maintain_order=True)
            .with_columns([pl.col(c).forward_fill().over("__ep__") for c in value_cols])
        )

    ts_out = (agg["__slot__"] * slot_us).cast(pl.Datetime("us")).alias(timestamp_col)
    out = agg.select(value_cols).with_columns(ts_out).select([timestamp_col, *value_cols])
    return out, agg["__ep__"].to_numpy().astype(np.int32)


__all__ = [
    "DEFAULT_CANDIDATES",
    "PRESETS",
    "RegimeFn",
    "RegimeSpec",
    "TimeConfig",
    "datetime_us",
    "discrete_candidates",
    "phase_regimes",
    "preset_regimes",
    "resample_to_slots",
    "resolve_regimes",
    "tick_phase_regimes",
    "time_array",
    "to_datetime64",
]
