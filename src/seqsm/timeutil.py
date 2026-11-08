"""Duration parsing shared by the estimator and the time-slot helpers."""

from __future__ import annotations

import re
from datetime import timedelta

_UNITS_US: dict[str, float] = {
    "us": 1.0,
    "µs": 1.0,
    "ms": 1_000.0,
    "s": 1_000_000.0,
    "m": 60_000_000.0,
    "h": 3_600_000_000.0,
    "d": 86_400_000_000.0,
    "w": 604_800_000_000.0,
}


def parse_duration(duration: str | timedelta) -> timedelta:
    """Parses a duration string (e.g. '1d', '12h', '30m', '15s', '500ms') into a timedelta."""
    if isinstance(duration, timedelta):
        return duration
    if not isinstance(duration, str):
        raise TypeError(f"Expected str or timedelta, got {type(duration).__name__}")

    s = duration.strip().lower()
    if not s:
        raise ValueError("Duration string cannot be empty.")

    match = re.fullmatch(r"^(\d+(?:\.\d+)?)\s*(ms|us|µs|s|m|h|d|w)$", s)
    if not match:
        raise ValueError(
            f"Invalid duration format: '{duration}'. Expected single unit format like '1d', '12h', '30m', '15s', '500ms' (compound strings like '1h30m' are not supported)."
        )

    val_str, unit = match.groups()
    val = float(val_str)
    if val <= 0.0:
        raise ValueError(f"Duration must be strictly positive, got '{duration}'.")
    return timedelta(microseconds=val * _UNITS_US[unit])


def duration_us(duration: str | timedelta) -> int:
    """Returns a duration as a positive integer number of microseconds."""
    delta = parse_duration(duration)
    return max(1, round(delta.total_seconds() * 1_000_000))


__all__ = ["duration_us", "parse_duration"]
