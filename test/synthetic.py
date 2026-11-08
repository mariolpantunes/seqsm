"""Shared synthetic generators for the time-as-signal tests and examples."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl


def daily_activity(n_days: int = 20, slot_min: int = 15, start: datetime = datetime(2026, 1, 5)) -> pl.DataFrame:
    """Deterministic routine: sleep 23-06, home 06-08, work 08-17, home 17-23 (Monday start)."""
    rows: list[dict[str, object]] = []
    steps = 24 * 60 // slot_min
    for d in range(n_days):
        for i in range(steps):
            ts = start + timedelta(days=d, minutes=i * slot_min)
            h = ts.hour
            act = "sleep" if h >= 23 or h < 6 else "home" if h < 8 or h >= 17 else "work"
            rows.append({"timestamp": ts, "activity": act})
    return pl.DataFrame(rows)


def weekday_weekend(n_weeks: int = 6, slot_min: int = 60, seed: int = 0) -> pl.DataFrame:
    """Two-state chain whose kernel depends on a day_type driver column (weekday vs weekend)."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    state = "out"
    ts = datetime(2026, 1, 5)
    for _ in range(n_weeks * 7 * 24 * 60 // slot_min):
        weekend = ts.weekday() >= 5
        p_in = 0.9 if weekend else 0.1  # P(next = in)
        state = "in" if rng.random() < p_in else "out"
        rows.append({"timestamp": ts, "state": state, "day_type": "weekend" if weekend else "weekday"})
        ts += timedelta(minutes=slot_min)
    return pl.DataFrame(rows)


def random_chain(n: int = 4000, seed: int = 1, slot_min: int = 15) -> pl.DataFrame:
    """Time-independent three-state chain (no regime should win BIC)."""
    rng = np.random.default_rng(seed)
    trans = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.3, 0.3, 0.4]])
    s = 0
    out: list[str] = []
    for _ in range(n):
        s = int(rng.choice(3, p=trans[s]))
        out.append("abc"[s])
    ts = [datetime(2026, 1, 5) + timedelta(minutes=slot_min * i) for i in range(n)]
    return pl.DataFrame({"timestamp": ts, "x": out})
