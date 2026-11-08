"""Context drivers and time-conditioned dwell times (semi-Markov event mode).

A `day_type` driver column conditions the kernel without being a state; in event mode the dwell
distributions also follow the regime of the visit start.

Run: PYTHONPATH=src venv/bin/python examples/context_day_type.py
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl

from seqsm import SeqSM, TimeConfig


def commuter_week(n_weeks: int = 8, seed: int = 0) -> pl.DataFrame:
    """Hourly home/office/out states: offices on weekdays, long outings on weekends."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    ts = datetime(2026, 1, 5)
    state = "home"
    for _ in range(n_weeks * 7 * 24):
        weekend = ts.weekday() >= 5
        hour = ts.hour
        if weekend:
            probs = (
                {"home": 0.6, "out": 0.4, "office": 0.0}
                if 10 <= hour < 20
                else {"home": 1.0, "out": 0.0, "office": 0.0}
            )
        elif 8 <= hour < 17:
            probs = {"home": 0.05, "out": 0.05, "office": 0.9}
        else:
            probs = {"home": 0.9, "out": 0.1, "office": 0.0}
        state = str(rng.choice(list(probs), p=list(probs.values())))
        rows.append({"timestamp": ts, "place": state, "day_type": "weekend" if weekend else "weekday"})
        ts += timedelta(hours=1)
    return pl.DataFrame(rows)


def main() -> None:
    df = commuter_week()

    # 1. day_type is a driver (context), `place` is the state.
    ctx = SeqSM(k=1, smoothing=0.5, state_cols=["place"], context_cols=["day_type"]).fit(df)
    names = ctx.state_encoder.vocabulary["place"].to_list()
    for day in ("weekday", "weekend"):
        p = ctx.predict_proba([names.index("home")], context={"day_type": day})
        print(day, {n: round(float(p[i]), 2) for i, n in enumerate(names)})

    # 2. Context + clock: 2 day types x 4 day-parts = 8 regimes; dwell times are regime-specific.
    full = SeqSM(
        k=1,
        mode="event",
        smoothing=0.5,
        timestamp_col="timestamp",
        state_cols=["place"],
        context_cols=["day_type"],
        time_config=TimeConfig(slot_duration="1h", regimes="day-part"),
    ).fit(df)
    print(repr(full), "-> regimes:", full.n_regimes)
    for day_type, when in (("weekday", datetime(2026, 3, 2, 8)), ("weekend", datetime(2026, 3, 7, 11))):
        p = full.predict_proba([names.index("home")], at=when, context={"day_type": day_type})
        print(f"{day_type} {when:%H:%M} P(next | home):", {n: round(float(p[i]), 2) for i, n in enumerate(names)})
    day = full.generate_schedule(24, start_time=datetime(2026, 3, 2), context={"day_type": "weekday"}, seed=3)
    print(day.select("timestamp", "place", "dwell_duration"))

    # 3. A regime's state machine can be exported on its own.
    morning_weekday = full.regime_of(datetime(2026, 3, 2, 8), {"day_type": "weekday"})
    print(full.to_state_machine(regime=morning_weekday, min_prob=0.05).to_mermaid())


if __name__ == "__main__":
    main()
