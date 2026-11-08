"""Time is already a discrete column: an integer tick index plus a period.

No datetime handling: `tick` counts 15-minute slots, the period is 96 ticks (one day) or detected.
`period="auto"` finds the fundamental cycle with categorical match autocorrelation.

Run: PYTHONPATH=src venv/bin/python examples/discrete_time_column.py
"""

from __future__ import annotations

import polars as pl

from seqsm import SeqSM, TimeConfig


def main() -> None:
    raw = pl.read_parquet("dataset/synthetic_routines.parquet")
    df = raw.select("room", "light").with_row_index("tick")  # discrete time column, 15-minute ticks

    # 1. Explicit period: one regime per phase value is too fine here, so use 12 bins (2 h each).
    model = SeqSM(
        k=1,
        smoothing=0.1,
        timestamp_col="tick",
        state_cols=["room", "light"],
        time_config=TimeConfig(regimes="phase", period=96, n_regimes=12),
    ).fit(df)
    print(repr(model), "| regime of tick 100:", model.regime_of(100))

    # 2. Detected period and automatically chosen number of phase bins (cross-validated).
    auto = SeqSM(
        k=1,
        smoothing=0.1,
        timestamp_col="tick",
        state_cols=["room", "light"],
        time_config=TimeConfig(regimes="auto", period="auto"),
    ).fit(df)
    print("detected period:", auto._regime_meta, "->", auto.n_regimes, "regimes")
    if auto.regime_report is not None:
        print(auto.regime_report.select("regime", "n_regimes", "cv_nll"))

    # 3. Ticks replace datetimes everywhere: `at` is an int, schedules start at an int tick.
    history = model.state_encoder.transform(df.tail(1))
    probs = model.predict_proba(list(history), at=8 * 4)  # 08:00 on a 15-minute grid
    print("most likely next state at 08:00:", model.state_encoder.state_to_dict(int(probs.argmax())))
    day = model.generate_schedule(96, start_time=0, seed=1)
    print(day.select("timestamp", "room", "light").head(6))


if __name__ == "__main__":
    main()
