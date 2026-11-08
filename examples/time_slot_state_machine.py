"""Time as a transition signal: 15-minute slots, auto-selected regimes, held-out comparison.

Run: PYTHONPATH=src venv/bin/python examples/time_slot_state_machine.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from seqsm import PhaseConditionedMarkovChain, SeqSM, TimeConfig
from seqsm.timeslots import datetime_us, preset_regimes

DATASET = Path("dataset/synthetic_routines.parquet")


def main() -> None:
    df = pl.read_parquet(DATASET)
    train = df.filter(pl.col("episode_id") < 10)
    test = df.filter(pl.col("episode_id") >= 10)

    # 1. The library infers roles (time, episode, states) and picks the regime definition by BIC.
    model = SeqSM(
        k=1,
        smoothing=0.1,
        auto_roles=True,
        time_config=TimeConfig(slot_duration="15m", regimes="auto"),
    ).fit(train)
    assert model.signal_roles is not None and model.regime_report is not None
    print("States :", model.state_encoder.feature_cols)
    print("Ignored:", model.signal_roles.ignored)
    print(model.regime_report.select("regime", "n_regimes", "n_params", "cv_nll"))
    print(repr(model))

    # 2. Held-out check: regime kernels vs the pooled (time-blind) kernel on unseen days.
    enc = model.state_encoder
    spec = model._regime_spec
    assert spec is not None
    phase: PhaseConditionedMarkovChain = model.phase_chain or PhaseConditionedMarkovChain()
    if model.phase_chain is not None:
        states = enc.transform(test)
        regimes = spec(datetime_us(test["timestamp"]))
        episodes = test["episode_id"].to_numpy().astype("int32")
        print(f"Held-out NLL (nats/step) pooled : {phase.average_nll(states, episode_ids=episodes):.3f}")
        print(f"Held-out NLL (nats/step) regimes: {phase.average_nll(states, regimes, episodes):.3f}")

    # 3. Next-state distribution depends on the clock.
    history = model.state_encoder.transform(train.tail(3))
    for hour in (3, 8, 19):
        probs = model.predict_proba(list(history), at=datetime(2026, 3, 2, hour, 0))
        top = int(probs.argmax())
        print(f"{hour:02d}:00 most likely next state -> {enc.state_to_dict(top)} (p={probs[top]:.2f})")

    # 4. One day, generated slot by slot with the time-of-day kernel.
    day = model.generate_schedule(96, start_time=datetime(2026, 3, 2), seed=7)
    print(day.select("timestamp", "room", "light").filter(pl.col("timestamp").dt.minute() == 0).head(8))

    # 5. Hand-picked regimes are available too (preset names or any callable).
    manual = SeqSM(
        k=1,
        smoothing=0.1,
        timestamp_col="timestamp",
        episode_col="episode_id",
        state_cols=["room", "light"],
        time_config=TimeConfig(slot_duration="15m", regimes="weekend-day-part"),
    )
    manual.fit(train)
    print(f"weekend-day-part regimes: {manual.n_regimes} ({preset_regimes('weekend-day-part').name})")


if __name__ == "__main__":
    main()
