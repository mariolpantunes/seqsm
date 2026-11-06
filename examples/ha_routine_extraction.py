"""Smart-home routine mining and schedule generation example using HomeAssistant sensor streams."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl

from seqsm import SeqSM


def generate_synthetic_home_data(n_days: int = 7) -> pl.DataFrame:
    """Generates synthetic multi-room smart-home sensor sequence with recurring routines."""
    records: list[dict[str, object]] = []
    base_time = datetime(2026, 1, 1, 0, 0, 0)
    current_time = base_time

    # Typical daily pattern: Sleep -> Kitchen -> Work -> Living -> Sleep
    routine_blocks = [
        ("Sleep", {"bedroom_motion": 0, "kitchen_motion": 0, "living_light": "off", "presence": "home"}, 420),  # 7h
        ("Morning", {"bedroom_motion": 1, "kitchen_motion": 1, "living_light": "dim", "presence": "home"}, 60),  # 1h
        ("Work", {"bedroom_motion": 0, "kitchen_motion": 0, "living_light": "off", "presence": "away"}, 480),  # 8h
        ("Evening", {"bedroom_motion": 0, "kitchen_motion": 1, "living_light": "on", "presence": "home"}, 180),  # 3h
        ("Night", {"bedroom_motion": 1, "kitchen_motion": 0, "living_light": "dim", "presence": "home"}, 300),  # 5h
    ]

    step_min = 5
    for day in range(n_days):
        for _block_name, sensors, duration_min in routine_blocks:
            n_steps = duration_min // step_min
            for _ in range(n_steps):
                row: dict[str, object] = {
                    "timestamp": current_time,
                    "day_idx": day,
                    **sensors,
                }
                records.append(row)
                current_time += timedelta(minutes=step_min)

    return pl.DataFrame(records)


def main() -> None:
    # 1. Generate synthetic sensor dataset
    df = generate_synthetic_home_data(n_days=5)

    # 2. Fit SeqSM estimator with K=2 history conditioning
    model = SeqSM(
        k=2,
        mode="event",
        smoothing=0.01,
        timestamp_col="timestamp",
        feature_cols=["bedroom_motion", "kitchen_motion", "living_light", "presence"],
        seed=42,
    )
    model.fit(df)

    # 3. Inspect stationary distribution
    _stat_df = model.stationary_distribution

    # 4. Generate 24-hour synthetic routine schedule
    sched_df = model.generate_schedule(
        window=timedelta(hours=24),
        time_unit="5m",
        start_time=datetime(2026, 1, 10, 0, 0, 0),
        method="sample",
    )

    # 5. Extract state machine and routine cycles
    fsm = model.to_state_machine(min_prob=0.05, min_count=2)
    _cycles = fsm.extract_cycles(max_length=6)
    _mermaid_spec = fsm.to_mermaid(diff_labels=True)

    # 6. Score behavioral anomalies
    _anomalies = model.score_anomalies(
        sched_df.select(["bedroom_motion", "kitchen_motion", "living_light", "presence"])
    )


if __name__ == "__main__":
    main()
