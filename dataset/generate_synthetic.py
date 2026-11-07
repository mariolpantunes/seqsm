"""Reproducible synthetic multi-sensor routine dataset generator for seqsm benchmarks."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl


def generate_synthetic_routines(
    num_days: int = 14,
    step_minutes: int = 15,
    seed: int = 42,
) -> pl.DataFrame:
    """Generates synthetic multivariate smart home sensor time series with realistic circadian routines.

    Routines simulate:
    - Sleeping (Night: 23:00 - 07:00): Bedroom, motion low, lights off, TV off, presence true
    - Morning Breakfast (07:00 - 08:30): Kitchen, motion high, lights bright, TV off, presence true
    - Work / Away (08:30 - 18:00): Weekday office/away vs weekend flexible
    - Evening Leisure (18:00 - 23:00): Living room, motion moderate, lights dim/normal, TV on, presence true
    """
    rng = np.random.default_rng(seed)
    steps_per_day = (24 * 60) // step_minutes
    total_steps = num_days * steps_per_day

    base_time = datetime(2026, 1, 1, 0, 0, 0)
    timestamps: list[datetime] = [base_time + timedelta(minutes=i * step_minutes) for i in range(total_steps)]

    episodes: list[int] = []
    rooms: list[str] = []
    motions: list[int] = []
    lights: list[str] = []
    tv_states: list[int] = []
    presences: list[bool] = []

    for idx, dt in enumerate(timestamps):
        day_idx = idx // steps_per_day
        episodes.append(day_idx)
        hour = dt.hour + dt.minute / 60.0
        is_weekend = dt.weekday() >= 5

        # Circadian phase transitions with random perturbations
        if hour < 7.0 or hour >= 23.0:
            # Sleeping phase
            room = "bedroom" if rng.random() > 0.05 else "bathroom"
            motion = 1 if (room == "bathroom" or rng.random() < 0.08) else 0
            light = "dim" if (room == "bathroom" and motion == 1) else "off"
            tv = 0
            presence = True
        elif 7.0 <= hour < 8.5:
            # Morning routine
            room = "kitchen" if rng.random() > 0.15 else "bathroom"
            motion = 1 if rng.random() > 0.10 else 0
            light = "bright" if motion == 1 else "normal"
            tv = 1 if rng.random() < 0.20 else 0
            presence = True
        elif 8.5 <= hour < 18.0:
            # Daytime work / away phase
            if is_weekend:
                # Weekend leisure
                room = rng.choice(["living_room", "kitchen", "office", "away"], p=[0.40, 0.25, 0.15, 0.20])
                presence = room != "away"
                motion = 1 if (presence and rng.random() > 0.25) else 0
                light = "normal" if (presence and motion == 1) else "off"
                tv = 1 if (room == "living_room" and rng.random() < 0.60) else 0
            else:
                # Weekday work
                is_home = rng.random() < 0.25  # Occasional remote work
                if is_home:
                    room = "office" if rng.random() > 0.15 else "kitchen"
                    presence = True
                    motion = 1 if rng.random() > 0.30 else 0
                    light = "normal" if motion == 1 else "off"
                    tv = 0
                else:
                    room = "away"
                    presence = False
                    motion = 0
                    light = "off"
                    tv = 0
        else:
            # Evening routine (18:00 - 23:00)
            if hour < 20.0:
                room = "kitchen" if rng.random() < 0.65 else "living_room"
            else:
                room = "living_room" if rng.random() > 0.15 else "bedroom"
            presence = True
            motion = 1 if rng.random() > 0.20 else 0
            light = "dim" if (room == "living_room" and rng.random() < 0.70) else "normal"
            tv = 1 if (room == "living_room" and rng.random() < 0.85) else 0

        rooms.append(str(room))
        motions.append(motion)
        lights.append(light)
        tv_states.append(tv)
        presences.append(presence)

    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "episode_id": np.array(episodes, dtype=np.int32),
            "room": rooms,
            "motion": np.array(motions, dtype=np.int32),
            "light": lights,
            "tv_active": np.array(tv_states, dtype=np.int32),
            "presence": presences,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic routine data for seqsm.")
    parser.add_argument("--days", type=int, default=14, help="Number of synthetic days to generate.")
    parser.add_argument("--step-minutes", type=int, default=15, help="Time step resolution in minutes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/synthetic_routines.parquet",
        help="Path for output parquet/csv file.",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_synthetic_routines(num_days=args.days, step_minutes=args.step_minutes, seed=args.seed)

    if out_path.suffix == ".parquet":
        df.write_parquet(out_path)
    elif out_path.suffix == ".csv":
        df.write_csv(out_path)
    else:
        df.write_parquet(out_path.with_suffix(".parquet"))
        df.write_csv(out_path.with_suffix(".csv"))

    print(f"Generated {len(df)} rows across {args.days} days into {out_path}.")


if __name__ == "__main__":
    main()
