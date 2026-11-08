"""Unit tests for slot quantization and regime functions."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from seqsm.timeslots import (
    TimeConfig,
    datetime_us,
    phase_regimes,
    preset_regimes,
    resample_to_slots,
    resolve_regimes,
)
from seqsm.timeutil import duration_us, parse_duration


def _us(*dts: datetime) -> np.ndarray:
    return np.array([np.datetime64(d, "us") for d in dts])


class TestPresets(unittest.TestCase):
    def test_day_part_boundaries(self) -> None:
        ts = _us(
            datetime(2026, 1, 5, 0), datetime(2026, 1, 5, 5, 59), datetime(2026, 1, 5, 6), datetime(2026, 1, 5, 23)
        )
        self.assertEqual(preset_regimes("day-part")(ts).tolist(), [0, 0, 1, 3])

    def test_hour_and_none(self) -> None:
        ts = _us(datetime(2026, 1, 5, 7, 30), datetime(2026, 1, 5, 23, 45))
        self.assertEqual(preset_regimes("hour")(ts).tolist(), [7, 23])
        self.assertEqual(preset_regimes("none")(ts).tolist(), [0, 0])

    def test_weekday_monday_is_zero(self) -> None:
        mon = datetime(2026, 1, 5)
        ts = _us(*[mon + timedelta(days=i) for i in range(7)])
        self.assertEqual(preset_regimes("weekday")(ts).tolist(), list(range(7)))
        self.assertEqual(preset_regimes("weekend")(ts).tolist(), [0, 0, 0, 0, 0, 1, 1])

    def test_weekend_day_part_ids(self) -> None:
        ts = _us(datetime(2026, 1, 10, 13), datetime(2026, 1, 5, 13))
        self.assertEqual(preset_regimes("weekend-day-part")(ts).tolist(), [4 + 2, 2])

    def test_unknown_preset(self) -> None:
        with self.assertRaises(ValueError):
            preset_regimes("lunar")

    def test_phase_regimes_bins_and_epoch_anchor(self) -> None:
        spec = phase_regimes("1d", 4)
        ts = _us(
            datetime(2026, 1, 5, 0), datetime(2026, 1, 5, 7), datetime(2026, 1, 9, 13), datetime(2026, 1, 9, 23, 59)
        )
        self.assertEqual(spec(ts).tolist(), [0, 1, 2, 3])
        with self.assertRaises(ValueError):
            phase_regimes("1d", 0)

    def test_regime_spec_rejects_out_of_range(self) -> None:
        cfg = TimeConfig(regimes=lambda ts: np.full(len(ts), 5, dtype=np.int32), n_regimes=2)
        with self.assertRaises(ValueError):
            resolve_regimes(cfg)(_us(datetime(2026, 1, 5)))


class TestTimeConfig(unittest.TestCase):
    def test_validation(self) -> None:
        with self.assertRaises(ValueError):
            TimeConfig(regimes="bogus")
        with self.assertRaises(ValueError):
            TimeConfig(min_regime_count=0)
        with self.assertRaises(ValueError):
            TimeConfig(fill="zero")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            TimeConfig(regimes=lambda ts: ts)  # callable needs n_regimes
        with self.assertRaises(ValueError):
            TimeConfig(candidates=("phase",))
        with self.assertRaises(ValueError):
            TimeConfig(criterion="aic")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            TimeConfig(cv_folds=1)
        with self.assertRaises(ValueError):
            TimeConfig(slot_duration="1h30m")

    def test_resolve_phase_needs_period(self) -> None:
        with self.assertRaises(ValueError):
            resolve_regimes(TimeConfig(regimes="phase"))
        spec = resolve_regimes(TimeConfig(regimes="phase", n_regimes=6), period="1d")
        self.assertEqual(spec.n_regimes, 6)

    def test_resolve_auto_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_regimes(TimeConfig(regimes="auto"))

    def test_durations(self) -> None:
        self.assertEqual(duration_us("15m"), 900_000_000)
        self.assertEqual(parse_duration(timedelta(seconds=3)), timedelta(seconds=3))
        with self.assertRaises(ValueError):
            parse_duration("0s")
        with self.assertRaises(TypeError):
            parse_duration(5)  # type: ignore[arg-type]


class TestResample(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pl.DataFrame(
            {
                "ts": [
                    datetime(2026, 1, 1, 0, 1),
                    datetime(2026, 1, 1, 0, 14),
                    datetime(2026, 1, 1, 0, 50),
                    datetime(2026, 1, 1, 1, 10),
                ],
                "x": ["a", "b", "c", "d"],
            }
        )

    def test_last_wins_and_ffill(self) -> None:
        out, ep = resample_to_slots(self.frame, "ts", "15m", ["x"])
        self.assertEqual(out["x"].to_list(), ["b", "b", "b", "c", "d"])
        self.assertEqual(
            out["ts"].to_list(),
            [datetime(2026, 1, 1, 0, 0) + timedelta(minutes=15 * i) for i in range(5)],
        )
        self.assertEqual(ep.tolist(), [0] * 5)

    def test_skip_leaves_gaps(self) -> None:
        out, _ = resample_to_slots(self.frame, "ts", "15m", ["x"], fill="skip")
        self.assertEqual(out["x"].to_list(), ["b", "c", "d"])

    def test_episodes_are_not_bridged(self) -> None:
        eps = np.array([0, 1, 1, 1], dtype=np.int32)
        out, ep = resample_to_slots(self.frame, "ts", "15m", ["x"], episode_ids=eps)
        self.assertEqual(ep.tolist(), [0, 1, 1, 1, 1, 1])
        self.assertEqual(out["x"].to_list(), ["a", "b", "b", "b", "c", "d"])

    def test_numeric_seconds_and_date(self) -> None:
        self.assertEqual(datetime_us(pl.Series([0, 60])).astype(np.int64).tolist(), [0, 60_000_000])
        d = datetime_us(pl.Series([datetime(2026, 1, 1).date()]))
        self.assertEqual(str(d[0]), "2026-01-01T00:00:00.000000")
        with self.assertRaises(TypeError):
            datetime_us(pl.Series(["x"]))


if __name__ == "__main__":
    unittest.main()
