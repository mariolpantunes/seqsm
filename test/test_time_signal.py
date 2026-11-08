"""Estimator-level tests: time as a transition signal, signal roles, context columns, serialization."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
from synthetic import daily_activity, random_chain, tick_routine, weekday_weekend

from seqsm import SeqSM, TimeConfig, load_model, save_model
from seqsm.timeutil import duration_us


class TestTimeConfigFit(unittest.TestCase):
    def setUp(self) -> None:
        self.df = daily_activity(n_days=14)
        self.model = SeqSM(
            k=1,
            smoothing=0.01,
            timestamp_col="timestamp",
            time_config=TimeConfig(slot_duration="15m", regimes="hour"),
        ).fit(self.df)

    def test_regimes_built(self) -> None:
        self.assertEqual(self.model.n_regimes, 24)
        self.assertIsNotNone(self.model.phase_chain)
        self.assertEqual(self.model.state_encoder.feature_cols, ("activity",))
        self.assertIn("n_regimes=24", repr(self.model))

    def test_predict_depends_on_time(self) -> None:
        names = self.model.state_encoder.vocabulary["activity"].to_list()
        sleep, home = names.index("sleep"), names.index("home")
        hist = [sleep]
        at_wake = self.model.predict_proba(hist, at=datetime(2026, 3, 2, 6, 0))
        at_night = self.model.predict_proba(hist, at=datetime(2026, 3, 2, 3, 0))
        pooled = self.model.predict_proba(hist)
        self.assertGreater(at_wake[home], 0.9)
        self.assertGreater(at_night[sleep], 0.9)
        self.assertLess(pooled[home], 0.2)
        self.assertEqual(self.model.predict(hist, method="map", at=datetime(2026, 3, 2, 6, 0))["activity"][0], "home")

    def test_multi_step_advances_the_slot_clock(self) -> None:
        names = self.model.state_encoder.vocabulary["activity"].to_list()
        probs = self.model.predict_proba([names.index("sleep")], steps=4, at=datetime(2026, 3, 2, 5, 15))
        self.assertEqual(probs.shape, (4, 3))
        # 05:15 sleep, 05:30 sleep, 05:45 sleep, 06:00 home
        self.assertEqual([names[int(i)] for i in probs.argmax(axis=1)], ["sleep", "sleep", "sleep", "home"])

    def test_generated_schedule_respects_time_of_day(self) -> None:
        sched = self.model.generate_schedule(96, start_time=datetime(2026, 4, 1), seed=3)
        by_hour = {r["timestamp"].hour: r["activity"] for r in sched.iter_rows(named=True)}
        self.assertEqual(by_hour[3], "sleep")
        self.assertEqual(by_hour[12], "work")
        self.assertEqual(by_hour[19], "home")
        self.assertEqual(sched["timestamp"][1] - sched["timestamp"][0], timedelta(minutes=15))

    def test_state_machine_per_regime(self) -> None:
        pooled = self.model.to_state_machine()
        wake = self.model.to_state_machine(regime=6)
        self.assertIsNotNone(pooled)
        names = self.model.state_encoder.vocabulary["activity"].to_list()
        edges = {(e.source_id, e.target_id): e.probability for e in wake.to_plot_data().edges}
        self.assertAlmostEqual(edges[(names.index("sleep"), names.index("home"))], 1.0, places=2)

    def test_regime_of_and_errors(self) -> None:
        self.assertEqual(self.model.regime_of(datetime(2026, 1, 1, 13, 30)), 13)
        with self.assertRaises(ValueError):
            self.model.regime_of()
        with self.assertRaises(ValueError):
            self.model.predict_proba([0], context={"x": 1})
        with self.assertRaises(ValueError):
            self.model.predict([0], method="viterbi", at=datetime(2026, 1, 1))
        with self.assertRaises(NotImplementedError):
            self.model.partial_fit(self.df)

    def test_homogeneous_model_rejects_time_arguments(self) -> None:
        plain = SeqSM(k=1).fit(self.df.select("activity"))
        with self.assertRaises(ValueError):
            plain.predict_proba([0], at=datetime(2026, 1, 1))
        with self.assertRaises(ValueError):
            plain.regime_of(datetime(2026, 1, 1))
        with self.assertRaises(ValueError):
            plain.to_state_machine(regime=0)
        self.assertEqual(plain.n_regimes, 1)

    def test_requires_timestamp(self) -> None:
        with self.assertRaises(ValueError):
            SeqSM(time_config=TimeConfig(regimes="hour")).fit(self.df.select("activity"))

    def test_state_cols_alias(self) -> None:
        m = SeqSM(state_cols=["activity"]).fit(self.df)
        self.assertEqual(m.state_cols, ("activity",))
        with self.assertRaises(ValueError):
            SeqSM(state_cols=["a"], feature_cols=["b"])

    def test_dict_time_config(self) -> None:
        m = SeqSM(timestamp_col="timestamp", time_config={"slot_duration": "30m", "regimes": "day-part"})
        self.assertEqual(m.time_config, TimeConfig(slot_duration="30m", regimes="day-part"))
        m.fit(self.df)
        self.assertEqual(m.n_regimes, 4)


class TestAutoRegimes(unittest.TestCase):
    def test_auto_picks_time_regime_and_reports(self) -> None:
        m = SeqSM(timestamp_col="timestamp", time_config=TimeConfig(slot_duration="15m", regimes="auto")).fit(
            daily_activity(n_days=15)
        )
        self.assertGreater(m.n_regimes, 1)
        assert m.regime_report is not None
        self.assertEqual(m.regime_report.columns[:2], ["regime", "n_regimes"])

    def test_auto_picks_none_for_time_independent_data(self) -> None:
        m = SeqSM(
            timestamp_col="timestamp", time_config=TimeConfig(regimes="auto", candidates=("hour", "day-part"))
        ).fit(random_chain())
        self.assertEqual(m.n_regimes, 1)
        self.assertEqual(m.regime_of(datetime(2026, 1, 1, 4)), 0)

    def test_phase_regimes_with_explicit_and_auto_period(self) -> None:
        df = daily_activity(n_days=20, slot_min=60)
        m = SeqSM(
            timestamp_col="timestamp",
            time_config=TimeConfig(slot_duration="1h", regimes="phase", period="1d", n_regimes=24),
        ).fit(df)
        self.assertEqual(m.n_regimes, 24)
        auto = SeqSM(
            timestamp_col="timestamp",
            time_config=TimeConfig(slot_duration="1h", regimes="phase", period="auto", n_regimes=24),
        ).fit(df)
        self.assertEqual(auto._regime_meta["period_us"], 86_400_000_000)

    def test_phase_auto_period_failure(self) -> None:
        df = pl.DataFrame(
            {"timestamp": [datetime(2026, 1, 1) + timedelta(hours=i) for i in range(6)], "x": list("aaaaaa")}
        )
        with self.assertRaises(ValueError):
            SeqSM(
                timestamp_col="timestamp",
                time_config=TimeConfig(slot_duration="1h", regimes="phase", period="auto"),
            ).fit(df)

    def test_callable_regimes(self) -> None:
        def office_hours(ts: np.ndarray) -> np.ndarray:
            hours = (ts.astype("datetime64[h]") - ts.astype("datetime64[D]")).astype(int)
            return ((hours >= 8) & (hours < 17)).astype(np.int32)

        m = SeqSM(
            timestamp_col="timestamp",
            time_config=TimeConfig(slot_duration="15m", regimes=office_hours, n_regimes=2),
        ).fit(daily_activity(n_days=6))
        self.assertEqual(m.regime_of(datetime(2026, 1, 1, 9)), 1)
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            save_model(m, Path(tmp) / "m.npz")

    def test_event_mode_with_time(self) -> None:
        m = SeqSM(
            k=1,
            mode="event",
            timestamp_col="timestamp",
            time_config=TimeConfig(slot_duration="15m", regimes="hour"),
        ).fit(daily_activity(n_days=10))
        sched = m.generate_schedule(96, start_time=datetime(2026, 4, 1), seed=1)
        self.assertGreater(sched.height, 1)
        self.assertEqual(int(sched["dwell_duration"].sum()), 96)


class TestContextColumns(unittest.TestCase):
    def setUp(self) -> None:
        self.df = weekday_weekend()
        self.model = SeqSM(k=1, smoothing=0.5, context_cols=["day_type"], state_cols=["state"]).fit(self.df)

    def test_context_conditions_kernel(self) -> None:
        names = self.model.state_encoder.vocabulary["state"].to_list()
        i_in = names.index("in")
        weekend = self.model.predict_proba([names.index("out")], context={"day_type": "weekend"})
        weekday = self.model.predict_proba([names.index("out")], context={"day_type": "weekday"})
        self.assertGreater(weekend[i_in], 0.8)
        self.assertLess(weekday[i_in], 0.2)
        self.assertEqual(self.model.n_regimes, 2)

    def test_context_errors(self) -> None:
        with self.assertRaises(ValueError):
            self.model.predict_proba([0], context={"day_type": "holiday"})
        with self.assertRaises(ValueError):
            self.model.predict_proba([0], context={})
        with self.assertRaises(ValueError):
            self.model.regime_of(context=None)
        with self.assertRaises(ValueError):
            self.model.generate_schedule(5)
        with self.assertRaises(ValueError):
            SeqSM(state_cols=["state"], context_cols=["state"]).fit(self.df)

    def test_context_in_schedule(self) -> None:
        sched = self.model.generate_schedule(200, context={"day_type": "weekend"}, seed=2)
        share_in = (sched["state"] == "in").mean()
        assert isinstance(share_in, float)
        self.assertGreater(share_in, 0.7)
        low = self.model.generate_schedule(200, context={"day_type": "weekday"}, seed=2)
        self.assertLess((low["state"] == "in").mean(), 0.3)  # type: ignore[operator]

    def test_context_plus_time(self) -> None:
        m = SeqSM(
            k=1,
            timestamp_col="timestamp",
            context_cols=["day_type"],
            state_cols=["state"],
            time_config=TimeConfig(regimes="day-part"),
        ).fit(self.df)
        self.assertEqual(m.n_regimes, 8)
        self.assertEqual(m.regime_of(datetime(2026, 1, 3, 13), {"day_type": "weekend"}), 1 * 4 + 2)
        with self.assertRaises(ValueError):
            m.regime_of(context={"day_type": "weekend"})

    def test_model_without_context_rejects_context(self) -> None:
        m = SeqSM(timestamp_col="timestamp", time_config=TimeConfig(regimes="day-part")).fit(self.df.drop("day_type"))
        with self.assertRaises(ValueError):
            m.regime_of(datetime(2026, 1, 1), {"x": 1})


class TestAutoRoles(unittest.TestCase):
    def test_auto_roles_end_to_end(self) -> None:
        df = pl.read_csv(Path(__file__).parent.parent / "dataset" / "synthetic_routines.csv", try_parse_dates=True)
        m = SeqSM(k=1, smoothing=0.1, auto_roles=True, time_config=TimeConfig(slot_duration="15m", regimes="auto"))
        m.fit(df)
        roles = m.signal_roles
        assert roles is not None
        self.assertEqual(roles.time, "timestamp")
        self.assertEqual(roles.episode, "episode_id")
        self.assertNotIn("presence", m.state_encoder.feature_cols)
        self.assertEqual(set(m.state_encoder.feature_cols), {"room", "motion", "light", "tv_active"})
        self.assertGreaterEqual(m.n_regimes, 1)
        self.assertEqual(m.generate_schedule(8, seed=1).height, 8)

    def test_auto_roles_options_are_forwarded(self) -> None:
        df = pl.read_csv(Path(__file__).parent.parent / "dataset" / "synthetic_routines.csv", try_parse_dates=True)
        m = SeqSM(auto_roles={"max_joint_states": 6}).fit(df)
        self.assertLess(len(m.state_encoder.feature_cols), 4)

    def test_auto_roles_without_time_config(self) -> None:
        df = pl.read_csv(Path(__file__).parent.parent / "dataset" / "synthetic_routines.csv", try_parse_dates=True)
        m = SeqSM(auto_roles=True).fit(df)
        self.assertEqual(m.n_regimes, 1)
        self.assertNotIn("timestamp", m.state_encoder.feature_cols)


class TestSerialization(unittest.TestCase):
    def _roundtrip(self, model: SeqSM) -> SeqSM:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.npz"
            save_model(model, path)
            return load_model(path)

    def test_time_model_roundtrip(self) -> None:
        model = SeqSM(
            k=1,
            mode="event",
            smoothing=0.01,
            timestamp_col="timestamp",
            time_config=TimeConfig(slot_duration="15m", regimes="hour"),
        ).fit(daily_activity(n_days=10))
        loaded = self._roundtrip(model)
        self.assertEqual(loaded.n_regimes, 24)
        at = datetime(2026, 3, 2, 6, 0)
        np.testing.assert_allclose(model.predict_proba([0], at=at), loaded.predict_proba([0], at=at))
        a = model.generate_schedule(48, start_time=datetime(2026, 4, 1), seed=5)
        b = loaded.generate_schedule(48, start_time=datetime(2026, 4, 1), seed=5)
        self.assertEqual(a["state_id"].to_list(), b["state_id"].to_list())

    def test_phase_and_context_roundtrip(self) -> None:
        model = SeqSM(
            k=1,
            timestamp_col="timestamp",
            context_cols=["day_type"],
            state_cols=["state"],
            time_config=TimeConfig(slot_duration="1h", regimes="phase", period="1d", n_regimes=6),
        ).fit(weekday_weekend(n_weeks=3))
        loaded = self._roundtrip(model)
        ctx = {"day_type": "weekend"}
        at = datetime(2026, 3, 7, 10)
        self.assertEqual(model.regime_of(at, ctx), loaded.regime_of(at, ctx))
        np.testing.assert_allclose(
            model.predict_proba([0], at=at, context=ctx), loaded.predict_proba([0], at=at, context=ctx)
        )
        assert loaded.time_config is not None and model.time_config is not None
        self.assertEqual(loaded.time_config.regimes, "phase")
        self.assertEqual(duration_us(loaded.time_config.slot_duration or "1s"), 3_600_000_000)
        self.assertEqual(loaded.n_regimes, model.n_regimes)

    def test_context_only_roundtrip(self) -> None:
        model = SeqSM(k=1, context_cols=["day_type"], state_cols=["state"]).fit(weekday_weekend(n_weeks=2))
        loaded = self._roundtrip(model)
        ctx = {"day_type": "weekday"}
        np.testing.assert_allclose(model.predict_proba([0], context=ctx), loaded.predict_proba([0], context=ctx))
        self.assertIsNone(loaded.time_config)

    def test_homogeneous_model_has_no_phase_after_load(self) -> None:
        loaded = self._roundtrip(SeqSM(k=1).fit(daily_activity(n_days=2).select("activity")))
        self.assertIsNone(loaded.phase_chain)


class TestDiscreteTimeColumn(unittest.TestCase):
    """Time is already a discrete integer column (a tick/slot index); only a period is needed."""

    def setUp(self) -> None:
        self.df = tick_routine(n_days=12)
        self.model = SeqSM(
            k=1,
            smoothing=0.01,
            timestamp_col="slot",
            state_cols=["activity"],
            time_config=TimeConfig(regimes="phase", period=24),
        ).fit(self.df)

    def test_one_regime_per_phase_value(self) -> None:
        self.assertEqual(self.model.n_regimes, 24)
        self.assertEqual(self.model.state_encoder.feature_cols, ("activity",))
        self.assertEqual(self.model.regime_of(24 * 5 + 7), 7)

    def test_predict_uses_tick(self) -> None:
        names = self.model.state_encoder.vocabulary["activity"].to_list()
        # last sleep hour is 5; the target slot 6 is "home"
        p = self.model.predict_proba([names.index("sleep")], at=24 * 40 + 6)
        self.assertEqual(names[int(p.argmax())], "home")
        p = self.model.predict_proba([names.index("sleep")], at=24 * 40 + 3)
        self.assertEqual(names[int(p.argmax())], "sleep")
        multi = self.model.predict_proba([names.index("sleep")], steps=3, at=5)
        self.assertEqual([names[int(i)] for i in multi.argmax(axis=1)], ["sleep", "home", "home"])

    def test_schedule_ticks(self) -> None:
        sched = self.model.generate_schedule(48, start_time=0, seed=1)
        by_slot = dict(zip(sched["timestamp"].to_list(), sched["activity"].to_list(), strict=True))
        self.assertEqual(by_slot[3], "sleep")
        self.assertEqual(by_slot[12], "work")
        self.assertEqual(by_slot[24 + 20], "home")
        with self.assertRaises(TypeError):
            self.model.generate_schedule(10, start_time=datetime(2026, 1, 1))
        with self.assertRaises(TypeError):
            self.model.generate_schedule(timedelta(hours=1))

    def test_type_errors(self) -> None:
        with self.assertRaises(TypeError):
            self.model.regime_of(datetime(2026, 1, 1))
        with self.assertRaises(ValueError):
            SeqSM(timestamp_col="slot", time_config=TimeConfig(slot_duration="1h", regimes="phase", period=24)).fit(
                self.df
            )
        with self.assertRaises(ValueError):
            SeqSM(timestamp_col="slot", time_config=TimeConfig(regimes="day-part")).fit(self.df)
        with self.assertRaises(ValueError):
            SeqSM(timestamp_col="slot", time_config=TimeConfig(regimes="phase")).fit(self.df)
        with self.assertRaises(ValueError):
            TimeConfig(regimes="phase", period=1)
        plain = SeqSM(timestamp_col="timestamp", time_config=TimeConfig(regimes="day-part")).fit(daily_activity(3))
        with self.assertRaises(TypeError):
            plain.regime_of(5)
        with self.assertRaises(TypeError):
            plain.generate_schedule(4, start_time=3)

    def test_coarser_bins_and_callable(self) -> None:
        m = SeqSM(timestamp_col="slot", time_config=TimeConfig(regimes="phase", period=24, n_regimes=4)).fit(self.df)
        self.assertEqual(m.n_regimes, 4)
        self.assertEqual(m.regime_of(13), 2)
        c = SeqSM(
            timestamp_col="slot",
            time_config=TimeConfig(regimes=lambda t: (t % 24 >= 12).astype(np.int32), n_regimes=2),
        ).fit(self.df)
        self.assertEqual(c.regime_of(13), 1)

    def test_auto_period_and_auto_bins(self) -> None:
        noisy = tick_routine(n_days=30, noise=0.1, seed=2)
        m = SeqSM(
            smoothing=0.1,
            timestamp_col="slot",
            state_cols=["activity"],
            time_config=TimeConfig(regimes="phase", period="auto", n_regimes=6),
        ).fit(noisy)
        self.assertEqual(m._regime_meta["period"], 24)
        a = SeqSM(
            smoothing=0.1,
            timestamp_col="slot",
            state_cols=["activity"],
            time_config=TimeConfig(regimes="auto", period=24),
        ).fit(noisy)
        assert a.regime_report is not None
        self.assertGreater(a.n_regimes, 1)
        self.assertIn("none", a.regime_report["regime"].to_list())
        flat = pl.DataFrame({"slot": list(range(200)), "x": ["a", "b", "c", "a", "c"] * 40})
        with self.assertRaises(ValueError):
            SeqSM(timestamp_col="slot", time_config=TimeConfig(regimes="phase", period="auto")).fit(
                pl.DataFrame({"slot": list(range(8)), "x": list("aaaaaaaa")})
            )
        self.assertIsNotNone(SeqSM(timestamp_col="slot", time_config=TimeConfig(regimes="auto", period=5)).fit(flat))

    def test_auto_roles_finds_tick_column(self) -> None:
        m = SeqSM(auto_roles=True, time_config=TimeConfig(regimes="phase", period=24)).fit(self.df)
        self.assertEqual(m.signal_roles.time if m.signal_roles else None, "slot")
        self.assertEqual(m.state_encoder.feature_cols, ("activity",))
        self.assertEqual(m.n_regimes, 24)

    def test_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.npz"
            save_model(self.model, path)
            loaded = load_model(path)
        np.testing.assert_allclose(self.model.predict_proba([0], at=30), loaded.predict_proba([0], at=30))
        a = self.model.generate_schedule(30, start_time=2, seed=4)
        b = loaded.generate_schedule(30, start_time=2, seed=4)
        self.assertEqual(a["state_id"].to_list(), b["state_id"].to_list())
        self.assertEqual(loaded.time_config.period if loaded.time_config else None, 24)

    def test_event_mode(self) -> None:
        m = SeqSM(
            mode="event",
            timestamp_col="slot",
            state_cols=["activity"],
            time_config=TimeConfig(regimes="phase", period=24),
        ).fit(self.df)
        sched = m.generate_schedule(24, start_time=0, seed=1)
        self.assertEqual(int(sched["dwell_duration"].sum()), 24)


if __name__ == "__main__":
    unittest.main()
