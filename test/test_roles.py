"""Unit tests for automatic signal-role inference and discretization."""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from seqsm.roles import (
    discretize_columns,
    infer_roles,
    mutual_information,
    next_step_predictability,
)


def _table(n: int = 400, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    room = np.repeat(rng.integers(0, 3, n // 20), 20)  # sticky => predictable
    return pl.DataFrame(
        {
            "timestamp": [datetime(2026, 1, 1) + timedelta(minutes=15 * i) for i in range(n)],
            "episode_id": np.repeat(np.arange(n // 100), 100),
            "room": [["bed", "kit", "liv"][i] for i in room],
            "presence": ["away" if i == 2 else "home" for i in room],
            "light": rng.choice(["on", "off"], n),
            "version": ["v1"] * n,
            "row_id": np.arange(n),
            "power": rng.normal(size=n),
            "hour_of_day": [(i // 4) % 24 for i in range(n)],
            "many": rng.integers(0, 40, n).astype(str),
        }
    )


class TestInferRoles(unittest.TestCase):
    def setUp(self) -> None:
        self.roles = infer_roles(_table())
        self.by_name = {p.name: p for p in self.roles.profiles}

    def test_time_and_episode(self) -> None:
        self.assertEqual(self.roles.time, "timestamp")
        self.assertEqual(self.roles.episode, "episode_id")

    def test_states(self) -> None:
        self.assertEqual(self.roles.state, ("room", "light"))

    def test_context_from_name(self) -> None:
        self.assertEqual(self.roles.context, ("hour_of_day",))

    def test_ignored_with_reasons(self) -> None:
        self.assertEqual(set(self.roles.ignored), {"presence", "version", "row_id", "power", "many"})
        self.assertIn("constant", self.by_name["version"].reason)
        self.assertIn("identifier", self.by_name["row_id"].reason)
        self.assertIn("discretize_columns", self.by_name["power"].reason)
        self.assertIn("high cardinality", self.by_name["many"].reason)
        self.assertIn("redundant with room", self.by_name["presence"].reason)

    def test_predictability_orders_columns(self) -> None:
        room = self.by_name["room"].predictability
        light = self.by_name["light"].predictability
        assert room is not None and light is not None
        self.assertGreater(room, 0.5)
        self.assertLess(light, 0.1)

    def test_table_shape(self) -> None:
        tab = self.roles.table()
        self.assertEqual(tab.height, 10)
        self.assertIn("reason", tab.columns)

    def test_explicit_columns_win(self) -> None:
        roles = infer_roles(_table(), episode_col="room", timestamp_col="row_id")
        self.assertEqual((roles.episode, roles.time), ("room", "row_id"))
        self.assertNotIn("room", roles.state)

    def test_noise_demotion(self) -> None:
        roles = infer_roles(_table(), min_predictability=0.2)
        self.assertEqual(roles.state, ("room",))
        self.assertIn("noise", {p.name: p for p in roles.profiles}["light"].reason)

    def test_joint_state_cap_drops_least_predictable(self) -> None:
        roles = infer_roles(_table(), max_joint_states=3)
        self.assertEqual(roles.state, ("room",))
        self.assertIn("joint state space", {p.name: p for p in roles.profiles}["light"].reason)

    def test_numeric_clock_column_is_time(self) -> None:
        df = pl.DataFrame({"time": list(range(50)), "x": ["a", "b"] * 25})
        self.assertEqual(infer_roles(df).time, "time")

    def test_custom_context_hint(self) -> None:
        import re

        roles = infer_roles(_table(), context_hints=re.compile("light"))
        self.assertEqual(roles.context, ("light",))


class TestInformation(unittest.TestCase):
    def test_mutual_information_identity_and_independence(self) -> None:
        a = np.array([0, 1, 0, 1] * 50, dtype=np.int64)
        mi, h_a, h_b = mutual_information(a, a)
        self.assertAlmostEqual(mi, 1.0)
        self.assertAlmostEqual(h_a, 1.0)
        self.assertAlmostEqual(h_b, 1.0)
        b = np.array([0, 0, 1, 1] * 50, dtype=np.int64)
        self.assertAlmostEqual(mutual_information(a, b)[0], 0.0, places=9)
        with self.assertRaises(ValueError):
            mutual_information(a, a[:-1])
        self.assertEqual(mutual_information(a[:0], a[:0]), (0.0, 0.0, 0.0))

    def test_next_step_predictability_extremes(self) -> None:
        alternating = np.array([0, 1] * 100, dtype=np.int64)
        self.assertAlmostEqual(next_step_predictability(alternating), 1.0)
        rng = np.random.default_rng(3)
        noise = rng.integers(0, 2, 5000).astype(np.int64)
        self.assertLess(next_step_predictability(noise), 0.01)
        self.assertEqual(next_step_predictability(np.array([0, 1], dtype=np.int64)), 0.0)

    def test_predictability_respects_episode_boundaries(self) -> None:
        codes = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        eps = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32)
        self.assertAlmostEqual(next_step_predictability(codes, eps), 1.0)


class TestDiscretize(unittest.TestCase):
    def test_quantile_bins_are_balanced(self) -> None:
        rng = np.random.default_rng(0)
        df = pl.DataFrame({"x": rng.normal(size=1000)})
        out = discretize_columns(df, ["x"], n_bins=4)
        counts = out["x"].value_counts()["count"].to_list()
        self.assertEqual(len(counts), 4)
        self.assertLess(max(counts) - min(counts), 10)

    def test_uniform_and_auto(self) -> None:
        df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]})
        out = discretize_columns(df, ["x"], n_bins=2, method="uniform")
        self.assertEqual(out["x"].to_list(), [0] * 5 + [1] * 5)
        auto = discretize_columns(df, ["x"])
        self.assertGreaterEqual(auto["x"].n_unique(), 2)

    def test_errors(self) -> None:
        with self.assertRaises(ValueError):
            discretize_columns(pl.DataFrame({"x": [1.0, 2.0]}), ["x"], n_bins=1)
        with self.assertRaises(ValueError):
            discretize_columns(pl.DataFrame({"x": pl.Series([None, None], dtype=pl.Float64)}), ["x"])

    def test_binned_float_becomes_state(self) -> None:
        df = _table()
        binned = discretize_columns(df, ["power"], n_bins=3)
        self.assertIn("power", infer_roles(binned).state)


if __name__ == "__main__":
    unittest.main()
