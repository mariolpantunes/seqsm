"""Unit tests for regime-conditioned (phase) Markov chains and BIC regime selection."""

from __future__ import annotations

import unittest
from datetime import datetime

import numpy as np
from synthetic import daily_activity, random_chain

from seqsm.encoding import StateEncoder
from seqsm.markov import KOrderMarkovChain
from seqsm.periodic import (
    PhaseConditionedMarkovChain,
    collapse_with_regimes,
    cross_validated_nll,
    regime_description_length,
    select_regimes,
)
from seqsm.timeslots import datetime_us, preset_regimes


def _encode(df, col):  # type: ignore[no-untyped-def]
    enc = StateEncoder(feature_cols=[col])
    return enc.fit_transform(df).astype(np.int32), enc


class TestCollapse(unittest.TestCase):
    def test_clock_is_identity(self) -> None:
        s = np.array([0, 0, 1], dtype=np.int32)
        r = np.array([0, 1, 1], dtype=np.int32)
        es, er, ee, ed = collapse_with_regimes(s, r, None, "clock")
        self.assertEqual((es.tolist(), er.tolist(), ee, ed.tolist()), ([0, 0, 1], [0, 1, 1], None, [1, 1, 1]))

    def test_event_runs_take_start_regime_and_split_episodes(self) -> None:
        s = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
        r = np.array([0, 1, 1, 2, 2, 3], dtype=np.int32)
        e = np.array([0, 0, 0, 0, 1, 1], dtype=np.int32)
        es, er, ee, ed = collapse_with_regimes(s, r, e, "event")
        self.assertEqual(es.tolist(), [0, 1, 1])
        self.assertEqual(er.tolist(), [0, 2, 2])
        self.assertEqual(ee.tolist() if ee is not None else None, [0, 0, 1])
        self.assertEqual(ed.tolist(), [3, 1, 2])

    def test_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            collapse_with_regimes(np.zeros(3, np.int32), np.zeros(2, np.int32), None, "clock")


class TestPhaseChain(unittest.TestCase):
    def setUp(self) -> None:
        self.df = daily_activity(n_days=12)
        self.states, self.enc = _encode(self.df, "activity")
        self.ts = datetime_us(self.df["timestamp"])
        self.spec = preset_regimes("hour")
        self.regimes = self.spec(self.ts)
        self.chain = PhaseConditionedMarkovChain(k=1, n_regimes=24, smoothing=0.01).fit_from_arrays(
            self.states, self.regimes, n_states=self.enc.n_states
        )

    def test_counts_sum_to_pooled(self) -> None:
        total = self.chain.regime_counts.sum(axis=0)
        np.testing.assert_allclose(total, self.chain.pooled.count_matrix)

    def test_single_regime_reduces_to_pooled(self) -> None:
        one = PhaseConditionedMarkovChain(k=1, n_regimes=1, smoothing=0.01).fit_from_arrays(
            self.states, np.zeros(len(self.states), np.int32), n_states=self.enc.n_states
        )
        np.testing.assert_allclose(one.chain_for(0).transition_matrix, one.pooled.transition_matrix)
        ref = KOrderMarkovChain(k=1, smoothing=0.01).fit_from_arrays(self.states, n_states=self.enc.n_states)
        np.testing.assert_allclose(one.pooled.transition_matrix, ref.transition_matrix)

    def test_every_regime_matrix_is_row_stochastic(self) -> None:
        for r in range(24):
            np.testing.assert_allclose(self.chain.chain_for(r).transition_matrix.sum(axis=1), 1.0)

    def test_regime_kernel_is_sharper_than_pooled(self) -> None:
        sleep = int(self.enc.dict_to_state({"activity": "sleep"}))
        home = int(self.enc.dict_to_state({"activity": "home"}))
        u = self.chain.pooled.encoder.encode_window((sleep,))
        p_morning = self.chain.predict_proba_window(u, 6)  # target slot at 06:xx: sleep -> home
        p_night = self.chain.predict_proba_window(u, 3)
        p_pool = self.chain.pooled.predict_proba_window(u)
        self.assertGreater(p_morning[home], 0.9)
        self.assertGreater(p_night[sleep], 0.9)
        self.assertLess(p_pool[home], 0.2)

    def test_unsupported_rows_fall_back_to_pooled(self) -> None:
        sparse = PhaseConditionedMarkovChain(k=1, n_regimes=24, smoothing=0.0, min_regime_count=10_000)
        sparse.fit_from_arrays(self.states, self.regimes, n_states=self.enc.n_states)
        np.testing.assert_allclose(sparse.chain_for(5).transition_matrix, sparse.pooled.transition_matrix)
        self.assertFalse(sparse.regime_support(5).any())

    def test_multi_step_follows_regime_sequence(self) -> None:
        sleep = int(self.enc.dict_to_state({"activity": "sleep"}))
        home = int(self.enc.dict_to_state({"activity": "home"}))
        u = self.chain.pooled.encoder.encode_window((sleep,))
        out = self.chain.multi_step_distribution(u, [6, 7, 8])
        self.assertEqual(out.shape, (3, self.enc.n_states))
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-9)
        self.assertEqual(int(np.argmax(out[0])), home)
        with self.assertRaises(ValueError):
            self.chain.multi_step_distribution(u, [])

    def test_heldout_nll_improves_with_regimes(self) -> None:
        train = slice(0, 96 * 8)
        test = slice(96 * 8, None)
        phase = PhaseConditionedMarkovChain(k=1, n_regimes=24, smoothing=0.1).fit_from_arrays(
            self.states[train], self.regimes[train], n_states=self.enc.n_states
        )
        nll_phase = phase.average_nll(self.states[test], self.regimes[test])
        nll_pool = phase.average_nll(self.states[test])
        self.assertLess(nll_phase, nll_pool)
        self.assertTrue(np.isnan(phase.average_nll(self.states[:1])))

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            PhaseConditionedMarkovChain(n_regimes=0)
        with self.assertRaises(ValueError):
            PhaseConditionedMarkovChain(min_regime_count=0)
        with self.assertRaises(ValueError):
            self.chain.chain_for(24)
        with self.assertRaises(ValueError):
            PhaseConditionedMarkovChain(n_regimes=2).fit_from_arrays(self.states, self.regimes)
        with self.assertRaises(ValueError):
            _ = PhaseConditionedMarkovChain().pooled
        with self.assertRaises(ValueError):
            _ = PhaseConditionedMarkovChain().regime_counts

    def test_set_fitted_roundtrip(self) -> None:
        clone = PhaseConditionedMarkovChain(k=1, n_regimes=24, smoothing=0.01)
        clone.set_fitted(self.chain.pooled, self.chain.regime_counts)
        np.testing.assert_allclose(clone.chain_for(6).transition_matrix, self.chain.chain_for(6).transition_matrix)
        with self.assertRaises(ValueError):
            PhaseConditionedMarkovChain(n_regimes=3).set_fitted(self.chain.pooled, self.chain.regime_counts)
        self.assertIn("n_regimes=24", repr(self.chain))

    def test_event_mode_regime_dwells(self) -> None:
        ev = PhaseConditionedMarkovChain(k=1, n_regimes=24, mode="event", min_regime_count=1)
        ev.fit_from_arrays(self.states, self.regimes, n_states=self.enc.n_states)
        work = int(self.enc.dict_to_state({"activity": "work"}))
        self.assertIn(8, ev.regime_dwell_samples)
        dwell = ev.chain_for(8).dwell_stats[work]
        self.assertEqual(dwell.count, 12)  # one work block start per day at 08:00
        self.assertEqual(dwell.mean, 36.0)  # 9 hours of 15-minute slots
        self.assertEqual(ev.chain_for(3).dwell_stats[work].count, ev.pooled.dwell_stats[work].count)


class TestRegimeSelection(unittest.TestCase):
    def test_time_dependent_data_selects_a_time_regime(self) -> None:
        df = daily_activity(n_days=15)
        states, _ = _encode(df, "activity")
        best, table = select_regimes(states, datetime_us(df["timestamp"]), ["day-part", "hour", "weekend"])
        self.assertNotEqual(best.name, "none")
        self.assertEqual(table["regime"][0], best.name)
        self.assertIn("none", table["regime"].to_list())

    def test_time_independent_data_selects_none(self) -> None:
        df = random_chain()
        states, _ = _encode(df, "x")
        best, _ = select_regimes(states, datetime_us(df["timestamp"]), ["hour", "day-part"])
        self.assertEqual(best.name, "none")

    def test_description_length_terms(self) -> None:
        df = daily_activity(n_days=4)
        states, _ = _encode(df, "activity")
        reg = preset_regimes("hour")(datetime_us(df["timestamp"]))
        bic, ll, n_params, n = regime_description_length(states, reg, 24)
        self.assertEqual(n, len(states) - 1)
        self.assertAlmostEqual(ll, 0.0)  # deterministic routine: every row has one successor
        self.assertEqual(n_params, 0)
        self.assertAlmostEqual(bic, 0.0)
        rnd = random_chain(n=500)
        r_states, _ = _encode(rnd, "x")
        bic, ll, n_params, n = regime_description_length(r_states, np.zeros(500, np.int32), 1)
        self.assertLess(ll, 0.0)
        self.assertEqual(n_params, 6)  # 3 rows x (3 successors - 1)
        self.assertAlmostEqual(bic, -2 * ll + n_params * np.log(n))
        self.assertEqual(regime_description_length(states[:1], reg[:1], 24)[3], 0)

    def test_custom_regime_spec_and_dedup(self) -> None:
        df = random_chain(n=300)
        states, _ = _encode(df, "x")
        spec = preset_regimes("weekend")
        _, table = select_regimes(states, datetime_us(df["timestamp"]), [spec, "weekend", "none"])
        self.assertEqual(table.filter(table["regime"] == "weekend").height, 1)
        self.assertEqual(datetime(2026, 1, 5).weekday(), 0)


class TestCrossValidation(unittest.TestCase):
    def test_cv_selects_time_regime_for_time_dependent_data(self) -> None:
        df = daily_activity(n_days=12)
        states, _ = _encode(df, "activity")
        best, table = select_regimes(
            states,
            datetime_us(df["timestamp"]),
            ["day-part", "hour"],
            criterion="cv",
            episode_ids=(np.arange(len(df)) // 96).astype(np.int32),
        )
        self.assertNotEqual(best.name, "none")
        self.assertFalse(table["cv_nll"].is_nan().any())
        self.assertEqual(table["regime"][0], best.name)

    def test_cv_selects_none_for_time_independent_data(self) -> None:
        df = random_chain(n=3000)
        states, _ = _encode(df, "x")
        best, _ = select_regimes(states, datetime_us(df["timestamp"]), ["hour", "weekend"], criterion="cv")
        self.assertEqual(best.name, "none")

    def test_cv_nll_prefers_informative_regimes(self) -> None:
        df = daily_activity(n_days=8)
        states, _ = _encode(df, "activity")
        ts = datetime_us(df["timestamp"])
        eps = (np.arange(len(df)) // 96).astype(np.int32)
        informative = cross_validated_nll(states, preset_regimes("hour")(ts), 24, episode_ids=eps)
        flat = cross_validated_nll(states, np.zeros(len(states), np.int32), 1, episode_ids=eps)
        self.assertLess(informative, flat)

    def test_invalid_criterion(self) -> None:
        with self.assertRaises(ValueError):
            select_regimes(np.zeros(10, np.int32), np.zeros(10, "datetime64[us]"), [], criterion="aic")  # type: ignore[arg-type]

    def test_nll_sum_matches_manual_for_order_two(self) -> None:
        df = daily_activity(n_days=6)
        states, enc = _encode(df, "activity")
        regimes = preset_regimes("day-part")(datetime_us(df["timestamp"]))
        chain = PhaseConditionedMarkovChain(k=2, n_regimes=4, smoothing=0.5).fit_from_arrays(
            states, regimes, n_states=enc.n_states
        )
        total, count = chain.nll_sum(states, regimes)
        manual, used = 0.0, 0
        for t in range(2, len(states)):
            u = chain.pooled.encoder.encode_window((int(states[t - 2]), int(states[t - 1])))
            if u < 0:
                continue
            row = chain.chain_for(int(regimes[t])).transition_matrix[u]
            manual += -float(np.log(max(float(row[states[t]]), 1e-6)))
            used += 1
        self.assertEqual(count, used)
        self.assertAlmostEqual(total, manual, places=6)


if __name__ == "__main__":
    unittest.main()
