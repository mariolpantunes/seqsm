"""Sequence motifs, behavioral cycle extraction, and anomaly scoring."""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from seqsm.linalg import jensen_shannon_divergence
from seqsm.markov import KOrderMarkovChain, VariableOrderMarkovChain


def detect_period(
    states: NDArray[np.int32],
    max_lag: int | None = None,
    min_rho: float = 0.05,
) -> int:
    """
    Detects dominant stream cycle lag via Categorical Match Autocorrelation (CMA).

    R_cat(tau) = 1/(T - tau) * sum_{t=0}^{T - tau - 1} I(s_t == s_{t+tau})
    rho_cat(tau) = (R_cat(tau) - P_baseline) / (1 - P_baseline)

    Parameters:
        states: Sequence of atomic state IDs.
        max_lag: Maximum lag horizon to scan (defaults to min(N//2, 500)).
        min_rho: Minimum correlation elevation over stationary baseline P_baseline = sum_s p(s)^2
            required to reject random background noise and accept lag tau as periodic (Jesus/Antunes FiCloud 2017).
    """
    arr = np.asarray(states, dtype=np.int32)
    n = len(arr)
    if n < 4:
        return 1

    max_tau = min(n // 2, 500) if max_lag is None else min(max_lag, n - 1)
    if max_tau < 2:
        return 1

    valid_states = arr[arr >= 0]
    if len(valid_states) == 0:
        return 1

    counts = np.bincount(valid_states)
    probs = counts.astype(np.float64) / len(valid_states)
    p_baseline = float(np.sum(probs**2))
    denom = max(1.0 - p_baseline, 1e-12)

    from seqsm.kernels import cma_detect_period_njit

    best_tau, _ = cma_detect_period_njit(arr, max_tau, p_baseline, denom, min_rho)
    return int(best_tau)


def score_anomalies(
    chain: KOrderMarkovChain | VariableOrderMarkovChain,
    encoded_states: NDArray[np.int32],
    dwell_times: NDArray[np.int32] | None = None,
    window_size: int = 5,
    threshold_quantile: float = 0.99,
    mode: Literal["two_factor", "nll"] = "two_factor",
    weight_trans: float = 0.5,
    weight_dwell: float = 0.5,
    penalty_unseen_trans: float = 20.0,
    penalty_unseen_dwell: float = 10.0,
    prob_floor: float = 1e-8,
) -> pl.DataFrame:
    """
    Computes joint semi-Markov negative log-likelihood (NLL) and flags behavioral anomalies.

    Decomposes into transition plausibility NLL_trans and dwell duration plausibility NLL_dwell.

    Parameters:
        chain: Fitted KOrderMarkovChain or VariableOrderMarkovChain model.
        encoded_states: Sequence of atomic state IDs.
        dwell_times: Optional dwell durations per state visit.
        window_size: Window size for rolling anomaly NLL averaging.
        threshold_quantile: Quantile cutoff on rolling NLL to flag anomaly steps.
        mode: 'two_factor' (combines transition and dwell NLL) or 'nll' (transition only).
        weight_trans: Weight for transition surprise (default 0.5 per semi-Markov ADL conventions).
        weight_dwell: Weight for dwell duration surprise (default 0.5 per Jesus/Antunes FiCloud 2017).
        penalty_unseen_trans: NLL penalty assigned to unseen/negative state transitions (default 20.0).
        penalty_unseen_dwell: NLL penalty assigned to unobserved dwell durations (default 10.0).
        prob_floor: Minimum probability floor to prevent log2(0) divergence (Begleiter et al., 2004).
    """
    states = np.asarray(encoded_states, dtype=np.int32)
    n = len(states)
    if n == 0:
        return pl.DataFrame(
            {
                "time_index": [],
                "state_id": [],
                "nll": [],
                "nll_trans": [],
                "nll_dwell": [],
                "rolling_nll": [],
                "is_anomaly": [],
            }
        )

    nll_trans = np.zeros(n, dtype=np.float64)
    nll_dwell = np.zeros(n, dtype=np.float64)

    dwells = np.asarray(dwell_times, dtype=np.int32) if dwell_times is not None else np.ones(n, dtype=np.int32)

    # Precompute O(1) dwell sample frequency tables
    from collections import Counter

    dwell_counts: dict[int, tuple[Counter[int], int, float]] = {}
    if isinstance(chain, KOrderMarkovChain):
        for sid, d_dist in chain.dwell_stats.items():
            if len(d_dist.samples) > 0:
                c = Counter(d_dist.samples)
                m = len(d_dist.samples)
                dwell_counts[sid] = (c, m, 1.0 / (m + 1.0))

    # Precompute marginal distribution
    mu = (
        chain.marginal_stationary_distribution()
        if isinstance(chain, KOrderMarkovChain)
        else chain.chains[0].marginal_stationary_distribution()
    )

    for i in range(n):
        target = int(states[i])
        if target < 0:
            nll_trans[i] = penalty_unseen_trans
            nll_dwell[i] = penalty_unseen_dwell
            continue

        if i == 0:
            p_trans = float(mu[target]) if target < len(mu) else 1e-6
        elif isinstance(chain, VariableOrderMarkovChain):
            max_k = max(c.k for c in chain.chains)
            hist = states[max(0, i - max_k) : i]
            probs = chain.predict_proba(hist)
            p_trans = float(probs[target]) if target < len(probs) else 1e-6
        else:
            k = chain.k
            if i >= k:
                win = tuple(int(x) for x in states[i - k : i])
                u = chain.encoder.encode_window(win)
                probs = chain.predict_proba_window(u)
                p_trans = float(probs[target]) if target < len(probs) else 1e-6
            else:
                p_trans = float(mu[target]) if target < len(mu) else 1e-6

        nll_trans[i] = -float(np.log2(max(p_trans, prob_floor)))

        d_val = int(dwells[i]) if i < len(dwells) else 1
        if target in dwell_counts:
            counts, m, fallback_p = dwell_counts[target]
            cnt = counts.get(d_val, 0)
            p_dwell = float(cnt) / float(m) if cnt > 0 else fallback_p
        else:
            p_dwell = 1.0
        nll_dwell[i] = -float(np.log2(max(p_dwell, prob_floor)))

    if mode == "two_factor":
        compound = weight_trans * nll_trans + weight_dwell * nll_dwell
    else:
        compound = nll_trans

    cumsum = np.cumsum(np.insert(compound, 0, 0.0))
    window_lengths = np.minimum(np.arange(1, n + 1), window_size)
    rolling_nll = (cumsum[1:] - cumsum[np.maximum(0, np.arange(1, n + 1) - window_size)]) / window_lengths

    thresh = float(np.quantile(rolling_nll, threshold_quantile)) if n > 1 else 10.0
    is_anomaly = rolling_nll >= thresh

    return pl.DataFrame(
        {
            "time_index": np.arange(n, dtype=np.int32),
            "state_id": states,
            "nll": compound,
            "nll_trans": nll_trans,
            "nll_dwell": nll_dwell,
            "rolling_nll": rolling_nll,
            "is_anomaly": is_anomaly,
        }
    )


def compare_chains(
    chain_a: KOrderMarkovChain,
    chain_b: KOrderMarkovChain,
) -> pl.DataFrame:
    """Computes Jensen-Shannon divergence and distribution differences between two chains."""
    mu_a = chain_a.marginal_stationary_distribution()
    mu_b = chain_b.marginal_stationary_distribution()
    max_states = max(len(mu_a), len(mu_b))

    pad_a = np.pad(mu_a, (0, max_states - len(mu_a)))
    pad_b = np.pad(mu_b, (0, max_states - len(mu_b)))
    jsd_val = jensen_shannon_divergence(pad_a, pad_b)

    return pl.DataFrame(
        {
            "state_id": np.arange(max_states, dtype=np.int32),
            "prob_a": pad_a,
            "prob_b": pad_b,
            "prob_diff": pad_a - pad_b,
            "jsd_total": [jsd_val] * max_states,
        }
    )


__all__ = ["compare_chains", "detect_period", "score_anomalies"]
