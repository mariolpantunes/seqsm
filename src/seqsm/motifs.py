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
) -> int:
    """
    Detects dominant stream cycle lag via Categorical Match Autocorrelation (CMA).

    R_cat(tau) = 1/(T - tau) * sum_{t=0}^{T - tau - 1} I(s_t == s_{t+tau})
    rho_cat(tau) = (R_cat(tau) - P_baseline) / (1 - P_baseline)
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

    best_tau = 1
    best_rho = 0.05

    for tau in range(2, max_tau + 1):
        matches = int(np.sum(arr[:-tau] == arr[tau:]))
        r_cat = float(matches) / float(n - tau)
        rho = (r_cat - p_baseline) / denom
        if rho > best_rho:
            best_rho = rho
            best_tau = tau

    return best_tau


def score_anomalies(
    chain: KOrderMarkovChain | VariableOrderMarkovChain,
    encoded_states: NDArray[np.int32],
    dwell_times: NDArray[np.int32] | None = None,
    window_size: int = 5,
    threshold_quantile: float = 0.99,
    mode: Literal["two_factor", "nll"] = "two_factor",
) -> pl.DataFrame:
    """
    Computes joint semi-Markov negative log-likelihood (NLL) and flags behavioral anomalies.

    Decomposes into transition plausibility NLL_trans and dwell duration plausibility NLL_dwell.
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
    dwell_stats = chain.dwell_stats if isinstance(chain, KOrderMarkovChain) else {}

    for i in range(n):
        target = int(states[i])
        if target < 0:
            nll_trans[i] = 20.0
            nll_dwell[i] = 10.0
            continue

        if i == 0:
            mu = chain.marginal_stationary_distribution() if isinstance(chain, KOrderMarkovChain) else np.empty(0)
            p_trans = float(mu[target]) if target < len(mu) else 1e-6
        elif isinstance(chain, VariableOrderMarkovChain):
            probs = chain.predict_proba(states[:i])
            p_trans = float(probs[target]) if target < len(probs) else 1e-6
        else:
            k = chain.k
            if i >= k:
                win = tuple(int(x) for x in states[i - k : i])
                u = chain.encoder.encode_window(win)
                probs = chain.predict_proba_window(u)
                p_trans = float(probs[target]) if target < len(probs) else 1e-6
            else:
                mu = chain.marginal_stationary_distribution()
                p_trans = float(mu[target]) if target < len(mu) else 1e-6

        nll_trans[i] = -float(np.log2(max(p_trans, 1e-8)))

        d_val = int(dwells[i]) if i < len(dwells) else 1
        if target in dwell_stats and len(dwell_stats[target].samples) > 0:
            samples = dwell_stats[target].samples
            match_count = samples.count(d_val)
            p_dwell = float(match_count) / float(len(samples)) if match_count > 0 else 1.0 / (len(samples) + 1.0)
        else:
            p_dwell = 1.0
        nll_dwell[i] = -float(np.log2(max(p_dwell, 1e-8)))

    if mode == "two_factor":
        compound = 0.5 * nll_trans + 0.5 * nll_dwell
    else:
        compound = nll_trans

    rolling_nll = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window_size + 1)
        rolling_nll[i] = float(np.mean(compound[start : i + 1]))

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
