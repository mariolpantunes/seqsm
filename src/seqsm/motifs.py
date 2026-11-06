"""Sequence motifs, behavioral cycle extraction, and anomaly scoring."""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray

from seqsm.linalg import jensen_shannon_divergence
from seqsm.markov import KOrderMarkovChain, VariableOrderMarkovChain


def score_anomalies(
    chain: KOrderMarkovChain | VariableOrderMarkovChain,
    encoded_states: NDArray[np.int32],
    window_size: int = 5,
    threshold_quantile: float = 0.99,
) -> pl.DataFrame:
    """
    Computes transition negative log-likelihood (NLL) and flags behavioral anomalies.

    Anomalies are detected when the rolling average NLL exceeds the threshold quantile.
    """
    states = np.asarray(encoded_states, dtype=np.int32)
    n = len(states)
    if n == 0:
        return pl.DataFrame(
            {
                "time_index": [],
                "state_id": [],
                "nll": [],
                "rolling_nll": [],
                "is_anomaly": [],
            }
        )

    nll_vals = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        target = int(states[i])
        if target < 0:
            nll_vals[i] = 20.0  # High penalty for unseen state
            continue

        if isinstance(chain, VariableOrderMarkovChain):
            probs = chain.predict_proba(states[:i])
            p = float(probs[target]) if target < len(probs) else 1e-6
        else:
            k = chain.k
            if i >= k:
                win = tuple(int(x) for x in states[i - k : i])
                u = chain.encoder.encode_window(win)
                probs = chain.predict_proba_window(u)
                p = float(probs[target]) if target < len(probs) else 1e-6
            else:
                mu = chain.marginal_stationary_distribution()
                p = float(mu[target]) if target < len(mu) else 1e-6

        p_safe = max(p, 1e-8)
        nll_vals[i] = -float(np.log2(p_safe))

    # Compute rolling average NLL
    rolling_nll = np.zeros(n, dtype=np.float64)
    for i in range(n):
        start = max(0, i - window_size + 1)
        rolling_nll[i] = float(np.mean(nll_vals[start : i + 1]))

    # Anomaly threshold
    thresh = float(np.quantile(rolling_nll, threshold_quantile)) if n > 1 else 10.0
    is_anomaly = rolling_nll >= thresh

    return pl.DataFrame(
        {
            "time_index": np.arange(n, dtype=np.int32),
            "state_id": states,
            "nll": nll_vals,
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


__all__ = ["compare_chains", "score_anomalies"]
