"""Linear algebra, stationary distribution, and information-theoretic operations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def companion_matrix_power(companion: NDArray[np.float64], steps: int) -> NDArray[np.float64]:
    """Computes companion matrix power (P^steps) using OpenBLAS-accelerated matrix power."""
    if steps < 0:
        raise ValueError(f"Steps must be non-negative, got {steps}.")
    if steps == 0:
        return np.eye(companion.shape[0], dtype=companion.dtype)
    return np.linalg.matrix_power(companion, steps)


def stationary_distribution_power_iteration(
    companion: NDArray[np.float64],
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> NDArray[np.float64]:
    """
    Computes the stationary distribution pi over K-windows satisfying pi * P = pi.

    Uses OpenBLAS matrix-vector products (dgemv) with L1-norm convergence check.
    Falls back to left-eigenvector decomposition if power iteration fails to converge.
    """
    n = companion.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        return np.array([1.0], dtype=np.float64)

    # Initialize uniform distribution
    pi = np.full(n, 1.0 / n, dtype=np.float64)

    for _ in range(max_iter):
        next_pi = pi @ companion
        norm = np.linalg.norm(next_pi, ord=1)
        if norm > 0.0:
            next_pi /= norm

        if np.linalg.norm(next_pi - pi, ord=1) < tol:
            return next_pi
        pi = next_pi

    # Fallback: exact left eigenvector for eigenvalue 1
    eigenvals, eigenvecs = np.linalg.eig(companion.T)
    # Find eigenvalue closest to 1.0
    idx = int(np.argmin(np.abs(eigenvals - 1.0)))
    pi_eig = np.real(eigenvecs[:, idx])
    pi_eig = np.abs(pi_eig)
    sum_pi = np.sum(pi_eig)
    if sum_pi > 0.0:
        pi_eig /= sum_pi
    else:
        pi_eig = np.full(n, 1.0 / n, dtype=np.float64)
    return pi_eig


def marginal_stationary_distribution(
    window_stationary: NDArray[np.float64],
    k_tuples: NDArray[np.int32],
    n_states: int,
) -> NDArray[np.float64]:
    """
    Partitions and sums K-window stationary probabilities onto atomic states.

    mu_s = sum_{u: (w_u)_K = s} pi_u
    """
    marginal = np.zeros(n_states, dtype=np.float64)
    n_windows = len(window_stationary)
    for u in range(n_windows):
        last_state = int(k_tuples[u, -1])
        if 0 <= last_state < n_states:
            marginal[last_state] += window_stationary[u]

    total = np.sum(marginal)
    if total > 0.0:
        marginal /= total
    return marginal


def conditional_entropy_rate(
    window_stationary: NDArray[np.float64],
    transition_matrix: NDArray[np.float64],
) -> float:
    """
    Computes conditional entropy rate H(S_{t+1} | W_t) = - sum_u pi_u sum_v P_{uv} log2(P_{uv}).

    Follows 0 * log2(0) = 0 convention.
    """
    n_windows, n_targets = transition_matrix.shape
    if n_windows == 0 or n_targets == 0:
        return 0.0

    entropy = 0.0
    for u in range(n_windows):
        pi_u = window_stationary[u]
        if pi_u <= 0.0:
            continue
        row_entropy = 0.0
        for v in range(n_targets):
            p = transition_matrix[u, v]
            if p > 0.0:
                row_entropy -= p * np.log2(p)
        entropy += pi_u * row_entropy

    return float(entropy)


def model_perplexity(entropy_rate: float) -> float:
    """Computes perplexity = 2^(H(S))."""
    return float(2.0**entropy_rate)


def jensen_shannon_divergence(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """Computes Jensen-Shannon divergence JSD(p || q) in bits."""
    p_norm = p / np.sum(p) if np.sum(p) > 0.0 else p
    q_norm = q / np.sum(q) if np.sum(q) > 0.0 else q
    m = 0.5 * (p_norm + q_norm)

    def kl(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        mask = a > 0.0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p_norm, m) + 0.5 * kl(q_norm, m)


__all__ = [
    "companion_matrix_power",
    "conditional_entropy_rate",
    "jensen_shannon_divergence",
    "marginal_stationary_distribution",
    "model_perplexity",
    "stationary_distribution_power_iteration",
]
