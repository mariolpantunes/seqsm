"""Linear algebra, stationary distribution, and information-theoretic operations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from seqsm.types import CSRMatrix


def companion_matrix_power(companion: NDArray[np.float64], steps: int) -> NDArray[np.float64]:
    """Computes companion matrix power (P^steps) using OpenBLAS-accelerated matrix power."""
    if steps < 0:
        raise ValueError(f"Steps must be non-negative, got {steps}.")
    if steps == 0:
        return np.eye(companion.shape[0], dtype=companion.dtype)
    if steps == 1:
        return companion.copy()
    return np.linalg.matrix_power(companion, steps)


def stationary_distribution_power_iteration(
    companion: NDArray[np.float64] | CSRMatrix,
    tol: float = 1e-10,
    max_iter: int = 10_000,
) -> NDArray[np.float64]:
    """
    Computes the stationary distribution pi over K-windows satisfying pi * P = pi.

    Uses OpenBLAS matrix-vector products (dgemv) with Aitken Delta^2 acceleration
    and L1-norm convergence check.
    Falls back to left-eigenvector decomposition if power iteration fails to converge.
    """
    n = companion.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    if n == 1:
        return np.array([1.0], dtype=np.float64)

    is_csr = isinstance(companion, CSRMatrix)
    # Initialize uniform distribution
    pi = np.full(n, 1.0 / n, dtype=np.float64)
    pi_history: list[NDArray[np.float64]] = []

    for _ in range(max_iter):
        if is_csr:
            next_pi = companion.dot_vector_left(pi)
        else:
            next_pi = pi @ companion

        norm = float(np.linalg.norm(next_pi, ord=1))
        if norm > 0.0:
            next_pi /= norm

        # Aitken Delta^2 acceleration every 3 steps
        pi_history.append(next_pi.copy())
        if len(pi_history) == 3:
            p0, p1, p2 = pi_history[0], pi_history[1], pi_history[2]
            denominator = p2 - 2.0 * p1 + p0
            valid_mask = np.abs(denominator) > 1e-14
            acc_pi = p2.copy()
            delta = p1 - p0
            acc_pi[valid_mask] = p0[valid_mask] - (delta[valid_mask] ** 2) / denominator[valid_mask]
            acc_pi = np.maximum(acc_pi, 0.0)
            norm_acc = float(np.linalg.norm(acc_pi, ord=1))
            if norm_acc > 0.0 and np.all(np.isfinite(acc_pi)):
                next_pi = acc_pi / norm_acc
            pi_history.clear()

        if float(np.linalg.norm(next_pi - pi, ord=1)) < tol:
            return next_pi
        pi = next_pi

    # Fallback: exact left eigenvector for eigenvalue 1
    dense_comp = companion.to_dense() if is_csr else companion
    eigenvals, eigenvecs = np.linalg.eig(dense_comp.T)
    idx = int(np.argmin(np.abs(eigenvals - 1.0)))
    pi_eig = np.real(eigenvecs[:, idx])
    pi_eig = np.abs(pi_eig)
    sum_pi = float(np.sum(pi_eig))
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
    if len(window_stationary) == 0 or n_states == 0:
        return np.zeros(n_states, dtype=np.float64)

    last_states = k_tuples[:, -1]
    valid_mask = (last_states >= 0) & (last_states < n_states)
    marginal = np.bincount(
        last_states[valid_mask],
        weights=window_stationary[valid_mask],
        minlength=n_states,
    ).astype(np.float64)

    total = float(np.sum(marginal))
    if total > 0.0:
        marginal /= total
    return marginal


def conditional_entropy_rate(
    window_stationary: NDArray[np.float64],
    transition_matrix: NDArray[np.float64],
) -> float:
    """
    Computes conditional entropy rate H(S_{t+1} | W_t) = - sum_u pi_u sum_v P_{uv} log2(P_{uv}).

    Follows 0 * log2(0) = 0 convention. Fully vectorized via OpenBLAS.
    """
    n_windows, n_targets = transition_matrix.shape
    if n_windows == 0 or n_targets == 0 or len(window_stationary) == 0:
        return 0.0

    log_p = np.zeros_like(transition_matrix)
    mask = transition_matrix > 0.0
    log_p[mask] = np.log2(transition_matrix[mask])
    row_entropy = -np.sum(transition_matrix * log_p, axis=1)
    return float(np.sum(window_stationary * row_entropy))


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
