"""Performance benchmark demonstrating OpenBLAS matrix operations and Numba kernel speed."""

from __future__ import annotations

import time

import numpy as np

from seqsm import KOrderMarkovChain
from seqsm.kernels import count_transitions_njit
from seqsm.linalg import companion_matrix_power, stationary_distribution_power_iteration


def run_benchmark() -> None:
    # 1. Benchmark Numba transition counting
    n_samples = 50_000
    n_windows = 500
    n_states = 50

    rng = np.random.default_rng(42)
    sources = rng.integers(0, n_windows, size=n_samples, dtype=np.int32)
    targets = rng.integers(0, n_states, size=n_samples, dtype=np.int32)

    # Warmup Numba JIT
    _ = count_transitions_njit(sources[:10], targets[:10], n_windows, n_states)

    t0 = time.perf_counter()
    counts = count_transitions_njit(sources, targets, n_windows, n_states)
    t_count = time.perf_counter() - t0

    # 2. Benchmark OpenBLAS Matrix Power (companion_matrix_power)
    # Generate random row-stochastic companion matrix
    raw_mat = rng.uniform(0.1, 1.0, size=(n_windows, n_windows))
    comp_mat = raw_mat / raw_mat.sum(axis=1, keepdims=True)

    t0 = time.perf_counter()
    _ = companion_matrix_power(comp_mat, 10)
    t_power = time.perf_counter() - t0

    # 3. Benchmark OpenBLAS Power Iteration (stationary_distribution_power_iteration)
    t0 = time.perf_counter()
    _ = stationary_distribution_power_iteration(comp_mat, tol=1e-8, max_iter=500)
    t_stat = time.perf_counter() - t0

    # 4. Fit KOrderMarkovChain
    chain = KOrderMarkovChain(k=2)
    states = rng.integers(0, 20, size=10_000, dtype=np.int32)
    t0 = time.perf_counter()
    chain.fit_from_arrays(states)
    t_fit = time.perf_counter() - t0

    _ = (counts, t_count, t_power, t_stat, t_fit)


if __name__ == "__main__":
    run_benchmark()
