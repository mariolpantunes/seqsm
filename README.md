# seqsm

**Sequential State Machines and $K$-Order Markov Chains from Aligned Sensor Sequences**

`seqsm` is a high-performance Python library designed to model, analyze, and simulate
multivariate discrete sequences using $K$-order Markov chains and Finite State Machines (FSMs).

## Features
- **Compact Sparse Indexing:** Avoids $|S|^K$ memory explosion by tracking observed $K$-histories.
- **Dual Representation:** Rectangular transition matrix and square De Bruijn companion matrix.
- **OpenBLAS Vectorization:** Fast multi-step prediction, power iteration stationary distribution, and entropy calculation.
- **Numba Acceleration:** High-speed sliding-window counting and valid-arc schedule simulation.
- **Scikit-Learn Style API:** `SeqSM` estimator with `fit`, `predict`, and `generate_schedule`.
- **Clock & Event Modes:** Handles uniform time ticks and semi-Markov run-length dwell times.
