<p align="center">
  <img src="assets/logo.svg" alt="SeqSM Logo" width="540" />
</p>

# seqsm: Sequential State Machines & $K$-Order Markov Chains

`seqsm` is a domain-agnostic, high-performance Python library designed to learn, analyze, reduce, and simulate **$K$-order Markov Chains** and **Finite State Machines (FSMs)** from aligned multivariate discrete sequences. It features an intuitive Scikit-Learn-style estimator interface (`SeqSM`) with `fit`, `predict`, and `generate_schedule` methods.

---

## Key Highlights

- **Sparse Observed $K$-Window Indexing ($U_K$):** Avoids $|S|^K$ memory explosion by tracking only observed $K$-histories ($N_K \ll |S|^K$).
- **Dual Operator Formulation:**
  - **Rectangular Operator ($P^{(K)} \in \mathbb{R}^{N_K \times |S|}$):** Memory-efficient transition accumulation and immediate next-state querying.
  - **De Bruijn Square Companion Operator ($\tilde{P}^{(K)} \in \mathbb{R}^{N_K \times N_K}$):** Isomorphic first-order representation supporting OpenBLAS Level-3 matrix powers $(\tilde{P}^{(K)})^h$ for multi-step forecasting.
- **Clock & Semi-Markov Event Modes:**
  - `mode="clock"`: Uniform discrete time steps $\Delta t$, retaining self-transitions.
  - `mode="event"`: Semi-Markov run-length collapse of consecutive identical states into visits paired with empirical dwell-time distributions ($F_{\text{dwell}}$).
- **Valid-Arc Schedule Rollouts:** Graph pruning ($P_{uv} \ge \tau_{\text{prob}}$, $C_{uv} \ge \tau_{\text{count}}$) with fast Numba JIT simulation to produce synthetic schedules that strictly honor learned physical constraints.
- **Interpretable FSM Graphs & Visualization:** Direct string printing (`__str__` / `summary()`), native Matplotlib graph visualization (`fsm.plot()`), and exports to Mermaid flowcharts and Graphviz DOT digraphs with cycle extraction and edge difference summaries.
- **Generative Fidelity & Similarity:** Multi-aspect model comparison (`compare_similarity`) evaluating Jensen-Shannon divergence, stationary cosine similarity, directed edge Jaccard index, and transition concordance.

---

## Installation

`seqsm` requires Python `>= 3.12`.

```bash
pip install seqsm
```

### Core Dependencies
- `polars >= 2.1.0` (High-performance tabular data ingestion and expressions)
- `numpy >= 2.5.4` (OpenBLAS LEVEL-2 `dgemv` and LEVEL-3 `dgemm` vectorization)
- `numba >= 0.68.1` (JIT-compiled sliding-window kernels and simulation loops)

---

## Quickstart

### 1. Mining Household Routines & Generating Schedules

```python
from datetime import timedelta
import polars as pl
from seqsm import SeqSM

# 1. Load aligned discrete sensor time series
df = pl.DataFrame({
    "bedroom_motion": [0, 0, 1, 1, 0, 0, 1, 1],
    "living_light": ["off", "off", "dim", "on", "off", "off", "dim", "on"],
    "presence": ["home", "home", "home", "home", "away", "away", "home", "home"],
})

# 2. Fit K=2 SeqSM estimator in semi-Markov event mode
model = SeqSM(k=2, mode="event", smoothing=0.01)
model.fit(df)

# 3. Generate a 24-hour synthetic routine schedule
schedule_df = model.generate_schedule(
    window=timedelta(hours=24),
    time_unit="5m",
    method="sample",
)
print(schedule_df.head())
```


### 2. Exporting Interpretable State Machine Graphs

```python
# Extract pruned StateMachine graph
fsm = model.to_state_machine(min_prob=0.05, min_count=2)

# Export as Mermaid flowchart diagram
mermaid_code = fsm.to_mermaid(diff_labels=True)
print(mermaid_code)

# Extract dominant routine cycles
cycles = fsm.extract_cycles(max_length=6)
for c in cycles:
    print("Cycle:", c.state_sequence, "Persistence:", c.persistence)
```

### 3. Detecting Behavioral Anomalies

```python
# Compute negative log-likelihood on test streams
anomalies = model.score_anomalies(df, window_size=5, threshold_quantile=0.95)
print(anomalies.filter(pl.col("is_anomaly")))
```

### 4. Time as a Transition Signal, Signal Roles, Context Drivers

```python
from datetime import datetime
from seqsm import SeqSM, TimeConfig, infer_roles

print(infer_roles(df).table())           # which columns are state / context / time / episode / ignored

model = SeqSM(
    k=1, smoothing=0.1, auto_roles=True,  # roles inferred where not given
    time_config=TimeConfig(slot_duration="15m", regimes="auto"),  # "day-part", "hour", "weekend", "phase", callable
).fit(df)
model.regime_report                       # candidates scored by blocked cross-validation (or BIC)

model.predict_proba(history, at=datetime(2026, 3, 2, 8, 0))     # time-of-day kernel
model.generate_schedule(96, start_time=datetime(2026, 3, 2))    # rollout advances the slot clock
model.to_state_machine(regime=model.regime_of(datetime(2026, 3, 2, 8)))
```

`state_cols` define the atomic state, `context_cols` (e.g. `day_type`) condition the kernel without being states
(`predict_proba(..., context={...})`), and `time_config` turns the timestamp into a regime signal.
`discretize_columns` bins continuous columns first.
If time is already a discrete integer column (a tick/slot index), pass it as `timestamp_col` and give only a period:
`TimeConfig(regimes="phase", period=96, n_regimes=12)`, or `TimeConfig(regimes="auto", period="auto")` to detect the cycle and
pick the bin count; `at=` and `start_time=` are then int ticks. `examples/discrete_time_column.py` shows it. Examples: `examples/time_slot_state_machine.py`,
`examples/auto_signal_roles.py`, `examples/context_day_type.py`, `examples/episode_constraints.py`, `examples/batch_process_episodes.py`.

### 5. Streaming & Live Sensor Viewer

For online event-by-event learning from live sensors, `StreamingSeqSM` wraps `SeqSM.partial_fit` with a rolling `(k+1)`-event buffer, optional fixed-edge discretization, next-state probability ranking, and JSON graph export:

```python
from seqsm import StreamingSeqSM

stream = StreamingSeqSM(k=2, mode="clock", decay=0.98, bins={"cpu": [0, 20, 50, 80, 100]})
update = stream.update({"cpu": 35.2})
print(update["predictions"])   # ranked next-state probabilities
print(update["graph"])         # JSON-serializable nodes and edges
```

Run the zero-dependency live sensor web viewer (reads `/proc/stat` & `/proc/meminfo`, or pass `--simulate`):

```bash
PYTHONPATH=src venv/bin/python examples/live_sensor_viewer.py
```

---

## Development & Quality Gate

Development in `seqsm` follows the strict quality gates defined in `bp-python` and `bp-git`:

```bash
# Set up virtual environment and install test dependencies
python3 -m venv venv
venv/bin/pip install -q --upgrade . --group test

# Install pre-commit hooks
pre-commit install

# Run the complete test suite with coverage gate (>= 85%)
NUMBA_DISABLE_JIT=1 PYTHONPATH=src venv/bin/python -m coverage run --source=seqsm -m unittest discover -s test
venv/bin/python -m coverage report --fail-under=85

# Run linters and type-checkers
ruff check src test examples
ruff format --check src test examples
basedpyright
vulture
```

---

## License

MIT License. Copyright (c) 2026 Mário Antunes.
