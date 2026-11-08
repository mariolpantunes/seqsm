"""Industrial batch process: episode boundaries, continuous simulation, and constrained runs.

Demonstrates:
1. Multi-batch manufacturing sequences where each batch is an isolated episode.
2. Boundary cuts: the terminal state of batch N does not transition to batch N+1.
3. Continuous simulation: the fitted Markov chain represents an infinite process.
4. Constrained batch execution: guiding a rollout to begin at 'load', avoid 'alarm',
   and terminate cleanly at 'unload' via `ScheduleConstraints`.

Run: PYTHONPATH=src venv/bin/python examples/batch_process_episodes.py
"""

from __future__ import annotations

import polars as pl

from seqsm import ScheduleConstraints, SeqSM


def manufacturing_stream(n_batches: int = 5) -> pl.DataFrame:
    """Simulates multiple manufacturing batches with occasional alarm excursions."""
    rows: list[dict[str, object]] = []
    for b in range(1, n_batches + 1):
        # Standard nominal flow: load -> heat -> react -> cool -> unload
        stages = ["load", "heat", "react", "react", "cool", "unload"]
        for s in stages:
            rows.append({"batch_id": b, "stage": s, "quality": "nominal"})
    return pl.DataFrame(rows)


def main() -> None:
    df = manufacturing_stream(n_batches=6)

    # 1. Fit SeqSM with episode_col to enforce batch boundary cuts.
    model = SeqSM(
        k=1,
        smoothing=0.01,
        episode_col="batch_id",
        state_cols=["stage", "quality"],
    ).fit(df)

    enc = model.state_encoder
    unload_state = enc.dict_to_state({"stage": "unload", "quality": "nominal"})

    # Inspection: unload is strictly an episode-ending state in the training data.
    # It has no outgoing transitions learned across batches (encoded window == -1).
    assert model.markov_chain is not None
    k_enc = model.markov_chain.encoder
    u_unload = k_enc.encode_window((unload_state,))
    print(f"Window index for 'unload': {u_unload} (-1: strictly cut at batch boundaries)")
    assert u_unload == -1

    # 2. Infinite continuous simulation:
    # Markov chain transitions run indefinitely for any requested window length.
    continuous = model.generate_schedule(window=12, seed=42)
    print("Continuous 12-step factory simulation:")
    print(continuous.select("time_step", "stage", "quality").head(6))

    # 3. Constrained single-batch execution:
    # Force the rollout to start from 'load', complete in 4 steps, and terminate at 'unload'.
    constraints = ScheduleConstraints(
        initial_state={"stage": "load", "quality": "nominal"},
        terminal_state={"stage": "unload", "quality": "nominal"},
        forbidden_states=[{"stage": "alarm", "quality": "defect"}],
    )
    single_batch = model.generate_schedule(
        window=4,
        constraints=constraints,
        method="constrained",
        seed=10,
    )
    print("\nConstrained single batch rollout (ends at unload):")
    print(single_batch.select("time_step", "stage", "transition_prob"))
    assert single_batch["stage"][-1] == "unload"


if __name__ == "__main__":
    main()
