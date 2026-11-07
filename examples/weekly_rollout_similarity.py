"""End-to-end example: Learn a State Machine, predict a full week, refit, and compare similarity."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import polars as pl

# Use non-interactive backend for headless execution and CI
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from seqsm import SeqSM


def main() -> None:
    dataset_path = Path("dataset/synthetic_routines.parquet")
    if not dataset_path.exists():
        from dataset.generate_synthetic import generate_synthetic_routines

        print("Generating synthetic routine dataset...")
        df_full = generate_synthetic_routines(num_days=14, seed=42)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        df_full.write_parquet(dataset_path)
    else:
        df_full = pl.read_parquet(dataset_path)

    # 1. Select training sequences (first 7 days: episodes 0..6)
    train_df = df_full.filter(pl.col("episode_id") < 7)
    print(f"Training dataset: {len(train_df)} steps across episodes {train_df['episode_id'].unique().to_list()}")

    feature_cols = ["room", "motion", "light", "tv_active", "presence"]

    # 2. Fit SeqSM on training sequences
    model_train = SeqSM(
        k=1,
        mode="event",
        feature_cols=feature_cols,
        episode_col="episode_id",
        min_prob=0.01,
        min_count=1,
        seed=42,
    )
    model_train.fit(train_df)
    print("\n--- Model 1 Fitted on Training Sequences ---")
    print(repr(model_train))

    # 3. Extract and print State Machine (using __str__ and __repr__)
    fsm_train = model_train.to_state_machine(min_prob=0.01)
    print("\n--- State Machine Summary (toString / __str__) ---")
    print(fsm_train)

    # 4. Generate Matplotlib visualization and save to assets/figures/
    figures_dir = Path("assets/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "state_machine.png"

    fig, ax = plt.subplots(figsize=(8, 6.5))
    fsm_train.plot(ax=ax, title="Learned Routine State Machine (Training Sequences)", show_edge_labels=True)
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved State Machine plot to {fig_path}")

    # 5. Predict / generate a full week schedule rollout (7 days = 672 steps at 15-min intervals)
    steps_one_week = 7 * 24 * 4  # 672 steps
    print(f"\nGenerating full-week schedule rollout ({steps_one_week} steps)...")
    synthetic_week = model_train.generate_schedule(
        window=steps_one_week,
        method="sample",
        seed=100,
    )
    print(f"Generated rollout shape: {synthetic_week.shape}")

    # 6. Fit another SeqSM model on the generated schedule
    print("\n--- Fitting Model 2 on Generated Weekly Schedule ---")
    model_gen = SeqSM(
        k=1,
        mode="event",
        feature_cols=feature_cols,
        min_prob=0.01,
        min_count=1,
        seed=42,
    )
    model_gen.fit(synthetic_week)
    print(repr(model_gen))

    # 7. Compare similarity between Model 1 and Model 2
    metrics = model_train.compare_similarity(model_gen)
    print("\n=== Model & State Machine Similarity Comparison ===")
    print(f"Overall Composite Similarity: {metrics['overall_similarity']:.4f} (scale [0, 1])")
    print(f"Stationary Cosine Similarity: {metrics['stationary_cosine']:.4f}")
    print(f"Directed Edge Jaccard Index:  {metrics['edge_jaccard']:.4f}")
    print(f"Stationary JSD (divergence):  {metrics['stationary_jsd']:.4f} bits")
    print(f"Transition Prob MAE on edges: {metrics['transition_mae']:.4f}")

    if metrics["overall_similarity"] >= 0.70:
        print("\nSUCCESS: Generated weekly trajectory faithfully matches learned state machine dynamics!")
    else:
        print("\nNOTE: Moderate similarity observed.")


if __name__ == "__main__":
    main()
