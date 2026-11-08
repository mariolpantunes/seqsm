"""Automatic signal roles: which columns are states, context, time, episode, or noise.

Builds a messy sensor table (identifier, constant, continuous and redundant columns), lets
`infer_roles` classify it, discretizes the continuous column and fits a model on the result.

Run: PYTHONPATH=src venv/bin/python examples/auto_signal_roles.py
"""

from __future__ import annotations

import numpy as np
import polars as pl

from seqsm import SeqSM, discretize_columns, infer_roles


def messy_table() -> pl.DataFrame:
    df = pl.read_parquet("dataset/synthetic_routines.parquet")
    rng = np.random.default_rng(0)
    return df.with_columns(
        pl.Series("row_id", np.arange(len(df))),
        pl.lit("v2").alias("firmware"),
        pl.Series("power_w", rng.gamma(2.0, 40.0, len(df))),
        pl.col("timestamp").dt.hour().alias("hour_of_day"),
        pl.Series("flicker", rng.integers(0, 2, len(df))),
    )


def main() -> None:
    df = messy_table()
    pl.Config.set_tbl_width_chars(160).set_fmt_str_lengths(60)

    roles = infer_roles(df)
    print(roles.table().select("column", "role", "predictability", "reason"))

    # Noise columns can be demoted by predictability, and the joint state space can be capped.
    strict = infer_roles(df, min_predictability=0.05, max_joint_states=40)
    print("strict states:", strict.state, "| context:", strict.context)

    # Continuous columns need binning first; the helper picks a Freedman-Diaconis bin count.
    binned = discretize_columns(df, ["power_w"], n_bins="auto")
    print("power_w bins:", binned["power_w"].n_unique())
    print("after binning:", infer_roles(binned).state)

    model = SeqSM(k=1, smoothing=0.1, auto_roles={"min_predictability": 0.05, "max_joint_states": 60}).fit(binned)
    print(repr(model))
    print("fitted on:", model.state_encoder.feature_cols)


if __name__ == "__main__":
    main()
