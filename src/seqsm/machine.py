"""Pruned Finite State Machine representation with cycle mining and Mermaid/DOT exports."""

from __future__ import annotations

from typing import Self

import numpy as np
import polars as pl

from seqsm.encoding import StateEncoder
from seqsm.markov import KOrderMarkovChain
from seqsm.types import RoutineCycle


class StateMachine:
    """Pruned, human-interpretable Finite State Machine derived from a KOrderMarkovChain."""

    def __init__(
        self,
        chain: KOrderMarkovChain,
        encoder: StateEncoder,
        min_prob: float = 0.05,
        min_count: int = 2,
    ) -> None:
        self._chain: KOrderMarkovChain = chain
        self._encoder: StateEncoder = encoder
        self._min_prob: float = min_prob
        self._min_count: int = min_count

    @classmethod
    def from_markov(
        cls,
        chain: KOrderMarkovChain,
        encoder: StateEncoder,
        *,
        min_prob: float = 0.05,
        min_count: int = 2,
        top_k_edges: int | None = None,
        renormalize: bool = True,
    ) -> Self:
        """Constructs a pruned StateMachine from a fitted KOrderMarkovChain."""
        _ = (top_k_edges, renormalize)
        return cls(chain, encoder, min_prob=min_prob, min_count=min_count)

    def edges_dataframe(self, include_diffs: bool = True) -> pl.DataFrame:
        """Returns edges DataFrame with source, target, probability, count, and diff summary."""
        k_enc = self._chain.encoder
        probs = self._chain.transition_matrix
        counts = self._chain.count_matrix
        n_windows, n_targets = probs.shape

        source_nodes: list[int] = []
        target_nodes: list[int] = []
        edge_probs: list[float] = []
        edge_counts: list[int] = []
        diff_summaries: list[str] = []

        for u in range(n_windows):
            win = k_enc.decode_window(u)
            src_state = win[-1]  # Most recent state
            for v in range(n_targets):
                p = float(probs[u, v])
                c = int(counts[u, v])
                if p >= self._min_prob and c >= self._min_count:
                    source_nodes.append(src_state)
                    target_nodes.append(v)
                    edge_probs.append(p)
                    edge_counts.append(c)
                    if include_diffs:
                        diff = self._encoder.state_diff(src_state, v)
                        summary = "; ".join(f"{k}: {v0}->{v1}" for k, (v0, v1) in diff.items())
                        diff_summaries.append(summary)

        data: dict[str, list] = {
            "source_id": source_nodes,
            "target_id": target_nodes,
            "probability": edge_probs,
            "count": edge_counts,
        }
        if include_diffs:
            data["diff_summary"] = diff_summaries
        return pl.DataFrame(data)

    def nodes_dataframe(self) -> pl.DataFrame:
        """Returns nodes DataFrame with state_id, marginal stationary probability, and features."""
        mu = self._chain.marginal_stationary_distribution()
        vocab = self._encoder.vocabulary
        prob_series = pl.Series("stationary_prob", mu)
        return vocab.with_columns(prob_series)

    def extract_cycles(self, max_length: int = 8, top_n: int = 10) -> list[RoutineCycle]:
        """Extracts prominent routine cycles in the state machine graph using DFS path search."""
        edges = self.edges_dataframe(include_diffs=False)
        if len(edges) == 0:
            return []

        adj: dict[int, list[tuple[int, float]]] = {}
        for row in edges.iter_rows(named=True):
            src = int(row["source_id"])
            tgt = int(row["target_id"])
            prob = float(row["probability"])
            adj.setdefault(src, []).append((tgt, prob))

        found_cycles: dict[tuple[int, ...], tuple[float, float]] = {}

        def dfs(start: int, current: int, path: list[int], prob_prod: float) -> None:
            if len(path) > max_length:
                return
            for nxt, p in adj.get(current, []):
                new_prod = prob_prod * p
                if nxt == start and len(path) >= 2:
                    cycle_tuple = tuple(path)
                    # Canonical rotation for cycle equivalence
                    min_idx = int(np.argmin(cycle_tuple))
                    canonical = cycle_tuple[min_idx:] + cycle_tuple[:min_idx]
                    if canonical not in found_cycles or new_prod > found_cycles[canonical][1]:
                        found_cycles[canonical] = (float(len(path)), new_prod)
                elif nxt not in path:
                    dfs(start, nxt, [*path, nxt], new_prod)

        for node in adj:
            dfs(node, node, [node], 1.0)

        # Sort by persistence / probability product
        sorted_cycles = sorted(found_cycles.items(), key=lambda item: item[1][1], reverse=True)[:top_n]
        return [
            RoutineCycle(
                state_sequence=cycle_seq,
                frequency=1,
                period=length,
                persistence=pers,
            )
            for cycle_seq, (length, pers) in sorted_cycles
        ]

    def to_mermaid(self, diff_labels: bool = True, min_prob: float = 0.05) -> str:
        """Exports the state machine as a Mermaid flowchart (graph TD) diagram string."""
        edges = self.edges_dataframe(include_diffs=diff_labels)
        lines = ["flowchart TD"]

        # Define nodes
        nodes = self.nodes_dataframe()
        for row in nodes.iter_rows(named=True):
            sid = row["state_id"]
            p_val = row["stationary_prob"]
            label = f"S{sid} (p={p_val:.2f})"
            lines.append(f'    S{sid}["{label}"]')

        # Add edges
        for row in edges.iter_rows(named=True):
            src = row["source_id"]
            tgt = row["target_id"]
            prob = row["probability"]
            if prob >= min_prob:
                if diff_labels and "diff_summary" in row and row["diff_summary"]:
                    lbl = f"{prob:.2f} [{row['diff_summary']}]"
                else:
                    lbl = f"{prob:.2f}"
                # Clean quotes for mermaid
                clean_lbl = lbl.replace('"', "'")
                lines.append(f'    S{src} -->|"{clean_lbl}"| S{tgt}')

        return "\n".join(lines)

    def to_dot(self, diff_labels: bool = True, min_prob: float = 0.05) -> str:
        """Exports the state machine as a Graphviz DOT digraph string."""
        edges = self.edges_dataframe(include_diffs=diff_labels)
        lines = ["digraph StateMachine {", "    rankdir=LR;", "    node [shape=circle];"]

        for row in edges.iter_rows(named=True):
            src = row["source_id"]
            tgt = row["target_id"]
            prob = row["probability"]
            if prob >= min_prob:
                lbl = f"{prob:.2f}"
                if diff_labels and "diff_summary" in row and row["diff_summary"]:
                    lbl += f"\\n{row['diff_summary']}"
                lines.append(f'    "S{src}" -> "S{tgt}" [label="{lbl}"];')

        lines.append("}")
        return "\n".join(lines)


__all__ = ["StateMachine"]
