"""Pruned Finite State Machine representation with cycle mining and Mermaid/DOT exports."""

from __future__ import annotations

from typing import Any, Literal, Self

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
        mode: Literal["atomic", "debruijn"] = "atomic",
        top_k_edges: int | None = None,
        renormalize: bool = True,
    ) -> None:
        self._chain: KOrderMarkovChain = chain
        self._encoder: StateEncoder = encoder
        self._min_prob: float = min_prob
        self._min_count: int = min_count
        self._mode: Literal["atomic", "debruijn"] = mode
        self._top_k_edges: int | None = top_k_edges
        self._renormalize: bool = renormalize

    @classmethod
    def from_markov(
        cls,
        chain: KOrderMarkovChain,
        encoder: StateEncoder,
        *,
        min_prob: float = 0.05,
        min_count: int = 2,
        mode: Literal["atomic", "debruijn"] = "atomic",
        top_k_edges: int | None = None,
        renormalize: bool = True,
    ) -> Self:
        """Constructs a pruned StateMachine from a fitted KOrderMarkovChain."""
        return cls(
            chain,
            encoder,
            min_prob=min_prob,
            min_count=min_count,
            mode=mode,
            top_k_edges=top_k_edges,
            renormalize=renormalize,
        )

    def edges_dataframe(self, include_diffs: bool = True) -> pl.DataFrame:
        """Returns edges DataFrame with source, target, probability, count, and diff summary."""
        k_enc = self._chain.encoder
        probs = self._chain.transition_matrix
        counts = self._chain.count_matrix
        n_windows, n_targets = probs.shape

        if self._mode == "debruijn":
            source_nodes: list[int] = []
            target_nodes: list[int] = []
            edge_probs: list[float] = []
            edge_counts: list[int] = []
            diff_summaries: list[str] = []

            for u in range(n_windows):
                for v in range(n_targets):
                    p = float(probs[u, v])
                    c = int(counts[u, v])
                    if p >= self._min_prob and c >= self._min_count:
                        source_nodes.append(u)
                        target_nodes.append(v)
                        edge_probs.append(p)
                        edge_counts.append(c)
                        if include_diffs:
                            try:
                                src_state = k_enc.decode_window(u)[-1]
                                diff = self._encoder.state_diff(src_state, v)
                                summary = "; ".join(f"{k}: {v0}->{v1}" for k, (v0, v1) in diff.items())
                            except (KeyError, ValueError):
                                summary = ""
                            diff_summaries.append(summary)

            data_debruijn: dict[str, list[Any]] = {
                "source_id": source_nodes,
                "target_id": target_nodes,
                "probability": edge_probs,
                "count": edge_counts,
            }
            if include_diffs:
                data_debruijn["diff_summary"] = diff_summaries
            return pl.DataFrame(data_debruijn)

        # mode == "atomic": stationary-weighted state-pair aggregation
        pi = self._chain.stationary_distribution()
        weighted_p: dict[tuple[int, int], float] = {}
        total_counts: dict[tuple[int, int], int] = {}
        src_marginal: dict[int, float] = {}

        for u in range(n_windows):
            src_state = int(k_enc.decode_window(u)[-1])
            pi_u = float(pi[u])
            src_marginal[src_state] = src_marginal.get(src_state, 0.0) + pi_u
            for v in range(n_targets):
                p_uv = float(probs[u, v])
                c_uv = int(counts[u, v])
                if c_uv > 0 or p_uv > 0.0:
                    pair = (src_state, v)
                    weighted_p[pair] = weighted_p.get(pair, 0.0) + pi_u * p_uv
                    total_counts[pair] = total_counts.get(pair, 0) + c_uv

        out_by_src: dict[int, list[tuple[int, float, int]]] = {}
        for (src_state, v), num_p in weighted_p.items():
            mu_s = src_marginal.get(src_state, 1.0)
            p_agg = num_p / mu_s if mu_s > 0.0 else 0.0
            c_agg = total_counts.get((src_state, v), 0)
            if p_agg >= self._min_prob and c_agg >= self._min_count:
                out_by_src.setdefault(src_state, []).append((v, p_agg, c_agg))

        final_sources: list[int] = []
        final_targets: list[int] = []
        final_probs: list[float] = []
        final_counts: list[int] = []
        final_diffs: list[str] = []

        for src_state in sorted(out_by_src.keys()):
            edges = out_by_src[src_state]
            edges.sort(key=lambda item: item[1], reverse=True)
            if self._top_k_edges is not None and len(edges) > self._top_k_edges:
                edges = edges[: self._top_k_edges]

            if self._renormalize and edges:
                denom = sum(p for _, p, _ in edges)
                if denom > 0.0:
                    edges = [(v, p / denom, c) for v, p, c in edges]

            for v, p, c in edges:
                final_sources.append(src_state)
                final_targets.append(v)
                final_probs.append(p)
                final_counts.append(c)
                if include_diffs:
                    try:
                        diff = self._encoder.state_diff(src_state, v)
                        summary = "; ".join(f"{k}: {v0}->{v1}" for k, (v0, v1) in diff.items())
                    except (KeyError, ValueError):
                        summary = ""
                    final_diffs.append(summary)

        data_atomic: dict[str, list[Any]] = {
            "source_id": final_sources,
            "target_id": final_targets,
            "probability": final_probs,
            "count": final_counts,
        }
        if include_diffs:
            data_atomic["diff_summary"] = final_diffs
        return pl.DataFrame(data_atomic)

    def nodes_dataframe(self) -> pl.DataFrame:
        """Returns nodes DataFrame with state_id, marginal stationary probability, and features."""
        mu = self._chain.marginal_stationary_distribution()
        vocab = self._encoder.vocabulary
        prob_series = pl.Series("stationary_prob", mu)
        return vocab.with_columns(prob_series)

    def extract_cycles(self, max_length: int = 8, top_n: int = 10) -> list[RoutineCycle]:
        """Extracts prominent routine cycles in the state machine graph using DFS path search.

        Note:
            Cycle search uses a depth-first traversal bounded by `max_length`. In dense or
            highly connected transition graphs, path expansion exhibits worst-case exponential
            complexity O(|V| * Δ^L), where Δ is the maximum out-degree and L is `max_length`.
            For large state spaces or dense graphs, keep `max_length` <= 8 to avoid explosive
            search times.

        Args:
            max_length: Maximum path length (depth) for cycle discovery. Must be >= 2.
            top_n: Number of top cycles to return, ranked by persistence (probability product).

        Returns:
            List of `RoutineCycle` dataclass instances sorted by descending persistence.
        """
        if max_length < 2 or top_n <= 0:
            return []

        edges = self.edges_dataframe(include_diffs=False)
        if len(edges) == 0:
            return []

        adj: dict[int, list[tuple[int, float]]] = {}
        for row in edges.iter_rows(named=True):
            src = int(row["source_id"])
            tgt = int(row["target_id"])
            prob = float(row["probability"])
            adj.setdefault(src, []).append((tgt, prob))

        found_cycles: dict[tuple[int, ...], tuple[float, float, int]] = {}

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
                    old_len, old_prod, old_count = found_cycles.get(canonical, (float(len(path)), 0.0, 0))
                    found_cycles[canonical] = (old_len, max(old_prod, new_prod), old_count + 1)
                elif nxt not in path:
                    dfs(start, nxt, [*path, nxt], new_prod)

        for node in adj:
            dfs(node, node, [node], 1.0)

        # Sort by persistence / probability product
        sorted_cycles = sorted(found_cycles.items(), key=lambda item: item[1][1], reverse=True)[:top_n]
        return [
            RoutineCycle(
                state_sequence=cycle_seq,
                frequency=freq,
                period=length,
                persistence=pers,
            )
            for cycle_seq, (length, pers, freq) in sorted_cycles
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
