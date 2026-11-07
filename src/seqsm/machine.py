"""Pruned Finite State Machine representation with cycle mining and Mermaid/DOT exports."""

from __future__ import annotations

from typing import Any, Literal, Self

import numpy as np
import polars as pl

from seqsm.encoding import StateEncoder
from seqsm.linalg import jensen_shannon_divergence
from seqsm.markov import KOrderMarkovChain
from seqsm.types import PlotEdge, PlotNode, RoutineCycle, StatePlotData


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

    def summary(self, top_n: int = 10) -> str:
        """Returns a formatted human-readable summary of states, top transitions, and routine cycles."""
        nodes = self.nodes_dataframe()
        edges = self.edges_dataframe(include_diffs=True)
        n_states = len(nodes)
        n_edges = len(edges)

        lines: list[str] = [
            f"StateMachine (mode={self._mode}, {n_states} states, {n_edges} transitions, min_prob={self._min_prob:.2f}):",
            "States (Marginal Stationary Distribution):",
        ]

        nodes_sorted = nodes.sort("stationary_prob", descending=True)
        vocab_cols = [c for c in nodes.columns if c not in ("state_id", "stationary_prob")]

        for row in nodes_sorted.iter_rows(named=True):
            sid = row["state_id"]
            p = row["stationary_prob"]
            feat_strs = [f"{col}={row[col]}" for col in vocab_cols]
            desc = ", ".join(feat_strs)
            lines.append(f"  S{sid} (p={p:.3f}): {desc}")

        lines.append(f"Top Transitions (ranked by probability, up to {top_n}):")
        if n_edges == 0:
            lines.append("  (no transitions meeting min_prob/min_count thresholds)")
        else:
            edges_sorted = edges.sort("probability", descending=True).head(top_n)
            for row in edges_sorted.iter_rows(named=True):
                src = row["source_id"]
                tgt = row["target_id"]
                prob = row["probability"]
                count = row["count"]
                diff = row.get("diff_summary", "")
                diff_str = f" [{diff}]" if diff else ""
                lines.append(f"  S{src} -> S{tgt} (p={prob:.3f}, count={count}){diff_str}")

        cycles = self.extract_cycles(max_length=6, top_n=5)
        if cycles:
            lines.append("Prominent Routine Cycles:")
            for c in cycles:
                seq_str = " -> ".join(f"S{s}" for s in c.state_sequence) + f" -> S{c.state_sequence[0]}"
                lines.append(f"  Cycle ({seq_str}): persistence={c.persistence:.4f}, period={c.period}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        nodes = self.nodes_dataframe()
        edges = self.edges_dataframe(include_diffs=False)
        return (
            f"StateMachine(mode={self._mode!r}, n_states={len(nodes)}, n_edges={len(edges)}, min_prob={self._min_prob})"
        )

    def __str__(self) -> str:
        return self.summary()

    def to_plot_data(
        self,
        layout: Literal["circular", "random"] = "circular",
        node_color: str = "#0284C7",
        min_prob: float | None = None,
        include_diffs: bool = True,
    ) -> StatePlotData:
        """Constructs a decoupled StatePlotData structure containing nodes and edges for Matplotlib rendering."""
        effective_min_prob = self._min_prob if min_prob is None else min_prob
        nodes_df = self.nodes_dataframe()
        edges_df = self.edges_dataframe(include_diffs=include_diffs)

        n_nodes = len(nodes_df)
        if n_nodes == 0:
            return StatePlotData(nodes=(), edges=(), title="Empty State Machine")

        positions: dict[int, tuple[float, float]] = {}
        if layout == "circular":
            for i, row in enumerate(nodes_df.iter_rows(named=True)):
                sid = int(row["state_id"])
                angle = (np.pi / 2.0) - (2.0 * np.pi * i / max(1, n_nodes))
                positions[sid] = (float(np.cos(angle)), float(np.sin(angle)))
        else:
            rng = np.random.default_rng(42)
            for row in nodes_df.iter_rows(named=True):
                sid = int(row["state_id"])
                positions[sid] = (float(rng.uniform(-0.9, 0.9)), float(rng.uniform(-0.9, 0.9)))

        plot_nodes: list[PlotNode] = []
        vocab_cols = [c for c in nodes_df.columns if c not in ("state_id", "stationary_prob")]

        for row in nodes_df.iter_rows(named=True):
            sid = int(row["state_id"])
            p_val = float(row["stationary_prob"])
            x, y = positions[sid]
            size = float(0.10 + 0.16 * np.sqrt(max(0.0, p_val)))
            feats = {col: row[col] for col in vocab_cols}
            plot_nodes.append(
                PlotNode(
                    id=sid,
                    label=f"S{sid}",
                    x=x,
                    y=y,
                    size=size,
                    color=node_color,
                    stationary_prob=p_val,
                    features=feats,
                )
            )

        plot_edges: list[PlotEdge] = []
        for row in edges_df.iter_rows(named=True):
            prob = float(row["probability"])
            if prob < effective_min_prob:
                continue
            src = int(row["source_id"])
            tgt = int(row["target_id"])
            if src not in positions or tgt not in positions:
                continue

            count = int(row["count"])
            diff = row.get("diff_summary")
            lbl = f"{prob:.2f}"

            plot_edges.append(
                PlotEdge(
                    source_id=src,
                    target_id=tgt,
                    source_pos=positions[src],
                    target_pos=positions[tgt],
                    probability=prob,
                    count=count,
                    label=lbl,
                    diff_summary=str(diff) if diff else None,
                )
            )

        return StatePlotData(
            nodes=tuple(plot_nodes),
            edges=tuple(plot_edges),
            title=f"State Machine ({n_nodes} states, {len(plot_edges)} transitions)",
        )

    def plot(
        self,
        ax: Any = None,
        layout: Literal["circular", "random"] = "circular",
        min_prob: float | None = None,
        figsize: tuple[float, float] = (10, 8),
        node_color: str = "#0284C7",
        edge_color: str = "#475569",
        show_edge_labels: bool = True,
        show_diffs: bool = False,
        title: str | None = None,
    ) -> Any:
        """Renders the state machine graph onto a Matplotlib Axes."""
        plot_data = self.to_plot_data(layout=layout, node_color=node_color, min_prob=min_prob, include_diffs=show_diffs)
        return plot_data.plot(
            ax=ax,
            figsize=figsize,
            node_color=node_color,
            edge_color=edge_color,
            show_edge_labels=show_edge_labels,
            show_diffs=show_diffs,
            title=title,
        )

    def compare_similarity(self, other: StateMachine) -> dict[str, float]:
        """Calculates multi-aspect structural and probabilistic similarity metrics against another StateMachine."""
        enc_self = self._encoder
        enc_other = other._encoder

        shared_features = (
            enc_self.feature_cols is not None
            and enc_other.feature_cols is not None
            and set(enc_self.feature_cols) == set(enc_other.feature_cols)
        )

        mu_self = self._chain.marginal_stationary_distribution()
        mu_other = other._chain.marginal_stationary_distribution()
        edges_self_df = self.edges_dataframe(include_diffs=False)
        edges_other_df = other.edges_dataframe(include_diffs=False)

        if shared_features:
            common_cols = sorted(enc_self.feature_cols)

            def get_tup_self(sid: int) -> tuple[Any, ...]:
                if sid < 0 or sid >= enc_self.n_states:
                    return (f"unknown_{sid}",)
                try:
                    d = enc_self.state_to_dict(sid)
                    return tuple(d[c] for c in common_cols)
                except KeyError:
                    return (f"unknown_{sid}",)

            def get_tup_other(sid: int) -> tuple[Any, ...]:
                if sid < 0 or sid >= enc_other.n_states:
                    return (f"unknown_{sid}",)
                try:
                    d = enc_other.state_to_dict(sid)
                    return tuple(d[c] for c in common_cols)
                except KeyError:
                    return (f"unknown_{sid}",)

            tups_self = [get_tup_self(i) for i in range(enc_self.n_states)]
            tups_other = [get_tup_other(i) for i in range(enc_other.n_states)]

            canonical_tuples = sorted(set(tups_self) | set(tups_other))
            tup_to_canonical = {t: idx for idx, t in enumerate(canonical_tuples)}
            num_canonical = len(canonical_tuples)

            p_self = np.zeros(num_canonical, dtype=np.float64)
            for i, p_val in enumerate(mu_self):
                if i < len(tups_self):
                    p_self[tup_to_canonical[tups_self[i]]] = p_val

            p_other = np.zeros(num_canonical, dtype=np.float64)
            for i, p_val in enumerate(mu_other):
                if i < len(tups_other):
                    p_other[tup_to_canonical[tups_other[i]]] = p_val

            edges_self = {
                (
                    tup_to_canonical[get_tup_self(int(r["source_id"]))],
                    tup_to_canonical[get_tup_self(int(r["target_id"]))],
                )
                for r in edges_self_df.iter_rows(named=True)
            }
            edges_other = {
                (
                    tup_to_canonical[get_tup_other(int(r["source_id"]))],
                    tup_to_canonical[get_tup_other(int(r["target_id"]))],
                )
                for r in edges_other_df.iter_rows(named=True)
            }

            prob_self = {
                (
                    tup_to_canonical[get_tup_self(int(r["source_id"]))],
                    tup_to_canonical[get_tup_self(int(r["target_id"]))],
                ): float(r["probability"])
                for r in edges_self_df.iter_rows(named=True)
            }
            prob_other = {
                (
                    tup_to_canonical[get_tup_other(int(r["source_id"]))],
                    tup_to_canonical[get_tup_other(int(r["target_id"]))],
                ): float(r["probability"])
                for r in edges_other_df.iter_rows(named=True)
            }
        else:
            max_s = max(len(mu_self), len(mu_other))
            p_self = np.pad(mu_self, (0, max_s - len(mu_self)))
            p_other = np.pad(mu_other, (0, max_s - len(mu_other)))

            edges_self = {(int(r["source_id"]), int(r["target_id"])) for r in edges_self_df.iter_rows(named=True)}
            edges_other = {(int(r["source_id"]), int(r["target_id"])) for r in edges_other_df.iter_rows(named=True)}

            prob_self = {
                (int(r["source_id"]), int(r["target_id"])): float(r["probability"])
                for r in edges_self_df.iter_rows(named=True)
            }
            prob_other = {
                (int(r["source_id"]), int(r["target_id"])): float(r["probability"])
                for r in edges_other_df.iter_rows(named=True)
            }

        jsd = jensen_shannon_divergence(p_self, p_other)

        norm_self = float(np.linalg.norm(p_self))
        norm_other = float(np.linalg.norm(p_other))
        denom = norm_self * norm_other
        cosine_sim = float(np.dot(p_self, p_other) / denom) if denom > 1e-12 else 0.0

        union = edges_self | edges_other
        intersect = edges_self & edges_other
        edge_jaccard = float(len(intersect) / max(1, len(union)))

        if intersect:
            diffs = [abs(prob_self[e] - prob_other[e]) for e in intersect]
            mae = float(np.mean(diffs))
        else:
            mae = 1.0

        overall_similarity = float(max(0.0, min(1.0, 0.5 * cosine_sim + 0.5 * edge_jaccard)))

        return {
            "stationary_jsd": float(jsd),
            "stationary_cosine": float(cosine_sim),
            "edge_jaccard": float(edge_jaccard),
            "transition_mae": float(mae),
            "overall_similarity": float(overall_similarity),
        }


__all__ = ["StateMachine"]
