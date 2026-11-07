"""Core types and dataclasses for seqsm."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import DTypeLike, NDArray

# Model operation modes
ModeType = Literal["clock", "event"]
PredictionMethod = Literal["sample", "map", "viterbi"]
ScheduleMethod = Literal["sample", "map", "constrained"]
BackoffMethod = Literal["witten_bell", "linear", "tuned"]


@dataclass(frozen=True)
class CSRMatrix:
    """Compressed Sparse Row (CSR) matrix representation.

    Note:
        Row pointers (`indptr`) and column indices (`indices`) are explicitly stored as
        32-bit signed integers (`np.int32`) to optimize cache locality and ensure direct
        compatibility with Numba JIT kernels. Consequently, the maximum number of non-zero
        elements (nnz) and maximum dimension size are bounded by 2^31 - 1 (~2.14 billion entries).
        For models exceeding this threshold, 64-bit sparse structures would be required.
    """

    data: NDArray[np.float64]
    indices: NDArray[np.int32]
    indptr: NDArray[np.int32]
    shape: tuple[int, int]

    def to_dense(self) -> NDArray[np.float64]:
        out = np.zeros(self.shape, dtype=np.float64)
        for i in range(self.shape[0]):
            start, end = int(self.indptr[i]), int(self.indptr[i + 1])
            out[i, self.indices[start:end]] = self.data[start:end]
        return out

    def dot_vector_left(self, vec: NDArray[np.float64]) -> NDArray[np.float64]:
        """Computes vector-matrix product: vec @ CSR of shape (shape[1],)."""
        if len(self.data) == 0:
            return np.zeros(self.shape[1], dtype=np.float64)
        row_counts = np.diff(self.indptr)
        row_indices = np.repeat(np.arange(self.shape[0], dtype=np.int32), row_counts)
        weights = vec[row_indices] * self.data
        return np.bincount(self.indices, weights=weights, minlength=self.shape[1]).astype(np.float64)


@dataclass(frozen=True)
class DwellDistribution:
    """Empirical dwell-time distribution for a given state."""

    state_id: int
    count: int
    mean: float
    std: float
    median: float
    min_dwell: int
    max_dwell: int
    samples: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ScheduleConstraints:
    """Constraints for guided schedule generation."""

    initial_state: int | dict[str, Any] | None = None
    terminal_state: int | dict[str, Any] | None = None
    mandatory_states: Sequence[tuple[int | dict[str, Any], int | tuple[int, int]]] = field(default_factory=tuple)
    forbidden_states: Sequence[int | dict[str, Any]] = field(default_factory=tuple)
    max_total_dwell: dict[int, int] | None = None


@dataclass(frozen=True)
class ScheduleTrajectory:
    """Raw trajectory output from schedule rollouts."""

    state_ids: tuple[int, ...]
    dwell_durations: tuple[int, ...]
    transition_probs: tuple[float, ...]
    arc_valid: tuple[bool, ...]
    constraint_violated: tuple[bool, ...] = field(default_factory=tuple)
    unmet_mandatory: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RoutineCycle:
    """Extracted behavioral routine cycle."""

    state_sequence: tuple[int, ...]
    frequency: int
    period: float
    persistence: float


@dataclass(frozen=True)
class PlotNode:
    """Node attributes for graph visualization."""

    id: int
    label: str
    x: float
    y: float
    size: float
    color: str
    stationary_prob: float
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlotEdge:
    """Directed edge attributes for graph visualization."""

    source_id: int
    target_id: int
    source_pos: tuple[float, float]
    target_pos: tuple[float, float]
    probability: float
    count: int
    label: str
    diff_summary: str | None = None


@dataclass(frozen=True)
class StatePlotData:
    """Graph structure representing state machine nodes and directed edges for Matplotlib rendering."""

    nodes: tuple[PlotNode, ...]
    edges: tuple[PlotEdge, ...]
    title: str = "State Machine"

    def plot(
        self,
        ax: Any = None,
        figsize: tuple[float, float] = (10, 8),
        node_color: str = "#0284C7",
        edge_color: str = "#475569",
        show_edge_labels: bool = True,
        show_diffs: bool = False,
        title: str | None = None,
    ) -> Any:
        """Renders the state machine graph onto a Matplotlib Axes."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle, FancyArrowPatch
        except ImportError as err:
            raise ImportError(
                "matplotlib is required for plotting state machines. Install with: pip install matplotlib"
            ) from err

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        ax.set_aspect("equal")
        ax.axis("off")

        node_dict = {n.id: n for n in self.nodes}

        # Draw directed edges
        for edge in self.edges:
            src = node_dict.get(edge.source_id)
            tgt = node_dict.get(edge.target_id)
            if src is None or tgt is None:
                continue

            x1, y1 = edge.source_pos
            x2, y2 = edge.target_pos
            prob = edge.probability
            lw = max(1.0, float(prob) * 4.5)
            alpha = max(0.35, min(0.95, float(prob) + 0.2))

            if edge.source_id == edge.target_id:
                # Self-loop arc
                loop_r = src.size * 1.4
                norm = float(np.hypot(x1, y1))
                dx = (x1 / norm) if norm > 1e-6 else 0.0
                dy = (y1 / norm) if norm > 1e-6 else 1.0
                cx = x1 + dx * loop_r
                cy = y1 + dy * loop_r
                loop_circle = Circle(
                    (cx, cy),
                    loop_r * 0.7,
                    fill=False,
                    edgecolor=edge_color,
                    linewidth=lw,
                    alpha=alpha,
                    linestyle="--",
                )
                ax.add_patch(loop_circle)
                if show_edge_labels:
                    ax.text(
                        cx,
                        cy,
                        f"{prob:.2f}",
                        fontsize=8,
                        color="#0F172A",
                        ha="center",
                        va="center",
                        bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85},
                    )
            else:
                r_src = src.size
                r_tgt = tgt.size
                dx = x2 - x1
                dy = y2 - y1
                dist = float(np.hypot(dx, dy))
                ux = (dx / dist) if dist > 1e-6 else 0.0
                uy = (dy / dist) if dist > 1e-6 else 1.0
                if dist > (r_src + r_tgt):
                    sx = x1 + ux * r_src
                    sy = y1 + uy * r_src
                    tx = x2 - ux * r_tgt
                    ty = y2 - uy * r_tgt
                else:
                    sx, sy, tx, ty = x1, y1, x2, y2

                arrow = FancyArrowPatch(
                    (sx, sy),
                    (tx, ty),
                    connectionstyle="arc3,rad=0.15",
                    arrowstyle="-|>",
                    mutation_scale=14,
                    linewidth=lw,
                    color=edge_color,
                    alpha=alpha,
                )
                ax.add_patch(arrow)

                if show_edge_labels:
                    mx = (sx + tx) / 2.0
                    my = (sy + ty) / 2.0
                    perp_x = -uy * 0.08
                    perp_y = ux * 0.08
                    lbl = f"{prob:.2f}"
                    if show_diffs and edge.diff_summary:
                        lbl += f"\n{edge.diff_summary}"
                    ax.text(
                        mx + perp_x,
                        my + perp_y,
                        lbl,
                        fontsize=8,
                        fontweight="bold",
                        color="#1E293B",
                        ha="center",
                        va="center",
                        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "#CBD5E1", "alpha": 0.85},
                    )

        # Draw nodes
        for node in self.nodes:
            circle = Circle(
                (node.x, node.y),
                node.size,
                facecolor=node.color or node_color,
                edgecolor="#0F172A",
                linewidth=1.75,
                zorder=4,
            )
            ax.add_patch(circle)

            ax.text(
                node.x,
                node.y + 0.02,
                f"S{node.id}",
                fontsize=11,
                fontweight="bold",
                color="white",
                ha="center",
                va="center",
                zorder=5,
            )
            ax.text(
                node.x,
                node.y - 0.04,
                f"p={node.stationary_prob:.2f}",
                fontsize=8,
                color="#F8FAFC",
                ha="center",
                va="center",
                zorder=5,
            )

        margin = 0.35
        all_x = [n.x for n in self.nodes] or [0.0]
        all_y = [n.y for n in self.nodes] or [0.0]
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

        ax.set_title(title or self.title, fontsize=13, fontweight="bold", pad=12)
        return ax


__all__ = [
    "BackoffMethod",
    "CSRMatrix",
    "DTypeLike",
    "DwellDistribution",
    "ModeType",
    "NDArray",
    "PlotEdge",
    "PlotNode",
    "PredictionMethod",
    "RoutineCycle",
    "ScheduleConstraints",
    "ScheduleMethod",
    "ScheduleTrajectory",
    "StatePlotData",
]
