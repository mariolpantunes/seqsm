"""Unit tests for StateMachine graph reduction and exports."""

from __future__ import annotations

import unittest

import numpy as np
import polars as pl

from seqsm.encoding import StateEncoder
from seqsm.machine import StateMachine
from seqsm.markov import KOrderMarkovChain


class TestStateMachine(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pl.DataFrame(
            {
                "motion": [0, 1, 1, 0, 1, 0],
                "light": ["off", "on", "on", "off", "on", "off"],
            }
        )
        self.encoder = StateEncoder()
        self.encoder.fit(self.df)
        self.states = self.encoder.transform(self.df)

        self.chain = KOrderMarkovChain(k=1, mode="clock")
        self.chain.fit_from_arrays(self.states)
        self.fsm = StateMachine.from_markov(self.chain, self.encoder, min_prob=0.0, min_count=1)

    def test_edges_dataframe(self) -> None:
        edges = self.fsm.edges_dataframe(include_diffs=True)
        self.assertGreater(len(edges), 0)
        self.assertIn("source_id", edges.columns)
        self.assertIn("target_id", edges.columns)
        self.assertIn("probability", edges.columns)
        self.assertIn("diff_summary", edges.columns)

        edges_no_diff = self.fsm.edges_dataframe(include_diffs=False)
        self.assertNotIn("diff_summary", edges_no_diff.columns)

    def test_nodes_dataframe(self) -> None:
        nodes = self.fsm.nodes_dataframe()
        self.assertEqual(len(nodes), self.encoder.n_states)
        self.assertIn("state_id", nodes.columns)
        self.assertIn("stationary_prob", nodes.columns)

    def test_extract_cycles(self) -> None:
        # Repeating cycle 0 -> 1 -> 0 -> 1
        cycle_chain = KOrderMarkovChain(k=1)
        cycle_chain.fit_from_arrays(np.array([0, 1, 0, 1, 0, 1], dtype=np.int32))
        cycle_fsm = StateMachine.from_markov(cycle_chain, self.encoder, min_prob=0.0, min_count=1)

        cycles = cycle_fsm.extract_cycles(max_length=4)
        self.assertGreater(len(cycles), 0)
        c0 = cycles[0]
        self.assertEqual(set(c0.state_sequence), {0, 1})

    def test_to_mermaid_and_dot(self) -> None:
        mermaid_str = self.fsm.to_mermaid(diff_labels=True)
        self.assertTrue(mermaid_str.startswith("flowchart TD"))
        self.assertIn("-->", mermaid_str)

        dot_str = self.fsm.to_dot(diff_labels=True)
        self.assertTrue(dot_str.startswith("digraph StateMachine {"))
        self.assertIn("->", dot_str)

    def test_edge_aggregation_unique_pairs_k2_h1(self) -> None:
        # K=2 chain: windows (0, 1) and (2, 1) both end in state 1.
        # Transitions (0, 1) -> 2 and (2, 1) -> 2 both represent state-pair (1 -> 2).
        # In atomic mode, edges_dataframe must have UNIQUE (source_id, target_id) pairs!
        chain_k2 = KOrderMarkovChain(k=2, smoothing=0.0)
        seq = np.array([0, 1, 2, 0, 1, 2, 2, 1, 2], dtype=np.int32)
        chain_k2.fit_from_arrays(seq)

        fsm_k2 = StateMachine.from_markov(chain_k2, self.encoder, min_prob=0.0, min_count=1, mode="atomic")
        edges = fsm_k2.edges_dataframe()

        pairs = list(zip(edges["source_id"].to_list(), edges["target_id"].to_list(), strict=True))
        self.assertEqual(len(pairs), len(set(pairs)), "Edges DataFrame must not contain duplicate state pairs.")

    def test_top_k_edges_and_renormalize_h2(self) -> None:
        chain = KOrderMarkovChain(k=1, smoothing=0.0)
        # From state 0, transitions to 1 and 2
        seq = np.array([0, 1, 0, 2, 0, 1, 0, 2], dtype=np.int32)
        chain.fit_from_arrays(seq)

        # top_k_edges=1: keeps only top outbound edge per state
        fsm_top1 = StateMachine.from_markov(
            chain, self.encoder, min_prob=0.0, min_count=1, top_k_edges=1, renormalize=True
        )
        edges = fsm_top1.edges_dataframe()

        # Each source should have at most 1 edge
        sources = edges["source_id"].to_list()
        self.assertEqual(len(sources), len(set(sources)))
        # Outbound probabilities should be renormalized to 1.0
        for p in edges["probability"].to_list():
            self.assertAlmostEqual(p, 1.0)

    def test_extract_cycles_frequency_r13(self) -> None:
        chain = KOrderMarkovChain(k=1)
        # Cycle 0 -> 1 -> 0
        seq = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int32)
        chain.fit_from_arrays(seq)
        fsm = StateMachine.from_markov(chain, self.encoder, min_prob=0.0)
        cycles = fsm.extract_cycles(max_length=4)
        self.assertGreater(len(cycles), 0)
        # Cycles found across rotations accumulate frequency
        self.assertGreaterEqual(cycles[0].frequency, 1)

    def test_state_machine_repr_and_str(self) -> None:
        repr_str = repr(self.fsm)
        self.assertIn("StateMachine(", repr_str)
        self.assertIn("n_states=", repr_str)

        summary_str = str(self.fsm)
        self.assertIn("StateMachine", summary_str)
        self.assertIn("States (Marginal Stationary Distribution):", summary_str)
        self.assertIn("Top Transitions", summary_str)
        self.assertEqual(summary_str, self.fsm.summary())

    def test_to_plot_data_and_plot(self) -> None:
        plot_data = self.fsm.to_plot_data(layout="circular", include_diffs=True)
        self.assertGreater(len(plot_data.nodes), 0)
        self.assertGreater(len(plot_data.edges), 0)

        # Verify node properties
        n0 = plot_data.nodes[0]
        self.assertIsInstance(n0.x, float)
        self.assertIsInstance(n0.y, float)
        self.assertGreater(n0.size, 0.0)

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 6))
        returned_ax = self.fsm.plot(ax=ax, show_edge_labels=True, show_diffs=True)
        self.assertIs(returned_ax, ax)
        plt.close(fig)

    def test_compare_similarity_identical_and_different(self) -> None:
        sim_self = self.fsm.compare_similarity(self.fsm)
        self.assertAlmostEqual(sim_self["stationary_cosine"], 1.0)
        self.assertAlmostEqual(sim_self["stationary_jsd"], 0.0)
        self.assertAlmostEqual(sim_self["edge_jaccard"], 1.0)
        self.assertAlmostEqual(sim_self["transition_mae"], 0.0)
        self.assertAlmostEqual(sim_self["overall_similarity"], 1.0)

        # Different FSM
        other_chain = KOrderMarkovChain(k=1)
        other_chain.fit_from_arrays(np.array([1, 1, 1, 0], dtype=np.int32))
        other_fsm = StateMachine.from_markov(other_chain, self.encoder, min_prob=0.0)
        sim_diff = self.fsm.compare_similarity(other_fsm)
        self.assertLess(sim_diff["overall_similarity"], 1.0)
        self.assertGreaterEqual(sim_diff["overall_similarity"], 0.0)


if __name__ == "__main__":
    unittest.main()
