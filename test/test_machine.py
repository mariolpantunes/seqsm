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


if __name__ == "__main__":
    unittest.main()
