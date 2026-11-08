"""Episode boundaries, infinite Markov processes, and begin/end schedule constraints.

Demonstrates:
1. Episode boundaries are transition cuts: transitions never cross between episodes.
2. The fitted chain is an infinite/stationary process (no absorbing sinks by default).
3. Enforcing begin/end states using `ScheduleConstraints` (initial_state, terminal_state).
4. Intermediate checkpoints using `mandatory_states`.

Run: PYTHONPATH=src venv/bin/python examples/episode_constraints.py
"""

from __future__ import annotations

import polars as pl

from seqsm import ScheduleConstraints, SeqSM


def session_data() -> pl.DataFrame:
    """Multi-session activity sequence with distinct sessions."""
    return pl.DataFrame(
        {
            "session_id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3],
            "action": [
                "login",
                "work",
                "work",
                "logout",
                "login",
                "browse",
                "work",
                "logout",
                "login",
                "browse",
                "logout",
            ],
        }
    )


def main() -> None:
    df = session_data()

    # 1. Episode boundaries act as cuts during training.
    # Transitions never cross between session 1 and session 2.
    model = SeqSM(
        k=1,
        smoothing=0.0,
        episode_col="session_id",
        feature_cols=["action"],
    ).fit(df)

    enc = model.state_encoder
    logout_id = enc.dict_to_state({"action": "logout"})
    login_id = enc.dict_to_state({"action": "login"})

    # Inspection: transition probability from logout to login across episodes is strictly 0.0
    assert model.markov_chain is not None
    p_logout_login = model.markov_chain.transition_matrix[logout_id, login_id]
    print(f"P(logout -> login across episode breaks): {p_logout_login:.4f} (strictly cut)")

    # 2. Infinite rollout: by default, simulation continues for any arbitrary window H
    # starting from stationary distribution or regime without absorbing terminal states.
    unconstrained = model.generate_schedule(window=10, seed=42)
    print("Unconstrained 10-step rollout actions:", unconstrained["action"].to_list())

    # 3. Constrained rollout: enforce starting condition and terminating end state.
    # - initial_state: specifies the conditioning pre-start state (e.g. user just logged in).
    # - terminal_state: forces the final transition to land in the designated ending state.
    constraints = ScheduleConstraints(
        initial_state={"action": "login"},
        terminal_state={"action": "logout"},
    )
    bounded_session = model.generate_schedule(window=5, constraints=constraints, seed=42)
    print("Constrained session (starts from login, ends at logout):")
    print(bounded_session.select("time_step", "action", "transition_prob"))
    assert bounded_session["action"][-1] == "logout"

    # 4. Mandatory intermediate checkpoint: ensure 'browse' occurs between steps 1 and 2.
    checkpoint_constraints = ScheduleConstraints(
        initial_state={"action": "login"},
        terminal_state={"action": "logout"},
        mandatory_states=[({"action": "browse"}, (1, 2))],
    )
    guided = model.generate_schedule(
        window=4,
        constraints=checkpoint_constraints,
        method="constrained",
        seed=10,
    )
    print("Guided session with mandatory checkpoint:", guided["action"].to_list())
    assert "browse" in guided["action"].to_list()
    assert guided["action"][-1] == "logout"


if __name__ == "__main__":
    main()
