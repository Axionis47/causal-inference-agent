"""The labelled turns: their shape and their kinds."""

from __future__ import annotations

from causal_agent.desk import turns as TD
from causal_agent.evals import turns


def test_the_held_out_turns_are_labelled_with_the_desk_kinds():
    held = turns.rows_of(turns.SET)
    assert len(held) >= 40
    for r in held:
        assert r["kind"] in TD.KINDS and r["message"].strip() and "at_ready" in r
    assert {r["kind"] for r in held} == set(TD.KINDS)
