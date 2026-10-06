"""The labelled turns: their shape, the split, and the rendering the decision model trains on."""

from __future__ import annotations

from causal_agent.desk import turns as TD
from causal_agent.evals import turns


def test_the_sets_are_labelled_with_the_desk_kinds_and_share_no_sentence():
    held, train = turns.rows_of(turns.SET), turns.rows_of(turns.TRAIN_SET)
    assert len(held) >= 40 and len(train) >= 200
    for r in held + train:
        assert r["kind"] in TD.KINDS and r["message"].strip() and "at_ready" in r
    sentences = {r["message"].lower() for r in held if len(r["message"].split()) > 3}
    assert not sentences & {r["message"].lower() for r in train}
    assert {r["kind"] for r in held} == set(TD.KINDS) == {r["kind"] for r in train}


def test_a_row_trains_on_the_state_and_the_questions_the_desk_sends():
    row = {"kind": "run", "message": "yes", "asked": None, "at_ready": True, "last_reply": "Everything the analysis needs is settled.\nmore"}
    out = turns.to_laya(row)
    assert out["state"] == TD.state_of("yes", None, True, "Everything the analysis needs is settled.\nmore")
    assert out["questions"] == TD.spec()["questions"]
    assert out["expected"] == {"kind": "run", "about_data": False}
    assert turns.to_laya({**row, "kind": "update"})["expected"] == {"kind": "update", "about_data": True}
