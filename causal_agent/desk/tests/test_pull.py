"""One pull over the shelves: the material normalises every address on the way in, says which shelf an item is on, and answers
`pull` by address and `shelf` by kind, so a cite in another spelling of a column still resolves."""

from __future__ import annotations

from causal_agent.common.contracts import RunRecord
from causal_agent.desk import explainer as X
from causal_agent.desk import material as M
from causal_agent.desk.contracts import AfterReply, NumberStated
from causal_agent.desk.handoff import forced
from causal_agent.desk.pull import shelf_of
from causal_agent.memory import store


def test_every_address_is_on_one_shelf():
    assert shelf_of("col:Math Score.when") == "understanding" and shelf_of("claim:assignment.kind") == "understanding"
    assert shelf_of("artifact:03e7a6a4.mean") == "artifact" and shelf_of("step:3") == "journal" and shelf_of("user:turn:2") == "words"
    assert shelf_of("matrix:adjustment.assignment") == "fit" and shelf_of("probe:data.timing") == "fit" and shelf_of("family:adjustment") == "family"
    assert (
        shelf_of("estimate:completed_vs_none.value") == "experiment"
        and shelf_of("figure:effect") == "experiment"
        and shelf_of("decision.family") == "experiment"
    )


def test_the_material_pulls_by_address_in_any_spelling_and_lists_a_shelf(tmp_path):
    m = store.migrate("students3", write=False)
    h = forced("students3", "q", "adjustment", "math score", "test preparation course", ["lunch"], memory=m)
    h.column("math score").facts.bounds = ["0", "100"]
    (tmp_path / "handoff.json").write_text(h.model_dump_json(indent=2))
    run = RunRecord(index=1, dataset="students3", question="q", family="adjustment", specialist="dowhy", status="done", design_dir=str(tmp_path))
    mat = M.render(run, m)
    low = mat.pull("[col:Math Score.profile.bounds.low]")  # brackets and the column's own spelling, as a model might write it
    assert low is not None and low.address == "col:math_score.profile.bounds.low" and low.value == 0.0 and low.shelf == "understanding"
    assert low.parent == "col:math_score.profile.bounds" and mat.pull("col:math_score.profile.bounds") is not None  # a parent is pullable
    assert mat.pull("estimate:nothing") is None
    assert [i.address for i in mat.shelf("experiment")][:2] == ["run.question", "decision.family"]
    assert all(i.shelf == "understanding" for i in mat.shelf("understanding")) and mat.shelf("artifact") == []


def test_the_cite_gate_resolves_a_cite_in_another_spelling_and_keeps_it_normalised():
    mat = M.Material()
    mat.add("col:Math Score.profile.bounds.low", "0", 0.0)
    mat.close()
    reply = AfterReply(
        kind="answer",
        text="The floor is 0.",
        cites=["col:math score.profile.bounds.low"],
        numbers=[NumberStated(address="[col:Math Score.profile.bounds.low]", value=0.0)],
    )
    assert (
        X._gate(reply, mat, "after") == []
        and reply.cites == ["col:math_score.profile.bounds.low"]
        and reply.numbers[0].address == "col:math_score.profile.bounds.low"
    )
    assert "cites not in the material" in X._gate(AfterReply(kind="answer", text="x", cites=["col:nothing.here"]), mat, "after")[0]
