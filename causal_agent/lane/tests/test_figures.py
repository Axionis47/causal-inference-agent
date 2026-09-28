"""The figures a lane leaves behind go through the check; a figure on an address the run did not produce is a decline."""

from __future__ import annotations

import json

from causal_agent.common.contracts import CheckResult, Estimate, Refutation
from causal_agent.desk.handoff import forced
from causal_agent.lane import figures as F
from causal_agent.memory import store
from causal_agent.viz.spec import Edge, FigureSpec, Node, Series


def test_ok_addresses_and_write(tmp_path):
    h = forced("students3", "q", "adjustment", "math score", "test preparation course", ["lunch"], memory=store.migrate("students3", write=False))
    state = {
        "checks": [CheckResult(contrast="c", name="overlap", level="pass")],
        "estimates": [Estimate(contrast="c", method="linear_regression", value=1.0), Estimate(contrast="c", method="psw", value=1.1, secondary=True)],
        "refutations": [Refutation(contrast="c", refuter="placebo_treatment_refuter", kind="falsification", new_effect=0.0, passed=True)],
        "check_facts": {"balance": {}},
        "declines": [],
    }
    ok = F.ok_addresses(h, state, "refute")
    assert {
        "col:lunch.when",
        "check:c.overlap",
        "estimate:c.value",
        "estimate:c.psw.value",
        "refute:c.placebo_treatment_refuter.new_effect",
        "check_facts.balance",
        "design.graph",
    } <= ok
    good = FigureSpec(id="g", kind="bars", title="t", series=[Series(name="s", x=["a"], y=[1.0])], draws_on=["estimate:c.value", "col:lunch.when"])
    bad = FigureSpec(id="b", kind="bars", title="t", series=[Series(name="s", x=["a"], y=[1.0])], draws_on=["estimate:c.made_up"])
    kept, declines = F.write(tmp_path, [good, None, bad], ok)
    assert [s.id for s in kept] == ["g"] and len(declines) == 1
    assert declines[0].about == "figure:b" and declines[0].check == "figure.check" and "estimate:c.made_up" in declines[0].reason
    assert [f["id"] for f in json.loads((tmp_path / "figures.json").read_text())] == ["g"]


def _graph() -> FigureSpec:
    return FigureSpec(
        id="causal_graph",
        kind="graph",
        title="the graph",
        draws_on=["design.graph", "col:lunch.when"],
        nodes=[Node(id="course", role="treatment"), Node(id="math", role="outcome"), Node(id="lunch", role="confounder")],
        edges=[Edge(src="course", dst="math"), Edge(src="lunch", dst="course", cites=["col:lunch.when"]), Edge(src="lunch", dst="math")],
    )


def test_check_spec_by_kind():
    ok = {"design.graph", "col:lunch.when", "estimate:c.value"}
    assert F.check_spec(_graph(), ok) == []
    g = _graph()
    g.edges.append(Edge(src="nope", dst="math"))
    assert any("nope -> math" in p for p in F.check_spec(g, ok))
    g = _graph()
    g.nodes = []
    assert any("no nodes" in p for p in F.check_spec(g, ok))
    g = _graph()
    g.draws_on.append("estimate:c.ci")
    assert any("estimate:c.ci" in p for p in F.check_spec(g, ok))
    bars = FigureSpec(id="b", kind="bars", title="t", series=[Series(name="s", x=["a"], y=[1.0])], draws_on=["estimate:c.value"])
    assert F.check_spec(bars, ok) == []
    assert F.check_spec(FigureSpec(id="b", kind="bars", title="t", draws_on=[]), ok) == ["no series to draw"]
    assert F.check_spec(FigureSpec(id="b", kind="bars", title="t", series=[Series(name="s", x=["a"], y=[None])]), ok) == ["every value is empty"]
    # a col: address resolves however the column is spelled
    assert F.check_spec(FigureSpec(id="b", kind="bars", title="t", series=[Series(name="s", x=["a"], y=[1.0])], draws_on=["col:Lunch.when"]), ok) == []
