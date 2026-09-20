"""The figure contract: the graph kind, the moment, the addresses, and the check by code."""

from __future__ import annotations

from causal_agent.viz.graph import check_spec
from causal_agent.viz.spec import Edge, FigureSpec, Node, Series


def graph() -> FigureSpec:
    return FigureSpec(id="causal_graph", kind="graph", title="the graph", draws_on=["design.graph", "col:lunch.when"],
                      nodes=[Node(id="course", label="test preparation course", role="treatment"), Node(id="math", label="math score", role="outcome"), Node(id="lunch", role="confounder")],
                      edges=[Edge(src="course", dst="math"), Edge(src="lunch", dst="course", cites=["col:lunch.when"]), Edge(src="lunch", dst="math")])


def test_graph_addresses_and_render():
    g = graph()
    assert g.moment == "run" and {"figure:causal_graph", "figure:causal_graph.node.0", "figure:causal_graph.edge.2"} <= g.addresses()
    text = g.render()
    assert "[figure:causal_graph.node.0] test preparation course (treatment)" in text and "[figure:causal_graph.edge.1] lunch -> course [col:lunch.when]" in text
    g.moment = "ready"
    assert "(before the run)" in g.render()


def test_check_spec_by_kind():
    ok = {"design.graph", "col:lunch.when", "estimate:c.value"}
    assert check_spec(graph(), ok) == []
    g = graph()
    g.edges.append(Edge(src="nope", dst="math"))
    assert any("nope -> math" in p for p in check_spec(g, ok))
    g = graph()
    g.nodes = []
    assert any("no nodes" in p for p in check_spec(g, ok))
    g = graph()
    g.draws_on.append("estimate:c.ci")
    assert any("estimate:c.ci" in p for p in check_spec(g, ok))
    bars = FigureSpec(id="b", kind="bars", title="t", series=[Series(name="s", x=["a"], y=[1.0])], draws_on=["estimate:c.value"])
    assert check_spec(bars, ok) == []
    assert check_spec(FigureSpec(id="b", kind="bars", title="t", draws_on=[]), ok) == ["no series to draw"]
    assert check_spec(FigureSpec(id="b", kind="bars", title="t", series=[Series(name="s", x=["a"], y=[None])]), ok) == ["every value is empty"]
    # a col: address resolves however the column is spelled
    assert check_spec(FigureSpec(id="b", kind="bars", title="t", series=[Series(name="s", x=["a"], y=[1.0])], draws_on=["col:Lunch.when"]), ok) == []
