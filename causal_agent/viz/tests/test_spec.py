"""The figure contract: the graph kind, the addresses, the render."""

from __future__ import annotations

from causal_agent.viz.spec import Edge, FigureSpec, Node


def graph() -> FigureSpec:
    return FigureSpec(
        id="causal_graph",
        kind="graph",
        title="the graph",
        draws_on=["design.graph", "col:lunch.when"],
        nodes=[
            Node(id="course", label="test preparation course", role="treatment"),
            Node(id="math", label="math score", role="outcome"),
            Node(id="lunch", role="confounder"),
        ],
        edges=[Edge(src="course", dst="math"), Edge(src="lunch", dst="course", cites=["col:lunch.when"]), Edge(src="lunch", dst="math")],
    )


def test_graph_addresses_and_render():
    g = graph()
    assert {"figure:causal_graph", "figure:causal_graph.node.0", "figure:causal_graph.edge.2"} <= g.addresses()
    text = g.render()
    assert (
        "[figure:causal_graph.node.0] test preparation course (treatment)" in text and "[figure:causal_graph.edge.1] lunch -> course [col:lunch.when]" in text
    )
