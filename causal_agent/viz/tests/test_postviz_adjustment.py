"""The adjustment lane's figures: the graph with roles and cited arrows, the balance before and after weighting."""

from __future__ import annotations

from causal_agent.viz.graph import check_spec
from causal_agent.viz.postviz import adjustment as A

GRAPH = {
    "treatment": "course",
    "outcome": "math",
    "nodes": ["course", "math", "lunch", "gender", "unobserved", "z", "m"],
    "edges": [
        {"src": "course", "dst": "math"},
        {"src": "lunch", "dst": "course", "cites": ["claim:assignment.depends_on"]},
        {"src": "lunch", "dst": "math", "cites": ["col:lunch.when"]},
        {"src": "gender", "dst": "math"},
        {"src": "unobserved", "dst": "course", "cites": ["claim:unobserved"]},
        {"src": "unobserved", "dst": "math", "cites": ["claim:unobserved"]},
        {"src": "z", "dst": "course", "cites": ["claim:exclusion"]},
        {"src": "course", "dst": "m"},
        {"src": "m", "dst": "math"},
    ],
    "excluded": [{"column": "reading", "why": "another measurement of the outcome"}],
}


def test_causal_graph_roles_edges_and_addresses():
    f = A.causal_graph(GRAPH, {"course": "test preparation course", "math": "math score"})
    roles = {n.id: n.role for n in f.nodes}
    assert roles == {
        "course": "treatment",
        "math": "outcome",
        "lunch": "confounder",
        "gender": "driver",
        "unobserved": "hidden",
        "z": "instrument",
        "m": "mediator",
        "reading": "excluded",
    }
    assert f.kind == "graph" and f.id == "causal_graph" and next(n for n in f.nodes if n.id == "course").label == "test preparation course"
    assert len(f.edges) == 9 and f.edges[1].cites == ["claim:assignment.depends_on"]
    assert "adjusted for lunch" in f.note and "hidden factor" in f.note
    assert set(f.draws_on) == {"design.graph", "claim:assignment.depends_on", "col:lunch.when", "claim:unobserved", "claim:exclusion"}
    assert check_spec(f, set(f.draws_on)) == []
    assert "figure:causal_graph.edge.8" in f.addresses() and "figure:causal_graph.node.7" in f.addresses()
    assert A.causal_graph({}) is None


def test_balance_before_and_after():
    f = A.balance(
        {"lunch": {"before": 0.31, "after": 0.04}, "parental": {"before": 0.12, "after": 0.02}},
        "completed_vs_none",
        0.1,
        {"parental": "parental level of education"},
    )
    assert f.kind == "bars" and f.id == "balance_completed_vs_none" and [s.name for s in f.series] == ["before adjustment", "after weighting on the score"]
    assert f.series[0].x == ["lunch", "parental level of education"] and f.series[1].y == [0.04, 0.02]
    assert f.marks[0].kind == "hline" and f.marks[0].at == 0.1 and "0.04" in f.note
    assert set(f.draws_on) == {"check:completed_vs_none.balance.lunch", "check:completed_vs_none.balance.parental", "check_facts.balance"}
    assert A.balance({}, "c") is None
