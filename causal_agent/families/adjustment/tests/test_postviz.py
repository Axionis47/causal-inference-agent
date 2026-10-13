"""Graph figures retain frozen design identity and relationship status."""

from causal_agent.families.adjustment import postviz
from causal_agent.lane.figures import check_spec


def test_graph_is_design_qualified():
    graph = {
        "treatment": "t",
        "outcome": "y",
        "nodes": ["t", "y", "hidden:u"],
        "edges": [
            {"src": "t", "dst": "y", "status": "proposed", "cites": ["rel:t_y"]},
            {"src": "hidden:u", "dst": "t", "status": "confirmed", "cites": ["claim:unobserved.exists"]},
        ],
        "excluded": [],
    }
    spec = postviz.causal_graph(graph, {}, design_id="d1")
    assert spec.id == "graph_d1" and spec.draws_on == ["design:d1.graph"]
    assert spec.edges[0].status == "proposed"
    assert next(n for n in spec.nodes if n.id == "hidden:u").role == "hidden"
    assert check_spec(spec, {"design:d1.graph"}) == []


def test_comparison_figures_separate_analysis_populations():
    from causal_agent.common.contracts import Estimate

    a = Estimate(
        design_id="d1",
        contrast="1_vs_0",
        method="linear",
        value=2,
        target={"type": "ate", "population": "sample", "scale": "units", "analysis_population_id": "rowsA"},
    )
    b = a.model_copy(update={"design_id": "d2", "target": {**a.target, "analysis_population_id": "rowsB"}})
    specs = postviz.comparisons([a, b])
    assert len(specs) == 2 and all(len(s.series[0].x) == 1 for s in specs)
