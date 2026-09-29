from causal_agent.families import registry as R
from causal_agent.families.base import render_preferences


def test_registry_loads():
    reg = R.knowledge()
    names = {f.name for f in reg}
    assert {"adjustment", "diff_in_diff", "discontinuity", "interrupted_series", "synthetic_control", "instrument", "root_cause"} <= names
    adj = next(f for f in reg if f.name == "adjustment")
    assert adj.status == "built" and adj.specialist == "dowhy"
    assert "prefer synthetic_control over diff_in_diff" in render_preferences(reg)
    assert "where the evidence usually lives" in adj.render()


def test_a_built_family_lists_its_decisions_and_a_declared_one_none():
    reg = {f.name: f for f in R.knowledge()}
    adj = reg["adjustment"]
    assert [d.name for d in adj.decisions] == ["who_is_treated", "mechanism", "road", "adjustment_set", "forbidden", "target", "run_at_all", "heterogeneity"]
    needs = R.needs()["adjustment"]
    assert "col:<column>.may_modify" in needs.asks and "claim:assignment.offer_column" in needs.asks and "probe:adjustment.overlap" not in needs.asks
    assert adj.decision("road") is not None and "claim:unobserved.exists" in adj.decision("road").rests_on and adj.decision("nope") is None
    assert [d.name for d in reg["diff_in_diff"].decisions] == ["groups", "periods", "comparison_holds", "controls", "cluster", "run_at_all"]
    assert [d.name for d in reg["discontinuity"].decisions] == ["score_and_line", "sharp_or_fuzzy", "line_is_clean", "covariates", "run_at_all"]
    assert reg["diff_in_diff"].decision("road") is None and reg["discontinuity"].decision("road") is None
    assert all(not reg[n].decisions for n in ("interrupted_series", "synthetic_control", "instrument", "root_cause"))
    text = adj.render()
    assert "decisions the design must make:" in text and "- road: the road: back door, front door, or instrument  (rests on claim:unobserved.exists" in text
    assert "decisions the design must make" not in reg["instrument"].render()
