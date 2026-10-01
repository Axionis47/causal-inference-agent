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
    assert [d.name for d in reg["diff_in_diff"].decisions] == ["groups", "periods", "mechanism", "comparison_holds", "controls", "cluster", "run_at_all"]
    assert [d.name for d in reg["discontinuity"].decisions] == ["score_and_line", "sharp_or_fuzzy", "line_is_clean", "covariates", "run_at_all"]
    assert reg["diff_in_diff"].decision("road") is None and reg["discontinuity"].decision("road") is None
    assert all(not reg[n].decisions for n in ("interrupted_series", "synthetic_control", "instrument", "root_cause"))
    text = adj.render()
    assert "decisions the design must make:" in text and "- road: the road: back door, front door, or instrument  (rests on claim:unobserved.exists" in text
    assert "decisions the design must make" not in reg["instrument"].render()


def test_what_a_family_requires_is_what_its_decisions_rest_on():
    """Ready means the inputs of every rung are settled: the required kinds are derived from the decisions, less the kinds the lane
    settles with its own question. The three built families still require what they did when the list was written by hand."""
    from causal_agent.families import registry as R

    needs = R.needs()
    for fam in R.knowledge():
        if fam.name not in needs or not fam.decisions:  # a declared family spells its list by hand until it is built
            continue
        kinds = [p[6:].split(".")[0] if p.startswith("claim:") else "measured" for d in fam.decisions for p in d.rests_on if p.startswith(("claim:", "col:"))]
        n = needs[fam.name]
        assert n.requires == [k for k in dict.fromkeys(kinds) if k not in n.lane_settles], fam.name
        assert set(n.lane_settles) <= set(kinds), fam.name
    assert set(needs["adjustment"].requires) == {"grain", "sampling", "change", "assignment", "measured", "missing", "unobserved", "spillover"}
    assert needs["adjustment"].lane_settles == ["exclusion", "mediator"]
    assert set(needs["diff_in_diff"].requires) == {"grain", "sampling", "change", "assignment", "measured", "missing", "spillover", "trend_continues"}
    assert set(needs["discontinuity"].requires) == {"grain", "sampling", "change", "assignment", "measured", "missing", "cutoff_only"}
