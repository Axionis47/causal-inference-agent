from causal_agent.knowledge import load_registry, render_preferences


def test_registry_loads():
    reg = load_registry()
    names = {f.name for f in reg}
    assert {"adjustment", "diff_in_diff", "discontinuity", "interrupted_series", "synthetic_control", "instrument", "root_cause"} <= names
    adj = next(f for f in reg if f.name == "adjustment")
    assert adj.status == "built" and adj.specialist == "dowhy"
    assert "prefer synthetic_control over diff_in_diff" in render_preferences(reg)
    assert "where the evidence usually lives" in adj.render()
