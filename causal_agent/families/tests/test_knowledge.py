"""Family readiness and decision material have one authority: semantic endpoints."""

from pathlib import Path

import yaml

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


def test_built_decisions_are_generated_from_endpoint_requirements():
    knowledge = {f.name: f for f in R.knowledge()}
    for name in ("adjustment", "diff_in_diff", "discontinuity"):
        needs = R.needs()[name]
        assert needs.endpoints is not None
        endpoints = needs.endpoints.endpoints
        assert [d.name for d in knowledge[name].decisions] == [e.id for e in endpoints]
        for e in endpoints:
            decision = knowledge[name].decision(e.id)
            assert decision.asks == e.title
            assert decision.rests_on == list(dict.fromkeys(r.address for r in e.requirements))
        assert knowledge[name].decision("not_an_endpoint") is None
        source = Path(__file__).resolve().parents[1] / name / "family.yaml"
        raw = yaml.safe_load(source.read_text())
        assert "decisions" not in raw and "lane_settles" not in raw.get("needs_claims", {})
    assert all(not knowledge[n].decisions for n in ("interrupted_series", "synthetic_control", "instrument", "root_cause"))


def test_required_kinds_and_question_addresses_come_from_the_same_catalogue():
    for name in ("adjustment", "diff_in_diff", "discontinuity"):
        needs = R.needs()[name]
        addresses = list(dict.fromkeys(r.address for e in needs.endpoints.endpoints for r in e.requirements))
        kinds = list(dict.fromkeys("measured" if a.startswith("col:") else a[6:].split(".")[0] for a in addresses if a.startswith(("claim:", "col:"))))
        assert needs.requires == kinds
        assert needs.asks == addresses
        assert not hasattr(needs, "lane_settles")
