"""A catalogue entry applies to facts by one rule: bounds, membership, equality, and a missing fact never matches."""

from __future__ import annotations

from causal_agent.lane.knowledge import fact_names, matches

FACTS = {"cohorts": 3, "periods_pre": 2, "kind": "long", "never_treated": True}


def test_bounds_membership_and_equality():
    assert matches({"cohorts_min": 2, "periods_pre_min": 1}, FACTS)
    assert not matches({"cohorts_min": 4}, FACTS)
    assert matches({"cohorts_max": 3}, FACTS) and not matches({"cohorts_max": 2}, FACTS)
    assert matches({"kind": ["long", "wide"]}, FACTS) and not matches({"kind": ["wide"]}, FACTS)
    assert matches({"never_treated": True}, FACTS) and not matches({"never_treated": [False]}, FACTS)
    assert matches({}, FACTS)


def test_a_fact_the_lane_did_not_compute_never_matches():
    assert not matches({"clusters_min": 1}, FACTS)
    assert not matches({"engine": ["feols"]}, FACTS)


def test_fact_names_strip_the_suffixes():
    assert fact_names({"cohorts_min": 2, "periods_pre_max": 9, "kind": ["long"]}) == {"cohorts", "periods_pre", "kind"}
