"""The in-house identification engine against textbook DAGs.

Each case fixes the right answer from causal-inference first principles:
adjust for confounders, never for mediators or colliders, and report
"not identifiable" when a backdoor path runs through an unobserved variable.
"""
from __future__ import annotations

from src.analysis_v2.spec import CausalDAG, CausalEdge, CausalNode


def _dag(edges, latent=(), treatment="T", outcome="Y") -> CausalDAG:
    nodes = {c for e in edges for c in e}
    return CausalDAG(
        nodes=[CausalNode(name=n, observed=n not in latent) for n in nodes],
        edges=[
            CausalEdge(source=s, target=t, mechanism="test") for s, t in edges
        ],
        treatment=treatment,
        outcome=outcome,
    )


def test_a_confounder_is_adjusted_for():
    dag = _dag([("C", "T"), ("C", "Y"), ("T", "Y")])
    z, _ = dag.adjustment_set()
    assert z == {"C"}
    assert dag.is_identifiable()[0] is True


def test_a_mediator_is_not_adjusted_for():
    dag = _dag([("T", "M"), ("M", "Y"), ("T", "Y")])
    z, _ = dag.adjustment_set()
    assert z == set()  # adjusting for M would block part of the effect
    assert dag.is_identifiable()[0] is True


def test_a_collider_is_not_adjusted_for():
    dag = _dag([("T", "Y"), ("T", "C"), ("Y", "C")])
    z, _ = dag.adjustment_set()
    assert z == set()  # conditioning on the collider C would open a path
    assert dag.is_identifiable()[0] is True


def test_a_latent_confounder_makes_the_effect_unidentifiable():
    dag = _dag([("U", "T"), ("U", "Y"), ("T", "Y")], latent=("U",))
    z, reason = dag.adjustment_set()
    assert z is None
    assert "unobserved" in reason
    assert dag.is_identifiable()[0] is False


def test_advertising_with_a_latent_market_size_is_unidentifiable():
    # the audit case: market size drives every channel and sales; unmeasured.
    dag = _dag(
        [
            ("market", "TV"), ("market", "Radio"), ("market", "Sales"),
            ("TV", "Sales"), ("Radio", "Sales"),
        ],
        latent=("market",),
        treatment="TV",
        outcome="Sales",
    )
    assert dag.is_identifiable()[0] is False


def test_advertising_without_a_latent_confounder_adjusts_for_the_parallel_cause():
    # no latent node: TV is identified; the parallel cause Radio is a harmless
    # precision covariate (a cause of the outcome, not a confounder of TV).
    dag = _dag(
        [("TV", "Sales"), ("Radio", "Sales")], treatment="TV", outcome="Sales"
    )
    z, _ = dag.adjustment_set()
    assert z == {"Radio"}
    assert dag.is_identifiable()[0] is True


def test_a_cycle_is_reported_not_crashed():
    dag = _dag([("T", "Y"), ("Y", "T")])
    z, reason = dag.adjustment_set()
    assert z is None and "cycle" in reason


def test_missing_treatment_or_outcome_is_not_identifiable():
    dag = CausalDAG(
        nodes=[CausalNode(name="T"), CausalNode(name="Y")],
        edges=[CausalEdge(source="T", target="Y", mechanism="t")],
        treatment="T",
        outcome="Z",  # not a node
    )
    z, reason = dag.adjustment_set()
    assert z is None and "missing" in reason


def test_testable_implications_lists_an_observed_independence():
    # T -> Y, and an isolated cause C -> Y: T and C are independent (no edge,
    # no common cause), an implication the data can check.
    dag = _dag([("T", "Y"), ("C", "Y")])
    implications = dag.testable_implications()
    pairs = {(a, b) for a, b, _ in implications}
    assert ("C", "T") in pairs or ("T", "C") in pairs
