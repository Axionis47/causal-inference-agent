"""Pure graph certificates and real numerical adapters, without model judgements."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_agent.common.contracts import Checks, ColumnBrief, Contrast, Fact, Handoff, Scope
from causal_agent.families.adjustment.lane.contracts import Design, Edge, Estimand, Graph
from causal_agent.families.adjustment.lane.engine import estimators
from causal_agent.families.adjustment.lane.engine.model import conditional_iv_valid, identify_candidates
from causal_agent.families.adjustment.lane.knowledge import load_estimators


@pytest.fixture
def relationship_context():
    """Declarative inputs for ledger tests, with no dataset generation or effect truth."""
    values = {
        "claim:story.text": "The study account names c as a common cause and x as a baseline measurement.",
        "claim:grain.row_is": "person",
        "claim:assignment.kind": "own_choice",
        "claim:sampling.how": "all",
        "claim:unobserved.exists": False,
        "claim:spillover.possible": False,
        "col:t.when": "at",
        "col:y.when": "after",
        "col:c.when": "before",
        "col:x.when": "before",
        "col:x.meaning": "A baseline measurement.",
        "col:c.feeds_treatment": True,
        "col:c.moves_outcome": True,
        "col:x.may_modify": True,
    }
    return Handoff(
        family="adjustment",
        specialist="dowhy",
        supported_now=True,
        outcome="y",
        treatment="t",
        scope=Scope(),
        pack_name="students",
        relevant_columns=[],
        chosen_assumption="ledger unit inputs",
        reasons=[],
        story=values["claim:story.text"],
        columns=[
            ColumnBrief(name="t", key="t", role="treatment", when="at"),
            ColumnBrief(name="y", key="y", role="outcome", when="after"),
            ColumnBrief(name="c", key="c", when="before"),
            ColumnBrief(name="x", key="x", when="before"),
        ],
        facts={address: Fact(value=value, status="confirmed", source="user:turn:1") for address, value in values.items()},
    )


def graph(edges, hidden=False):
    return Graph(
        treatment="t",
        outcome="y",
        nodes=sorted({v for pair in edges for v in pair}),
        edges=[Edge(src=a, dst=b, cites=["claim:story.text"], status="supported") for a, b in edges],
    )


def design(endpoint, graph, certificate, method, **extra):
    entry = next(e for e in load_estimators() if e.name == method)
    return Design(
        id="dtest",
        endpoint_id=endpoint,
        context_id="context",
        ledger_revision=3,
        target={"type": certificate.target, "population": "sample", "scale": "outcome_units"},
        contrasts=[Contrast(control="0", treated="1", reason="protocol", cites=["claim:story.text"])],
        graph=graph,
        estimand=certificate,
        checks=Checks(),
        estimator=entry,
        target_units=certificate.target,
        bootstrap_draws=20,
        **extra,
    )


def test_measured_backdoor_and_general_certificates_preserve_sets():
    rng = np.random.default_rng(2)
    n = 1000
    c = rng.normal(size=n)
    t = rng.binomial(1, 0.5, n)
    table = pd.DataFrame({"treated": t, "c": c, "y": 4 * t + 2 * c + rng.normal(size=n)})
    g = graph([("t", "y"), ("c", "t"), ("c", "y")])
    certs, errors = identify_candidates(g, table, ["minimal-adjustment", "efficient-adjustment"])
    backdoor = next(e for e in certs if e.kind == "backdoor")
    assert backdoor.adjustment_set == ["c"]
    assert any(e.kind == "general_adjustment" for e in certs)
    d = design("total.backdoor", g, backdoor, "linear_regression")
    result, _ = estimators.estimate(table, d, d.estimator, "1_vs_0")
    assert result.error is None and result.value == pytest.approx(4, abs=0.2)
    assert result.design_id == "dtest" and result.endpoint_id == "total.backdoor"


def test_hidden_cause_cannot_be_removed_to_manufacture_backdoor():
    table = pd.DataFrame({"treated": [0, 1] * 30, "y": np.arange(60)})
    g = graph([("hidden:u", "t"), ("hidden:u", "y"), ("t", "y")])
    certs, _ = identify_candidates(g, table, ["minimal-adjustment"])
    assert not certs
    assert "hidden:u" in g.nodes


def test_frontdoor_and_mediation_are_distinct_targets():
    rng = np.random.default_rng(3)
    n = 1500
    t = rng.binomial(1, 0.5, n)
    m = 1.5 * t + rng.normal(size=n)
    y = 2 * t + 2 * m + rng.normal(size=n)
    table = pd.DataFrame({"treated": t, "m": m, "y": y})
    g = graph([("t", "m"), ("m", "y"), ("t", "y")])
    certs, _ = identify_candidates(g, table, ["minimal-adjustment"], mediator="m")
    assert not any(e.kind == "frontdoor" for e in certs)
    for target, value in [("nie", 3), ("nde", 2)]:
        certificate = next(e for e in certs if e.kind == "mediation" and e.target == target)
        d = design("mediation.linear_" + target, g, certificate, "mediation_two_stage")
        result, _ = estimators.estimate(table, d, d.estimator, "1_vs_0")
        assert result.error is None, result.error
        assert result.value == pytest.approx(value, abs=0.3)


def test_conditional_multi_iv_has_intercept_and_controls_in_both_matrices():
    rng = np.random.default_rng(5)
    n = 4000
    c = rng.normal(size=n)
    z1 = c + rng.normal(size=n)
    z2 = rng.normal(size=n)
    u = rng.normal(size=n)
    t = (0.7 * z1 + 0.7 * z2 + 0.5 * c + 0.8 * u + rng.normal(size=n) > 0).astype(int)
    y = 11 + 4 * t + 2 * c + u + rng.normal(size=n)
    table = pd.DataFrame({"treated": t, "z1": z1, "z2": z2, "c": c, "y": y})
    g = graph([("c", "z1"), ("c", "y"), ("c", "t"), ("z1", "t"), ("z2", "t"), ("hidden:u", "t"), ("hidden:u", "y"), ("t", "y")])
    assert not conditional_iv_valid(g, ["z1", "z2"], [])
    assert conditional_iv_valid(g, ["z1", "z2"], ["c"])
    d = design(
        "iv.linear_conditional",
        g,
        Estimand(kind="iv", instruments=["z1", "z2"], adjustment_set=["c"], identification_method="conditional_iv_dseparation"),
        "iv_2sls",
    )
    result, _ = estimators.estimate(table, d, d.estimator, "1_vs_0")
    assert result.error is None, result.error
    assert result.value == pytest.approx(4, abs=0.3)
    assert result.ci_low < result.value < result.ci_high


def test_exclusion_violation_and_post_treatment_controls_reject_conditional_iv():
    g = graph([("z", "t"), ("z", "y"), ("t", "y")])
    assert not conditional_iv_valid(g, ["z"], [])
    g = graph([("z", "t"), ("t", "c"), ("c", "y"), ("t", "y")])
    assert not conditional_iv_valid(g, ["z"], ["c"])


def test_structural_zero_is_graph_implication_without_interval():
    g = graph([("t", "x"), ("c", "y")])
    table = pd.DataFrame({"treated": [0, 1] * 30, "x": np.arange(60), "c": np.arange(60), "y": np.arange(60)})
    certs, _ = identify_candidates(g, table, ["minimal-adjustment"])
    assert len(certs) == 1 and certs[0].structural_zero
    d = design("structural.zero", g, certs[0], "structural_zero")
    result, bundle = estimators.estimate(table, d, d.estimator, "1_vs_0")
    assert result.value == 0 and result.ci_low is None and bundle is None


def test_frozen_design_top_level_is_immutable():
    from pydantic import ValidationError

    d = design("structural.zero", graph([("t", "x"), ("c", "y")]), Estimand(kind="zero", structural_zero=True), "structural_zero")
    with pytest.raises(ValidationError):
        d.context_id = "different"


def test_binary_instrument_wald_effect_is_explicitly_local():
    rng = np.random.default_rng(19)
    n = 2000
    z = rng.binomial(1, 0.5, n)
    t = z * rng.binomial(1, 0.7, n)
    y = 4 * t + rng.normal(size=n)
    table = pd.DataFrame({"treated": t, "z": z, "y": y})
    g = graph([("z", "t"), ("t", "y")])
    certificate = Estimand(kind="iv", target="late", instruments=["z"], identification_method="minimal-adjustment")
    d = design("iv.wald_late", g, certificate, "wald")
    result, _ = estimators.estimate(table, d, d.estimator, "1_vs_0")
    assert result.error is None, result.error
    assert result.value == pytest.approx(4, abs=0.3) and result.target_units == "late"


def test_frontdoor_total_effect_with_hidden_confounder():
    rng = np.random.default_rng(12)
    n = 2000
    u = rng.normal(size=n)
    t = (u + rng.normal(size=n) > 0).astype(int)
    m = 1.5 * t + rng.normal(size=n)
    y = 2 * m + u + rng.normal(size=n)
    table = pd.DataFrame({"treated": t, "m": m, "y": y})
    g = graph([("t", "m"), ("m", "y"), ("hidden:u", "t"), ("hidden:u", "y")])
    certificates, _ = identify_candidates(g, table, ["minimal-adjustment"])
    certificate = next(e for e in certificates if e.kind == "frontdoor")
    assert certificate.frontdoor_set == ["m"]
    d = design("frontdoor.linear_total", g, certificate, "frontdoor_two_stage")
    result, _ = estimators.estimate(table, d, d.estimator, "1_vs_0")
    assert result.error is None, result.error
    assert result.value == pytest.approx(3, abs=0.5)


def test_nested_mutation_breaks_frozen_integrity_before_fit():
    d = design("structural.zero", graph([("t", "x"), ("c", "y")]), Estimand(kind="zero", structural_zero=True), "structural_zero")
    d.graph.edges.append(Edge(src="t", dst="y"))
    result, _ = estimators.estimate(pd.DataFrame({"treated": [0, 1], "x": [1, 2], "c": [1, 2], "y": [1, 2]}), d, d.estimator, "1_vs_0")
    assert "frozen specification changed" in result.error
    assert result.value is None


def test_conditional_effects_use_one_interaction_model_with_frozen_groups():
    rng = np.random.default_rng(17)
    n = 1000
    x = rng.normal(size=n)
    t = rng.binomial(1, 0.5, n)
    y = (2 + 0.5 * x) * t + x + rng.normal(size=n)
    table = pd.DataFrame({"treated": t, "x": x, "y": y})
    g = graph([("x", "y"), ("t", "y")])
    d = design(
        "adjustment.linear_conditional",
        g,
        Estimand(kind="backdoor", target="conditional", model_covariates=["x"], identification_method="minimal-adjustment"),
        "linear_interaction",
        modifiers=["x"],
        modifier_groups={
            "x": [
                {"label": "negative", "low": float(x.min()), "high": 0, "include_low": True},
                {"label": "positive", "low": 0, "high": float(x.max()), "include_low": False},
            ]
        },
    )
    results = estimators.conditional_estimates(table, d, "1_vs_0")
    assert len(results) == 2 and all(r.error is None for r in results)
    assert results[0].value < results[1].value
    assert results[0].value == pytest.approx(2 + 0.5 * float(table.loc[table.x <= 0, "x"].mean()), abs=0.25)
    assert results[1].value == pytest.approx(2 + 0.5 * float(table.loc[table.x > 0, "x"].mean()), abs=0.25)
    assert all(r.target["subgroup"]["column"] == "x" for r in results)
    from causal_agent.families.adjustment.lane.artifacts import _material

    text, _, _ = _material({"designs": {d.id: d}, "estimates": results})
    assert '"identification_adjustment_set": []' in text and '"model_covariates": ["x"]' in text
    assert "not that the fitted model omits these covariates" in text


def test_relationship_write_classifies_provenance_and_preserves_confirmed_reads(relationship_context):
    from causal_agent.families.adjustment.lane.contracts import RelationshipProposal, Relationships
    from causal_agent.families.adjustment.lane.reason import _draft_entries, seed

    h = relationship_context
    state = {"handoff": h, "columns": {b.key: b.name for b in h.columns}}
    state.update(seed(state))
    proposal = Relationships(
        entries=[
            RelationshipProposal(kind="edge", src="c", dst="t", status="supported", reason="consistent read", cites=["claim:story.text"]),
            RelationshipProposal(kind="edge", src="t", dst="y", status="supported", reason="supplied causal mechanism", cites=["claim:story.text"]),
        ],
        considered=["c", "x"],
        reason="read the context",
        cites=["claim:story.text"],
    )
    entries = _draft_entries(state, proposal)
    assert len(entries) == 1 and entries[0].src == "t" and entries[0].dst == "y"
    assert entries[0].evidence[0].source == "pack" and entries[0].evidence[0].category == "study_account"
    assert state["ledger"].entries["rel:edge.c.t"].status == "confirmed"


def test_unsettled_absence_has_an_edge_present_rival(relationship_context):
    from causal_agent.families.adjustment.lane.reason import _graphs
    from causal_agent.lane.ledger import Entry, Ledger

    h = relationship_context
    ledger = Ledger(
        entries={
            "rel:effect": Entry(id="rel:effect", kind="edge", src="t", dst="y", status="supported", reason="study mechanism"),
            "rel:uncertain": Entry(id="rel:uncertain", kind="no_edge", src="c", dst="t", status="proposed", reason="unsettled absence"),
        }
    )
    graphs, errors = _graphs({"handoff": h, "columns": {b.key: b.name for b in h.columns}, "ledger": ledger})
    assert len(graphs) == 2 and not errors
    assert {("c", "t") in {(e.src, e.dst) for e in g.edges} for g in graphs} == {True, False}


def test_id_expression_is_observable_diagnostic_and_blocks_latent_graphs():
    from causal_agent.families.adjustment.lane.engine.model import id_diagnostic

    observed = id_diagnostic(graph([("c", "t"), ("c", "y"), ("t", "y")]), {"t", "c", "y"})
    assert observed["status"] == "diagnostic"
    assert set(observed["variables"]) <= {"t", "c", "y"}
    assert set(observed["cites"]) == {"col:" + v for v in observed["variables"]}
    hidden = id_diagnostic(graph([("hidden:u", "t"), ("hidden:u", "y"), ("t", "y")]), {"t", "y"})
    assert hidden["status"] == "unsupported" and "observed_nodes" in hidden["reason"]


def test_candidate_transitions_retain_obligations_from_real_certificate(relationship_context):
    from causal_agent.common.contracts import CheckResult
    from causal_agent.families.adjustment.lane.contracts import Candidate
    from causal_agent.families.adjustment.lane.knowledge import load_endpoint_catalogue
    from causal_agent.families.adjustment.lane.reason import _transitions
    from causal_agent.lane.endpoints import admit
    from causal_agent.lane.ledger import Ledger

    h = admit(relationship_context, load_endpoint_catalogue())
    g = graph([("t", "y"), ("c", "t"), ("c", "y")])
    table = pd.DataFrame({"treated": [0, 1], "c": [1, 2], "y": [2, 3]})
    certs, _ = identify_candidates(g, table, ["minimal-adjustment"])
    c = Candidate(
        id="dcertificate",
        endpoint_id="total.backdoor",
        graph_id="g",
        graph=g,
        estimand=certs[0],
        target={"type": "ate"},
        assumptions=[],
        cites=["claim:story.text"],
        checks=Checks(results=[CheckResult(design_id="dcertificate", contrast="1_vs_0", name="overlap", level="hard", detail="unsupported cells")]),
    )
    events = _transitions({"handoff": h, "ledger": Ledger(), "candidates": []}, [c], "check_design")
    assert any(o["id"] == "overlap" and o["status"] == "failed" for o in events[0]["obligations"])
    assert any(o["id"] == "identification" and o["status"] == "passed" for o in events[0]["obligations"])
    assert events[0]["endpoint_id"] == "total.backdoor" and events[0]["declared_trajectory"]


def test_propensity_placebo_refits_scores_without_polluting_raw_data():
    from causal_agent.families.adjustment.lane.engine import falsify
    from causal_agent.families.adjustment.lane.knowledge import load_falsifications

    rng = np.random.default_rng(23)
    n = 700
    c = rng.normal(size=n)
    t = (c + rng.normal(size=n) > 0).astype(int)
    table = pd.DataFrame({"treated": t, "c": c, "y": 4 * t + 2 * c + rng.normal(size=n)})
    g = graph([("t", "y"), ("c", "t"), ("c", "y")])
    certs, _ = identify_candidates(g, table, ["minimal-adjustment"])
    certificate = next(e for e in certs if e.kind == "backdoor")
    d = design("total.backdoor", g, certificate, "propensity_score_weighting")
    estimate, bundle = estimators.estimate(table, d, d.estimator, "1_vs_0")
    assert estimate.error is None and set(table.columns) == {"treated", "c", "y"}
    entry = next(e for e in load_falsifications() if e.name == "placebo_treatment_refuter")
    entry = entry.model_copy(update={"params": {**entry.params, "num_simulations": 20}})
    null = falsify.run(bundle[0], bundle, estimate, entry, "1_vs_0")
    assert null.new_effect is not None and abs(null.new_effect) < 0.6 and null.passed


def test_frozen_support_records_original_rows_and_detects_changed_data():
    from causal_agent.families.adjustment.lane.support import freeze, verify

    table = pd.DataFrame({"t": [0, 1, 0], "c": [2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0]})
    contrast = Contrast(control="0", treated="1", reason="protocol", cites=["claim:story.text"])
    support = freeze(table, "t", [contrast], [5, 9, 11], "context")
    assert support["1_vs_0"]["source_row_ids"] == [5, 9, 11]
    g = graph([("t", "y")])
    d = design("total.unadjusted", g, Estimand(kind="backdoor", adjustment_set=[]), "linear_regression", support={"by_contrast": support})
    from causal_agent.families.adjustment.lane.support import contrast_table

    sub = contrast_table(table, "t", contrast)
    verify(sub, d, "1_vs_0")
    changed = sub.copy()
    changed.loc[0, "y"] += 1
    with pytest.raises(ValueError, match="frozen support"):
        verify(changed, d, "1_vs_0")


def test_structural_zero_needs_positive_domain_admission(relationship_context):
    from causal_agent.families.adjustment.lane.knowledge import load_endpoint_catalogue
    from causal_agent.lane.endpoints import admit

    hidden = admit(relationship_context, load_endpoint_catalogue())
    zero = admit(
        relationship_context.model_copy(
            update={"facts": {**relationship_context.facts, "claim:change.no_causal_path": Fact(value=True, status="confirmed", source="user:turn:1")}}
        ),
        load_endpoint_catalogue(),
    )
    assert not next(e for e in hidden.admission.endpoints if e.endpoint_id == "structural.zero").admitted
    assert next(e for e in zero.admission.endpoints if e.endpoint_id == "structural.zero").admitted


def test_graph_cannot_violate_explicit_absence_of_every_causal_path(relationship_context):
    from causal_agent.families.adjustment.lane.reason import _graphs
    from causal_agent.lane.ledger import Entry, Ledger

    h = relationship_context.model_copy(
        update={"facts": {**relationship_context.facts, "claim:change.no_causal_path": Fact(value=True, status="confirmed", source="user:turn:1")}}
    )
    ledger = Ledger(
        entries={"rel:effect": Entry(id="rel:effect", kind="edge", src="t", dst="y", status="supported", reason="contradicts supplied no-path assertion")}
    )
    graphs, errors = _graphs({"handoff": h, "columns": {b.key: b.name for b in h.columns}, "ledger": ledger})
    assert not graphs and any("explicit supplied absence" in error for error in errors)


def test_drafted_model_context_cannot_support_a_causal_relationship(relationship_context):
    from causal_agent.families.adjustment.lane.contracts import RelationshipProposal, Relationships
    from causal_agent.families.adjustment.lane.reason import _draft_entries
    from causal_agent.lane.ledger import Ledger, apply

    h = relationship_context
    facts = dict(h.facts)
    facts["col:x.meaning"] = facts["col:x.meaning"].model_copy(update={"status": "drafted", "source": "model:infer"})
    h = h.model_copy(update={"facts": facts})
    record = Relationships(
        entries=[RelationshipProposal(kind="edge", src="x", dst="y", status="supported", reason="AI interpretation", cites=["col:x.note"])],
        considered=["c", "x"],
        reason="draft",
        cites=["col:x.note"],
    )
    entries = _draft_entries({"handoff": h, "ledger": Ledger()}, record)
    _, errors = apply(Ledger(), entries, by="relate", resolves=lambda _: True)
    assert errors and entries[0].evidence[0].source == "judgement"


def test_pre_freeze_tools_refuse_all_outcome_relations_including_aliases():
    from causal_agent.families.adjustment.lane.reason import RestrictedTools
    from causal_agent.lane.tools import Refused, Tools

    table = pd.DataFrame({"t": [0, 1] * 20, "c": np.arange(40), "y": np.arange(40) ** 2})
    tools = Tools(df=table, outcome="y", treatment="t", treated=table.t == 1, aliases={"grade": "y"})
    restricted = RestrictedTools(tools, ["describe", "association", "redundancy", "cells", "by_arm", "timing"])
    for name, args in [
        ("association", {"a": "y", "b": "c"}),
        ("association", {"a": "grade", "b": "c"}),
        ("redundancy", {"a": "c", "b": "y"}),
        ("cells", {"columns": ["c", "grade"]}),
        ("by_arm", {"column": "grade"}),
        ("timing", {"a": "y", "b": "c"}),
    ]:
        with pytest.raises(Refused, match="only after design freeze"):
            restricted.call(name, args)
    assert restricted.call("describe", {"column": "y"}).text
    assert restricted.call("association", {"a": "c", "b": "t"}).text
    assert restricted.call("timing", {}).text


def test_frozen_contrast_labels_are_citable_at_the_estimate_evidence():
    from causal_agent.common.contracts import Estimate
    from causal_agent.families.adjustment.lane.artifacts import _interpret_errors, _material
    from causal_agent.families.adjustment.lane.contracts import ComparisonInterpretation

    d = design("total.unadjusted", graph([("t", "y")]), Estimand(kind="backdoor"), "linear_regression")
    e = Estimate(design_id=d.id, endpoint_id=d.endpoint_id, contrast=d.contrasts[0].key, method="linear_regression", value=4, ci_low=3.5, ci_high=4.5)
    material, addresses, numbers = _material({"designs": {d.id: d}, "estimates": [e]})
    record = ComparisonInterpretation(
        answer="Receiving t=1 rather than t=0 has an estimated effect of 4 (95% interval 3.5 to 4.5).", caveats=[], cites=[e.tag + ".value", e.tag + ".ci"]
    )
    assert '"treated": "1", "control": "0"' in material
    assert _interpret_errors(record, addresses, numbers, []) == []
    invented = record.model_copy(update={"answer": "Receiving t=2 rather than t=0 has an estimated effect of 4."})
    assert any("number 2" in error for error in _interpret_errors(invented, addresses, numbers, []))


def test_propensity_stratification_cannot_change_frozen_population_or_select_bootstrap_draws(monkeypatch):
    from causal_agent.families.adjustment.lane.engine.model import stratification_support
    from causal_agent.families.adjustment.lane.support import contrast_table

    table = pd.read_csv("data/raw/students-performance-in-exams/StudentsPerformance.csv")
    table = pd.DataFrame(
        {
            "t": table["test preparation course"].eq("completed").astype(int),
            "c": pd.Categorical(table["parental level of education"]).codes,
            "x": table["lunch"].eq("standard").astype(int),
            "y": table["math score"],
        }
    )
    contrast = Contrast(treated="1", control="0", reason="protocol", cites=[])
    table = contrast_table(table, "t", contrast)
    g = graph([("c", "t"), ("c", "y"), ("x", "y"), ("t", "y")])
    certificate = Estimand(kind="backdoor", adjustment_set=["c", "x"], identification_method="maximal-adjustment")
    d = design("total.backdoor", g, certificate, "propensity_score_stratification")
    specification = d.model_dump(exclude={"integrity"})
    specification["estimator"]["params"]["num_strata"] = 50
    d = Design.model_validate(specification)
    support = stratification_support(table, g, certificate, d.estimator.params)
    assert support["retained_rows"] < support["input_rows"] == len(table)
    changed_outcomes = table.assign(y=table["y"].iloc[::-1].to_numpy() * 1000)
    assert stratification_support(changed_outcomes, g, certificate, d.estimator.params) == support
    record, _ = estimators.estimate(table, d, d.estimator, contrast.key)
    assert record.value is None and record.n_treated + record.n_control == len(table)
    assert "support: propensity stratification" in record.error

    calls = []

    def fit_once_then_clip(sample, design, entry):
        calls.append(len(sample))
        if len(calls) == 1:
            return 4.0, None
        raise estimators.FrozenSupportError("support: one bootstrap draw clipped the frozen population")

    monkeypatch.setattr(estimators, "_dowhy_fit", fit_once_then_clip)
    record, _ = estimators.estimate(table, d, d.estimator, contrast.key)
    assert len(calls) == 2
    assert record.error and record.ci_low is None and record.ci_high is None


def test_lead_ranking_uses_support_then_assumptions_catalogue_and_stable_identity():
    from causal_agent.common.contracts import CheckResult
    from causal_agent.families.adjustment.lane.contracts import Candidate
    from causal_agent.families.adjustment.lane.knowledge import load_endpoint_catalogue
    from causal_agent.families.adjustment.lane.reason import _lead_rank

    catalogue = load_endpoint_catalogue()
    g = graph([("t", "y")])
    weaker = Candidate(
        id="d000",
        graph_id="g",
        endpoint_id="total.backdoor",
        graph=g,
        estimand=Estimand(kind="backdoor"),
        target={"type": "ate"},
        checks=Checks(results=[CheckResult(contrast="1_vs_0", name="overlap", level="soft", value=0.8)]),
    )
    stronger = weaker.model_copy(update={"id": "d999", "checks": Checks(results=[CheckResult(contrast="1_vs_0", name="overlap", level="pass", value=1.0)])})
    assert min([weaker, stronger], key=lambda c: _lead_rank(c, catalogue)).id == stronger.id
    assumed = stronger.model_copy(update={"id": "d001", "assumptions": ["additional homogeneous-effect assumption"]})
    assert _lead_rank(stronger, catalogue) < _lead_rank(assumed, catalogue)
    earlier_endpoint = stronger.model_copy(update={"id": "dzzz", "endpoint_id": "total.unadjusted"})
    assert _lead_rank(earlier_endpoint, catalogue) < _lead_rank(stronger, catalogue)
    tie = stronger.model_copy(update={"id": "d111"})
    assert min([stronger, tie], key=lambda c: _lead_rank(c, catalogue)).id == tie.id


def test_lead_completion_requires_every_frozen_primary_contrast_and_records_fallback():
    from causal_agent.common.contracts import Estimate
    from causal_agent.families.adjustment.lane.artifacts import _execution_summary

    d = design("total.unadjusted", graph([("t", "y")]), Estimand(kind="backdoor"), "linear_regression")
    specification = d.model_dump(exclude={"integrity"})
    specification["contrasts"] = [
        Contrast(treated="1", control="0", reason="protocol", cites=[]),
        Contrast(treated="2", control="0", reason="protocol", cites=[]),
    ]
    d = Design.model_validate(specification)
    good = Estimate(design_id=d.id, contrast="1_vs_0", method="linear_regression", value=4)
    failed = Estimate(design_id=d.id, contrast="2_vs_0", method="linear_regression", error="fit failed")
    summary = _execution_summary({"designs": {d.id: d}, "estimates": [good, failed]})[d.id]
    assert not summary["complete"] and summary["failed"] == ["2_vs_0"]
    missing = _execution_summary({"designs": {d.id: d}, "estimates": [good]})[d.id]
    assert not missing["complete"] and missing["missing"] == ["2_vs_0"]
    fallback = failed.model_copy(update={"method": "frozen_fallback", "value": 4, "error": None})
    resolved = _execution_summary({"designs": {d.id: d}, "estimates": [good, failed.model_copy(update={"secondary": True}), fallback]})[d.id]
    assert resolved["complete"] and resolved["executed"] == ["1_vs_0", "2_vs_0"]


def test_conditional_completion_requires_every_requested_group_and_ignores_overall_reference():
    from causal_agent.common.contracts import Estimate
    from causal_agent.families.adjustment.lane.artifacts import _execution_summary

    d = design(
        "adjustment.linear_conditional",
        graph([("x", "y"), ("t", "y")]),
        Estimand(kind="backdoor", target="conditional", model_covariates=["x"]),
        "linear_interaction",
        modifiers=["x"],
        modifier_groups={"x": [{"label": "low", "values": ["0"]}, {"label": "high", "values": ["1"]}]},
    )
    overall = Estimate(design_id=d.id, contrast="1_vs_0", method=d.estimator.name, value=2, secondary=True)
    low = overall.model_copy(update={"modifier": "x", "level": "low", "secondary": False})
    high = low.model_copy(update={"level": "high", "value": None, "error": "no treatment arm in requested subgroup"})
    failed = _execution_summary({"designs": {d.id: d}, "estimates": [overall, low, high]})[d.id]
    assert not failed["complete"] and failed["failed"] == ["1_vs_0.x.high"]
    missing = _execution_summary({"designs": {d.id: d}, "estimates": [overall, low]})[d.id]
    assert not missing["complete"] and missing["missing"] == ["1_vs_0.x.high"]
    only_mean = _execution_summary({"designs": {d.id: d}, "estimates": [overall.model_copy(update={"secondary": False})]})[d.id]
    assert not only_mean["complete"] and only_mean["executed"] == []
    complete = _execution_summary({"designs": {d.id: d}, "estimates": [overall, low, high.model_copy(update={"value": 3, "error": None})]})[d.id]
    assert complete["complete"] and complete["executed"] == ["1_vs_0.x.high", "1_vs_0.x.low"]


def test_conditional_unsupported_group_cannot_be_hidden_by_finite_overall_fit():
    from causal_agent.families.adjustment.lane.artifacts import _execution_summary

    rng = np.random.default_rng(71)
    n = 400
    x = np.r_[np.zeros(10), np.ones(n - 10)]
    treatment = rng.binomial(1, 0.5, n)
    treatment[:10] = 0
    table = pd.DataFrame({"treated": treatment, "x": x, "y": 2 * treatment + 3 * x + rng.normal(size=n)})
    d = design(
        "adjustment.linear_conditional",
        graph([("x", "y"), ("t", "y")]),
        Estimand(kind="backdoor", target="conditional", model_covariates=["x"]),
        "linear_interaction",
        modifiers=["x"],
        modifier_groups={"x": [{"label": "rare", "values": ["0.0"]}, {"label": "regular", "values": ["1.0"]}]},
    )
    reference, bundle = estimators.estimate(table, d, d.estimator, "1_vs_0")
    assert reference.error is None and np.isfinite(reference.value) and reference.secondary
    groups = estimators.conditional_estimates(table, d, "1_vs_0", bundle=bundle)
    rare, regular = groups
    assert rare.error == "fewer than twenty rows in a subgroup arm" and rare.n_treated == 0 and rare.n_control == 10
    assert regular.error is None and not regular.secondary
    summary = _execution_summary({"designs": {d.id: d}, "estimates": [reference, *groups]})[d.id]
    assert not summary["complete"] and summary["failed"] == ["1_vs_0.x.rare"]
    model_failures = estimators.conditional_estimates(table, d, "1_vs_0", error="frozen interaction model failed")
    assert len(model_failures) == 2 and all(record.error == "frozen interaction model failed" and not record.secondary for record in model_failures)


def test_confirmed_modifier_is_retained_without_model_restatement_and_invalid_value_refuses(relationship_context):
    from causal_agent.families.adjustment.lane.contracts import RelationshipProposal, Relationships
    from causal_agent.families.adjustment.lane.reason import _relationship_gate, seed
    from causal_agent.families.adjustment.lane.views import build

    h = relationship_context
    state = {"handoff": h, "columns": {column.key: column.name for column in h.columns}}
    state.update(seed(state))
    original = state["ledger"].model_copy(deep=True)
    view = build("relate", h, original)
    assert "READ-ONLY CONFIRMED LEDGER" in view.text and "rel:modifies.x.value" in view.text
    retained = Relationships(entries=[], considered=["c", "x"], reason="supplied modifier is already retained", cites=["rel:modifies.x.value"])
    assert _relationship_gate(state, retained, view.resolve, all_columns=True) == []
    for value in (None, ["t", "y"], False):
        proposal = RelationshipProposal(
            kind="modifies", src="x", dst="", value=value, status="supported", reason="incorrect restatement", cites=["claim:story.text"]
        )
        attempted = retained.model_copy(update={"entries": [proposal]})
        assert _relationship_gate(state, attempted, view.resolve, all_columns=True)
        assert state["ledger"] == original


def test_model_material_uses_effective_controls_and_keeps_stage_sets_separate():
    from causal_agent.families.adjustment.lane.artifacts import _model_material

    g = graph([("c", "t"), ("c", "y"), ("t", "y")])
    for kind, endpoint, method in (
        ("backdoor", "total.backdoor", "linear_regression"),
        ("general_adjustment", "total.general_adjustment", "general_linear"),
        ("iv", "iv.linear_conditional", "instrumental_variable"),
    ):
        d = design(endpoint, g, Estimand(kind=kind, adjustment_set=["c"], instruments=["z"] if kind == "iv" else []), method)
        assert d.estimand.model_covariates == []
        material = _model_material(d)
        assert material["model_covariates"] == ["c"]
        if kind == "iv":
            assert material["excluded_instruments"] == ["z"]
    for kind, endpoint, method in (
        ("frontdoor", "frontdoor.linear_total", "frontdoor_two_stage"),
        ("mediation", "mediation.linear_nie", "mediation_two_stage"),
    ):
        certificate = Estimand(kind=kind, mediator="m", frontdoor_set=["m"], first_stage_set=["c"], second_stage_set=["x"], total_set=["c", "x"])
        d = design(endpoint, g, certificate, method)
        material = _model_material(d)
        assert "model_covariates" not in material
        assert material["stage_adjustment_sets"] == {"first": ["c"], "second": ["x"], "total": ["c", "x"]}
        assert material["mediator_variables"] == ["m"]


def test_internal_and_lossy_artifact_names_never_overwrite_admitted_variables():
    from causal_agent.families.adjustment.lane.reason import _groups
    from causal_agent.families.adjustment.lane.support import contrast_table, unique_contrast_addresses

    table = pd.DataFrame({"t": [0, 1], "treated": [8, 9], "y": [3, 4]})
    original = table.copy(deep=True)
    with pytest.raises(ValueError, match="overwrite an unrelated"):
        contrast_table(table, "t", Contrast(treated="1", control="0", reason="protocol", cites=[]))
    pd.testing.assert_frame_equal(table, original)
    contrasts = [Contrast(treated="0", control=control, reason="protocol", cites=[]) for control in ["-1", "1"]]
    assert not unique_contrast_addresses(contrasts)
    with pytest.raises(ValueError, match="collide at artifact"):
        _groups(pd.DataFrame({"x": ["-1", "1"]}), ["x"])
    with pytest.raises(ValueError, match="more than four categorical"):
        _groups(pd.DataFrame({"x": list("abcde")}), ["x"])


def test_pair_profiles_and_checked_timing_cannot_be_laundered_into_causal_support(relationship_context):
    from causal_agent.families.adjustment.lane.contracts import RelationshipProposal, Relationships
    from causal_agent.families.adjustment.lane.reason import _draft_entries, _evidence, _relationship_gate, seed

    h = relationship_context
    facts = dict(h.facts)
    facts.pop("claim:story.text")
    facts["col:x.when"] = facts["col:x.when"].model_copy(update={"source": "data:timing_check"})
    facts["col:c.feeds_treatment"] = facts["col:c.feeds_treatment"].model_copy(update={"source": "data:association"})
    h = h.model_copy(update={"story": None, "facts": facts})
    state = {"handoff": h, "columns": {b.key: b.name for b in h.columns}}
    state.update(seed(state))
    assert state["ledger"].entries["rel:edge.c.t"].status == "proposed"
    assert state["ledger"].entries["rel:when.x.value"].status == "confirmed"
    assert _evidence(state, "rel:when.x.value", "edge").category == "association"
    assert _evidence(state, "col:x.profile.numeric.mean", "edge").source == "data"
    assert _evidence(state, "pair.target", "edge").category == "assumption"
    for address in ["pair.target", "col:x.profile.numeric.mean", "rel:when.x.value"]:
        record = Relationships(
            entries=[RelationshipProposal(kind="edge", src="t", dst="y", status="supported", reason="candidate relationship", cites=[address])],
            considered=["c", "x"],
            reason="candidate",
            cites=[address],
        )
        assert any("specific study account" in error for error in _relationship_gate(state, record, lambda _: True, all_columns=True))
    record = Relationships(
        entries=[
            RelationshipProposal(
                kind="edge",
                src="t",
                dst="y",
                status="proposed",
                reason="conditional target path, not established by the question",
                cites=["pair.target", "pair.treatment", "pair.outcome"],
            )
        ],
        considered=["c", "x"],
        reason="source account does not establish this path",
        cites=["pair.target"],
    )
    assert _relationship_gate(state, record, lambda _: True, all_columns=True) == []
    assert all(e.source == "judgement" and e.category == "assumption" for e in _draft_entries(state, record)[0].evidence)
    phantom = record.model_copy(update={"cites": ["QUESTION", "TARGET"]})
    assert any("unresolved citation" in error for error in _relationship_gate(state, phantom, lambda a: a not in {"QUESTION", "TARGET"}, all_columns=True))
