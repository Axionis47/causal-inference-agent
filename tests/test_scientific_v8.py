"""Pure scientific counterexamples, without database writes, models or runners."""

import csv
import io
from uuid import uuid4

import pytest

from backend.analysis import domain, lineage
from backend.analysis.contracts import (
    Assumption,
    CausalAccount,
    DiagnosticWindow,
    Frame,
    IdentificationObligation,
    Plan,
    RelationshipDisposition,
    RequestedEffect,
    Roles,
    Target,
    Treatments,
    Variable,
)
from backend.context.contracts import DatasetStatement, Participant, Relationship, empty_content


@pytest.fixture(autouse=True)
def isolated():
    # Override the integration suite's database-reset fixture for these pure checks.
    yield


def claim(statement="Supplied fixture account"):
    return {"statement": statement, "basis": "supplied", "references": ["question"]}


def decision():
    return {"status": "ready", "summary": "Fixture", "citations": ["question"], "changes": "Bound", "permits": "Check"}


def study(names=("t", "y", "c")):
    context = empty_content(uuid4(), list(names))
    ids = {c.identity.name: c.identity.column_id for c in context.columns.values()}
    variables = {name: Variable(id=cid, label=name, column_ids=[cid], meaning=claim()) for name, cid in ids.items()}
    target = Target(**decision(), outcome=variables["y"], measurement=claim(), population=claim(), unit=claim(), timing=claim(), effect_question="Effect of the intervention")
    treatments = Treatments(**decision(), treatments=[{"variable": variables["t"], "change": claim(), "timing": claim()}],
        effects=[{"id": "primary", "treatment_ids": [ids["t"]], "contrast": claim(), "estimand": "Total effect", "priority": "primary"}],
        assignments=[{"id": "assignment", "treatment_ids": [ids["t"]], "kind": "point", "unit": claim(), "timing": claim(), "mechanism": claim()}])
    roles = Roles(**decision(), assignments=[{
        "variable": variables[name], "roles": ["confounder"], "timing": "before_treatment", "account": claim(),
        "timing_evidence": claim(), "reason": "Common cause", "effect_ids": ["primary"], "treatment_ids": [ids["t"]], "time_scope": "Before assignment",
    } for name in names if name not in {"t", "y"}])
    return context, ids, target, treatments, roles


def prior(target, treatments, roles):
    return {"target": target.model_dump(), "treatments": treatments.model_dump(), "roles": roles.model_dump()}


def graph(ids, target, treatments, roles, edges, adjustment=(), strategy="backdoor"):
    proposal = CausalAccount(**decision(), nodes=[{"id": cid} for cid in ids.values()],
        edges=[{"source": ids[a], "target": ids[b], "basis": "supplied", "references": ["question"], "explanation": "Fixture edge"} for a, b in edges],
        identification="Conditional on the accepted account", identifications=[{
            "effect_id": "primary", "strategy": strategy, "adjustment": [ids[name] for name in adjustment], "evidence": claim(),
        }])
    return domain.canonical_graph(proposal, target, treatments, roles)


def failures(value, context, prerequisites):
    return [check.detail for check in domain.checks(value, context, prerequisites, scientific_version=8) if not check.passed]


def conditional_assignment():
    context, ids, target, treatments, roles = study()
    treatments.assumptions = [Assumption(id="assignment", statement="Assignment is independent.",
        rationale="Explicit design premise.", affects=["primary"])]
    treatments.assignments[0].mechanism.references = ["question", "assumption:assignment"]
    return context, ids, target, treatments, roles


def test_resolve_only_demotes_explicit_assumption_dependencies_and_preserves_proposal():
    context, _, target, treatments, _ = conditional_assignment()
    treatments.assignments[0].mechanism.references[-1] = "a0"
    original = treatments.model_dump()
    resolved = domain.resolve(treatments, {"a0": "assumption:assignment"})
    mechanism = resolved.assignments[0].mechanism
    assert mechanism.basis == "assumption"
    assert mechanism.statement == treatments.assignments[0].mechanism.statement
    assert mechanism.references == ["question", "assumption:assignment"]
    assert treatments.model_dump() == original
    assert domain.resolve(resolved, {}) == resolved
    assert failures(resolved, context, {"target": target.model_dump()}) == []


@pytest.mark.parametrize("basis", ["unknown", "assumption"])
def test_resolve_never_promotes_claim_authority(basis):
    _, _, _, treatments, _ = conditional_assignment()
    treatments.assignments[0].mechanism.basis = basis
    assert domain.resolve(treatments, {}).assignments[0].mechanism.basis == basis


def test_exact_declared_bare_assumption_references_resolve_without_changing_prose_or_ids():
    context, _, target, treatments, _ = conditional_assignment()
    treatments.assumptions[0].depends_on = ["scope"]
    treatments.assumptions.append(Assumption(id="scope", statement="Scope premise.", rationale="Explicit.", affects=["primary"]))
    treatments.assignments[0].mechanism.references = ["question", "assignment"]
    treatments.citations = ["assignment"]
    treatments.summary = "assignment and scope are unchanged prose"
    original = treatments.model_dump()
    normalized = domain.resolve(treatments, {})
    assert normalized.assignments[0].mechanism.references == ["question", "assumption:assignment"]
    assert normalized.citations == ["assumption:assignment"]
    assert normalized.assumptions[0].depends_on == ["assumption:scope"]
    assert normalized.assignments[0].id == treatments.assignments[0].id
    assert normalized.summary == treatments.summary
    assert normalized.assumptions[0].statement == treatments.assumptions[0].statement
    assert treatments.model_dump() == original
    assert failures(normalized, context, {"target": target.model_dump()}) == []


@pytest.mark.parametrize("defect", ["unknown_case", "prior_only", "duplicate_identity"])
def test_bare_assumption_resolution_never_guesses_missing_or_ambiguous_identities(defect):
    context, _, target, treatments, _ = conditional_assignment()
    treatments.assignments[0].mechanism.references = ["assignment"]
    prerequisites = {"target": target.model_dump()}
    if defect == "unknown_case":
        treatments.assignments[0].mechanism.references = ["Assignment"]
    elif defect == "prior_only":
        prerequisites["target"]["assumptions"] = [a.model_dump() for a in treatments.assumptions]
        treatments.assumptions = []
    else:
        treatments.assumptions.append(treatments.assumptions[0].model_copy(deep=True))
    normalized = domain.resolve(treatments, {})
    assert normalized.assignments[0].mechanism.references == treatments.assignments[0].mechanism.references
    assert any(not check.passed and check.rule == "evidence_authority_and_dependencies"
               for check in domain.checks(normalized, context, prerequisites, scientific_version=10))


@pytest.mark.parametrize("identity,mapping", [("question", {}), ("dataset.study_process.source", {}), ("assignment", {"assignment": "question"})])
def test_bare_assumption_reference_collisions_require_an_explicit_namespace(identity, mapping):
    _, _, _, treatments, _ = conditional_assignment()
    treatments.assumptions[0].id = identity
    treatments.assignments[0].mechanism.references = [identity]
    with pytest.raises(ValueError, match="collides with an evidence reference"):
        domain.resolve(treatments, mapping)


def test_bare_prior_assumption_requires_explicit_offered_ledger_and_still_validates_definition():
    context, _, target, treatments, _ = conditional_assignment()
    target.assumptions = treatments.assumptions
    treatments.assumptions = []
    treatments.assignments[0].mechanism.references = ["assignment"]
    prerequisites = {"target": target.model_dump()}
    normalized = domain.resolve(treatments, {}, known_assumptions={"assignment"})
    assert normalized.assignments[0].mechanism.references == ["assumption:assignment"]
    assert normalized.assignments[0].mechanism.basis == "assumption"
    assert failures(normalized, context, prerequisites) == []
    assert domain.resolve(treatments, {}).assignments[0].mechanism.references == ["assignment"]
    # The offered ID permits decoding, not manufacturing a missing or conflicting premise.
    assert failures(normalized, context, {"target": target.model_copy(update={"assumptions": []}).model_dump()})
    treatments.assumptions = [target.assumptions[0].model_copy(update={"statement": "Conflicting premise."})]
    normalized = domain.resolve(treatments, {}, known_assumptions={"assignment"})
    assert any("silently changed" in failure for failure in failures(normalized, context, prerequisites))


def test_offered_prior_assumption_does_not_resolve_unknown_or_duplicate_proposal_id():
    _, _, _, treatments, _ = conditional_assignment()
    treatments.assignments[0].mechanism.references = ["assignment", "not_offered"]
    treatments.assumptions.append(treatments.assumptions[0].model_copy(deep=True))
    normalized = domain.resolve(treatments, {}, known_assumptions={"assignment", "unrelated"})
    assert normalized.assignments[0].mechanism.references == ["assignment", "not_offered"]


def test_offered_prior_assumption_decoding_is_typed_and_collision_checked():
    _, _, _, treatments, _ = conditional_assignment()
    treatments.assumptions[0].depends_on = ["prior"]
    treatments.summary = "prior"
    treatments.citations = ["prior"]
    normalized = domain.resolve(treatments, {}, known_assumptions={"prior"})
    assert normalized.summary == "prior"
    assert normalized.citations == ["assumption:prior"]
    assert normalized.assumptions[0].depends_on == ["assumption:prior"]
    with pytest.raises(ValueError, match="collides with an evidence reference"):
        domain.resolve(treatments, {"prior": "question"}, known_assumptions={"prior"})


def aliased_graph():
    context, ids, target, treatments, roles = study()
    value = graph(ids, target, treatments, roles, [("c", "t"), ("c", "y"), ("t", "y")], ["c"])
    aliases = {identity: name + "_node" for name, identity in ids.items()}
    for node in value.nodes:
        node.id = aliases[node.id]
        node.label = "Non-authoritative label"
        node.roles = ["Non-authoritative role"]
    for edge in value.edges:
        edge.source = aliases[edge.source]
        edge.target = aliases[edge.target]
    value.identifications[0].adjustment = [aliases[ids["c"]]]
    value.assumptions = [Assumption(id="scope", statement="c_node is unchanged prose", rationale="Explicit.", affects=["c_node", "primary"])]
    return context, ids, target, treatments, roles, value


def test_exact_graph_alias_preserves_science_and_only_changes_declared_typed_identities():
    context, ids, target, treatments, roles, proposal = aliased_graph()
    original = proposal.model_dump()
    normalized = domain.canonical_graph(proposal, target, treatments, roles)
    assert [(edge.source, edge.target) for edge in normalized.edges] == [(ids["c"], ids["t"]), (ids["c"], ids["y"]), (ids["t"], ids["y"])]
    assert [edge.model_dump(exclude={"source", "target"}) for edge in normalized.edges] == [edge.model_dump(exclude={"source", "target"}) for edge in proposal.edges]
    assert normalized.identifications[0].adjustment == [ids["c"]]
    assert normalized.assumptions[0].affects == [ids["c"], "primary"]
    assert normalized.assumptions[0].statement == proposal.assumptions[0].statement
    assert {node.id: node.label for node in normalized.nodes} == {identity: name for name, identity in ids.items()}
    assert proposal.model_dump() == original
    assert failures(normalized, context, prior(target, treatments, roles)) == []
    assert domain.canonical_graph(normalized, target, treatments, roles) == normalized


@pytest.mark.parametrize("defect", ["conflicting_binding", "unknown_binding", "partial_binding", "duplicate_node", "duplicate_binding", "effect_collision", "assumption_collision"])
def test_graph_alias_conflicts_and_collisions_are_rejected(defect):
    _, ids, target, treatments, roles, value = aliased_graph()
    if defect == "conflicting_binding":
        value.nodes[0].id = ids["y"]
    elif defect == "unknown_binding":
        value.nodes[0].column_ids = ["unoffered_column"]
    elif defect == "partial_binding":
        value.nodes[0].column_ids.append(ids["y"])
    elif defect == "duplicate_node":
        value.nodes.append(value.nodes[0].model_copy(deep=True))
    elif defect == "duplicate_binding":
        value.nodes.append(value.nodes[0].model_copy(update={"id": "different_alias"}))
    elif defect == "effect_collision":
        value.nodes[0].id = "primary"
    else:
        value.nodes[0].id = "scope"
    with pytest.raises(ValueError):
        domain.canonical_graph(value, target, treatments, roles)


def test_graph_alias_does_not_guess_from_labels_or_undeclared_endpoints():
    context, _, target, treatments, roles, value = aliased_graph()
    value.nodes[0].column_ids = []
    value.nodes[0].label = treatments.treatments[0].variable.label
    value.edges[0].source = "undeclared_c_alias"
    normalized = domain.canonical_graph(value, target, treatments, roles)
    assert normalized.nodes[0] == value.nodes[0]
    assert normalized.edges[0].source == "undeclared_c_alias"
    assert failures(normalized, context, prior(target, treatments, roles))


def test_graph_alias_cannot_collapse_distinct_derived_measurements_of_one_column():
    _, ids, target, treatments, roles, value = aliased_graph()
    derived = roles.assignments[0].model_copy(deep=True)
    derived.variable = derived.variable.model_copy(update={"id": "derived_c", "derivation": "Square of the same measurement"})
    roles.assignments.append(derived)
    with pytest.raises(ValueError, match="exact column binding to one accepted variable"):
        domain.canonical_graph(value, target, treatments, roles)
    # An explicit canonical identity still distinguishes the original measurement.
    value.nodes[-1].id = ids["c"]
    normalized = domain.canonical_graph(value, target, treatments, roles)
    assert normalized.nodes[-1].id == ids["c"]


@pytest.mark.parametrize("defect", ["missing_ledger", "unsupported_reference", "conflicting_ledger", "cycle"])
def test_demoted_claim_still_requires_valid_explicit_provenance(defect):
    context, _, target, treatments, _ = conditional_assignment()
    prerequisites = {"target": target.model_dump()}
    if defect == "missing_ledger":
        treatments.assumptions = []
    elif defect == "unsupported_reference":
        treatments.assignments[0].mechanism.references.append("invented_source")
    elif defect == "conflicting_ledger":
        prerequisites["target"]["assumptions"] = [treatments.assumptions[0].model_copy(update={"statement": "A different premise."}).model_dump()]
    else:
        treatments.assumptions[0].depends_on = ["assumption:assignment"]
    checks = domain.checks(domain.resolve(treatments, {}), context, prerequisites, scientific_version=8)
    assert any(check.rule == "evidence_authority_and_dependencies" and not check.passed for check in checks)


def test_decision_reference_alone_cannot_be_recast_as_supplied_or_assumed_evidence():
    context, _, target, treatments, _ = study()
    treatments.assignments[0].unit.references = ["decision:target"]
    resolved = domain.resolve(treatments, {})
    assert resolved.assignments[0].unit.basis == "supplied"
    assert any("direct supplied" in error for error in failures(resolved, context, {"target": target.model_dump()}))


def test_demotion_cannot_satisfy_a_supplied_only_essential_meaning():
    context, _, target, treatments, _ = conditional_assignment()
    treatments.treatments[0].variable.meaning.references.append("assumption:assignment")
    resolved = domain.resolve(treatments, {})
    assert resolved.treatments[0].variable.meaning.basis == "assumption"
    assert any("supplied meaning" in error for error in failures(resolved, context, {"target": target.model_dump()}))


@pytest.mark.parametrize("edges,adjustment,expected", [
    ({("c", "t"), ("c", "y"), ("t", "y")}, set(), False),
    ({("c", "t"), ("c", "y"), ("t", "y")}, {"c"}, True),
    ({("u", "t"), ("u", "c"), ("v", "c"), ("v", "y"), ("t", "y")}, {"c"}, False),
    ({("t", "c"), ("c", "y")}, {"c"}, False),
    ({("u", "t"), ("u", "c"), ("v", "c"), ("v", "y"), ("t", "y")}, set(), True),
    ({("u", "t"), ("u", "c"), ("v", "c"), ("v", "y"), ("c", "d")}, {"d"}, False),
])
def test_backdoor_checks_paths_colliders_and_treatment_descendants(edges, adjustment, expected):
    nodes = {node for edge in edges for node in edge}
    assert domain.backdoor_valid(nodes, edges, {"t"}, "y", adjustment) is expected


def test_joint_treatments_need_adjustment_for_each_factor():
    nodes = {"a", "b", "c", "d", "y"}
    edges = {("c", "a"), ("c", "y"), ("d", "b"), ("d", "y"), ("a", "y"), ("b", "y")}
    assert not domain.backdoor_valid(nodes, edges, {"a", "b"}, "y", {"c"})
    assert domain.backdoor_valid(nodes, edges, {"a", "b"}, "y", {"c", "d"})


def source_assignment_graph():
    context, ids, target, treatments, roles = study()
    context.dataset.study_process["assignment"] = DatasetStatement(
        value="The supplied assignment mechanism.", status="reviewed",
        source={"kind": "review", "id": uuid4(), "author_id": uuid4()},
    )
    treatments.assignments[0].mechanism.references = ["dataset.study_process.assignment"]
    value = graph(ids, target, treatments, roles, [("t", "y"), ("c", "y")], strategy="source_assignment")
    value.assumptions = [Assumption(id="fidelity", statement="The supplied assignment operated faithfully.",
        rationale="Explicit conditional premise, not a supplied fact.", affects=["primary"])]
    value.identifications[0].evidence.basis = "assumption"
    value.identifications[0].evidence.references = ["dataset.study_process.assignment", "assumption:fidelity"]
    return context, ids, target, treatments, roles, value


def test_conditional_source_assignment_preserves_assumption_with_exact_supplied_process_link():
    context, _, target, treatments, roles, value = source_assignment_graph()
    original = value.model_dump()
    assert failures(value, context, prior(target, treatments, roles)) == []
    assert value.model_dump() == original
    assert value.identifications[0].evidence.basis == "assumption"
    assert value.assumptions[0].id == "fidelity"
    # A directly supplied account remains supported when the same source is in the question.
    treatments.assignments[0].mechanism.references = ["question"]
    value.identifications[0].evidence.basis = "supplied"
    value.identifications[0].evidence.references = ["question"]
    assert failures(value, context, prior(target, treatments, roles)) == []


@pytest.mark.parametrize("defect", ["unrelated_source", "indirect_source", "unknown_mechanism", "assumed_mechanism", "unknown_process", "unsupported_process", "unsupported_mechanism_source", "missing_factor_coverage", "duplicate_factor_coverage", "missing_ledger", "open_confounding_path"])
def test_conditional_source_assignment_rejects_missing_or_unsupported_scientific_dependencies(defect):
    context, ids, target, treatments, roles, value = source_assignment_graph()
    if defect == "unrelated_source":
        value.identifications[0].evidence.references = ["question", "assumption:fidelity"]
    elif defect == "indirect_source":
        value.assumptions[0].depends_on = ["dataset.study_process.assignment"]
        value.identifications[0].evidence.references = ["assumption:fidelity"]
    elif defect == "unknown_mechanism":
        treatments.assignments[0].mechanism.basis = "unknown"
    elif defect == "assumed_mechanism":
        treatments.assignments[0].mechanism.basis = "assumption"
    elif defect == "unknown_process":
        treatments.assignments[0].kind = "unknown"
    elif defect == "unsupported_process":
        treatments.assignments[0].kind = "dynamic_regime"
    elif defect == "unsupported_mechanism_source":
        treatments.assignments[0].mechanism.references = ["invented_source"]
    elif defect == "missing_factor_coverage":
        treatments.assignments = []
    elif defect == "duplicate_factor_coverage":
        treatments.assignments.append(treatments.assignments[0].model_copy(update={"id": "duplicate"}))
    elif defect == "missing_ledger":
        value.assumptions = []
    else:
        value.edges.append(value.edges[1].model_copy(update={"target": ids["t"]}))
    assert failures(value, context, prior(target, treatments, roles))


@pytest.mark.parametrize("defect", [None, "missing_second_process", "only_other_process_source"])
def test_source_assignment_requires_coverage_and_direct_evidence_for_every_selected_process(defect):
    context, ids, target, treatments, roles, value = source_assignment_graph()
    second = treatments.treatments[0].model_copy(deep=True)
    second.variable = roles.assignments[0].variable
    treatments.treatments.append(second)
    roles.assignments = []
    treatments.effects[0].treatment_ids.append(ids["c"])
    process = treatments.assignments[0].model_copy(deep=True)
    process.id = "second_assignment"
    process.treatment_ids = [ids["c"]]
    process.mechanism.references = ["question"]
    treatments.assignments.append(process)
    value = domain.canonical_graph(value, target, treatments, roles)
    value.identifications[0].evidence.references.append("question")
    if defect == "missing_second_process":
        treatments.assignments.pop()
    elif defect == "only_other_process_source":
        value.identifications[0].evidence.references.remove("dataset.study_process.assignment")
    errors = failures(value, context, prior(target, treatments, roles))
    assert bool(errors) is (defect is not None)


@pytest.mark.parametrize("kind", ["switching", "dynamic_regime"])
def test_unimplemented_changing_regime_is_a_visible_stop(kind):
    context, _, target, treatments, _ = study()
    treatments.assignments[0].kind = kind
    assert any("not an implemented parity" in error for error in failures(treatments, context, {"target": target.model_dump()}))
    treatments.status = "unsupported"
    assert not any("not an implemented parity" in error for error in failures(treatments, context, {"target": target.model_dump()}))


def test_assignment_process_must_cover_factors_without_duplication():
    context, _, target, treatments, _ = study()
    assert failures(treatments, context, {"target": target.model_dump()}) == []
    treatments.assignments.append(treatments.assignments[0].model_copy(update={"id": "duplicate"}))
    assert any("partition" in error for error in failures(treatments, context, {"target": target.model_dump()}))
    treatments.assignments = []
    assert any("linked assignment" in error for error in failures(treatments, context, {"target": target.model_dump()}))


def test_history_registry_closes_at_aggregate_roles_without_blocking_other_column_tasks():
    context, ids, target, treatments, roles = study(("t", "y", "c", "history"))
    treatments.assignments[0].history_variables = [ids["history"]]
    local = roles.model_copy(deep=True)
    local.assignments = [item for item in local.assignments if item.variable.id == ids["c"]]
    local_context = context.model_copy(deep=True)
    local_context.columns.pop(ids["history"])
    local_checks = domain.checks(local, local_context, prior(target, treatments, local),
        role_columns={ids[name] for name in ("t", "y", "c")}, scientific_version=8)
    assert all(check.passed for check in local_checks)
    assert failures(roles, context, prior(target, treatments, roles)) == []
    # The aggregate gate still rejects an omitted or invented history identity.
    assert any("unbound history" in error for error in failures(local, context, prior(target, treatments, local)))
    treatments.assignments[0].history_variables = ["invented_history"]
    assert any("unbound history" in error for error in failures(roles, context, prior(target, treatments, roles)))


def requested_target(target):
    target.requested_effects = [RequestedEffect(id="primary", question=claim("The supplied causal effect question."), priority="primary")]
    return target


def failures_v9(value, context, prerequisites):
    return [check.detail for check in domain.checks(value, context, prerequisites, scientific_version=9) if not check.passed]


def test_v9_freezes_requested_effects_before_treatment_binding_but_keeps_historical_reads():
    context, _, target, treatments, _ = study()
    assert failures(target, context, {}) == []
    assert any("requested effect identities" in error for error in failures_v9(target, context, {}))
    requested_target(target)
    assert failures_v9(target, context, {}) == []
    assert failures_v9(treatments, context, {"target": target.model_dump()}) == []


@pytest.mark.parametrize("defect", ["duplicate", "no_primary", "two_primaries", "unknown_question", "assumed_question"])
def test_v9_target_requires_unique_supplied_questions_and_exactly_one_primary(defect):
    context, _, target, _, _ = study()
    requested_target(target)
    first = target.requested_effects[0]
    if defect == "duplicate":
        target.requested_effects.append(first.model_copy(update={"priority": "secondary"}, deep=True))
    elif defect == "no_primary":
        first.priority = "secondary"
    elif defect == "two_primaries":
        target.requested_effects.append(first.model_copy(update={"id": "another"}, deep=True))
    elif defect == "unknown_question":
        first.question.basis = "unknown"
    else:
        first.question.basis = "assumption"
        first.question.references = ["assumption:requested"]
        target.assumptions = [Assumption(id="requested", statement="Invented user request.", rationale="Not authorized.", affects=["primary"])]
    assert any("requested effect identities" in error or "supplied meaning" in error for error in failures_v9(target, context, {}))


@pytest.mark.parametrize("defect", ["renamed", "added_auxiliary", "omitted", "priority_swap"])
def test_v9_treatments_cannot_change_the_accepted_requested_effects(defect):
    context, _, target, treatments, _ = study()
    requested_target(target)
    target.requested_effects.append(RequestedEffect(id="secondary", question=claim("A second supplied effect."), priority="secondary"))
    treatments.effects.append(treatments.effects[0].model_copy(update={"id": "secondary", "priority": "secondary"}, deep=True))
    assert failures_v9(treatments, context, {"target": target.model_dump()}) == []
    if defect == "renamed":
        treatments.effects[1].id = "changed"
    elif defect == "added_auxiliary":
        treatments.effects.append(treatments.effects[1].model_copy(update={"id": "treatment_jump"}, deep=True))
    elif defect == "omitted":
        treatments.effects.pop()
    else:
        treatments.effects[0].priority, treatments.effects[1].priority = "secondary", "primary"
    assert any("Preserve the accepted target" in error for error in failures_v9(treatments, context, {"target": target.model_dump()}))


@pytest.mark.parametrize("role_names", [["identifier"], ["time"], ["identifier", "irrelevant"], ["time", "irrelevant"]])
def test_single_effect_operational_scope_is_derived_without_inventing_timing(role_names):
    context, ids, target, treatments, roles = study()
    item = roles.assignments[0]
    item.roles = role_names
    item.operational_use = ["time_key" if "time" in role_names else "unit_key"]
    item.effect_ids, item.treatment_ids, item.time_scope = [], [], ""
    item.timing = "unknown"
    item.timing_evidence.basis = "unknown"
    original = roles.model_dump()
    normalized = domain.normalize_role_scope(roles, prior(target, treatments, roles))
    bound = normalized.assignments[0]
    assert bound.effect_ids == ["primary"] and bound.treatment_ids == [ids["t"]]
    assert bound.time_scope == "Operational metadata for the bound effect"
    assert bound.timing == "unknown" and bound.timing_evidence == item.timing_evidence
    assert roles.model_dump() == original
    assert domain.normalize_role_scope(normalized, prior(target, treatments, normalized)) == normalized
    assert failures_v9(normalized, context, prior(target, treatments, normalized)) == []


@pytest.mark.parametrize("ambiguity", ["causal_role", "no_operational_use", "multiple_effects", "conflicting_effect", "conflicting_treatment"])
def test_operational_scope_normalizer_does_not_fill_semantic_or_ambiguous_scopes(ambiguity):
    _, _, target, treatments, roles = study()
    item = roles.assignments[0]
    item.roles = ["identifier"]
    item.operational_use = ["unit_key"]
    item.effect_ids, item.treatment_ids, item.time_scope = [], [], ""
    if ambiguity == "causal_role":
        item.roles.append("confounder")
    elif ambiguity == "no_operational_use":
        item.operational_use = []
    elif ambiguity == "multiple_effects":
        treatments.effects.append(treatments.effects[0].model_copy(update={"id": "secondary", "priority": "secondary"}, deep=True))
    elif ambiguity == "conflicting_effect":
        item.effect_ids = ["other"]
    else:
        item.treatment_ids = ["other"]
    assert domain.normalize_role_scope(roles, prior(target, treatments, roles)) == roles


def test_graph_nodes_are_derived_and_identification_is_conditional_and_enforced():
    context, ids, target, treatments, roles = study()
    value = graph(ids, target, treatments, roles, [("c", "t"), ("c", "y"), ("t", "y")], ["c"])
    assert all(node.column_ids == [node.id] and node.label for node in value.nodes)
    assert failures(value, context, prior(target, treatments, roles)) == []
    value.identifications[0].adjustment = []
    assert any("backdoor criterion" in error for error in failures(value, context, prior(target, treatments, roles)))


def test_role_scope_cannot_reference_an_unrequested_effect_or_treatment():
    context, _, target, treatments, roles = study()
    assert failures(roles, context, prior(target, treatments, roles)) == []
    roles.assignments[0].effect_ids = ["invented"]
    assert any("effect identities" in error for error in failures(roles, context, prior(target, treatments, roles)))


def test_irrelevant_label_does_not_remove_scientific_scope_obligations():
    context, ids, target, treatments, roles = study()
    item = roles.assignments[0]
    item.roles = ["pre_outcome", "irrelevant"]
    item.effect_ids, item.treatment_ids = [], []
    item.time_scope = "Earlier outcome period before treatment"
    errors = failures(roles, context, prior(target, treatments, roles))
    assert len(errors) == 1
    assert ids["c"] in errors[0]
    assert "missing required scope fields: effect_ids, treatment_ids." in errors[0]
    assert "time_scope" not in errors[0]
    assert domain.normalize_role_scope(roles, prior(target, treatments, roles)) == roles


def test_v8_access_rederives_source_lineage_instead_of_trusting_legacy_proxy_flag():
    context, ids, target, treatments, roles = study()
    context.relationships["source"] = relation(ids["y"], ids["c"])
    roles.assignments[0].outcome_proxy = False
    frame = Frame(target=target, treatments=treatments, roles=roles,
        causal=CausalAccount(**decision(), identification="Fixture"), graph_hash="frozen")
    raw, manifest = domain.projection(frame, context, b"t,y,c\n1,999,888\n", scientific_version=8)
    assert manifest["columns"] == [ids["t"]]
    assert list(csv.reader(io.StringIO(raw.decode()))) == [[ids["t"]], ["1"]]


def test_pre_treatment_scope_rejects_descendant_even_if_role_label_is_not_confounder():
    context, ids, target, treatments, roles = study()
    roles.assignments[0].roles = ["effect_modifier"]
    value = graph(ids, target, treatments, roles, [("t", "c"), ("c", "y")], strategy="other")
    assert any("pre-treatment measurement" in error for error in failures(value, context, prior(target, treatments, roles)))


def relation(source, destination, **extra):
    return Relationship(value="Sourced relation", kind="derivation", status="reviewed",
        source={"kind": "review", "id": uuid4(), "author_id": uuid4()},
        participants=[{"kind": "column", "id": source, "role": "from"}, {"kind": "column", "id": destination, "role": "to"}], **extra)


@pytest.mark.parametrize("qualification", [{"conditions": "Only after disease onset"}, {"timing": "Follow-up only"}, {"qualifications": "Valid only in a qualified subpopulation"}])
def test_qualified_lineage_is_withheld_without_becoming_an_unconditional_proxy(qualification):
    context, ids, target, _, roles = study()
    context.relationships["qualified"] = relation(ids["y"], ids["c"], **qualification)
    result = lineage.classify(roles, context, target)
    assert result.assignments[0].outcome_proxy is None
    assert result.assignments[0].proxy_evidence.references == ["relationships.qualified"]


def test_multi_party_lineage_is_not_cartesian_pairwise_causation():
    context, ids, target, _, roles = study()
    source = relation(ids["y"], ids["c"])
    source.participants.append(Participant(kind="column", id=ids["t"], role="from"))
    context.relationships["combined"] = Relationship.model_validate(source.model_dump())
    assert lineage.classify(roles, context, target).assignments[0].outcome_proxy is None


def test_source_absence_scope_preserves_indirect_path_distinction():
    context, ids, target, treatments, roles = study()
    roles.assignments[0].roles = ["mediator"]
    roles.assignments[0].timing = "after_treatment"
    value = graph(ids, target, treatments, roles, [("t", "c"), ("c", "y")], strategy="other")
    value.identifications[0].obligations = [IdentificationObligation(name="identification", claim=claim())]
    source = relation(ids["t"], ids["y"])
    source.kind = "no_causal_influence"
    source.absence_scope = "direct_effect"
    context.relationships["absence"] = source
    value = CausalAccount.model_validate({**value.model_dump(), "relationships": [{
        "reference": "relationships.absence", "effect_ids": ["primary"], "disposition": "applicable", "source_id": ids["t"], "target_id": ids["y"],
        "rationale": {"statement": "No direct edge", "basis": "supplied", "references": ["relationships.absence"]},
    }]})
    assert failures(value, context, prior(target, treatments, roles)) == []
    source.absence_scope = "any_pathway"
    assert any("every causal pathway" in error for error in failures(value, context, prior(target, treatments, roles)))


def test_authority_demotion_cannot_override_an_applicable_source_contradiction():
    context, ids, target, treatments, roles = study()
    value = graph(ids, target, treatments, roles, [("t", "y")])
    source = relation(ids["t"], ids["y"])
    source.kind = "no_causal_influence"
    source.absence_scope = "any_pathway"
    context.relationships["absence"] = source
    value.assumptions = [Assumption(id="causal_edge", statement="Treatment affects the outcome.",
        rationale="Proposed causal premise.", affects=["primary"])]
    value.edges[0].references.append("assumption:causal_edge")
    value = CausalAccount.model_validate({**value.model_dump(), "relationships": [{
        "reference": "relationships.absence", "effect_ids": ["primary"], "disposition": "applicable", "source_id": ids["t"], "target_id": ids["y"],
        "rationale": {"statement": "The supplied absence applies.", "basis": "supplied", "references": ["relationships.absence"]},
    }]})
    resolved = domain.resolve(value, {})
    assert resolved.edges[0].basis == "assumption"
    assert any("every causal pathway" in error for error in failures(resolved, context, prior(target, treatments, roles)))


def accepted_relationship_fixture():
    context, ids, target, treatments, roles = study()
    source = relation(ids["t"], ids["y"])
    source.kind = "causal_influence"
    source.directness = "direct"
    context.relationships["shared"] = source
    value = graph(ids, target, treatments, roles, [("t", "y")])
    record = {**decision(), "relationships": [{
        "reference": "relationships.shared", "effect_ids": ["primary"], "disposition": "applicable",
        "source_id": ids["t"], "target_id": ids["y"], "scope": "Accepted target and treatment window",
        "rationale": {"basis": "assumption", "statement": "This qualified relationship applies under the declared scope.",
                      "references": ["relationships.shared", "assumption:interpretation"]},
    }], "assumptions": [
        {"id": "interpretation", "statement": "The source interpretation applies.", "rationale": "Explicit interpretation.", "affects": ["primary"], "depends_on": ["assumption:scope"]},
        {"id": "scope", "statement": "The source and target scopes agree.", "rationale": "Explicit scope premise.", "affects": ["primary"], "depends_on": ["question"]},
        {"id": "unused", "statement": "Unrelated premise.", "rationale": "Not needed here.", "affects": ["other"], "depends_on": []},
    ]}
    return context, target, treatments, roles, value, record


def test_causal_assembly_adopts_exact_shared_interpretations_and_dependency_closed_ledger():
    context, target, treatments, roles, value, record = accepted_relationship_fixture()
    original = value.model_dump()
    normalized = domain.adopt_relationships(value, [record])
    assert normalized.relationships[0] == RelationshipDisposition.model_validate(record["relationships"][0])
    assert {item.id for item in normalized.assumptions} == {"interpretation", "scope"}
    assert value.model_dump() == original
    assert domain.adopt_relationships(normalized, [record]) == normalized
    assert failures_v9(normalized, context, prior(target, treatments, roles)) == []


def test_causal_assembly_with_no_shared_relationships_preserves_the_proposal():
    _, _, _, _, value, _ = accepted_relationship_fixture()
    assert domain.adopt_relationships(value, []) == value


@pytest.mark.parametrize("kind", ["association", "membership", "unclassified"])
def test_non_directed_relationships_do_not_force_endpoints_or_causal_edges(kind):
    context, target, treatments, roles, value, record = accepted_relationship_fixture()
    source = context.relationships["shared"]
    source.kind = kind
    for participant in source.participants:
        participant.role = "participant"
    record["relationships"][0].update(source_id="", target_id="")
    normalized = domain.adopt_relationships(value, [record])
    normalized.edges = []
    domain.validate_relationships(normalized.relationships, context, {"primary"})
    assert failures_v9(normalized, context, prior(target, treatments, roles)) == []
    normalized.relationships[0].source_id = source.participants[0].id
    normalized.relationships[0].target_id = source.participants[1].id
    with pytest.raises(ValueError, match="does not define directed"):
        domain.validate_relationships(normalized.relationships, context, {"primary"})


@pytest.mark.parametrize("defect", ["foreign_effect", "foreign_endpoint", "reversed_endpoints", "missing_scope", "missing_source_citation"])
def test_relationship_worker_can_reject_structural_errors_before_graph_authoring(defect):
    context, _, _, _, value, record = accepted_relationship_fixture()
    normalized = domain.adopt_relationships(value, [record])
    item = normalized.relationships[0]
    if defect == "foreign_effect":
        item.effect_ids = ["not_requested"]
    elif defect == "foreign_endpoint":
        item.target_id = "invented_column"
    elif defect == "reversed_endpoints":
        item.source_id, item.target_id = item.target_id, item.source_id
    elif defect == "missing_scope":
        context.relationships["shared"].qualifications = "Applies only in a specified subgroup."
        item.scope = ""
    else:
        item.rationale.references = ["assumption:interpretation"]
    with pytest.raises(ValueError):
        domain.validate_relationships(normalized.relationships, context, {"primary"})


@pytest.mark.parametrize("field,new", [
    ("disposition", "not_applicable"), ("source_id", "different"), ("target_id", "different"),
    ("effect_ids", ["different"]), ("scope", "A different time window"), ("reference", "relationships.new"),
])
def test_causal_proposal_cannot_reinterpret_an_accepted_relationship(field, new):
    _, _, _, _, value, record = accepted_relationship_fixture()
    normalized = domain.adopt_relationships(value, [record])
    normalized.relationships[0] = normalized.relationships[0].model_copy(update={field: new})
    with pytest.raises(ValueError, match="cannot add, duplicate or change"):
        domain.adopt_relationships(normalized, [record])


def test_causal_assembly_rejects_conflicting_accepted_or_model_assumption_definitions():
    _, _, _, _, value, record = accepted_relationship_fixture()
    value.assumptions = [Assumption(**{**record["assumptions"][0], "statement": "Changed premise."})]
    with pytest.raises(ValueError, match="redefine an accepted assumption"):
        domain.adopt_relationships(value, [record])
    value.assumptions = []
    contradictory = {**record, "assumptions": [{**record["assumptions"][0], "statement": "Changed premise."}]}
    with pytest.raises(ValueError, match="conflicting assumption definitions"):
        domain.adopt_relationships(value, [record, contradictory])


def test_causal_assembly_does_not_bypass_missing_assumptions_or_supplied_relationships():
    context, target, treatments, roles, value, record = accepted_relationship_fixture()
    missing = {**record, "assumptions": []}
    assert any("outside this task" in error for error in failures_v9(domain.adopt_relationships(value, [missing]), context, prior(target, treatments, roles)))
    normalized = domain.adopt_relationships(value, [record])
    normalized.edges = []
    assert any("omits an applicable supplied direct" in error for error in failures_v9(normalized, context, prior(target, treatments, roles)))


def test_relevant_unresolved_relationship_is_preserved_as_a_stop():
    _, _, _, _, value, record = accepted_relationship_fixture()
    record["relationships"][0]["disposition"] = "unresolved"
    with pytest.raises(ValueError, match="unresolved relationship"):
        domain.adopt_relationships(value, [record])
    value.status = "needs_input"
    value.unresolved = ["Which time scope does the source relationship describe?"]
    assert domain.adopt_relationships(value, [record]).relationships[0].disposition == "unresolved"


def test_operational_identifiers_survive_as_grouping_tokens_but_outcome_lineage_does_not():
    context, ids, target, treatments, roles = study()
    key = roles.assignments[0]
    key.roles = ["identifier"]
    key.timing = "unknown"
    key.operational_use = ["unit_key"]
    frame = Frame(target=target, treatments=treatments, roles=roles,
        causal=CausalAccount(**decision(), identification="Fixture"), graph_hash="frozen")
    raw, manifest = domain.projection(frame, context, b"t,y,c\n1,999,alice\n0,888,alice\n1,777,bob\n", purpose="unit_structure")
    rows = list(csv.reader(io.StringIO(raw.decode())))
    assert rows == [[ids["c"]], ["unit_0"], ["unit_0"], ["unit_1"]]
    assert manifest["pseudonymized_group_keys"] == [ids["c"]]
    key.outcome_proxy = True
    with pytest.raises(ValueError, match="No safe"):
        domain.projection(frame, context, b"t,y,c\n1,999,alice\n", purpose="unit_structure")


def plan(kind="estimate"):
    return Plan(**decision(), effects=[{"effect_id": "primary", "estimator": "Specified Python", "adjustment_reason": "Specified", "inference": "Specified", "result_kind": kind, "result_reason": "Prespecified result shape"}],
        population="Specified", missingness="Specified", exclusions="Specified", multiplicity="One", reporting="All", limitations=["Conditional"])


@pytest.mark.parametrize("kind,facts", [
    ("estimate", {"primary_estimate": 2, "primary_ci_low": 1, "primary_ci_high": 3, "primary_n": 10}),
    ("structural_zero", {"primary_estimate": 0}),
    ("estimate_without_inference", {"primary_estimate": 2, "primary_n": 10}),
    ("confidence_set", {"primary_n": 10, "primary_confidence_status": 1, "primary_region_count": 2, "primary_region_0_low": -3, "primary_region_0_high": -1, "primary_region_1_low": 1, "primary_region_1_high": 3}),
    ("confidence_set", {"primary_n": 10, "primary_confidence_status": 2, "primary_region_count": 0}),
    ("confidence_set", {"primary_n": 10, "primary_confidence_status": 3, "primary_region_count": 0, "primary_searched_low": -10, "primary_searched_high": 10}),
])
def test_valid_result_alternatives_do_not_fabricate_ordinary_intervals(kind, facts):
    _, _, _, treatments, _ = study()
    assert domain.execution_error(facts, treatments, plan(kind)) == ""


def test_result_shape_cannot_change_after_freeze_or_hide_nonfinite_values():
    _, _, _, treatments, _ = study()
    assert domain.execution_error({"primary_estimate": 0}, treatments)
    assert domain.execution_error({"primary_estimate": float("nan"), "primary_n": 10}, treatments, plan("estimate_without_inference"))
    assert domain.execution_error({"primary_n": 10, "primary_confidence_status": 1, "primary_region_count": 1}, treatments, plan("confidence_set"))


def longitudinal_study():
    context, ids, target, treatments, roles = study(("t", "y", "unit", "time", "adoption", "proxy"))
    for role in roles.assignments:
        name = role.variable.label
        role.roles = ["time"] if name in {"time", "adoption"} else ["identifier"] if name == "unit" else ["unknown"]
        role.timing = "unknown"
        role.operational_use = {"unit": ["unit_key"], "time": ["time_key"], "adoption": ["assignment_key"], "proxy": []}[name]
        role.outcome_proxy = name == "proxy"
    treatments.assignments[0].kind = "absorbing_adoption"
    treatments.assignments[0].diagnostic_window = DiagnosticWindow(
        unit_variable_id=ids["unit"], time_variable_id=ids["time"], cutoff=2,
        assignment_encoding="indicator", treated_values=["1"], control_values=["0"],
        adoption_variable_id=ids["adoption"], never_adoption_values=["never"], evidence=claim(), history_evidence=claim(),
    )
    frame = Frame(target=target, treatments=treatments, roles=roles,
        causal=CausalAccount(**decision(), identification="DiD fixture"), graph_hash="frozen")
    raw = b"t,y,unit,time,adoption,proxy\n0,11,a,1,2,901\n1,92,a,2,2,902\n0,21,b,1,never,903\n0,94,b,2,never,904\n"
    return context, ids, frame, raw


def test_longitudinal_projection_masks_target_periods_for_treated_and_never_treated():
    context, ids, frame, raw = longitudinal_study()
    projected, manifest = domain.projection(frame, context, raw, purpose="adoption_structure")
    records = list(csv.DictReader(io.StringIO(projected.decode())))
    assert [row[ids["y"]] for row in records] == ["11", "", "21", ""]
    assert len(records) == 4 and ids["proxy"] not in records[0]
    assert [row[ids["unit"]] for row in records] == ["unit_0", "unit_0", "unit_1", "unit_1"]
    assert manifest["preperiod_window"]["visible_rows"] == 2
    changed = raw.replace(b",92,", b",992,").replace(b",94,", b",994,").replace(b",904", b",999")
    assert domain.projection(frame, context, changed, purpose="adoption_structure")[0] == projected


@pytest.mark.parametrize("defect", ["late_cutoff", "switching", "non_numeric_time", "duplicate_period", "unknown_never_code", "contradictory_adoption", "outcome_as_time"])
def test_longitudinal_projection_blocks_unverified_or_unsafe_period_mapping(defect):
    context, ids, frame, raw = longitudinal_study()
    window = frame.treatments.assignments[0].diagnostic_window
    assert window is not None
    if defect == "late_cutoff":
        window.cutoff = 3
    elif defect == "switching":
        raw += b"0,99,a,3,2,999\n"
    elif defect == "non_numeric_time":
        raw = raw.replace(b",a,1,", b",a,yesterday,")
    elif defect == "duplicate_period":
        raw += b"0,99,a,1,2,999\n"
    elif defect == "unknown_never_code":
        window.never_adoption_values = []
    elif defect == "contradictory_adoption":
        raw = raw.replace(b",a,1,2,", b",a,1,3,").replace(b",a,2,2,", b",a,2,3,")
    else:
        window.time_variable_id = ids["y"]
    with pytest.raises(ValueError):
        domain.projection(frame, context, raw, purpose="adoption_structure")
