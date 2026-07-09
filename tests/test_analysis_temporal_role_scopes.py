"""Finite, explicitly dated adoption events and inherited aggregate obligations."""

import pytest
from test_analysis_role_scopes import scoped_roles
from test_scientific_v8 import claim, decision, prior, study

from backend.analysis import domain
from backend.analysis.contracts import (
    CausalAccount,
    ContrastBinding,
    EffectRequest,
    EffectWeight,
    Frame,
    Roles,
    TreatmentValue,
)


@pytest.fixture(autouse=True)
def isolated():
    yield


def adoption_effects():
    context, ids, target, treatments, roles = study(("t", "y", "x", "cohort", "time", "later"))
    process = treatments.assignments[0]
    process.kind = "absorbing_adoption"
    process.history_variables = [ids["cohort"], ids["time"]]
    treatments.values = [TreatmentValue(id=key, variable_id=ids["cohort"], value={"kind": "text", "literal": literal}, meaning=claim())
                         for key, literal in [("g2", "early_cohort"), ("g3", "later_cohort"), ("never", "never_cohort")]]
    treatments.effects = []
    for identity, cohort, adoption, period, baseline in [
        ("g2_t2", "g2", "2", "2", "1"), ("g2_t3", "g2", "2", "3", "1"),
        ("g3_t3", "g3", "3", "3", "2"),
    ]:
        treatments.effects.append(EffectRequest(id=identity, treatment_ids=[ids["t"]], contrast=claim(),
            estimand="Group-time ATT", priority="secondary", comparison=ContrastBinding(kind="group_time", evidence=claim(),
                group_time={"cohort_value_id": cohort, "comparison_value_ids": ["never"], "time_variable_id": ids["time"],
                            "adoption_period": {"kind": "number", "literal": adoption},
                            "period": {"kind": "number", "literal": period}, "baseline": {"kind": "number", "literal": baseline}})))
    treatments.effects.append(EffectRequest(id="average", treatment_ids=[ids["t"]], contrast=claim(), estimand="Declared average", priority="primary",
        comparison=ContrastBinding(kind="aggregate", evidence=claim(), aggregate={"terms": [
            {"effect_id": effect.id, "weight": 1 / 3} for effect in treatments.effects]})))
    return context, ids, target, treatments, roles


def test_shared_cohort_event_groups_effects_but_aggregate_owns_each_distinct_event():
    context, ids, target, treatments, _ = adoption_effects()
    scopes = [item for item in domain.role_scopes(context, target, treatments) if item["column_id"] == ids["x"]]
    assert len(scopes) == 2
    effect_sets = {frozenset(scope["effect_treatments"]) for scope in scopes}
    assert effect_sets == {frozenset({"g2_t2", "g2_t3", "average"}), frozenset({"g3_t3", "average"})}
    assert len({key for scope in scopes for key in scope["obligation_keys"]}) == 5
    assert len([key for scope in scopes for key in scope["obligation_keys"] if ":average:" in key]) == 2
    assert all(scope["event"]["group_time"]["time_variable_id"] == ids["time"] for scope in scopes)


def test_event_identity_ignores_value_aliases_prose_names_order_and_numeric_spelling():
    context, _, target, treatments, _ = adoption_effects()
    before = domain.role_scopes(context, target, treatments)
    for value in treatments.values:
        value.id = "renamed_" + value.id
        value.meaning.statement = "The same accepted value described differently."
    treatments.values.reverse()
    for effect in treatments.effects:
        if effect.comparison.group_time:
            group = effect.comparison.group_time
            group.cohort_value_id = "renamed_" + group.cohort_value_id
            group.comparison_value_ids = ["renamed_" + identity for identity in group.comparison_value_ids]
            group.adoption_period.literal += ".00"
    treatments.effects.reverse()
    treatments.assignments[0].id = "new_process_alias"
    treatments.assignments[0].timing.statement = "Changed prose does not create new event credit."
    context.columns = dict(reversed(list(context.columns.items())))
    for column in context.columns.values():
        column.identity.name = "opaque_" + column.identity.column_id
    assert domain.role_scopes(context, target, treatments) == before


def test_outcome_period_and_baseline_do_not_create_extra_assignment_events():
    context, _, target, treatments, _ = adoption_effects()
    before = domain.role_scopes(context, target, treatments)
    treatments.effects[1].comparison.group_time.period.literal = "4"
    treatments.effects[1].comparison.group_time.baseline.literal = "0"
    assert domain.role_scopes(context, target, treatments) == before


def test_cohort_literal_and_explicit_adoption_date_both_define_event_identity():
    context, ids, target, treatments, _ = adoption_effects()
    before = {scope["key"] for scope in domain.role_scopes(context, target, treatments) if scope["column_id"] == ids["x"]}
    changed_date = treatments.model_copy(deep=True)
    changed_date.effects[2].comparison.group_time.adoption_period.literal = "2"
    after = {scope["key"] for scope in domain.role_scopes(context, target, changed_date) if scope["column_id"] == ids["x"]}
    assert len(after) == 2 and len(before & after) == 1
    changed_cohort = treatments.model_copy(deep=True)
    changed_cohort.values[1].value.literal = "another_later_cohort"
    after = {scope["key"] for scope in domain.role_scopes(context, target, changed_cohort) if scope["column_id"] == ids["x"]}
    assert len(after) == 2 and len(before & after) == 1


def test_earlier_and_later_selected_factor_roles_remain_distinct_for_component_events():
    context, ids, target, treatments, roles = adoption_effects()
    later = treatments.treatments[0].model_copy(deep=True)
    later.variable = next(role.variable for role in roles.assignments if role.variable.id == ids["later"])
    treatments.treatments.append(later)
    process = treatments.assignments[0].model_copy(deep=True)
    process.id, process.kind, process.treatment_ids, process.history_variables = "later", "point", [ids["later"]], []
    treatments.assignments.append(process)
    treatments.effects.append(EffectRequest(id="later_effect", treatment_ids=[ids["later"]], contrast=claim(), estimand="Later point effect", priority="secondary"))
    aggregate = next(effect for effect in treatments.effects if effect.id == "average")
    aggregate.treatment_ids.append(ids["later"])
    aggregate.comparison.aggregate.terms.append(EffectWeight(effect_id="later_effect", weight=.25))
    for term in next(effect for effect in treatments.effects if effect.id == "average").comparison.aggregate.terms:
        term.weight = .25
    scopes = domain.role_scopes(context, target, treatments)
    earlier = [scope for scope in scopes if scope["column_id"] == ids["t"]]
    later_scopes = [scope for scope in scopes if scope["column_id"] == ids["later"]]
    assert len(earlier) == 1 and earlier[0]["event"]["kind"] == "point"
    assert earlier[0]["effect_treatments"] == {"average": [ids["later"]], "later_effect": [ids["later"]]}
    assert len(later_scopes) == 2 and all(scope["event"]["kind"] == "absorbing_adoption" for scope in later_scopes)
    assert all(scope["effect_treatments"]["average"] == [ids["t"]] for scope in later_scopes)


@pytest.mark.parametrize("defect", ["unknown_component", "cycle", "scope_change", "unknown_cohort"])
def test_role_event_closure_refuses_unknown_or_inconsistent_frozen_dependencies(defect):
    context, ids, target, treatments, _ = adoption_effects()
    aggregate = treatments.effects[-1]
    if defect == "unknown_component":
        aggregate.comparison.aggregate.terms[0].effect_id = "missing"
    elif defect == "cycle":
        aggregate.comparison.aggregate.terms[0].effect_id = "average"
    elif defect == "scope_change":
        aggregate.treatment_ids = [ids["later"]]
    else:
        treatments.effects[0].comparison.group_time.cohort_value_id = "missing"
    with pytest.raises(ValueError):
        domain.role_scopes(context, target, treatments)


def assembled_roles(context, target, treatments, original):
    return Roles(**decision(), assignments=[
        assignment for scope in domain.role_scopes(context, target, treatments)
        for assignment in scoped_roles(scope, target, treatments, original).assignments
    ])


def test_harness_stamps_exact_event_before_any_operational_normalization():
    context, ids, target, treatments, original = adoption_effects()
    scope = next(scope for scope in domain.role_scopes(context, target, treatments) if scope["column_id"] == ids["x"])
    proposal = scoped_roles(scope, target, treatments, original)
    for assignment in proposal.assignments:
        assignment.assignment_event_key = None
    raw = proposal.model_dump()
    normalized = domain.normalize_role_scope(proposal, prior(target, treatments, original), scope)
    assert proposal.model_dump() == raw
    assert all(item.assignment_event_key == domain.fingerprint(scope["event"]) for item in normalized.assignments)
    domain.validate_role_scope(normalized, scope)
    with pytest.raises(ValueError, match="harness-bound"):
        domain.validate_role_scope(proposal, scope)
    domain.validate_role_scope(proposal, scope, allow_legacy=True)
    proposal.assignments[0].assignment_event_key = "different-event"
    with pytest.raises(ValueError, match="substitute"):
        domain.normalize_role_scope(proposal, prior(target, treatments, original), scope)


def test_assembled_aggregate_keeps_distinct_event_dispositions_for_same_effect_and_factor():
    context, ids, target, treatments, original = adoption_effects()
    roles = assembled_roles(context, target, treatments, original)
    domain.validate_role_events(roles, context, target, treatments)
    average = [item for item in roles.assignments if item.variable.id == ids["x"] and item.effect_ids == ["average"]]
    assert len(average) == 2
    assert len({item.assignment_event_key for item in average}) == 2
    assert average[0].treatment_ids == average[1].treatment_ids == [ids["t"]]
    roles.assignments.remove(average[0])
    with pytest.raises(ValueError, match="omitted.*average"):
        domain.validate_role_events(roles, context, target, treatments)


@pytest.mark.parametrize("replacement", [None, "unaccepted-event"])
def test_aggregate_assembly_rejects_unstamped_or_substituted_event(replacement):
    context, _, target, treatments, original = adoption_effects()
    roles = assembled_roles(context, target, treatments, original)
    roles.assignments[0].assignment_event_key = replacement
    with pytest.raises(ValueError, match="no accepted"):
        domain.validate_role_events(roles, context, target, treatments)


def test_combined_v15_gate_requires_event_closure_but_column_tasks_defer_global_coverage():
    context, ids, target, treatments, original = adoption_effects()
    roles = assembled_roles(context, target, treatments, original)
    prerequisites = prior(target, treatments, original)
    before = domain.checks(roles, context, prerequisites, scientific_version=15)
    assert all(check.passed for check in before), [(check.rule, check.detail) for check in before if not check.passed]
    scope = next(scope for scope in domain.role_scopes(context, target, treatments) if scope["column_id"] == ids["x"])
    local = scoped_roles(scope, target, treatments, original)
    # The worker separately validates its exact owned scope. Other columns and
    # assignment history are closed only when the coordinator assembles all roles.
    local_columns = set(target.outcome.column_ids) | {ids["t"], ids["x"]}
    local_checks = domain.checks(local, context, prerequisites, scientific_version=15, role_columns=local_columns)
    assert all(check.passed for check in local_checks), [(check.rule, check.detail) for check in local_checks if not check.passed]
    missing = next(item for item in roles.assignments if item.variable.id == ids["x"] and item.effect_ids == ["average"])
    roles.assignments.remove(missing)
    combined = domain.checks(roles, context, prerequisites, scientific_version=15)
    assert any(not check.passed and check.rule == "semantic_bindings" and "omitted" in check.detail for check in combined)


def baseline_frame():
    context, ids, target, treatments, original = adoption_effects()
    roles = assembled_roles(context, target, treatments, original)
    for item in roles.assignments:
        if item.variable.id == ids["x"]:
            item.roles = ["pre_outcome"]
            item.timing = "before_treatment"
            item.timing_evidence = item.account.model_copy()
            item.outcome_proxy = False
    return context, ids, Frame(target=target, treatments=treatments, roles=roles, causal=CausalAccount(**decision(), identification="Fixture"), graph_hash="fixture")


def test_baseline_union_requires_every_aggregate_event_instead_of_collapsing_effect_factor_pairs():
    context, ids, frame = baseline_frame()
    before = domain._baseline_access(frame, context, {"average"}, scientific_version=15)
    assert ids["x"] in before
    assert len(before[ids["x"]]["effect_treatment_events"]) == 2
    assert before[ids["x"]]["effect_treatment_pairs"] == [["average", ids["t"]]]
    missing = next(item for item in frame.roles.assignments if item.variable.id == ids["x"] and item.effect_ids == ["average"])
    frame.roles.assignments.remove(missing)
    assert ids["x"] not in domain._baseline_access(frame, context, {"average"}, scientific_version=15)
    # The historical pair-only rule would grant access, proving the missing event matters.
    assert ids["x"] in domain._baseline_access(frame, context, {"average"}, scientific_version=14)


@pytest.mark.parametrize("defect", ["after", "unknown", "proxy"])
def test_event_scoped_timing_or_proxy_blocks_aggregate_without_contaminating_other_event(defect):
    context, ids, frame = baseline_frame()
    scopes = domain.role_scopes(context, frame.target, frame.treatments)
    earlier = next(scope for scope in scopes if scope["column_id"] == ids["x"] and "g2_t2" in scope["effect_treatments"])
    for item in frame.roles.assignments:
        if item.variable.id == ids["x"] and item.assignment_event_key == domain.fingerprint(earlier["event"]):
            if defect == "proxy":
                item.outcome_proxy = True
            else:
                item.timing = "after_treatment" if defect == "after" else "unknown"
    assert ids["x"] not in domain._baseline_access(frame, context, {"average"}, scientific_version=15)
    assert ids["x"] not in domain._baseline_access(frame, context, {"g2_t2"}, scientific_version=15)
    assert ids["x"] in domain._baseline_access(frame, context, {"g3_t3"}, scientific_version=15)


def test_projection_exposes_only_baselines_covering_the_requested_event():
    context, ids, frame = baseline_frame()
    raw = b"t,y,x,cohort,time,later\n0,9001,17,early_cohort,1,0\n1,9002,19,later_cohort,3,1\n"
    projected, receipt = domain.projection(frame, context, raw, effect_ids={"average"}, scientific_version=15)
    assert ids["x"] in projected.decode().splitlines()[0]
    assert "9001" not in projected.decode() and "9002" not in projected.decode()
    assert ids["x"] in receipt["columns"]
    missing = next(item for item in frame.roles.assignments if item.variable.id == ids["x"] and item.effect_ids == ["average"])
    frame.roles.assignments.remove(missing)
    projected, receipt = domain.projection(frame, context, raw, effect_ids={"average"}, scientific_version=15)
    assert ids["x"] not in receipt["columns"]
    assert "17" not in projected.decode().splitlines()[1].split(",")
