"""Frozen treatment operands distinguish levels, joint states, history and thresholds."""

from copy import deepcopy

import pytest
from test_scientific_v8 import claim, study

from backend.analysis import contrasts, domain, references
from backend.analysis.contracts import (
    ContrastBinding,
    DiagnosticWindow,
    RequestedEffect,
    TreatmentValue,
    TypedValue,
)
from backend.analysis.provider import provider_schema


@pytest.fixture(autouse=True)
def isolated():
    yield


def literal(text, kind="number"):
    return {"kind": kind, "literal": str(text)}


def level(identity, variable, text, kind="number"):
    return TreatmentValue(id=identity, variable_id=variable, value=literal(text, kind), meaning=claim())


def binding(kind, **payload):
    return ContrastBinding(kind=kind, evidence=claim(), **{kind: payload})


def fixture():
    context, ids, target, treatments, roles = study(("t", "y", "b", "cohort", "time", "score", "eligibility"))
    target.requested_effects = [RequestedEffect(id="primary", question=claim(), priority="primary")]
    treatments.values = [level("zero", ids["t"], 0), level("one", ids["t"], 1)]
    treatments.effects[0].comparison = binding("static", terms=[
        {"weight": 1, "value_ids": ["one"]}, {"weight": -1, "value_ids": ["zero"]},
    ])
    return context, ids, target, treatments, roles


def accepted(treatments, context, target):
    resolved = domain.resolve(treatments, {})
    checks = domain.checks(resolved, context, {"target": target.model_dump()}, scientific_version=15)
    assert all(check.passed for check in checks), [(check.rule, check.detail) for check in checks if not check.passed]
    return resolved


def rejected(treatments, context, target, expected):
    resolved = domain.resolve(treatments, {})
    checks = domain.checks(resolved, context, {"target": target.model_dump()}, scientific_version=15)
    assert any(not check.passed and expected in check.detail for check in checks), checks


def test_one_categorical_factor_two_requested_effects_freeze_exact_ordered_arms():
    context, ids, target, treatments, _ = fixture()
    treatments.values = [level("control", ids["t"], "No E-Mail", "text"), level("mens", ids["t"], "Mens E-Mail", "text"), level("womens", ids["t"], "Womens E-Mail", "text")]
    first = treatments.effects[0]
    first.comparison = binding("static", terms=[{"weight": 1, "value_ids": ["mens"]}, {"weight": -1, "value_ids": ["control"]}])
    second = first.model_copy(deep=True)
    second.id, second.priority = "secondary", "secondary"
    second.comparison.static.terms[0].value_ids = ["womens"]
    treatments.effects.append(second)
    target.requested_effects.append(RequestedEffect(id="secondary", question=claim(), priority="secondary"))
    raw = treatments.model_dump()
    resolved = accepted(treatments, context, target)
    assert len(resolved.treatments) == 1 and len(resolved.values) == 3
    assert [term.weight for term in resolved.effects[0].comparison.static.terms] == [1, -1]
    assert resolved.effects[0].comparison.static.terms[1].value_ids == resolved.effects[1].comparison.static.terms[1].value_ids
    assert treatments.model_dump() == raw
    assert domain.resolve(resolved, {}) == resolved


def test_numeric_identity_is_exact_without_rounding_or_text_coercion():
    def number(text):
        return TypedValue(kind="number", literal=text)
    assert contrasts.value_id("x", number("1")) == contrasts.value_id("x", number("+01.000e0"))
    assert contrasts.value_id("x", number("-0")) == contrasts.value_id("x", number("0"))
    assert contrasts.value_id("x", number("1")) != contrasts.value_id("x", TypedValue(kind="text", literal="1"))
    assert contrasts.value_id("x", number("1")) != contrasts.value_id("other", number("1"))
    assert contrasts.value_id("x", number("1.00000000000000000000000000001")) != contrasts.value_id("x", number("1"))
    assert contrasts.value_id("x", TypedValue(kind="boolean", literal="true")) != contrasts.value_id("x", number("1"))


@pytest.mark.parametrize("kind,text", [("number", "NaN"), ("number", "inf"), ("number", "1 or 2"), ("number", ""), ("boolean", "1"), ("boolean", "True")])
def test_unparseable_or_nonfinite_value_representations_fail_closed(kind, text):
    with pytest.raises(ValueError):
        contrasts.value_id("x", TypedValue(kind=kind, literal=text))


def test_value_identity_survives_local_alias_reorder_and_column_rename():
    context, ids, target, treatments, _ = fixture()
    before = accepted(treatments, context, target)
    treatments.treatments[0].variable.id = "factor_alias"
    treatments.effects[0].treatment_ids = ["factor_alias"]
    treatments.assignments[0].treatment_ids = ["factor_alias"]
    for item in treatments.values:
        item.variable_id = "factor_alias"
    treatments.values.reverse()
    context.columns[ids["t"]].identity.name = "opaque_91"
    after = accepted(treatments, context, target)
    assert {item.id for item in before.values} == {item.id for item in after.values}
    assert before.effects[0].comparison == after.effects[0].comparison


@pytest.mark.parametrize("duplicate", ["local_id", "numeric_identity", "digest_collision"])
def test_ambiguous_value_identity_rejected_without_merging(monkeypatch, duplicate):
    _, _, _, treatments, _ = fixture()
    if duplicate == "local_id":
        treatments.values[1].id = treatments.values[0].id
    elif duplicate == "numeric_identity":
        treatments.values[1].value.literal = "0.00"
    else:
        monkeypatch.setattr(contrasts, "value_id", lambda *_: "value_collision")
    with pytest.raises(ValueError, match="unique|collision"):
        domain.resolve(treatments, {})


def factorial():
    context, ids, target, treatments, roles = fixture()
    other = treatments.treatments[0].model_copy(deep=True)
    other.variable = next(role.variable for role in roles.assignments if role.variable.id == ids["b"])
    treatments.treatments.append(other)
    # The secondary marginal asks about B, while primary asks about A and still
    # explicitly averages over both B levels. Neither implies independent assignment.
    second = treatments.effects[0].model_copy(deep=True)
    second.id, second.priority, second.treatment_ids = "secondary", "secondary", [ids["b"]]
    treatments.effects.append(second)
    target.requested_effects.append(RequestedEffect(id="secondary", question=claim(), priority="secondary"))
    treatments.assignments[0].kind = "joint"
    treatments.assignments[0].treatment_ids.append(ids["b"])
    treatments.values.extend([level("b0", ids["b"], 0), level("b1", ids["b"], 1)])
    terms = [(["zero", "b0"], -.5), (["zero", "b1"], -.5), (["one", "b0"], .5), (["one", "b1"], .5)]
    treatments.effects[0].comparison = binding("static", terms=[{"value_ids": state, "weight": weight} for state, weight in terms])
    treatments.effects[1].comparison = binding("static", terms=[{"value_ids": state, "weight": weight} for state, weight in [(x[0], w) for x, w in zip(terms, [-.5, .5, -.5, .5], strict=True)]])
    return context, ids, target, treatments, roles


def test_factorial_marginal_preserves_joint_states_and_declared_weights():
    context, _, target, treatments, _ = factorial()
    before = treatments.assignments[0].mechanism.model_dump()
    resolved = accepted(treatments, context, target)
    assert [term.weight for term in resolved.effects[0].comparison.static.terms] == [-.5, -.5, .5, .5]
    assert resolved.assignments[0].mechanism.model_dump() == before
    assert len(resolved.effects[0].treatment_ids) == 1
    assert all(len(term.value_ids) == 2 for term in resolved.effects[0].comparison.static.terms)


@pytest.mark.parametrize("defect,message", [
    ("missing_factor", "same complete"), ("duplicate_state", "Duplicate joint"),
    ("two_levels_one_factor", "exactly one value"), ("constant_effect_factor", "must vary"),
    ("unknown_value", "exact declared"), ("nonfinite_weight", "finite and nonzero"),
    ("unbalanced", "sum to 0"), ("zero_weight", "finite and nonzero"),
])
def test_joint_state_defects_are_deterministically_rejected(defect, message):
    context, _, target, treatments, _ = factorial()
    terms = treatments.effects[0].comparison.static.terms
    if defect == "missing_factor":
        terms[0].value_ids.pop()
    elif defect == "duplicate_state":
        terms[1].value_ids = terms[0].value_ids.copy()
    elif defect == "two_levels_one_factor":
        terms[0].value_ids = ["zero", "one"]
    elif defect == "constant_effect_factor":
        terms[:] = terms[:2]
        terms[0].weight = -1
        terms[1].weight = 1
    elif defect == "unknown_value":
        terms[0].value_ids[0] = "zero_guess"
    elif defect == "nonfinite_weight":
        terms[0].weight = float("nan")
    elif defect == "zero_weight":
        terms[0].weight = 0
    else:
        terms[0].weight = -.4
    rejected(treatments, context, target, message)


def test_large_opposing_weights_do_not_hide_a_nonzero_total_through_decimal_rounding():
    context, ids, target, treatments, _ = fixture()
    treatments.values.append(level("two", ids["t"], 2))
    treatments.effects[0].comparison = binding("static", terms=[
        {"weight": 1e30, "value_ids": ["zero"]}, {"weight": 1, "value_ids": ["one"]}, {"weight": -1e30, "value_ids": ["two"]},
    ])
    rejected(treatments, context, target, "sum to 0")


def adoption():
    context, ids, target, treatments, roles = fixture()
    treatments.assignments[0].kind = "absorbing_adoption"
    treatments.assignments[0].history_variables = [ids["cohort"], ids["time"]]
    treatments.values = [level("g2", ids["cohort"], 2), level("never", ids["cohort"], 0)]
    treatments.effects[0].comparison = binding("group_time", cohort_value_id="g2", comparison_value_ids=["never"], time_variable_id=ids["time"], adoption_period=literal(2), period=literal(2), baseline=literal(1))
    return context, ids, target, treatments, roles


def test_group_time_selectors_keep_cohort_and_time_separate_from_indicator_factor():
    context, ids, target, treatments, _ = adoption()
    resolved = accepted(treatments, context, target)
    comparison = resolved.effects[0].comparison.group_time
    assert comparison.time_variable_id == ids["time"]
    assert comparison.period.literal == "2" and comparison.baseline.literal == "1"
    assert resolved.values[0].variable_id == ids["cohort"] != resolved.treatments[0].variable.id


def test_group_code_never_becomes_the_adoption_time():
    context, _, target, treatments, _ = adoption()
    treatments.values[0].value.literal = "999"
    resolved = accepted(treatments, context, target)
    assert resolved.effects[0].comparison.group_time.adoption_period.literal == "2"
    assert resolved.values[0].value.literal == "999"


@pytest.mark.parametrize("adoption_time", ["1", "3", "NaN"])
def test_group_time_requires_an_explicit_adoption_time_between_baseline_and_period(adoption_time):
    context, _, target, treatments, _ = adoption()
    treatments.effects[0].comparison.group_time.adoption_period.literal = adoption_time
    rejected(treatments, context, target, "adoption_period" if adoption_time != "NaN" else "finite decimal")


@pytest.mark.parametrize("encoding", ["indicator", "group"])
def test_fresh_diagnostic_window_requires_explicit_code_variable_before_projection(encoding):
    context, ids, target, treatments, _ = adoption()
    treatments.assignments[0].diagnostic_window = DiagnosticWindow(
        unit_variable_id=ids["b"], time_variable_id=ids["time"], cutoff=1,
        assignment_encoding=encoding, treated_values=["1"], control_values=["0"],
        adoption_variable_id=ids["cohort"], never_adoption_values=["0"], evidence=claim(), history_evidence=claim(),
    )
    rejected(treatments, context, target, "explicit assignment_variable_id")
    treatments.assignments[0].diagnostic_window.assignment_variable_id = ids["t"]
    accepted(treatments, context, target)


@pytest.mark.parametrize("defect,message", [("missing_history", "history_variables"), ("same_period", "strictly earlier"), ("wrong_cohort", "same exact cohort"), ("text_period", "numeric period"), ("wrong_process", "absorbing-adoption"), ("target_metadata", "never target-outcome")])
def test_group_time_mapping_cannot_be_guessed(defect, message):
    context, ids, target, treatments, _ = adoption()
    comparison = treatments.effects[0].comparison.group_time
    if defect == "missing_history":
        treatments.assignments[0].history_variables = []
    elif defect == "same_period":
        comparison.baseline = comparison.period
    elif defect == "wrong_cohort":
        treatments.values[1].variable_id = ids["t"]
    elif defect == "text_period":
        comparison.period.kind = "text"
    elif defect == "wrong_process":
        treatments.assignments[0].kind = "point"
    else:
        comparison.time_variable_id = ids["y"]
    rejected(treatments, context, target, message)


def discontinuity(observed_assignment=False):
    context, ids, target, treatments, roles = fixture()
    treatments.assignments[0].history_variables = [ids["score"]]
    treatments.effects[0].comparison = binding("discontinuity", running_variable_id=ids["score"], cutoff=literal(0), cutoff_in="above", direction="above_minus_below", estimand="local_wald", treatment_value_ids=["zero", "one"])
    if observed_assignment:
        treatments.values.extend([level("ineligible", ids["eligibility"], 0), level("eligible", ids["eligibility"], 1)])
        treatments.assignments[0].history_variables.append(ids["eligibility"])
        treatments.effects[0].comparison.discontinuity.assignment_value_ids = ["ineligible", "eligible"]
    return context, ids, target, treatments, roles


@pytest.mark.parametrize("observed_assignment", [False, True])
def test_fuzzy_discontinuity_freezes_threshold_direction_without_auxiliary_effects(observed_assignment):
    context, _, target, treatments, _ = discontinuity(observed_assignment)
    resolved = accepted(treatments, context, target)
    assert len(resolved.effects) == 1
    assert resolved.effects[0].comparison.discontinuity.estimand == "local_wald"
    assert resolved.effects[0].comparison.discontinuity.cutoff_in == "above"
    assert len(resolved.effects[0].comparison.discontinuity.assignment_value_ids) == (2 if observed_assignment else 0)


@pytest.mark.parametrize("defect,message", [("receipt_as_assignment", "distinct single assignment"), ("score_as_factor", "distinct bound running"), ("unknown_score", "exact offered"), ("text_cutoff", "finite numeric"), ("one_assignment_level", "distinct single assignment")])
def test_discontinuity_does_not_infer_assignment_or_exchange_receipt_and_score(defect, message):
    context, ids, target, treatments, _ = discontinuity(True)
    comparison = treatments.effects[0].comparison.discontinuity
    if defect == "receipt_as_assignment":
        comparison.assignment_value_ids = ["zero", "one"]
    elif defect == "score_as_factor":
        comparison.running_variable_id = ids["t"]
    elif defect == "unknown_score":
        comparison.running_variable_id = "guessed_score"
    elif defect == "text_cutoff":
        comparison.cutoff.kind = "text"
    else:
        comparison.assignment_value_ids = ["eligible"]
    rejected(treatments, context, target, message)


def aggregate():
    context, ids, target, treatments, roles = adoption()
    component = treatments.effects[0]
    second = component.model_copy(deep=True)
    second.id, second.priority = "later", "secondary"
    second.comparison.group_time.period.literal = "3"
    combined = component.model_copy(deep=True)
    combined.id, combined.priority = "average", "secondary"
    combined.comparison = binding("aggregate", terms=[{"effect_id": "primary", "weight": .5}, {"effect_id": "later", "weight": .5}])
    treatments.effects.extend([second, combined])
    target.requested_effects.extend([RequestedEffect(id=identity, question=claim(), priority="secondary") for identity in ("later", "average")])
    return context, ids, target, treatments, roles


def test_requested_aggregate_preserves_explicit_component_ids_and_equal_weights():
    context, _, target, treatments, _ = aggregate()
    resolved = accepted(treatments, context, target)
    assert [(term.effect_id, term.weight) for term in resolved.effects[-1].comparison.aggregate.terms] == [("primary", .5), ("later", .5)]


def test_same_cohort_and_time_binding_cannot_mint_conflicting_adoption_events():
    context, _, target, treatments, _ = aggregate()
    treatments.effects[1].comparison.group_time.adoption_period.literal = "3"
    rejected(treatments, context, target, "conflicting adoption periods")
    treatments.effects[1].comparison.group_time.adoption_period.literal = "2.00"
    accepted(treatments, context, target)


@pytest.mark.parametrize("defect,message", [("self", "cannot include itself"), ("duplicate", "distinct declared"), ("unknown", "distinct declared"), ("negative", "positive"), ("sum", "sum to 1"), ("cycle", "cycle")])
def test_aggregate_closes_exact_dependency_graph(defect, message):
    context, _, target, treatments, _ = aggregate()
    terms = treatments.effects[-1].comparison.aggregate.terms
    if defect == "self":
        terms[0].effect_id = "average"
    elif defect == "duplicate":
        terms[0].effect_id = "later"
    elif defect == "unknown":
        terms[0].effect_id = "invented_auxiliary"
    elif defect == "negative":
        terms[0].weight, terms[1].weight = -1, 2
    elif defect == "sum":
        terms[0].weight = .3
    else:
        treatments.effects[0].comparison = binding("aggregate", terms=[{"effect_id": "average", "weight": 1}])
    rejected(treatments, context, target, message)


def test_historical_missing_bindings_remain_readable_but_fresh_ready_decision_blocks():
    context, _, target, treatments, _ = fixture()
    treatments.values, treatments.effects[0].comparison = [], None
    assert all(check.passed for check in domain.checks(treatments, context, {"target": target.model_dump()}, scientific_version=9))
    rejected(treatments, context, target, "typed comparison")


def test_unsupported_history_is_retained_but_never_promoted_to_static_ready():
    context, _, target, treatments, _ = fixture()
    treatments.values = []
    treatments.effects[0].comparison = ContrastBinding(kind="unsupported", evidence=claim(), unsupported=claim("History-adaptive treatment rule based on earlier outcomes."))
    rejected(treatments, context, target, "unsupported")
    treatments.status = "unsupported"
    checks = domain.checks(treatments, context, {"target": target.model_dump()}, scientific_version=15)
    assert [check.rule for check in checks if not check.passed] == ["semantic_sufficiency"]
    assert treatments.effects[0].comparison.unsupported.statement.startswith("History-adaptive")


def test_unknown_or_unoffered_value_authority_still_fails_existing_provenance_gate():
    context, _, target, treatments, _ = fixture()
    treatments.values[0].meaning.references = ["columns.unoffered.codes.invented"]
    rejected(treatments, context, target, "References outside")
    treatments.values[0].meaning.basis = "unknown"
    treatments.values[0].meaning.references = []
    rejected(treatments, context, target, "Treatment value meaning")


def test_unused_metadata_value_still_requires_role_registry_closure():
    context, ids, target, treatments, _ = fixture()
    treatments.values.append(level("metadata", ids["cohort"], 4))
    rejected(treatments, context, target, "including unused registry levels")
    treatments.assignments[0].history_variables.append(ids["cohort"])
    accepted(treatments, context, target)


def test_ambiguous_payload_kind_and_target_value_are_rejected():
    context, ids, target, treatments, _ = fixture()
    treatments.effects[0].comparison.aggregate = binding("aggregate", terms=[{"effect_id": "primary", "weight": 1}]).aggregate
    rejected(treatments, context, target, "exactly the payload")
    treatments.effects[0].comparison.aggregate = None
    treatments.values[0].variable_id = ids["y"]
    rejected(treatments, context, target, "never the target outcome")


def test_later_packets_offer_only_accepted_value_ids_and_exact_metadata_handles():
    context, ids, target, treatments, _ = discontinuity(True)
    resolved = accepted(treatments, context, target)
    payload = {"column_identities": [column.identity.model_dump() for column in context.columns.values()], "decisions": {"treatments": resolved.model_dump()}}
    before = deepcopy(payload)
    encoded, reverse, choices = references.selections(payload)
    assert choices["value_ids"] == [item.id for item in resolved.values]
    assert references.handle("c", ids["score"]) in choices["running_variable_id"]
    assert references.translate(encoded["decisions"]["treatments"], reverse) == resolved.model_dump()
    assert payload == before
    # Native provider schema remains one object with kind plus nullable payloads;
    # local checks enforce the exact-one rule without repeated conditional branches.
    schema = provider_schema(type(treatments), choices)
    comparison = schema["properties"]["effects"]["items"]["properties"]["comparison"]["anyOf"][0]
    assert comparison["properties"]["kind"]["enum"] == ["static", "group_time", "discontinuity", "aggregate", "unsupported"]


@pytest.mark.parametrize("literal_is_handle", [False, True])
def test_reference_translation_cannot_rewrite_an_exact_data_literal(literal_is_handle):
    context, ids, _, treatments, _ = fixture()
    durable = ids["t"]
    alias = references.handle("c", durable)
    exact = alias if literal_is_handle else durable
    treatments.values[0].value = TypedValue(kind="text", literal=exact)
    packet = {"column_identities": [column.identity.model_dump() for column in context.columns.values()], "decisions": {"treatments": treatments.model_dump()}}
    encoded, reverse, _ = references.selections(packet)
    assert encoded["decisions"]["treatments"]["values"][0]["value"]["literal"] == exact
    decoded = references.translate(encoded, reverse)
    assert decoded["decisions"]["treatments"]["values"][0]["value"]["literal"] == exact
    assert decoded["decisions"]["treatments"]["values"][0]["variable_id"] == durable
