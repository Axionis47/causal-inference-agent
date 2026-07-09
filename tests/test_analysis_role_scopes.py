"""Pure role-event coverage and coordinator checks, without database or models."""

from copy import deepcopy
from types import SimpleNamespace
from uuid import uuid4

import pytest
from historical_analysis import tasks, work
from historical_analysis.runtime import Runtime
from model_double import model_double
from test_scientific_v8 import decision, prior, relation, study

from backend import model_access
from backend.analysis import domain
from backend.analysis.contracts import CausalAccount, Frame, Roles


@pytest.fixture(autouse=True)
def isolated():
    yield


def two_events():
    context, ids, target, treatments, roles = study(("t", "y", "c", "z"))
    second = treatments.treatments[0].model_copy(deep=True)
    second.variable = roles.assignments[0].variable
    treatments.treatments.append(second)
    treatments.effects[0].treatment_ids.append(ids["c"])
    event = treatments.assignments[0].model_copy(deep=True)
    event.id = "later_event"
    event.treatment_ids = [ids["c"]]
    treatments.assignments.append(event)
    roles.assignments = roles.assignments[1:]
    return context, ids, target, treatments, roles


def scoped_roles(scope, target, treatments, roles):
    registry = domain.variables(target, treatments, roles)
    variable = next(item for item in registry.values() if item.column_ids == [scope["column_id"]])
    assignments = []
    for effect, selected in scope["effect_treatments"].items():
        assignments.append({
            "variable": variable, "roles": ["irrelevant"], "timing": "unknown",
            "assignment_event_key": domain.fingerprint(scope["event"]),
            "account": {"statement": "Explicitly irrelevant to this scope.", "basis": "supplied", "references": ["question"]},
            "timing_evidence": {"statement": "No timing premise needed for exclusion.", "basis": "unknown"},
            "reason": "Explicit exclusion", "effect_ids": [effect], "treatment_ids": selected,
        })
    return Roles(**decision(), assignments=assignments)


def test_effects_at_one_event_share_a_task_but_own_distinct_obligations():
    context, ids, target, treatments, _ = study()
    treatments.effects.append(treatments.effects[0].model_copy(update={"id": "secondary", "priority": "secondary"}))
    scopes = domain.role_scopes(context, target, treatments)
    assert len(scopes) == 1
    assert scopes[0]["column_id"] == ids["c"]
    assert scopes[0]["effect_treatments"] == {"primary": [ids["t"]], "secondary": [ids["t"]]}
    assert len(scopes[0]["obligation_keys"]) == 2


def test_role_scope_identities_ignore_prose_process_aliases_names_and_column_order():
    context, _, target, treatments, _ = two_events()
    before = domain.role_scopes(context, target, treatments)
    treatments.assignments.reverse()
    treatments.treatments.reverse()
    treatments.effects[0].treatment_ids.reverse()
    for event in treatments.assignments:
        event.id = "renamed_" + event.id
        event.timing.statement = "Same accepted event described with different prose"
    context.columns = dict(reversed(list(context.columns.items())))
    for column in context.columns.values():
        column.identity.name = "opaque_" + column.identity.column_id
    assert domain.role_scopes(context, target, treatments) == before


def test_distinct_events_create_distinct_scopes_and_cross_treatment_roles():
    context, ids, target, treatments, _ = two_events()
    scopes = domain.role_scopes(context, target, treatments)
    assert len(scopes) == 4
    assert len([item for item in scopes if item["column_id"] == ids["z"]]) == 2
    assert next(item for item in scopes if item["column_id"] == ids["t"])["effect_treatments"] == {"primary": [ids["c"]]}
    assert next(item for item in scopes if item["column_id"] == ids["c"])["effect_treatments"] == {"primary": [ids["t"]]}
    assert not any(item["column_id"] == ids["y"] for item in scopes)


def test_joint_event_marginals_assess_other_factor_without_claiming_its_own_role():
    context, ids, target, treatments, _ = two_events()
    treatments.assignments = [treatments.assignments[0]]
    treatments.assignments[0].kind = "joint"
    treatments.assignments[0].treatment_ids.append(ids["c"])
    treatments.effects.extend([
        treatments.effects[0].model_copy(update={"id": "a", "treatment_ids": [ids["t"]], "priority": "secondary"}),
        treatments.effects[0].model_copy(update={"id": "b", "treatment_ids": [ids["c"]], "priority": "secondary"}),
    ])
    scopes = domain.role_scopes(context, target, treatments)
    assert len(scopes) == 3
    assert next(item for item in scopes if item["column_id"] == ids["t"])["effect_treatments"] == {"b": [ids["c"]]}
    assert next(item for item in scopes if item["column_id"] == ids["c"])["effect_treatments"] == {"a": [ids["t"]]}
    assert len(next(item for item in scopes if item["column_id"] == ids["z"])["obligation_keys"]) == 3


@pytest.mark.parametrize("defect", ["missing_effect", "implicit_irrelevant", "wrong_event", "wrong_column", "duplicate"])
def test_scoped_acceptance_requires_explicit_complete_and_nonconflicting_coverage(defect):
    context, _, target, treatments, roles = study()
    treatments.effects.append(treatments.effects[0].model_copy(update={"id": "secondary", "priority": "secondary"}))
    scope = domain.role_scopes(context, target, treatments)[0]
    value = scoped_roles(scope, target, treatments, roles)
    domain.validate_role_scope(value, scope)
    if defect == "missing_effect":
        value.assignments.pop()
    elif defect == "implicit_irrelevant":
        value.assignments[0].effect_ids = []
    elif defect == "wrong_event":
        value.assignments[0].treatment_ids = ["other_event_factor"]
    elif defect == "wrong_column":
        value.assignments[0].variable = target.outcome
    else:
        value.assignments.append(value.assignments[0].model_copy(deep=True))
    with pytest.raises(ValueError):
        domain.validate_role_scope(value, scope)


def test_operational_scope_normalization_uses_the_owned_event_not_all_effect_treatments():
    context, ids, target, treatments, roles = two_events()
    scope = next(item for item in domain.role_scopes(context, target, treatments) if item["column_id"] == ids["z"])
    item = roles.assignments[0]
    item.roles = ["identifier"]
    item.operational_use = ["unit_key"]
    item.effect_ids, item.treatment_ids, item.time_scope = [], [], ""
    normalized = domain.normalize_role_scope(roles, prior(target, treatments, roles), scope)
    assert normalized.assignments[0].treatment_ids == scope["effect_treatments"]["primary"]
    domain.validate_role_scope(normalized, scope)


def test_repeated_measurement_meanings_and_established_treatment_bindings_cannot_change():
    context, ids, target, treatments, roles = two_events()
    scopes = [item for item in domain.role_scopes(context, target, treatments) if item["column_id"] == ids["z"]]
    left, right = [scoped_roles(scope, target, treatments, roles) for scope in scopes]
    combined = Roles(**decision(), assignments=left.assignments + right.assignments)
    assert domain.variables(target, treatments, combined)[ids["z"]] == roles.assignments[0].variable
    combined.assignments[1].variable = combined.assignments[1].variable.model_copy(deep=True)
    combined.assignments[1].variable.meaning.statement = "A conflicting meaning"
    with pytest.raises(ValueError, match="different binding or meaning"):
        domain.variables(target, treatments, combined)
    treatment_scope = next(item for item in domain.role_scopes(context, target, treatments) if item["column_id"] == ids["t"])
    value = scoped_roles(treatment_scope, target, treatments, roles).model_copy(deep=True)
    value.assignments[0].variable.meaning.statement = "A new incompatible treatment meaning"
    with pytest.raises(ValueError, match="different binding or meaning"):
        domain.variables(target, treatments, value)


@pytest.mark.parametrize("qualification", [None, "direct", "qualified"])
def test_selected_factor_diagnostics_ignore_other_event_timing_but_respect_actual_lineage(qualification):
    context, ids, target, treatments, roles = two_events()
    scopes = [item for item in domain.role_scopes(context, target, treatments) if item["column_id"] in {ids["t"], ids["c"]}]
    assignments = [assignment for scope in scopes for assignment in scoped_roles(scope, target, treatments, roles).assignments]
    for item in assignments:
        item.timing = "after_treatment"
    if qualification:
        context.relationships["proxy"] = relation(ids["y"], ids["t"], **({"qualifications": "Only within a qualified subgroup"} if qualification == "qualified" else {}))
    frame = Frame(target=target, treatments=treatments, roles=Roles(**decision(), assignments=assignments), causal=CausalAccount(**decision(), identification="Fixture"), graph_hash="fixed")
    _, receipt = domain.projection(frame, context, b"t,y,c,z\n1,999,0,4\n", scientific_version=12)
    assert set(receipt["columns"]) == ({ids["c"]} if qualification else {ids["t"], ids["c"]})
    assert ids["y"] in receipt["withheld"]


def baseline_frame():
    context, ids, target, treatments, roles = two_events()
    scopes = [item for item in domain.role_scopes(context, target, treatments) if item["column_id"] == ids["z"]]
    assignments = [assignment for scope in scopes for assignment in scoped_roles(scope, target, treatments, roles).assignments]
    for item in assignments:
        item.roles = ["pre_outcome"]
        item.timing = "before_treatment"
        item.timing_evidence = item.account.model_copy(update={"statement": "This measurement precedes the scoped event."})
        item.time_scope = "Baseline before the explicitly selected event"
    frame = Frame(target=target, treatments=treatments, roles=Roles(**decision(), assignments=assignments), causal=CausalAccount(**decision(), identification="Fixture"), graph_hash="fixed")
    return context, ids, frame


def test_baseline_projection_unions_supplied_event_proofs_without_reading_target_values():
    context, ids, frame = baseline_frame()
    original = frame.model_dump()
    raw, receipt = domain.projection(frame, context, b"t,y,c,z\n1,999,0,4\n", scientific_version=12)
    assert set(receipt["columns"]) == {ids["t"], ids["c"], ids["z"]}
    assert receipt["access"][ids["z"]]["effect_treatment_pairs"] == sorted([["primary", ids["t"]], ["primary", ids["c"]]])
    assert frame.model_dump() == original
    other, _ = domain.projection(frame, context, b"t,y,c,z\n1,-12345,0,4\n", scientific_version=12)
    assert raw == other


@pytest.mark.parametrize("defect", ["missing_event", "after", "at", "unknown", "assumed_timing", "missing_refs", "unsupported_refs", "actual_proxy", "qualified_proxy"])
def test_baseline_union_requires_every_pair_and_no_relevant_conflicting_access_evidence(defect):
    context, ids, frame = baseline_frame()
    item = frame.roles.assignments[0]
    if defect == "missing_event":
        frame.roles.assignments.pop()
    elif defect in {"after", "at", "unknown"}:
        item.timing = {"after": "after_treatment", "at": "at_treatment", "unknown": "unknown"}[defect]
    elif defect == "assumed_timing":
        item.timing_evidence.basis = "assumption"
    elif defect == "missing_refs":
        item.timing_evidence.references = []
    elif defect == "unsupported_refs":
        item.timing_evidence.references = ["invented_source"]
    else:
        context.relationships["proxy"] = relation(ids["y"], ids["z"], **({"qualifications": "Only for a subgroup"} if defect == "qualified_proxy" else {}))
    _, receipt = domain.projection(frame, context, b"t,y,c,z\n1,999,0,4\n", scientific_version=12)
    assert ids["z"] in receipt["withheld"]


def test_baseline_union_requires_exact_shared_measurement_meaning():
    context, _, frame = baseline_frame()
    frame.roles.assignments[1].variable = frame.roles.assignments[1].variable.model_copy(deep=True)
    frame.roles.assignments[1].variable.meaning.statement = "A conflicting measurement meaning"
    with pytest.raises(ValueError, match="different binding or meaning"):
        domain.projection(frame, context, b"t,y,c,z\n1,999,0,4\n", scientific_version=12)


def test_baseline_union_is_effect_scoped_and_never_applies_to_structure_only_diagnostics():
    context, ids, frame = baseline_frame()
    secondary = frame.treatments.effects[0].model_copy(update={"id": "secondary", "priority": "secondary"})
    frame.treatments.effects.append(secondary)
    conflicting = frame.roles.assignments[0].model_copy(deep=True)
    conflicting.effect_ids = ["secondary"]
    conflicting.timing = "after_treatment"
    frame.roles.assignments.append(conflicting)
    _, all_effects = domain.projection(frame, context, b"t,y,c,z\n1,999,0,4\n", scientific_version=12)
    _, primary_only = domain.projection(frame, context, b"t,y,c,z\n1,999,0,4\n", effect_ids={"primary"}, scientific_version=12)
    _, structure = domain.projection(frame, context, b"t,y,c,z\n1,999,0,4\n", effect_ids={"primary"}, purpose="adoption_structure", scientific_version=12)
    assert ids["z"] in all_effects["withheld"]
    assert ids["z"] in primary_only["columns"]
    assert ids["z"] in structure["withheld"]


def test_coordinator_creates_canonical_grouped_tasks_and_supplies_exact_cross_treatment_variables(monkeypatch):
    context, ids, target, treatments, roles = two_events()
    runtime = Runtime.__new__(Runtime)
    saved = {"decision:" + key: value for key, value in prior(target, treatments, roles).items() if key != "roles"}
    runtime.context = context
    runtime.db = SimpleNamespace(commit=lambda: None)
    runtime.version = SimpleNamespace(id=uuid4())
    runtime.run = SimpleNamespace(id=uuid4(), inputs={"model": "test", "roles": "Assess roles", "contract": "checkpoint-harness-19"}, question="Fixed question", context_revision_id=uuid4())
    runtime.h = SimpleNamespace(saved=saved, accepted=lambda name: SimpleNamespace(id="accepted:" + name))
    records = []
    screen_records = [{"id": "screen:" + cid, "key": "screen:" + cid, "kind": "screen", "scope": {}, "column_ids": [cid], "status": "accepted",
        "attempts": [{"id": "screen-receipt:" + cid, "status": "accepted", "complete": False, "checks": [{"passed": True}], "output": {
            "target_candidate": False, "treatment_candidate": False, "summary": "Bound", "variable": {
                "id": cid, "label": column.identity.name, "column_ids": [cid], "meaning": {"statement": "Unknown fixture meaning", "basis": "unknown"}, "derivation": "",
            },
        }}]} for cid, column in context.columns.items()]
    monkeypatch.setattr(tasks, "list_tasks", lambda *args, **kwargs: screen_records)
    monkeypatch.setattr(tasks, "ensure", lambda run_id, **kwargs: records.append(deepcopy(kwargs)))
    assert runtime.prepare_work("role") == {"work_kind": "role"}
    expected = domain.role_scopes(context, target, treatments)
    assert {item["key"] for item in records} == {item["key"] for item in expected}
    assert all(item["packet"]["role_scope"] == item["scope"]["role_scope"] for item in records)
    treatment_task = next(item for item in records if item["column_ids"] == [ids["t"]])
    assert treatment_task["packet"]["established_variables"] == [treatments.treatments[0].variable.model_dump(mode="json")]
    assert all(item["dependency_receipt_ids"] == ["accepted:target", "accepted:treatments", "screen-receipt:" + item["column_ids"][0]] for item in records)


def test_worker_provider_choices_and_acceptance_are_limited_to_its_event(monkeypatch):
    context, ids, target, treatments, roles = two_events()
    scope = next(item for item in domain.role_scopes(context, target, treatments) if item["column_id"] == ids["z"])
    prerequisites = {key: value for key, value in prior(target, treatments, roles).items() if key != "roles"}
    packet = work.packet(context, {ids["z"]}, prerequisites, "Fixed question")
    packet.update({"role_scope": scope, "_model": "test", "_prompt": "Assess roles", "assumption_namespace": "role_fixture_"})
    packet["established_variables"] = [item.variable.model_dump(mode="json") for item in roles.assignments if item.variable.id == ids["z"]]
    task = {"id": str(uuid4()), "kind": "role", "packet": packet, "attempts": [], "scope": {"role_scope": scope}, "column_ids": [ids["z"]]}
    receipts = []
    monkeypatch.setattr(tasks, "view", lambda identity: task)
    monkeypatch.setattr(tasks, "claim", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(work, "get_current_run_tree", lambda: None)

    def propose(schema, prompt, offered, model, **kwargs):
        owned = offered["role_scope"]["effect_treatments"]
        fields = kwargs["prepared"].schema["properties"]["assignments"]["items"]["properties"]
        assert fields["effect_ids"]["items"]["enum"] == ["primary"]
        assert fields["treatment_ids"]["items"]["enum"] == owned["primary"]
        assert len(fields["treatment_ids"]["items"]["enum"]) == 1
        assert "reuse that exact Variable unchanged" in prompt
        return scoped_roles(scope, target, treatments, roles)

    def finish(claim, output, checks, **kwargs):
        assert kwargs["status"] == "accepted"
        receipts.append({"id": "accepted-role", "output": output, "checks": checks})
        return receipts[-1]

    monkeypatch.setattr(model_access, "structured", model_double(propose))
    monkeypatch.setattr(tasks, "finish", finish)
    assert work.run(task["id"]) == {"completed": ["accepted-role"]}
    assert any(item["rule"] == "role_effect_event_coverage" and item["passed"] for item in receipts[0]["checks"])


@pytest.mark.parametrize("defect", [None, "missing_root", "missing_effect", "changed_child_scope", "fresh_role_credit"])
def test_coordinator_closes_original_scopes_through_exact_widened_children(monkeypatch, defect):
    # These saved fixtures predate claim admissions. This test owns frontier
    # closure and effect/event accounting; v19 authority has separate regressions.
    checks = domain.checks
    monkeypatch.setattr(domain, "checks", lambda *args, **kwargs: checks(*args, **{**kwargs, "scientific_version": 15}))
    context, ids, target, treatments, roles = study()
    treatments.effects.append(treatments.effects[0].model_copy(update={"id": "secondary", "priority": "secondary"}))
    scope = domain.role_scopes(context, target, treatments)[0]
    role_scope = {field: scope[field] for field in ("column_id", "event", "effect_treatments")}
    value = scoped_roles(scope, target, treatments, roles)
    if defect == "missing_effect":
        value.assignments.pop()
    parent = {"id": "parent", "key": scope["key"], "kind": "role", "column_ids": [ids["c"]], "obligation_keys": scope["obligation_keys"], "scope": {"role_scope": role_scope}, "status": "expanded", "attempts": []}
    child = {"id": "child", "key": "widen:" + scope["key"], "kind": "role", "column_ids": [ids["c"]], "obligation_keys": ["widen:" + ids["c"]], "scope": {"parent": "parent", "role_scope": deepcopy(role_scope)}, "status": "accepted", "attempts": [{"id": "accepted", "output": value.model_dump(mode="json")}]}
    if defect == "changed_child_scope":
        child["scope"]["role_scope"]["event"]["kind"] = "absorbing_adoption"
    elif defect == "fresh_role_credit":
        child["obligation_keys"] = scope["obligation_keys"]
    records = [child] if defect == "missing_root" else [parent, child]
    runtime = Runtime.__new__(Runtime)
    saved = {"decision:" + key: value for key, value in prior(target, treatments, roles).items() if key != "roles"}
    runtime.context = context
    runtime.db = SimpleNamespace(commit=lambda: None)
    runtime.run = SimpleNamespace(id=uuid4())
    completed = []
    runtime.h = SimpleNamespace(saved=saved, remember=lambda key, value, **kwargs: saved.update({key: value}), finish=lambda *args, **kwargs: completed.append((args, kwargs)))
    monkeypatch.setattr(tasks, "list_tasks", lambda *args, **kwargs: records)
    if defect:
        with pytest.raises(ValueError):
            runtime.complete_work({"work_kind": "role"})
    else:
        assert runtime.complete_work({"work_kind": "role"}) == {}
        assert completed[0][0][2]["coverage"]["effect_event_obligations"] == 2
        assert len(saved["decision:roles"]["assignments"]) == 2
