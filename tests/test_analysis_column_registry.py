"""One accepted source binding per column, reused across independent event roles."""

from copy import deepcopy
from uuid import uuid4

import pytest
from historical_analysis import work
from model_double import model_double
from test_scientific_v8 import study

from backend import model_access
from backend.analysis import domain, provider
from backend.analysis.contracts import Claim, Roles


@pytest.fixture(autouse=True)
def isolated():
    yield


def record(variable, *, key=None, parent=None, status="accepted"):
    cid = variable.id
    return {"id": key or "screen:" + cid, "key": key or "screen:" + cid, "kind": "screen",
            "column_ids": [cid], "scope": {"parent": parent} if parent else {}, "status": status,
            "attempts": [{"id": "receipt:" + (key or cid), "status": "accepted" if status == "accepted" else "rejected", "complete": False,
                          "checks": [{"rule": "owned_column_binding", "passed": True}], "output": {
                              "target_candidate": False, "treatment_candidate": False, "summary": "This observed column is bound.",
                              "variable": variable.model_dump(mode="json"),
                          }}]}


def fixture():
    context, ids, target, treatments, roles = study()
    variable = roles.assignments[0].variable
    screen = work.ColumnScreen(target_candidate=False, treatment_candidate=False, summary="Source-qualified meaning.", variable=variable, citations=["question"])
    return context, ids, target, treatments, roles, screen


@pytest.mark.parametrize("unknown", [False, True])
def test_screen_accepts_only_its_owned_observed_binding_without_guessing_unknown_meaning(unknown):
    context, ids, _, _, _, screen = fixture()
    if unknown:
        screen.variable.meaning.basis = "unknown"
        screen.variable.meaning.references = []
    before = screen.model_dump()
    work.validate_screen(screen, context, [ids["c"]])
    assert screen.model_dump() == before


@pytest.mark.parametrize("defect", ["missing", "foreign_id", "foreign_column", "derived", "assumed", "unoffered", "decision_authority", "no_source", "several_columns"])
def test_screen_cannot_invent_or_rebind_measurement(defect):
    context, ids, _, _, _, screen = fixture()
    if defect == "missing":
        screen.variable = None
    elif defect == "foreign_id":
        screen.variable.id = ids["y"]
    elif defect == "foreign_column":
        screen.variable.column_ids = [ids["y"]]
    elif defect == "derived":
        screen.variable.derivation = "Invented transformation"
    elif defect == "assumed":
        screen.variable.meaning.basis = "assumption"
    elif defect == "unoffered":
        screen.variable.meaning.references = ["columns.foreign.definition.invented"]
    elif defect == "decision_authority":
        screen.variable.meaning.prerequisites = ["decision:target"]
    elif defect == "no_source":
        screen.variable.meaning.references = []
    else:
        screen.variable.column_ids = [ids["c"], ids["t"]]
    with pytest.raises(ValueError):
        work.validate_screen(screen, context, [ids["c"]])


def test_mapping_uses_accepted_widening_child_and_retains_its_exact_evidence():
    _, ids, _, _, roles, _ = fixture()
    variable = roles.assignments[0].variable
    variable.meaning.references = ["columns." + ids["c"] + ".definition.newly_retrieved"]
    parent = record(variable, status="expanded")
    child = record(variable, key="widen:" + parent["key"], parent=parent["id"])
    unrelated = {"kind": "role"}
    before = deepcopy([parent, child, unrelated])
    mapping = work.accepted_screen_mapping([unrelated, child, parent], {ids["c"]})
    assert mapping == {ids["c"]: {"variable": variable.model_dump(mode="json"), "receipt_id": child["attempts"][0]["id"]}}
    assert [parent, child, unrelated] == before
    assert mapping[ids["c"]]["receipt_id"] != parent["attempts"][0]["id"]


@pytest.mark.parametrize("defect", ["missing_root", "duplicate_root", "foreign_column", "rejected", "failed_gate", "proposal_only", "missing_binding", "two_children", "unexpected_child", "orphan", "expanded_no_child"])
def test_registry_has_no_first_wins_or_incomplete_receipt_fallback(defect):
    _, ids, _, _, roles, _ = fixture()
    original = record(roles.assignments[0].variable)
    records = [original]
    if defect == "missing_root":
        records = []
    elif defect == "duplicate_root":
        records.append(deepcopy(original))
    elif defect == "foreign_column":
        original["column_ids"] = [ids["y"]]
    elif defect == "rejected":
        original["status"] = "rejected"
    elif defect == "failed_gate":
        original["attempts"][0]["checks"][0]["passed"] = False
    elif defect == "proposal_only":
        original["attempts"][0]["status"] = "rejected"
    elif defect == "missing_binding":
        original["attempts"][0]["output"]["variable"] = None
    elif defect in {"two_children", "unexpected_child", "orphan"}:
        original["status"] = "expanded" if defect != "unexpected_child" else "accepted"
        records.append(record(roles.assignments[0].variable, key="widen:first", parent="foreign" if defect == "orphan" else original["id"]))
        if defect == "two_children":
            records.append(record(roles.assignments[0].variable, key="widen:second", parent=original["id"]))
    else:
        original["status"] = "expanded"
    with pytest.raises(ValueError):
        work.accepted_screen_mapping(records, {ids["c"]})


def test_two_event_accounts_reuse_one_measurement_without_erasing_semantic_differences():
    _, _, target, treatments, roles, _ = fixture()
    bound = roles.assignments[0].variable.model_dump(mode="json")
    other = roles.model_copy(deep=True)
    other.assignments[0].account.statement = "This event has a different role account."
    other.assignments[0].time_scope = "A later assignment event."
    work.validate_role_variables(roles, [bound])
    work.validate_role_variables(other, [bound])
    combined = roles.model_copy(deep=True)
    combined.assignments.extend(other.assignments)
    registry = domain.variables(target, treatments, combined)
    assert registry[bound["id"]].model_dump(mode="json") == bound
    assert combined.assignments[0].account != combined.assignments[1].account


@pytest.mark.parametrize("field", ["id", "label", "columns", "meaning", "references", "basis", "derivation", "prerequisites"])
def test_role_proposal_cannot_reauthor_any_binding_field(field):
    _, ids, _, _, roles, _ = fixture()
    bound = roles.assignments[0].variable.model_dump(mode="json")
    proposed = roles.assignments[0].variable
    if field == "id":
        proposed.id = "invented"
    elif field == "label":
        proposed.label = "Paraphrased label"
    elif field == "columns":
        proposed.column_ids = [ids["y"]]
    elif field == "meaning":
        proposed.meaning.statement = "Paraphrased or conflicting definition"
    elif field == "references":
        proposed.meaning.references = ["question", "invented"]
    elif field == "basis":
        proposed.meaning.basis = "unknown"
    elif field == "prerequisites":
        proposed.meaning.prerequisites = ["decision:target"]
    else:
        proposed.derivation = "Alternative computation"
    with pytest.raises(ValueError, match="exact established Variable"):
        work.validate_role_variables(roles, [bound])


def test_final_selected_binding_overrides_screen_only_through_exact_offered_authority():
    _, _, _, _, roles, _ = fixture()
    screen = roles.assignments[0].variable.model_copy(deep=True)
    selected = screen.model_copy(deep=True)
    selected.meaning.statement = "Final accepted selected-variable meaning."
    roles.assignments[0].variable = selected
    work.validate_role_variables(roles, [selected.model_dump(mode="json")])
    with pytest.raises(ValueError):
        work.validate_role_variables(roles, [screen.model_dump(mode="json")])
    with pytest.raises(ValueError, match="one authoritative definition"):
        work.validate_role_variables(roles, [screen.model_dump(mode="json"), selected.model_dump(mode="json")])


def test_worker_bounds_correction_and_retains_rejected_raw_binding(monkeypatch):
    context, ids, target, treatments, roles, _ = fixture()
    scope = next(item for item in domain.role_scopes(context, target, treatments) if item["column_id"] == ids["c"])
    source = work.packet(context, {ids["c"]}, {"target": target.model_dump(), "treatments": treatments.model_dump()}, "Fixed question")
    source.update(role_scope=scope, established_variables=[roles.assignments[0].variable.model_dump(mode="json")], _model="test", _prompt="Assess roles", assumption_namespace="registry_")
    task = {"id": str(uuid4()), "kind": "role", "column_ids": [ids["c"]], "scope": {"role_scope": scope}, "packet": source, "attempts": []}
    receipts, requests = [], []
    original = roles.model_dump(mode="json")

    def propose(schema, prompt, payload, model, **kwargs):
        requests.append(deepcopy(payload))
        value = roles.model_copy(deep=True)
        if len(requests) == 1:
            value.assignments[0].variable.label = "Rejected independent paraphrase"
        return value

    def finish(claimed, output, checks, **kwargs):
        receipt = {"id": "receipt:" + str(len(receipts)), "output": deepcopy(output), "checks": deepcopy(checks), **kwargs}
        receipts.append(receipt)
        task["attempts"].append(receipt)
        return receipt

    monkeypatch.setattr(work.tasks, "view", lambda _: deepcopy(task))
    monkeypatch.setattr(work.tasks, "claim", lambda *args, **kwargs: object())
    monkeypatch.setattr(work.tasks, "finish", finish)
    monkeypatch.setattr(model_access, "structured", model_double(propose))
    work.run(task["id"])
    assert [item["status"] for item in receipts] == ["rejected", "accepted"]
    assert "Rejected independent paraphrase" in receipts[0]["raw_output"]
    assert receipts[1]["output"]["assignments"][0]["variable"] == original["assignments"][0]["variable"]
    assert requests[0]["established_variables"] == requests[1]["established_variables"]
    assert any(item["rule"] == "immutable_variable_binding" for item in requests[1]["correction"]["gate_feedback"])
    assert roles.model_dump(mode="json") == original


def test_unknown_screen_does_not_require_a_manufactured_source_citation(monkeypatch):
    context, ids, _, _, _, screen = fixture()
    screen.variable.meaning = Claim(statement="Meaning is not supplied.", basis="unknown")
    screen.citations = []
    source = work.packet(context, {ids["c"]}, {}, "Fixed question")
    source.update(_model="test", _prompt="Screen one column", assumption_namespace="registry_")
    task = {"id": str(uuid4()), "kind": "screen", "column_ids": [ids["c"]], "scope": {}, "packet": source, "attempts": []}
    accepted = []
    monkeypatch.setattr(work.tasks, "view", lambda _: task)
    monkeypatch.setattr(work.tasks, "claim", lambda *args, **kwargs: object())
    monkeypatch.setattr(model_access, "structured", model_double(lambda *args, **kwargs: screen))
    monkeypatch.setattr(work.tasks, "finish", lambda claimed, output, checks, **kwargs: accepted.append({"id": "accepted", "output": output, "checks": checks, **kwargs}) or accepted[-1])
    work.run(task["id"])
    assert len(accepted) == 1 and accepted[0]["status"] == "accepted"
    assert accepted[0]["output"]["variable"]["meaning"]["basis"] == "unknown"
    assert accepted[0]["output"]["variable"]["meaning"]["references"] == []


def test_role_event_stamp_is_absent_from_provider_schema_with_or_without_choices():
    for choices in (None, {"effect_ids": ["primary"], "treatment_ids": ["t"]}):
        schema = provider.provider_schema(Roles, choices)
        assert "assignment_event_key" not in schema["properties"]["assignments"]["items"]["properties"]
