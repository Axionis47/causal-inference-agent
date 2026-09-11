"""Tests for response schemas and the shared model-result correction boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, cast

import pytest
from pydantic import BaseModel, ConfigDict, create_model

from causal.shared.agenttask import TaskRunner, evidence_availability, result_schema
from causal.shared.contracts import (
    ArtifactEnvelopeV1,
    ArtifactRef,
    ReferenceKind,
    SensitivityClass,
    reference_field,
)
from causal.shared.envelope import (
    AgentTaskEnvelopeV1,
    AgentTaskResultV1,
    ContextRequirementV1,
    Criticality,
    EvidenceClass,
    MissingAction,
    RequirementScopeKind,
    SupportRequirement,
    TaskBudgets,
    TaskStatus,
)
from causal.shared.gateway import GatewayResultV1
from causal.shared.validation import (
    FIX_ACTIONS,
    ValidationReport,
    make_issue,
    validate_references,
)

HASH = "a" * 64
REF = ArtifactRef(artifact_id="manifest-1", content_hash=HASH)


class Anchor(BaseModel):
    label: str


class Draft(BaseModel):
    """A draft payload: one definition shared with the result schema, one of its own."""

    reason: str
    anchor: Anchor


class ReferenceRow(BaseModel):
    result: Annotated[str, reference_field(ReferenceKind.DIAGNOSTIC_RESULT)]


class ReferenceDraft(BaseModel):
    source_ids: Annotated[tuple[str, ...], reference_field(ReferenceKind.EVIDENCE)]
    diagnostics: Annotated[tuple[str, ...], reference_field(ReferenceKind.DIAGNOSTIC)]
    assessments: tuple[ReferenceRow, ...]
    requirements: Annotated[tuple[str, ...], reference_field(ReferenceKind.REQUIREMENT)]
    columns: Annotated[tuple[str, ...], reference_field(ReferenceKind.COLUMN)]
    concepts: Annotated[tuple[str, ...], reference_field(ReferenceKind.CONCEPT)]
    edges: Annotated[tuple[str, ...], reference_field(ReferenceKind.GRAPH_EDGE)]
    alternatives: Annotated[tuple[str, ...], reference_field(ReferenceKind.ALTERNATIVE)]
    methods: Annotated[tuple[str, ...], reference_field(ReferenceKind.METHOD)]
    artifacts: Annotated[tuple[str, ...], reference_field(ReferenceKind.ARTIFACT)]
    supporting_evidence_ids: tuple[str, ...]


BARE: dict[str, Any] = AgentTaskResultV1.model_json_schema()


class TestResultSchema:
    def test_the_payload_property_becomes_the_draft(self) -> None:
        merged: Any = result_schema(Draft)
        assert set(merged["properties"]) == {
            "status", "payload", "missing_requirements", "conflicts", "warnings"}
        assert not ({"envelope_id", "task_id", "parent_artifact_ids", "tool_receipts",
                     "output_hash", "validation_target"} & set(merged["properties"]))
        assert set(merged["properties"]["payload"]["properties"]) == {
            "reason",
            "anchor",
        }
        assert "$defs" not in merged["properties"]["payload"]

    def test_the_defs_union_keeps_both_sides(self) -> None:
        defs: Any = result_schema(Draft)["$defs"]
        assert {"Anchor", "ContextRequirementV1"} <= set(defs)
        assert not {"ToolReceiptV1", "ToolCallStatus"} & set(defs)

    def test_a_colliding_definition_raises(self) -> None:
        clash = create_model(
            "Clash", requirement=(create_model("ContextRequirementV1", note=(str, ...)), ...))
        with pytest.raises(ValueError, match="redefines"):
            result_schema(clash)

    def test_memoized_per_draft_and_the_source_schema_is_untouched(self) -> None:
        assert result_schema(Draft) is result_schema(Draft)
        assert result_schema(Draft) is not result_schema(Anchor)
        assert AgentTaskResultV1.model_json_schema() == BARE

    def test_a_many_payload_is_an_items_array_of_the_draft(self) -> None:  # D-066
        single: Any = result_schema(Draft)
        merged: Any = result_schema(Draft, many=True)
        assert merged["properties"]["payload"] == {
            "type": "object",
            "required": ["items"],
            "properties": {"items": {"type": "array", "items": single["properties"]["payload"]}},
        }
        assert merged["$defs"] == single["$defs"]
        assert merged is result_schema(Draft, many=True) is not single

    @pytest.mark.parametrize("item_count", (1, 2))
    def test_column_batch_schema_has_the_exact_assigned_card_count(self, item_count: int) -> None:
        schema: Any = result_schema(Draft, many=True, item_count=item_count)
        items = schema["properties"]["payload"]["properties"]["items"]
        assert items["minItems"] == items["maxItems"] == item_count
        unbounded: Any = result_schema(Draft, many=True)
        assert "maxItems" not in unbounded["properties"]["payload"]["properties"]["items"]

    def test_requirement_ids_are_constrained_for_the_current_task(self) -> None:
        schema: Any = result_schema(Draft, requirement_ids=("design.cutoff",))
        requirement = schema["$defs"]["ContextRequirementV1"]
        assert requirement["properties"]["requirement_id"]["enum"] == ["design.cutoff"]

    def test_reference_fields_are_constrained_to_the_task_catalog(self) -> None:
        schema: Any = result_schema(
            ReferenceDraft, evidence_ids=("ev:shown",), diagnostic_ids=("arm_counts",),
            diagnostic_result_ids=("dr:arm_counts:abc",),
            requirement_ids=("design.assignment",), column_ids=("treat",),
            concept_ids=("c-treatment",), graph_edge_ids=("edge-1",),
            alternative_ids=("alt-1",), method_ids=("did",),
            artifact_ids=("artifact-1",))
        payload = schema["properties"]["payload"]["properties"]
        expected = {
            "source_ids": "ev:shown", "diagnostics": "arm_counts",
            "requirements": "design.assignment", "columns": "treat",
            "concepts": "c-treatment", "edges": "edge-1", "alternatives": "alt-1",
            "methods": "did", "artifacts": "artifact-1",
        }
        assert all(payload[field]["items"]["enum"] == [value]
                   for field, value in expected.items())
        result_id = schema["$defs"]["ReferenceRow"]["properties"]["result"]
        assert result_id["enum"] == ["dr:arm_counts:abc"]
        assert "enum" not in payload["supporting_evidence_ids"]["items"]
        attempted = schema["$defs"]["AttemptedEvidenceV1"]["properties"]["evidence_id"]
        assert attempted["enum"] == ["ev:shown"]
        requirement = schema["$defs"]["ContextRequirementV1"]["properties"]
        assert requirement["attempted_evidence"]["maxItems"] == 0

    def test_evidence_availability_is_harness_owned_and_budgeted(self) -> None:
        assert evidence_availability({"ev:empty": "", "ev:shown": "fact"}) == {
            "ev:empty": "empty", "ev:shown": "evidenced"}


class PayloadDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    value: str
    supporting_evidence_ids: Annotated[
        tuple[str, ...], reference_field(ReferenceKind.EVIDENCE)
    ] = ()


@dataclass(frozen=True)
class Spec:
    task_kind: str = "bounded_task"
    output_artifact_type: str = "bounded_draft"
    output_schema_version: str = "bounded-draft.v1"
    prompt_version: str = "bounded-prompt.v1"
    token_budget: int = 100
    tool_call_budget: int = 0
    correction_budget: int = 1
    wall: int = 1
    allowed_stopping_states: tuple[TaskStatus, ...] = (TaskStatus.COMPLETE,)
    allowed_requirement_ids: tuple[str, ...] = ()


def result(**overrides: object) -> dict[str, object]:
    body: dict[str, object] = {"status": "complete", "payload": {"value": "accepted"},
                              "missing_requirements": [], "conflicts": [], "warnings": []}
    body.update(overrides)
    return body


class Gateway:
    def __init__(self, replies: list[dict[str, object]]) -> None:
        self.replies = replies
        self.calls: list[AgentTaskEnvelopeV1] = []

    def invoke(self, envelope: AgentTaskEnvelopeV1, prompt: str,
               response_schema: dict[str, object]) -> GatewayResultV1:
        self.calls.append(envelope)
        body = self.replies.pop(0)
        return GatewayResultV1(
            text=json.dumps(body), parsed=body, token_usage={}, attempts=1, seed=1)


class Harness:
    def __init__(self, first: dict[str, object],
                 allowed: tuple[TaskStatus, ...], *, second: dict[str, object] | None = None,
                 evidence: frozenset[str] = frozenset(),
                 requirement_ids: tuple[str, ...] = ()) -> None:
        self.spec = Spec(
            allowed_stopping_states=allowed, allowed_requirement_ids=requirement_ids)
        self.gateway = Gateway([first, second or result()])
        self.evidence = evidence
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.commits: list[dict[str, Any]] = []
        self.upserts: list[tuple[Any, ...]] = []
        self.state: dict[str, Any] = {"analysis_id": "an-1", "stage_run_id": "run-1",
                                      "design_revision": 1, "corrections": {},
                                      "open_requirement_ids": []}
        self.runner = TaskRunner(
            gateway=self.gateway, tasks={self.spec.task_kind: self.spec},
            tools={self.spec.task_kind: ()}, evals={self.spec.task_kind: ("EV-TEST-001",)},
            prompts_root=Path("."), envelope=self.envelope, prompt=lambda *args: "test prompt",
            validate=lambda *args: ValidationReport(wall=1, issues=()), context=lambda state: None,
            manifest=lambda state: REF, evidence=lambda state: self.evidence,
            parents=lambda state, *kinds: (), commit=self.commit,
            emit=lambda state, name, evals, **kwargs: self.events.append((name, kwargs)),
            record=lambda *args: None, exhausted=lambda *args: None,
            upsert=lambda *args: self.upserts.append(args))

    def envelope(self, spec: Spec, **values: Any) -> AgentTaskEnvelopeV1:
        return AgentTaskEnvelopeV1(
            envelope_id=f"env:{values['attempt_id']}",
            schema_version="agent-task-envelope.v1",
            analysis_id=values["analysis_id"],
            stage_run_id=values["stage_run_id"],
            task_id=values["task_id"],
            attempt_id=values["attempt_id"],
            context_manifest=values["manifest_ref"],
            task_kind=spec.task_kind,
            scope_kind=values["scope_kind"],
            scope_ids=tuple(values["scope_ids"]),
            parent_artifacts=tuple(values["parent_artifacts"]),
            allowed_evidence_ids=tuple(values["allowed_evidence_ids"]),
            allowed_retrieval_ids=(),
            allowed_tool_ids=tuple(values["allowed_tool_ids"]),
            output_schema_version=spec.output_schema_version,
            validator_version="bounded-validator.v1",
            prompt_version=spec.prompt_version,
            model_profile_version="test-model.v1",
            budgets=TaskBudgets(
                token_budget=spec.token_budget,
                tool_call_budget=spec.tool_call_budget,
                correction_budget=spec.correction_budget,
            ),
            allowed_stopping_states=spec.allowed_stopping_states,
            error_vocabulary=("schema_invalid",),
            forbidden_payload_classes=(),
            payload_type=values["payload_type"],
            payload=dict(values["payload"]),
        )

    def commit(self, state: Any, kind: str, payload: Any,
               parents: tuple[ArtifactEnvelopeV1, ...]) -> ArtifactEnvelopeV1:
        self.commits.append(dict(payload))
        return ArtifactEnvelopeV1(
            artifact_id="artifact-1",
            artifact_type=kind,
            schema_version="bounded-draft.v1",
            content_hash=HASH,
            analysis_id=state["analysis_id"],
            stage_run_id=state["stage_run_id"],
            producer_component="test",
            producer_version="test.v1",
            parent_artifacts=(),
            sensitivity_class=SensitivityClass.INTERNAL,
            created_at_utc=datetime(2026, 9, 6, tzinfo=UTC),
            payload_locator="memory:artifact-1",
        )

    def run(self) -> dict[str, object]:
        completed = self.runner.run(
            self.state,
            self.spec.task_kind,
            PayloadDraft,
            scope_kind="dataset",
            scope_ids=(),
            parent_kinds=(),
            payload={"request": "draft"},
        )
        assert completed is not None
        return self.gateway.calls[1].payload["correction"] if len(self.gateway.calls) > 1 else {"normalized": completed[0][0].value}  # type: ignore[return-value]


@pytest.mark.parametrize(
    ("first", "allowed", "path", "error_type"),
    [
        (result(status="conflict", conflicts=["facts disagree"]),
         (TaskStatus.COMPLETE,), "/status", "stopping_state_not_allowed"),
        (result(status="needs_context"),
         (TaskStatus.COMPLETE, TaskStatus.NEEDS_CONTEXT), "/missing_requirements",
         "missing_requirements_required_for_needs_context"),
        (result(status="conflict"),
         (TaskStatus.COMPLETE, TaskStatus.CONFLICT), "/conflicts",
         "conflicts_required_for_conflict"),
    ],
)
def test_result_status_contracts_request_a_targeted_correction(
        first: dict[str, object], allowed: tuple[TaskStatus, ...], path: str,
        error_type: str) -> None:
    correction = Harness(first, allowed).run()
    assert correction == ({"issues": [{"code": "schema_invalid", "json_path": path, "error_type": error_type}]} if path else {"normalized": "accepted"})


def test_pydantic_errors_expose_bounded_paths_and_types_without_rejected_values() -> None:
    broken = result()
    broken.pop("status")
    broken.update({f"unexpected_{index}": f"secret-{index}" for index in range(20)})

    correction = Harness(broken, (TaskStatus.COMPLETE,)).run()
    issues = correction["issues"]
    assert isinstance(issues, list) and len(issues) == 8
    assert {"code": "schema_invalid", "json_path": "/status",
            "error_type": "missing"} in issues
    assert all(set(issue) == {"code", "json_path", "error_type"} for issue in issues)
    assert "secret-" not in json.dumps(correction)


def test_schema_events_carry_only_the_safe_repair_coordinates() -> None:
    harness = Harness(result(status="conflict"), (TaskStatus.COMPLETE,))
    harness.run()
    failed = next(fields for name, fields in harness.events if name == "agent.schema_failed")
    assert failed["safe_dimensions"] == {
        "json_path": "/status", "error_type": "stopping_state_not_allowed"}


def test_reference_only_correction_cannot_change_the_semantic_decision() -> None:
    first = result(payload={"value": "original",
                            "supporting_evidence_ids": ["invented"]})
    harness = Harness(
        first, (TaskStatus.COMPLETE,), second=result(payload={"value": "changed"}),
        evidence=frozenset({"ev:shown"}),
    )
    completed = harness.runner.run(
        harness.state, harness.spec.task_kind, PayloadDraft,
        scope_kind="dataset", scope_ids=(), parent_kinds=(),
        payload={"request": "draft"})
    assert completed is None
    assert harness.state["corrections"][
        "bounded_draft:reference_repair_changed_semantics"] == 2
    assert not any(name == "task.completed" for name, _ in harness.events)


def test_reference_repair_cannot_change_an_unrelated_valid_reference() -> None:
    harness = Harness(
        result(payload={"value": "same", "supporting_evidence_ids": [
            "invented", "ev:keep"]}),
        (TaskStatus.COMPLETE,),
        second=result(payload={"value": "same", "supporting_evidence_ids": [
            "ev:fixed", "ev:changed"]}),
        evidence=frozenset({"ev:keep", "ev:fixed", "ev:changed"}),
    )
    completed = harness.runner.run(
        harness.state, harness.spec.task_kind, PayloadDraft,
        scope_kind="dataset", scope_ids=(), parent_kinds=(), payload={"request": "draft"})
    assert completed is None
    assert harness.state["corrections"][
        "bounded_draft:reference_repair_changed_unrelated_reference"] == 2
    assert not harness.commits and not harness.upserts


def test_reference_repair_can_change_only_the_exact_failing_path() -> None:
    harness = Harness(
        result(payload={"value": "same", "supporting_evidence_ids": [
            "invented", "ev:keep"]}),
        (TaskStatus.COMPLETE,),
        second=result(payload={"value": "same", "supporting_evidence_ids": [
            "ev:fixed", "ev:keep"]}),
        evidence=frozenset({"ev:keep", "ev:fixed"}),
    )
    completed = harness.runner.run(
        harness.state, harness.spec.task_kind, PayloadDraft,
        scope_kind="dataset", scope_ids=(), parent_kinds=(), payload={"request": "draft"})
    assert completed is not None
    assert completed[0][0].supporting_evidence_ids == ("ev:fixed", "ev:keep")
    assert len(harness.commits) == len(harness.upserts) == 1


@pytest.mark.parametrize(("original_refs", "repaired_refs", "allowed"), (
    (["invented", "ev:keep"], ["ev:keep"], True),
    (["invented", "ev:keep"], ["ev:fixed", "ev:keep"], True),
    (["invented", "ev:keep"], ["ev:fixed"], False),
    (["invented", "ev:keep"], ["ev:fixed", "ev:changed"], False),
    (["invented", "ev:keep", "ev:keep"], ["ev:keep", "ev:keep"], True),
    (["invented", "ev:keep", "ev:keep"], ["ev:keep"], False),
    (["ev:keep", "invented", "ev:changed"], ["ev:changed", "ev:keep"], False),
    (["invented", "ev:keep", "invented-again", "ev:changed"],
     ["ev:keep", "ev:fixed", "ev:changed"], True),
))
def test_reference_array_repair_preserves_valid_values_order_and_multiplicity(
    original_refs: list[str], repaired_refs: list[str], allowed: bool,
) -> None:
    harness = Harness(
        result(payload={"value": "same", "supporting_evidence_ids": original_refs}),
        (TaskStatus.COMPLETE,),
        second=result(payload={"value": "same", "supporting_evidence_ids": repaired_refs}),
        evidence=frozenset({"ev:keep", "ev:fixed", "ev:changed"}))
    completed = harness.runner.run(
        harness.state, harness.spec.task_kind, PayloadDraft,
        scope_kind="dataset", scope_ids=(), parent_kinds=(), payload={"request": "draft"})
    assert (completed is not None) is allowed
    assert len(harness.commits) == len(harness.upserts) == int(allowed)
    if not allowed:
        assert harness.state["corrections"][
            "bounded_draft:reference_repair_changed_unrelated_reference"] == 2


def missing_requirement(requirement_id: str) -> ContextRequirementV1:
    return ContextRequirementV1(
        requirement_id=requirement_id, registry_version="requirements.v1",
        scope_kind=RequirementScopeKind.DESIGN, scope_id="design",
        fact_required="assignment mechanism", why_required="selects the method",
        decisions_blocked=("method",), criticality=Criticality.BLOCKING,
        acceptable_evidence_types=(EvidenceClass.SOURCE_STATEMENT,),
        required_support=SupportRequirement.DIRECT, methods_required_for=("did",),
        attempted_evidence=(), user_may_know=True, expected_answer_schema="text.v1",
        missing_action=MissingAction.ASK_USER)


@pytest.mark.parametrize("second_failure", ("semantic_change", "schema_invalid"))
@pytest.mark.parametrize("many", (False, True))
def test_reference_correction_keeps_the_complete_original_decision_across_attempts(
    second_failure: str, many: bool,
) -> None:
    original = result(
        status="needs_context", payload={"value": "original",
            "supporting_evidence_ids": ["invented", "ev:keep"]},
        missing_requirements=[missing_requirement("assignment.known").model_dump(mode="json")],
        conflicts=["retain this conflict"], warnings=["retain this warning"])
    repaired = original | {"payload": {"value": "original",
                                      "supporting_evidence_ids": ["ev:fixed", "ev:keep"]}}
    invalid = result(payload={"value": "changed", "supporting_evidence_ids": ["ev:fixed"]})
    if many:
        for decision in (original, repaired, invalid):
            decision["payload"] = {"items": [decision["payload"]]}
    harness = Harness(
        original, (TaskStatus.COMPLETE, TaskStatus.NEEDS_CONTEXT),
        second=invalid if second_failure == "semantic_change" else {"payload": {}},
        evidence=frozenset({"ev:fixed", "ev:keep"}), requirement_ids=("assignment.known",))
    harness.spec = replace(harness.spec, correction_budget=2)
    harness.gateway.replies.append(repaired)
    runner = replace(harness.runner, tasks={harness.spec.task_kind: harness.spec})
    completed = runner.run(
        harness.state, harness.spec.task_kind, PayloadDraft,
        scope_kind="dataset", scope_ids=(), parent_kinds=(), payload={"request": "draft"}, many=many)
    assert completed is not None
    for request in harness.gateway.calls[1:]:
        correction = cast(dict[str, Any], request.payload["correction"])
        assert correction["baseline_decision"] == original
        prefix = "/payload/items/0" if many else "/payload"
        assert correction["editable_reference_paths"] == [f"{prefix}/supporting_evidence_ids/0"]
        assert "failing_payload" not in correction
        assert "all other fields" in correction["instruction"]
    retry = cast(dict[str, Any], harness.gateway.calls[2].payload["correction"])
    expected = "reference_repair_changed_semantics" if second_failure == "semantic_change" else "schema_invalid"
    assert expected in {issue["code"] for issue in retry["issues"]}
    assert len(harness.commits) == len(harness.upserts) == 1
    assert harness.upserts[0][0][0].requirement_id == "assignment.known"


@pytest.mark.parametrize("boundary", ("wall", "precommit"))
@pytest.mark.parametrize("change_during_reference_repair", (False, True))
def test_repaired_references_allow_the_next_requested_semantic_correction(
    boundary: str, change_during_reference_repair: bool,
) -> None:
    requirement = missing_requirement("assignment.known").model_dump(mode="json")
    value = "changed" if change_during_reference_repair else "original"
    harness = Harness(
        result(status="needs_context", payload={"value": "original",
               "supporting_evidence_ids": ["invented"]}, missing_requirements=[requirement]),
        (TaskStatus.COMPLETE, TaskStatus.NEEDS_CONTEXT),
        second=result(status="needs_context", payload={"value": value,
                      "supporting_evidence_ids": ["ev:shown"]},
                      missing_requirements=[requirement]),
        evidence=frozenset({"ev:shown"}), requirement_ids=("assignment.known",))
    harness.spec = replace(harness.spec, correction_budget=2)
    harness.gateway.replies.append(result(
        payload={"value": value, "supporting_evidence_ids": ["ev:shown"]}))

    def semantic_issues(sealed: AgentTaskResultV1) -> tuple[Any, ...]:
        return (make_issue(
            "invalid_context_requirement", "/missing_requirements/0",
            "wall3.requirement_search", FIX_ACTIONS, False,
            detail="This requirement is already resolved; remove it."),
        ) if sealed.missing_requirements else ()

    def validate(wall: int, kind: str, model: type[BaseModel],
                 sealed: AgentTaskResultV1, ctx: Any) -> ValidationReport:
        references = validate_references(
            model, sealed.payload, {ReferenceKind.EVIDENCE: ("ev:shown",)})
        issues = references or (semantic_issues(sealed) if boundary == "wall" else ())
        return ValidationReport(wall=2 if references else 3, issues=issues)

    runner = replace(harness.runner, tasks={harness.spec.task_kind: harness.spec},
                     validate=validate)
    completed = runner.run(
        harness.state, harness.spec.task_kind, PayloadDraft,
        scope_kind="dataset", scope_ids=(), parent_kinds=(), payload={"request": "draft"},
        precommit_admission=(lambda items, sealed: semantic_issues(sealed))
        if boundary == "precommit" else None)
    if change_during_reference_repair:
        assert completed is None
        assert not harness.commits and not harness.upserts
        assert harness.state["corrections"][
            "bounded_draft:reference_repair_changed_semantics"] == 3
    else:
        assert completed is not None and completed[0][0].value == "original"
        assert len(harness.commits) == len(harness.upserts) == 1
        assert harness.upserts[0][0] == ()
        assert harness.state["open_requirement_ids"] == []
        assert harness.state["corrections"]["bounded_draft:invalid_context_requirement"] == 2
        assert "bounded_draft:reference_repair_changed_semantics" not in harness.state["corrections"]


def test_fake_gateway_cannot_escape_the_task_requirement_allowlist() -> None:
    rejected = result(
        status="needs_context", payload={"value": "same"},
        missing_requirements=[missing_requirement("outside.task").model_dump(mode="json")])
    harness = Harness(
        rejected, (TaskStatus.COMPLETE, TaskStatus.NEEDS_CONTEXT), second=rejected,
        requirement_ids=("inside.task",))
    completed = harness.runner.run(
        harness.state, harness.spec.task_kind, PayloadDraft,
        scope_kind="dataset", scope_ids=(), parent_kinds=(), payload={"request": "draft"})
    assert completed is None
    issue = cast(dict[str, Any], harness.gateway.calls[1].payload["correction"])["issues"][0]
    assert issue["code"] == "unresolved_requirement"
    assert issue["json_path"] == "/missing_requirements/0/requirement_id"
    assert not harness.commits and not harness.upserts


def test_precommit_admission_rejects_without_upsert_or_commit_then_corrects() -> None:
    harness = Harness(result(), (TaskStatus.COMPLETE,))
    calls = 0

    def admit(
        items: tuple[BaseModel, ...], sealed: AgentTaskResultV1,
    ) -> tuple[Any, ...]:
        nonlocal calls
        calls += 1
        assert tuple(cast(PayloadDraft, item).value for item in items) == ("accepted",)
        assert sealed.status is TaskStatus.COMPLETE
        if calls == 1:
            return (make_issue(
                "fact_binding_failed", "/payload/value", "precommit.fact_binding",
                FIX_ACTIONS, False),)
        return ()

    completed = harness.runner.run(
        harness.state, harness.spec.task_kind, PayloadDraft,
        scope_kind="dataset", scope_ids=(), parent_kinds=(), payload={"request": "draft"},
        precommit_admission=admit)
    assert completed is not None and calls == 2
    assert len(harness.upserts) == len(harness.commits) == 1
    assert harness.state["corrections"]["bounded_draft:fact_binding_failed"] == 1


def test_semantic_correction_includes_every_prior_model_owned_decision_field() -> None:
    requirements = [missing_requirement(name).model_dump(mode="json")
                    for name in ("assignment.known", "unit.known")]
    original = result(status="conflict", missing_requirements=requirements,
        conflicts=["A conflicting interpretation"], warnings=["A retained qualification"])
    harness = Harness(original, (TaskStatus.COMPLETE, TaskStatus.CONFLICT),
        requirement_ids=("assignment.known", "unit.known"))
    completed = harness.runner.run(harness.state, harness.spec.task_kind, PayloadDraft,
        scope_kind="dataset", scope_ids=(), parent_kinds=(),
        payload={"correction": {"code": "existing_task_feedback", "detail": "Preserve this"}},
        precommit_admission=lambda items, sealed: (make_issue(
            "requirement_conflict", "/missing_requirements/1", "precommit.requirements",
            FIX_ACTIONS, False),) if sealed.conflicts else ())
    assert completed is not None
    correction = harness.gateway.calls[1].payload["correction"]
    assert correction["failing_decision"] == original
    assert set(correction["failing_decision"]) == {
        "status", "payload", "missing_requirements", "conflicts", "warnings"}
    assert correction["failing_decision"]["missing_requirements"][1]["requirement_id"] == "unit.known"
    assert correction["failing_payload"] == original["payload"]
    assert correction["task_feedback"] == {"code": "existing_task_feedback", "detail": "Preserve this"}
    assert len(harness.commits) == len(harness.upserts) == 1
