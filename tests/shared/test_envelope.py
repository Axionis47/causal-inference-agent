"""Tests for the shared agent-task envelope contracts (T-009, EV-SYS-001 unit layer)."""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum

import pytest
from pydantic import ValidationError

from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef
from causal.shared.envelope import (
    AgentTaskEnvelopeV1,
    AgentTaskResultV1,
    AttemptedEvidenceV1,
    CausalFrameV1,
    ContextRequirementV1,
    Criticality,
    EpistemicStatus,
    EvidenceClass,
    MissingAction,
    RequirementScopeKind,
    SupportClass,
    SupportRequirement,
    TaskBudgets,
    TaskStatus,
)

Model = type[AgentTaskEnvelopeV1] | type[AgentTaskResultV1]
Kwargs = Callable[..., dict[str, object]]

HASH = "a" * 64
REF = ArtifactRef(artifact_id="art-1", content_hash=HASH)
FRAME = CausalFrameV1(treatment="promo", outcome="revenue", population="stores", timeframe="2024")
BUDGETS = TaskBudgets(token_budget=8000, tool_call_budget=4)
REQUIREMENT = ContextRequirementV1(
    requirement_id="req-1",
    registry_version="requirements.v1",
    scope_kind=RequirementScopeKind.COLUMN,
    scope_id="stores.promo_flag",
    fact_required="whether promo_flag is recorded before the promotion starts",
    why_required="pre-treatment timing decides whether the column is a confounder",
    decisions_blocked=("role_assignment",),
    criticality=Criticality.BLOCKING,
    acceptable_evidence_types=(EvidenceClass.DATA_DICTIONARY, EvidenceClass.USER_CONFIRMATION),
    required_support=SupportRequirement.DIRECT_OR_CORROBORATED,
    methods_required_for=("difference-in-differences",),
    attempted_evidence=(AttemptedEvidenceV1(evidence_id="ev-1", availability_status="empty"),),
    user_may_know=True,
    expected_answer_schema="free-text.v1",
    missing_action=MissingAction.ASK_USER,
)


def envelope_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "envelope_id": "env-1",
        "schema_version": "agent-task-envelope.v1",
        "analysis_id": "an-1",
        "stage_run_id": "run-1",
        "task_id": "task-1",
        "attempt_id": "attempt-1",
        "context_manifest": REF,
        "task_kind": "column_card",
        "scope_kind": "column",
        "scope_ids": ("stores.promo_flag",),
        "parent_artifacts": (REF,),
        "allowed_evidence_ids": ("ev-1",),
        "allowed_retrieval_ids": ("surface-1",),
        "allowed_tool_ids": ("column-profile",),
        "output_schema_version": "column-semantic-card.v1",
        "validator_version": "column-semantic-card-validator.v1",
        "prompt_version": "column-card.v1",
        "model_profile_version": "design-worker.v1",
        "budgets": BUDGETS,
        "allowed_stopping_states": (TaskStatus.COMPLETE, TaskStatus.NEEDS_CONTEXT),
        "error_vocabulary": ("SCHEMA_INVALID",),
        "forbidden_payload_classes": ("raw_rows",),
        "payload_type": "column-card-request.v1",
        "payload": {"column_name": "promo_flag", "ordinal": 3},
    }
    base.update(overrides)
    return base


def result_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "envelope_id": "env-1",
        "schema_version": "agent-task-result.v1",
        "task_id": "task-1",
        "status": TaskStatus.NEEDS_CONTEXT,
        "artifact_type": "ColumnSemanticCard",
        "artifact_schema_version": "column-semantic-card.v1",
        "parent_artifact_ids": ("art-1",),
        "payload": {"column_name": "promo_flag"},
        "missing_requirements": (REQUIREMENT,),
        "conflicts": (),
        "warnings": ("timing slot left unknown",),
        "validation_target": "column-semantic-card.v1",
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize(
    ("model", "kwargs"),
    [(AgentTaskEnvelopeV1, envelope_kwargs), (AgentTaskResultV1, result_kwargs)],
)
class TestEnvelopeAndResult:
    def test_roundtrip(self, model: Model, kwargs: Kwargs) -> None:
        built = model(**kwargs())  # type: ignore[arg-type]
        assert model.model_validate(built.model_dump()) == built

    def test_canonical_hash_stable(self, model: Model, kwargs: Kwargs) -> None:
        built = model(**kwargs())  # type: ignore[arg-type]
        assert content_hash(built.canonical_payload()) == content_hash(built.canonical_payload())

    def test_extra_field_rejected(self, model: Model, kwargs: Kwargs) -> None:
        with pytest.raises(ValidationError):
            model(**kwargs(surprise="x"))  # type: ignore[arg-type]


def test_result_carries_full_requirement() -> None:
    result = AgentTaskResultV1(**result_kwargs())  # type: ignore[arg-type]
    requirement = result.missing_requirements[0]
    assert requirement.missing_action is MissingAction.ASK_USER
    assert requirement.attempted_evidence[0].availability_status == "empty"
    assert result.canonical_payload()["missing_requirements"][0]["scope_kind"] == "column"


class TestBudgets:
    def test_defaults(self) -> None:
        assert (BUDGETS.transient_attempt_budget, BUDGETS.correction_budget) == (3, 2)

    @pytest.mark.parametrize(
        ("field", "bad"), [("token_budget", 0), ("token_budget", -1), ("tool_call_budget", -1)]
    )
    def test_bounds_enforced(self, field: str, bad: int) -> None:
        kwargs: dict[str, int] = {"token_budget": 100, "tool_call_budget": 0, field: bad}
        with pytest.raises(ValidationError):
            TaskBudgets(**kwargs)

    def test_zero_tool_calls_allowed(self) -> None:
        assert TaskBudgets(token_budget=1, tool_call_budget=0).tool_call_budget == 0


@pytest.mark.parametrize(
    ("enum_type", "expected"),
    [
        (TaskStatus, ("complete", "conflict", "needs_context", "refused")),
        (EpistemicStatus, ("disputed", "evidenced", "hypothesis", "unknown")),
        (SupportClass, ("conflicting", "corroborated_source_inference",
                        "direct_source_statement", "direct_user_confirmation",
                        "measured_observation", "model_hypothesis", "unknown")),
        (MissingAction, ("ask_user", "refuse", "retain_as_sensitivity")),
    ],
)
def test_enum_vocabulary(enum_type: type[StrEnum], expected: tuple[str, ...]) -> None:
    assert tuple(sorted(member.value for member in enum_type)) == expected
