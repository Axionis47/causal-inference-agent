"""Incomplete provider output consumes ordinary corrections, never transport retries."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from causal.shared.agenttask import result_schema
from causal.shared.envelope import TaskStatus
from causal.shared.gateway import GATEWAY_ERROR_CODES, MODEL_OUTPUT_TRUNCATED, GatewayError
from causal.shared.validation import ValidationIssueV1, ValidationReport, parse_strict
from tests.shared.test_agenttask import Gateway, Harness, PayloadDraft, result


class AttemptGateway(Gateway):
    def invoke(self, envelope: Any, prompt: str, response_schema: dict[str, object]) -> Any:
        if isinstance(self.replies[0], GatewayError):
            self.calls.append(envelope)
            raise self.replies.pop(0)
        return super().invoke(envelope, prompt, response_schema)


def setup(replies: list[Any]) -> tuple[Harness, list[Any]]:
    harness = Harness(result(), (TaskStatus.COMPLETE,), evidence=frozenset({"ev:valid"}))
    harness.gateway = AttemptGateway(replies)
    spec = replace(harness.spec, correction_budget=2)
    exhausted: list[Any] = []
    harness.runner = replace(harness.runner, gateway=harness.gateway,
        tasks={spec.task_kind: spec}, exhausted=lambda *args: exhausted.append(args))
    return harness, exhausted


def run(harness: Harness) -> Any:
    # The exact diagnostic correction in the captured fifth RCT method task. It must survive.
    return harness.runner.run(harness.state, harness.spec.task_kind, PayloadDraft,
        scope_kind="design", scope_ids=("agent_design_proposal",), parent_kinds=(),
        payload={"correction": {"code": "diagnostic_tool_budget_exhausted",
                                "detail": "diagnostic tool-call budget exhausted"}})


def truncated() -> GatewayError:
    return GatewayError("untrusted provider details must not be copied", MODEL_OUTPUT_TRUNCATED)


def test_truncation_then_complete_commits_once_with_same_identity_and_existing_budget() -> None:
    harness, exhausted = setup([truncated(), result()])
    assert run(harness) is not None
    assert len(harness.commits) == len(harness.upserts) == 1
    assert not exhausted
    first, second = harness.gateway.calls
    assert first.task_id == second.task_id
    assert first.attempt_id.endswith(":1") and second.attempt_id.endswith(":2")
    assert first.budgets == second.budgets and first.budgets.correction_budget == 2
    assert first.model_profile_version == second.model_profile_version
    assert first.parent_artifacts == second.parent_artifacts
    correction = second.payload["correction"]
    assert correction["code"] == "diagnostic_tool_budget_exhausted"
    assert correction["detail"] == "diagnostic tool-call budget exhausted"
    assert "compact" in correction["output_constraint"]
    assert "untrusted provider details" not in str(second.payload)
    assert any(name == "agent.correction_requested" and row["error_code"] == MODEL_OUTPUT_TRUNCATED
               for name, row in harness.events)


def test_all_truncated_exhausts_three_attempts_without_artifact_or_requirement_mutation() -> None:
    harness, exhausted = setup([truncated(), truncated(), truncated()])
    assert run(harness) is None
    assert len(harness.gateway.calls) == 3
    assert not harness.commits and not harness.upserts
    assert len(exhausted) == 1
    assert exhausted[0][-1][0].code == MODEL_OUTPUT_TRUNCATED
    assert harness.state["corrections"][f"bounded_draft:{MODEL_OUTPUT_TRUNCATED}"] == 3
    correction = harness.gateway.calls[-1].payload["correction"]
    assert len(correction["issues"]) == 1  # Repeated output failures do not grow the prompt.


@pytest.mark.parametrize("drift", (False, True))
def test_reference_only_freeze_survives_intervening_truncation(drift: bool) -> None:
    first = result(payload={"value": "original scientific claim",
                            "supporting_evidence_ids": ["ev:valid", "ev:invalid"]})
    repaired = deepcopy(first)
    repaired["payload"]["supporting_evidence_ids"] = ["ev:valid"]
    if drift:
        repaired["payload"]["value"] = "changed scientific claim"
    harness, exhausted = setup([first, truncated(), repaired])
    completed = run(harness)
    second, third = harness.gateway.calls[1:]
    correction_before = second.payload["correction"]
    correction_after = third.payload["correction"]
    for key in ("baseline_decision", "editable_reference_paths", "instruction"):
        assert correction_after[key] == correction_before[key]
    assert "compact" in correction_after["output_constraint"]
    if drift:
        assert completed is None and not harness.commits and not harness.upserts
        assert exhausted[0][-1][0].code == "reference_repair_changed_semantics"
    else:
        assert completed is not None and not exhausted
        assert harness.commits == [repaired["payload"]]


@pytest.mark.parametrize("code", sorted(GATEWAY_ERROR_CODES - {MODEL_OUTPUT_TRUNCATED}))
def test_other_gateway_failures_propagate_without_correction(code: str) -> None:
    error = GatewayError("private provider details", code)
    harness, exhausted = setup([error])
    with pytest.raises(GatewayError) as caught:
        run(harness)
    assert caught.value is error
    assert len(harness.gateway.calls) == 1
    assert not harness.commits and not harness.upserts and not exhausted
    assert not harness.state["corrections"]


def test_schema_correction_preserves_original_feedback_and_reports_latest_issue() -> None:
    invalid = result(status="needs_context", missing_requirements=[])
    harness, exhausted = setup([invalid, result()])
    assert run(harness) is not None
    assert harness.gateway.calls[1].payload["correction"]["task_feedback"] == {
        "code": "diagnostic_tool_budget_exhausted", "detail": "diagnostic tool-call budget exhausted"}
    harness, exhausted = setup([truncated(), invalid, invalid])
    assert run(harness) is None
    assert exhausted[0][-1][0].code == "schema_invalid"
    assert not harness.commits and not harness.upserts


def test_exact_focused_replay_keeps_task_feedback_after_wall_failure_and_reports_schema_failure() -> None:
    from causal.design import packs, v2

    case = json.loads((Path(__file__).parent / "fixtures/rct_method_feedback_loss.json").read_text())
    original, final = case["attempt2"], case["attempt3"]
    evidence = frozenset(row["evidence_id"] for decision in (original, final)
                         for row in decision["payload"]["source_interpretations"])
    harness, exhausted = setup([truncated(), original, final])
    spec = replace(harness.spec, correction_budget=2,
                   allowed_stopping_states=(TaskStatus.COMPLETE, TaskStatus.NEEDS_CONTEXT),
                   allowed_requirement_ids=tuple(row["requirement_id"]
                                                 for row in original["missing_requirements"]))
    root = Path(__file__).resolve().parents[2]
    context = SimpleNamespace(packs=packs.load_method_packs(root / "registries/method-packs.v1.json"))
    diagnostic_results = [{"diagnostic_result_id": row["diagnostic_result_id"]}
                          for row in original["payload"]["diagnostic_assessments"]]
    known_issue = parse_strict(ValidationIssueV1, case["validation_issue"])
    # Reproduce the exact saved wall result; schema parsing and all typed references still
    # run normally. The source response bodies are never repaired by this test harness.
    runner = replace(harness.runner, tasks={spec.task_kind: spec},
        evidence=lambda _: evidence, context=lambda _: context,
        validate=lambda wall, kind, model, result, ctx: ValidationReport(
            wall=3, issues=(known_issue,) if result.missing_requirements else ()))
    completed = runner.run(harness.state, spec.task_kind, v2.AgentDesignProposalV2,
        scope_kind="design", scope_ids=("agent_design_proposal",), parent_kinds=(),
        payload={"correction": case["original_task_correction"],
                 "agent_loop": case["original_agent_loop"], "diagnostic_results": diagnostic_results,
                 "available_diagnostics": {"randomized_experiment": ["outcome_missingness", "compliance_availability"]}})
    assert completed is None and len(harness.gateway.calls) == 3
    assert not harness.commits and not harness.upserts
    for call in harness.gateway.calls[1:]:
        assert call.payload["correction"]["task_feedback"] == case["original_task_correction"]
        assert call.payload["agent_loop"] == case["original_agent_loop"]
    latest_feedback = harness.gateway.calls[-1].payload["correction"]
    assert [row["code"] for row in latest_feedback["issues"]] == ["invalid_context_requirement"]
    assert "output_constraint" not in latest_feedback  # Superseded corrections do not accumulate.
    assert exhausted[0][-1][0].code == "schema_invalid"
    assert "missing_requirements_required_for_needs_context" in exhausted[0][-1][0].detail


@pytest.mark.parametrize("remaining,expected", ((0, 0), (2, 2), (10, 4)))
def test_diagnostic_request_limit_only_narrows_request_count(remaining: int, expected: int) -> None:
    from causal.design.v2 import AgentDesignProposalV2

    original = result_schema(AgentDesignProposalV2, diagnostic_ids=("arm_counts",),
                             diagnostic_result_ids=("dr:observed",))
    narrowed = result_schema(AgentDesignProposalV2, diagnostic_ids=("arm_counts",),
        diagnostic_result_ids=("dr:observed",), diagnostic_request_limit=remaining)
    expected_schema = deepcopy(original)
    expected_schema["properties"]["payload"]["properties"]["requested_diagnostic_ids"]["maxItems"] = expected
    assert narrowed == expected_schema
    assert original["properties"]["payload"]["properties"]["requested_diagnostic_ids"]["maxItems"] == 4
