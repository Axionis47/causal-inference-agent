"""Method diagnostic requests cannot exceed the remaining budget before commit."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from causal.design import diagnostics
from causal.design.harness_nodes import PipelineNodes, _diagnostic_budget_issues
from causal.design.packs import load_method_packs
from causal.design.v2 import AgentDesignProposalV2, DiagnosticAssessmentV2
from causal.shared.envelope import AgentTaskResultV1, TaskStatus
from tests.shared.test_agenttask import REF, Harness, missing_requirement, result

PACKS = load_method_packs(Path("registries/method-packs.v1.json"))
METHODS = ("randomized_experiment", "aipw", "did", "sharp_rdd")
OBSERVED = ("arm_counts", "assignment_unit_uniqueness", "cluster_sizes", "baseline_availability")
REQUESTED = ("outcome_missingness", "compliance_availability", "power_precision_feasibility")


def proposal(requested: tuple[str, ...], observed: int = 4) -> AgentDesignProposalV2:
    return AgentDesignProposalV2(
        assignment_mechanism="randomized", requested_estimand="itt", comparator="0",
        ranked_method_ids=METHODS, method_facts=(), optional_assumption_ids=(),
        optional_risk_ids=("rct.noncompliance_risk",), optional_sensitivity_ids=(),
        requested_diagnostic_ids=requested, diagnostic_assessments=tuple(
            DiagnosticAssessmentV2(diagnostic_result_id=f"dr:{name}:observed", judgment="supports")
            for name in OBSERVED[:observed]))


@pytest.mark.parametrize(("observed", "requested", "allowed"), (
    (0, OBSERVED, True), (2, REQUESTED[:2], True), (2, REQUESTED, False),
    (4, (), True), (4, REQUESTED[:1], False),
))
def test_only_remaining_requests_are_allowed_without_changing_assessments(
    observed: int, requested: tuple[str, ...], allowed: bool,
) -> None:
    draft = proposal(requested, observed)
    original = draft.model_dump(mode="json")
    issues = _diagnostic_budget_issues(draft, observed)
    assert (not issues) == allowed
    assert draft.model_dump(mode="json") == original
    if issues:
        assert issues[0].code == diagnostics.DIAGNOSTIC_TOOL_BUDGET_EXHAUSTED
        assert issues[0].json_path == "/payload/requested_diagnostic_ids"


@pytest.mark.parametrize("status", (TaskStatus.COMPLETE, TaskStatus.NEEDS_CONTEXT, TaskStatus.CONFLICT))
def test_actual_investigation_admission_checks_budget_before_every_status(
    monkeypatch: pytest.MonkeyPatch, status: TaskStatus,
) -> None:
    # Starting at zero exposes the actual closure's exhausted-budget path without model or DB IO.
    monkeypatch.setattr(diagnostics, "DIAGNOSTIC_TOOL_CALL_LIMIT", 0)
    nodes: Any = PipelineNodes.__new__(PipelineNodes)
    nodes.deps = SimpleNamespace(packs=PACKS)
    nodes._ref = lambda *args: REF
    nodes._payload = lambda *args: {}
    nodes._ctx = lambda *args, **kwargs: None
    nodes._model = lambda *args: None
    checked = []

    def run_task(*args: Any, **kwargs: Any) -> None:
        issues = kwargs["precommit_admission"](
            (proposal(REQUESTED[:1], 0),), AgentTaskResultV1.model_construct(status=status))
        checked.extend(issue.code for issue in issues)

    nodes._run_task = run_task
    assert nodes._investigate({"artifacts": {"RoleLedger": "ledger"}}, None) is None
    assert checked == [diagnostics.DIAGNOSTIC_TOOL_BUDGET_EXHAUSTED]


@pytest.mark.parametrize("status", (TaskStatus.COMPLETE, TaskStatus.NEEDS_CONTEXT, TaskStatus.CONFLICT))
def test_overbudget_decision_cannot_upsert_requirements_or_commit_before_correction(
    status: TaskStatus,
) -> None:
    requirement = missing_requirement("design.unit_identity").model_dump(mode="json")
    extras: dict[str, Any] = {"missing_requirements": [requirement]} if status is TaskStatus.NEEDS_CONTEXT else (
        {"conflicts": ["context conflicts"]} if status is TaskStatus.CONFLICT else {})
    draft = proposal(REQUESTED)
    first = result(status=status.value, payload=draft.model_dump(mode="json"), **extras)
    corrected = draft.model_copy(update={"requested_diagnostic_ids": ()})
    harness = Harness(first, (TaskStatus.COMPLETE, TaskStatus.NEEDS_CONTEXT, TaskStatus.CONFLICT),
        second=result(payload=corrected.model_dump(mode="json")), requirement_ids=("design.unit_identity",))
    spec = replace(harness.spec, task_kind="method_design")
    runner = replace(harness.runner, tasks={spec.task_kind: spec}, tools={spec.task_kind: ()},
        evals={spec.task_kind: ("EV-TEST-001",)})
    invoke = harness.gateway.invoke

    def guarded_invoke(*args: Any, **kwargs: Any) -> Any:
        assert not harness.commits and not harness.upserts
        return invoke(*args, **kwargs)

    harness.gateway.invoke = guarded_invoke
    completed = runner.run(harness.state, spec.task_kind, AgentDesignProposalV2,
        scope_kind="design", scope_ids=("agent_design_proposal",), parent_kinds=(),
        payload={"agent_loop": {"remaining_tool_calls": 0},
            "available_diagnostics": {"randomized_experiment": [*OBSERVED, *REQUESTED]},
            "diagnostic_results": [{"diagnostic_result_id": row.diagnostic_result_id}
                                   for row in draft.diagnostic_assessments]},
        ctx=SimpleNamespace(packs=PACKS),
        precommit_admission=lambda items, sealed: _diagnostic_budget_issues(items[0], 4))
    assert completed is not None
    assert len(harness.gateway.calls) == 2
    assert len(harness.commits) == len(harness.upserts) == 1
    assert harness.upserts[0][0] == ()
    assert completed[0][0].diagnostic_assessments == draft.diagnostic_assessments
    assert completed[0][0].ranked_method_ids == draft.ranked_method_ids
    assert harness.state["corrections"][
        f"bounded_draft:{diagnostics.DIAGNOSTIC_TOOL_BUDGET_EXHAUSTED}"] == 1
