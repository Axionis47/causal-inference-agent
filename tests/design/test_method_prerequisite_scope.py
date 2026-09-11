"""Replay the actual RCT proposal that asked for lower-ranked methods' prerequisites."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from causal.design import contracts, packs, semantics, v2, validators
from causal.design.compile import load_task_table
from causal.shared.agenttask import _seal_result
from causal.shared.envelope import ContextRequirementV1, TaskStatus
from causal.shared.validation import parse_strict
from tests.shared.test_agenttask import REF, Harness

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/shared/fixtures/rct_method_prerequisite_scope.json"


def captured() -> tuple[dict[str, Any], validators.ValidationContext, dict[str, Any]]:
    case = json.loads(FIXTURE.read_text())
    latest = {row["artifact_type"]: row for row in case["artifacts"]}
    sections = dict(part.split("\n", 1) for part in case["captured_request"]["prompt"].split("\n\n## ")[1:])
    evidence: dict[str, str] = {}
    key = ""
    for line in sections["allowed_evidence"].splitlines():
        if line.startswith("  "):
            evidence[key] += ("\n" if evidence[key] else "") + line[2:]
        else:
            key = line.removesuffix(":")
            evidence[key] = ""
    catalog = json.loads(sections["reference_catalog"])
    graph = parse_strict(semantics.CausalContextV1, latest["CausalContext"]["payload"])
    context = validators.ValidationContext(
        manifest=parse_strict(contracts.DesignContextManifestV1, latest["DesignContextManifest"]["payload"]),
        intent=parse_strict(contracts.DesignIntentV1, latest["DesignIntent"]["payload"]),
        measurement_map=parse_strict(semantics.MeasurementMapV1, latest["MeasurementMap"]["payload"]),
        causal_context=graph,
        role_ledger=parse_strict(semantics.RoleLedgerV1, latest["RoleLedger"]["payload"]),
        rules=validators.load_validation_rules(ROOT / "registries/design-validation-rules.v1.json"),
        parents={row["artifact_id"]: row["content_hash"] for row in case["artifacts"]},
        evidence_ids=frozenset(catalog["source_evidence_ids"]), evidence_text=evidence,
        concept_ids=frozenset(graph.concept_ids),
        templates=packs.load_requirement_templates(ROOT / "registries/context-requirements.v1.json"),
        packs=packs.load_method_packs(ROOT / "registries/method-packs.v1.json"),
        diagnostic_ids=frozenset(catalog["registered_diagnostic_ids"]))
    return case["captured_decision"], context, case


def test_exact_model_requests_are_corrected_before_requirement_upsert_and_proposal_commit() -> None:
    original, context, _ = captured()
    corrected = deepcopy(original)
    corrected.update(status="complete", missing_requirements=[])
    assert corrected["payload"] == original["payload"]  # Scientific proposal/ranking unchanged.
    harness = Harness(original, (TaskStatus.COMPLETE, TaskStatus.NEEDS_CONTEXT), second=corrected,
                      evidence=frozenset(context.evidence_ids), requirement_ids=tuple(context.templates))
    spec = load_task_table(ROOT / "registries/design-tasks.v1.json")["method_design"]
    runner = replace(harness.runner, tasks={spec.task_kind: spec}, tools={spec.task_kind: ()},
                     evals={spec.task_kind: ("EV-P2-006",)}, context=lambda _: context,
                     evidence=lambda _: context.evidence_text, validate=validators.validate_result)
    completed = runner.run(harness.state, spec.task_kind, v2.AgentDesignProposalV2,
                           scope_kind="design", scope_ids=("agent_design_proposal",), parent_kinds=(),
                           payload={"available_diagnostics": {pack.method_id: list(pack.allowed_prerepair_diagnostic_ids)
                                    for pack in context.packs.all()}})
    assert completed is not None
    assert len(harness.gateway.calls) == 2
    assert harness.commits == [original["payload"]]
    assert len(harness.upserts) == 1 and harness.upserts[0][0] == ()
    correction = harness.gateway.calls[1].payload["correction"]
    issues = correction["issues"]
    assert {row["code"] for row in issues} == {"method_requirement_outside_preferred_design"}
    assert {row["artifact_ids"][0] for row in issues} == {
        "design.cutoff", "design.sharp_assignment", "design.adoption_time"}
    assert all(row["rule_id"] == "wall3.preferred_method_requirements" for row in issues)


@pytest.mark.parametrize("method,assignment,estimand,requirement", (
    ("randomized_experiment", "randomized", "itt", "design.population_comparator"),
    ("aipw", "self_selected", "ate", "design.population_comparator"),
    ("did", "time_of_adoption", "att_group_time_aggregate", "design.adoption_time"),
    ("sharp_rdd", "policy_cutoff", "late_at_cutoff", "design.cutoff"),
    ("sharp_rdd", "policy_cutoff", "late_at_cutoff", "design.sharp_assignment"),
))
def test_missing_preferred_method_prerequisite_remains_requestable_for_each_family(
    method: str, assignment: str, estimand: str, requirement: str,
) -> None:
    original, context, _ = captured()
    template = context.templates[requirement]
    request = next(row for row in original["missing_requirements"] if row["requirement_id"] == requirement)
    assert method in template.methods_required_for
    body = original["payload"] | {"assignment_mechanism": assignment,
        "requested_estimand": estimand, "source_interpretations": [], "optional_risk_ids": [],
        "ranked_method_ids": [method] + [name for name in original["payload"]["ranked_method_ids"] if name != method]}
    proposal = parse_strict(v2.AgentDesignProposalV2, body)
    harness = Harness(original, (TaskStatus.NEEDS_CONTEXT,))
    envelope = harness.envelope(harness.spec, analysis_id="an-1", stage_run_id="run-1",
        task_id="task", attempt_id="task:1", manifest_ref=REF,
        scope_kind="design", scope_ids=("design",), parent_artifacts=(), allowed_evidence_ids=(),
        allowed_tool_ids=(), payload_type="method_design-context", payload={})
    result = _seal_result(envelope, "AgentDesignProposal", original | {
        "payload": body, "missing_requirements": [parse_strict(ContextRequirementV1, request).model_dump(mode="json")]})
    assert validators.wall_evidence(proposal, result, context).passed


def test_saved_settlement_and_applicability_failures_share_complete_wall3_feedback() -> None:
    case = json.loads((ROOT / "tests/shared/fixtures/rct_semantic_correction.json").read_text())
    original, task_payload = case["decision"], case["request_context"]
    _, base_context, _ = captured()
    context = replace(base_context,
        resolved_requirements={f"{row['requirement_id']}::{row['scope_id']}": row["state"]
            for row in task_payload["prerequisite_context"]["settled_requirements"]},
        diagnostic_result_ids=frozenset(row["diagnostic_result_id"]
                                       for row in task_payload["diagnostic_results"]))
    corrected = deepcopy(original) | {"status": "complete", "missing_requirements": []}
    harness = Harness(original, (TaskStatus.COMPLETE, TaskStatus.NEEDS_CONTEXT), second=corrected,
        evidence=frozenset(context.evidence_ids), requirement_ids=tuple(context.templates))
    spec = load_task_table(ROOT / "registries/design-tasks.v1.json")["method_design"]
    runner = replace(harness.runner, tasks={spec.task_kind: spec}, tools={spec.task_kind: ()},
        evals={spec.task_kind: ("EV-P2-006",)}, context=lambda _: context,
        evidence=lambda _: context.evidence_text, validate=validators.validate_result)
    completed = runner.run(harness.state, spec.task_kind, v2.AgentDesignProposalV2,
        scope_kind="design", scope_ids=("agent_design_proposal",), parent_kinds=(), payload=task_payload)
    assert completed is not None
    assert len(harness.gateway.calls) == 2 and len(harness.commits) == len(harness.upserts) == 1
    correction = harness.gateway.calls[1].payload["correction"]
    assert {(row["code"], row["json_path"]) for row in correction["issues"]} == {
        ("invalid_context_requirement", "/missing_requirements/1"),
        ("method_requirement_outside_preferred_design", "/missing_requirements/2")}
    assert all(row["rule_id"].startswith("wall3.") for row in correction["issues"])
    assert correction["failing_decision"] == original
    assert correction["failing_decision"]["missing_requirements"][1]["requirement_id"] == "design.unit_identity"
    assert correction["failing_payload"] == original["payload"]
    assert correction["task_feedback"] == task_payload["correction"]
    assert completed[0][0].comparator == original["payload"]["comparator"] == "treatment=0"


def test_reference_failure_still_precedes_requirement_settlement_and_applicability() -> None:
    original, context, _ = captured()
    changed = deepcopy(original)
    changed["payload"]["ranked_method_ids"][0] = "invented_method"
    envelope = Harness(changed, (TaskStatus.NEEDS_CONTEXT,)).envelope(
        load_task_table(ROOT / "registries/design-tasks.v1.json")["method_design"],
        analysis_id="an-1", stage_run_id="run-1", task_id="task", attempt_id="task:1",
        manifest_ref=REF, scope_kind="design", scope_ids=("design",), parent_artifacts=(),
        allowed_evidence_ids=tuple(context.evidence_ids), allowed_tool_ids=(),
        payload_type="method_design-context", payload={})
    report = validators.validate_result(6, "method_design", v2.AgentDesignProposalV2,
        _seal_result(envelope, "AgentDesignProposal", changed), context)
    assert report.wall == 2
    assert {row.code for row in report.issues} == {"unresolved_method"}
