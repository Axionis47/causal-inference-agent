"""Observed comparator spelling is corrected before a completed proposal is committed."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from causal.design import compiler_v2, contracts, resolution_v2, v2, validators
from causal.design.compile import load_task_table
from causal.design.harness_nodes import PipelineNodes, _observed_comparator_issues
from causal.shared.envelope import TaskStatus
from causal.shared.validation import parse_strict
from tests.design.test_method_prerequisite_scope import ROOT, captured
from tests.design.test_v2_compiler import fact
from tests.shared.test_agenttask import REF, Harness


def saved_case() -> tuple[Any, ...]:
    fixture = json.loads((ROOT / "tests/shared/fixtures/rct_semantic_correction.json").read_text())
    _, context, _ = captured()
    payload = fixture["request_context"]
    observed = tuple(parse_strict(contracts.DiagnosticResultV1, row["observation"])
                     for row in payload["diagnostic_results"])
    context = replace(context, diagnostic_result_ids=frozenset(
        row["diagnostic_result_id"] for row in payload["diagnostic_results"]))
    draft = parse_strict(v2.AgentDesignProposalV2, fixture["decision"]["payload"])
    facts = resolution_v2.compile_fact_set(selected_csv=observed[0].csv_artifact,
        ledger=context.role_ledger, ledger_ref=REF, facts=(
            fact("assignment_mechanism", "randomized"), fact("estimand", "itt"),
            fact("grain", "one_row_per_unit"), fact("comparator", draft.comparator)))
    pack = context.packs.get("randomized_experiment")
    return fixture, context, draft, facts, pack, compiler_v2.compile_diagnostic_plan(pack, facts), observed


@pytest.mark.parametrize("observed_count", (0, 1))
def test_absent_observed_treatment_levels_do_not_block_initial_investigation(observed_count: int) -> None:
    _, _, draft, _, pack, plan, observed = saved_case()
    # Unit-uniqueness observations do not establish treatment levels.
    assert not _observed_comparator_issues(pack, draft.comparator, plan, observed[1:1 + observed_count])


@pytest.mark.parametrize("method", ("did", "sharp_rdd"))
def test_comparator_guard_does_not_change_did_or_rdd(method: str) -> None:
    _, context, draft, _, _, plan, observed = saved_case()
    assert not _observed_comparator_issues(context.packs.get(method), draft.comparator, plan, observed)


def test_aipw_uses_its_observed_prevalence_levels_for_the_same_existing_contrast_check() -> None:
    _, context, draft, _, _, plan, observed = saved_case()
    prevalence = observed[0].model_copy(update={"diagnostic_id": "treatment_prevalence"})
    plan = plan.model_copy(update={"candidate_method_id": "aipw"})
    issues = _observed_comparator_issues(context.packs.get("aipw"), draft.comparator, plan, (prevalence,))
    assert [row.code for row in issues] == ["comparator_level_unresolved"]
    assert not _observed_comparator_issues(context.packs.get("aipw"), "0", plan, (prevalence,))


@pytest.mark.parametrize("status", (TaskStatus.COMPLETE, TaskStatus.NEEDS_CONTEXT))
def test_actual_admission_corrects_saved_complete_comparator_before_commit_and_leaves_context_requests(
    status: TaskStatus,
) -> None:
    fixture, context, draft, facts, _, _, observed = saved_case()
    original = deepcopy(fixture["decision"])
    original["status"] = status.value
    # A genuine preferred-method request remains allowed; the comparator guard applies only to complete.
    original["missing_requirements"] = original["missing_requirements"][:1] if status is TaskStatus.NEEDS_CONTEXT else []
    corrected = deepcopy(original)
    corrected["payload"]["comparator"] = "0"
    for row in corrected["payload"]["source_interpretations"]:
        if row["fact_key"] == "comparator":
            row["value"] = "0"
    harness = Harness(original, (TaskStatus.COMPLETE, TaskStatus.NEEDS_CONTEXT), second=corrected,
        evidence=context.evidence_ids, requirement_ids=tuple(context.templates))
    spec = load_task_table(ROOT / "registries/design-tasks.v1.json")["method_design"]
    runner = replace(harness.runner, tasks={spec.task_kind: spec}, tools={spec.task_kind: ()},
        evals={spec.task_kind: ("EV-P2-006",)}, context=lambda _: context,
        evidence=lambda _: context.evidence_text, validate=validators.validate_result)
    nodes: Any = PipelineNodes.__new__(PipelineNodes)
    nodes.deps = SimpleNamespace(packs=context.packs)
    nodes._ref = lambda *args: REF
    nodes._payload = lambda *args: {}
    nodes._ctx = lambda *args, **kwargs: context
    nodes._model = lambda *args: context.causal_context
    nodes._fact_set = lambda state, proposal, **kwargs: facts.model_copy(update={"facts": tuple(
        row.model_copy(update={"value": proposal.comparator}) if row.fact_id == "comparator" else row
        for row in facts.facts)})
    completed = []

    def run_task(*args: Any, **kwargs: Any) -> None:
        found = runner.run(harness.state, spec.task_kind, v2.AgentDesignProposalV2,
            scope_kind="design", scope_ids=("agent_design_proposal",), parent_kinds=(),
            payload=fixture["request_context"],
            precommit_admission=lambda items, sealed: kwargs["precommit_admission"](
                items, sealed, observed_results=observed))
        completed.append(found)

    nodes._run_task = run_task
    nodes._investigate({"artifacts": {"RoleLedger": "ledger"}}, context.role_ledger)
    assert completed[0] is not None and len(harness.commits) == len(harness.upserts) == 1
    assert draft.comparator == "treatment=0"  # No compiler normalization or mutation.
    assert completed[0][0][0].ranked_method_ids == draft.ranked_method_ids
    assert completed[0][0][0].diagnostic_assessments == draft.diagnostic_assessments
    if status is TaskStatus.COMPLETE:
        assert len(harness.gateway.calls) == 2 and completed[0][0][0].comparator == "0"
        correction = harness.gateway.calls[1].payload["correction"]
        assert correction["failing_decision"] == original
        assert [(row["code"], row["json_path"]) for row in correction["issues"]] == [
            ("comparator_level_unresolved", "/payload/comparator")]
        assert "('0', '1')" in correction["issues"][0]["detail"]
        assert harness.commits == [corrected["payload"]]
    else:
        assert len(harness.gateway.calls) == 1 and completed[0][0][0].comparator == "treatment=0"
