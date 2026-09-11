"""Deterministic tests for the evaluation-only live model quality command."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from causal.shared.gateway import GatewayResultV1
from tests.shared.test_gateway import SCHEMA, make_envelope
from tools import model_quality as mq


@pytest.mark.parametrize(("stress", "ids"), (
    (False, mq.LIVE_IDS), (True, (*mq.LIVE_IDS, *mq.STRESS_IDS))))
def test_validate_checks_schemas_hashes_gold_separation_and_suite(stress: bool, ids: tuple[str, ...]) -> None:
    report = mq.validate(verify_sources=False, stress=stress)
    assert report["passed"] is True and report["suite"] == ("stress" if stress else "release")
    assert [row["case_id"] for row in report["checks"]] == list(ids)
    assert len(report["fixture_hash"]) == len(report["gold_hash"]) == 64


def test_an_unnecessary_context_stop_is_scored_as_ask_policy_not_old_retry_noise() -> None:
    assert mq._score({}, {}, {}, "needs_context", [], [], [{"event_name":
        "task.validation_failed", "error_code": "shape_invalid"}]) == [{
            "classification": "ask_policy", "detail": "terminal:needs_context"}]


@pytest.mark.parametrize("stress", (False, True))
def test_live_submission_carries_every_fixture_context(stress: bool) -> None:
    fixtures, _ = mq._documents(stress=stress)
    assert all(mq._intake_submission(case).context_text == case["context"]
               for case in fixtures["cases"])


def test_matrix_expands_two_datasets_per_family_into_exact_context_pairs() -> None:
    cells, gold = mq._matrix_documents()
    assert len(cells) == len({row["cell_id"] for row in cells}) == 16
    assert {row["context_variant"] for row in cells} == {"rich", "sparse"}
    for family in mq.FAMILIES:
        found = [row for row in cells if row["family"] == family]
        assert len({row["case_id"] for row in found}) == 2
        assert len(found) == 4
    assert set(gold["matrix_expectations"]) == {"rich", "sparse"}
    assert gold["matrix_expectations"]["sparse"]["minimum_question_count"] > 0


def test_cell_filter_advances_only_one_cell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = mq._initial_matrix_state("matrix-test")
    state_file = tmp_path / "state.json"
    mq._save_state(state_file, state, create=True)
    monkeypatch.setattr(mq, "validate_environment", lambda **_: [])
    monkeypatch.setenv("CAUSAL_LANGSMITH_PROJECT", "test-project")
    monkeypatch.setattr(mq.LangSmithTracer, "preflight", lambda self: None)
    monkeypatch.setattr(mq.LangSmithTracer, "flush", lambda self: None)

    def terminal(*args: Any, **kwargs: Any) -> None:
        cell = args[4]
        cell["status"] = "terminal"
        cell["result"] = {"cell_id": cell["cell_id"]}

    monkeypatch.setattr(mq, "_advance_matrix_cell", terminal)
    selected = state["cells"][6]["cell_id"]
    advanced = mq._progress_matrix(state_file, tmp_path / "out", None, selected)
    assert [row["cell_id"] for row in advanced["cells"] if row["status"] == "terminal"] == [
        selected]


def test_sparse_cell_withholds_fixture_context_and_has_a_stable_cell_id() -> None:
    cells, _ = mq._matrix_documents()
    source = next(row for row in cells if row["cell_id"] == "nhefs_aipw--sparse")
    sparse = mq._matrix_case(source)
    rich = mq._matrix_case(source | {"context_variant": "rich"})
    assert sparse["_submission_context"] is None
    assert sparse["context"] == mq.NEUTRAL_CONTEXT
    assert rich["_submission_context"] == source["context"]
    assert mq._cell_id("nhefs_aipw", "sparse") == "nhefs_aipw--sparse"


def test_matrix_validation_flags_weak_fixtures_without_failing_manifest_shape() -> None:
    report = mq.matrix_validate(verify_sources=False)
    assert report["passed"] is True and report["cell_count"] == 16
    assert report["scientific_positive_ready"] is False
    assert report["scientific_suitability"]["groupon_aipw"]["status"] == "unsuitable"
    assert report["scientific_suitability"]["card_krueger_did"]["status"] == "unsuitable"


def test_fixture_prompt_cannot_contain_the_gold_canary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixtures, gold = mq._documents()
    fixtures["cases"][0]["context"] += str(gold["prompt_leak_canary"])
    path = tmp_path / "fixtures.json"
    path.write_text(json.dumps(fixtures), encoding="utf-8")
    monkeypatch.setattr(mq, "FIXTURES", path)
    with pytest.raises(ValueError, match="gold leak"):
        mq._documents()


def test_audit_gateway_injects_trace_ids_but_stores_only_hashes_and_metrics() -> None:
    class Gateway:
        def __init__(self) -> None:
            self.envelope: Any = None

        def invoke(self, envelope: Any, prompt: str, response_schema: dict[str, object]) -> GatewayResultV1:
            self.envelope = envelope
            return GatewayResultV1(text="", parsed=None, token_usage={"total": 3}, attempts=1, seed=1)

    gateway = Gateway()
    audit = mq._AuditGateway(gateway, "GOLD_CANARY", {  # type: ignore[arg-type]
        "run_id": "run-1", "case_id": "case-1", "mode": "gate"})
    audit.invoke(make_envelope(), "private prompt", SCHEMA)
    assert gateway.envelope.payload["evaluation"]["case_id"] == "case-1"
    assert gateway.envelope.payload["evaluation"]["seed_key"] == "case-1:column_card:stores.promo_flag"
    assert set(audit.calls[0]) >= {"prompt_hash", "envelope_hash", "tokens", "case_id"}
    assert audit.calls[0]["response_shape_issues"]
    assert not {"prompt", "response", "reasoning", "gold_labels"} & set(audit.calls[0])


def test_context_pair_uses_the_same_dataset_level_model_seed_key() -> None:
    class Gateway:
        def __init__(self) -> None:
            self.keys: list[str] = []

        def invoke(self, envelope: Any, prompt: str,
                   response_schema: dict[str, object]) -> GatewayResultV1:
            self.keys.append(envelope.payload["evaluation"]["seed_key"])
            return GatewayResultV1(
                text="{}", parsed={}, token_usage={"total": 1}, attempts=1, seed=1)

    gateway = Gateway()
    for variant in mq.MATRIX_VARIANTS:
        mq._AuditGateway(gateway, "GOLD_CANARY", {
            "run_id": "matrix-1", "case_id": f"dataset--{variant}",
            "seed_case_id": "dataset", "context_variant": variant, "mode": "matrix",
        }).invoke(make_envelope(), "private prompt", SCHEMA)
    assert gateway.keys[0] == gateway.keys[1]
    assert gateway.keys[0].startswith("dataset:")


def test_diagnostic_loop_report_is_complete_typed_and_redacted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    digest, common = "a" * 64, {"analysis_id": "an-1", "raw_rows": [["secret"]]}
    rows = [common | {"event_name": "agent.diagnostic_requested", "task_id": "task-1", "attempt_number": 1, "safe_dimensions": {"diagnostic_id": "rough_overlap", "remaining_tool_calls": 3, "request_hash": digest, "reasoning": "hidden"}},
        common | {"event_name": "diagnostic.completed", "status": "pass", "safe_dimensions": {"diagnostic_id": "rough_overlap", "result_hash": digest, "warning_count": 0, "used_row_count": 614, "gold_labels": "hidden"}},
        common | {"event_name": "agent.design_revised", "task_id": "task-1", "attempt_number": 2, "safe_dimensions": {"prior_proposal_hash": digest, "revised_proposal_hash": digest, "triggering_diagnostic_ids": "rough_overlap"}},
        common | {"event_name": "agent.escalated", "task_id": "task-1", "error_code": "diagnostic_tool_budget_exhausted", "safe_dimensions": {"responsible_actor": "product", "exhausted_budget": True}}]
    monkeypatch.setattr(mq, "REPORTS", tmp_path)
    (tmp_path / "run-1.events.ndjson").write_text("\n".join(map(json.dumps, rows)))
    found = mq._diagnostic_events("run-1", "an-1")
    assert [row["decision_source"] for row in found] == ["llm_decision", "deterministic_normalization", "llm_decision", "compiler_failure"]
    assert all(row["complete"] for row in found) and found[1]["used_row_count"] == 614
    assert not any(word in json.dumps(found) for word in ("secret", "reasoning", "gold_labels"))
    allowed = [{"task_kind": "intent", "allowed_tool_count": 0}, {"task_kind": "method_design", "allowed_tool_count": 1}]
    assert mq._tool_policy(allowed) and not mq._tool_policy([{"task_kind": "intent", "allowed_tool_count": 1}])


def test_reports_are_content_hashed_and_immutable(tmp_path: Path) -> None:
    payload = {"report_id": "run-1", "passed": True}
    path, digest = mq._commit(tmp_path, "report.json", payload)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert digest == stored["report_hash"] == mq.content_hash(payload)
    with pytest.raises(FileExistsError):
        mq._commit(tmp_path, "report.json", payload)


def test_hitl_response_must_be_external_isolated_and_bound_to_request(tmp_path: Path) -> None:
    body = {
        "schema_version": mq.HITL_REQUEST_SCHEMA, "run_id": "matrix-1",
        "cell_id": "case--sparse", "case_id": "case", "context_variant": "sparse",
        "kind": "table_selection",
        "interrupt": {"interrupt_id": "artifact-1", "expected_interrupt_hash": "a" * 64,
                      "expected_revision": 1, "interrupt_kind": "table_selection"},
        "payload": [{"logical_name": "table.csv"}],
    }
    request = mq._write_request(tmp_path / "outbox", body)
    response_path = mq._response_path(tmp_path / "inbox", request)
    response_path.parent.mkdir(parents=True)
    response = {
        "schema_version": mq.HITL_RESPONSE_SCHEMA, "run_id": "matrix-1",
        "cell_id": "case--sparse", "request_hash": request["request_hash"],
        "reviewer": "independent-reviewer", "isolation_attestation": "cell_only",
        "decision": {"selected_table": "table.csv"},
    }
    response_path.write_text(json.dumps(response), encoding="utf-8")
    found = mq._hitl_response(
        response_path, {"run_id": "matrix-1"}, {"cell_id": "case--sparse"}, request)
    assert found == response
    response["isolation_attestation"] = "saw_gold"
    response_path.write_text(json.dumps(response), encoding="utf-8")
    with pytest.raises(ValueError, match="isolated HITL request"):
        mq._hitl_response(
            response_path, {"run_id": "matrix-1"}, {"cell_id": "case--sparse"}, request)


def test_matrix_state_rejects_anything_other_than_exactly_16_cells(tmp_path: Path) -> None:
    state = {"schema_version": mq.MATRIX_STATE_SCHEMA, "cells": [{}] * 15}
    path = tmp_path / "state.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly 16"):
        mq._load_state(path)


@pytest.mark.parametrize(("kind", "value"), (("value", "Assigned at enrollment."), ("unknown", None)))
def test_clarification_json_reaches_runtime_without_changing_reviewer_answer(
        kind: str, value: str | None) -> None:
    calls = []

    class Runtime:
        def answer_context(self, *args: Any) -> str:
            calls.append(args)
            return "resumed"

    request = {"kind": "clarification", "request_hash": "b" * 64, "interrupt": {
        "interrupt_id": "question-1", "expected_interrupt_hash": "a" * 64,
        "expected_revision": 1, "interrupt_kind": "clarification"}}
    response = json.loads(json.dumps({
        "reviewer": "simulated-human:isolated", "isolation_attestation": "cell_only",
        "decision": {"schema_version": "user-context-answer.v1", "packet_id": "packet-1",
                     "answers": [{"question_id": "question-1", "answer_kind": kind,
                                  "value": value}], "provenance": "user"}}))
    original_hash = mq.content_hash(response)
    cell: dict[str, Any] = {"cell_id": "case--rich", "analysis_id": "analysis-1",
                            "hitl": [], "pending_request": request}
    assert mq._apply_hitl(Runtime(), cell, request, response) == "resumed"
    assert calls[0][0] == "analysis-1"
    assert calls[0][1].model_dump(mode="json") == response["decision"]
    assert calls[0][2].expected_interrupt_hash == "a" * 64
    assert calls[0][3] == "matrix:case--rich:" + "b" * 16
    assert mq.content_hash(response) == original_hash == cell["hitl"][0]["response_hash"]
    assert cell["pending_request"] is None

    response["decision"]["answers"][0]["value"] = 42
    before = len(calls)
    with pytest.raises(ValidationError):
        mq._apply_hitl(Runtime(), cell, request, response)
    assert len(calls) == before


def test_matrix_finalize_commits_one_fresh_16_cell_report(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state = mq._initial_matrix_state("matrix-test")
    for cell in state["cells"]:
        cell["status"] = "terminal"
        cell["result"] = {
            "cell_id": cell["cell_id"], "hard_failures": [], "task_results": [],
            "execution_passed": True, "positive_analysis_passed": True,
            "scientific_suitability": {"status": "suitable", "reasons": []},
        }
    state_file = tmp_path / "matrix.state.json"
    mq._save_state(state_file, state, create=True)
    reports = tmp_path / "reports"
    approved = tmp_path / "approved.sha256"
    approved.write_text(state["gold_hash"], encoding="utf-8")
    monkeypatch.setattr(mq, "REPORTS", reports)
    monkeypatch.setattr(mq, "MATRIX_APPROVED", approved)
    monkeypatch.setattr(mq, "_worktree", lambda: state["worktree_at_start"])
    path, report = mq.matrix_finalize(state_file)
    assert report["cell_count"] == 16 and report["passed"] is True
    assert report["historical_report_inputs"] == []
    assert list(reports.glob("*.json")) == [path]
    assert mq.matrix_finalize(state_file)[0] == path


def test_hidden_context_specific_ask_policy() -> None:
    _, gold = mq._matrix_documents()
    case = {"expected_initial_question_ids": []}
    rich = {"ask_expectation": gold["matrix_expectations"]["rich"]}
    sparse = {"ask_expectation": gold["matrix_expectations"]["sparse"]}
    assert mq._ask_policy(case, rich, [])
    assert not mq._ask_policy(case, rich, ["q:design.treatment_meaning"])
    assert not mq._ask_policy(case, sparse, [])
    assert mq._ask_policy(case, sparse, ["q:design.treatment_meaning"])


def test_live_modes_fail_closed_when_observability_is_missing(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    for name in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY", "CAUSAL_LANGSMITH_PROJECT", "CAUSAL_EVAL_POSTGRES_ADMIN_DSN", "CAUSAL_EVAL_S3_ENDPOINT", "CAUSAL_EVAL_SOURCE_S3_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    assert mq.main(["calibrate"]) == 2
    assert "langsmith_api_key_missing" in capsys.readouterr().out


def test_source_s3_endpoint_is_an_explicit_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CAUSAL_EVAL_SOURCE_S3_ENDPOINT", raising=False)
    assert "source_s3_endpoint_missing" in mq.validate_environment(
        require_live=False, require_source_s3=True)


@pytest.fixture
def historical_scoring_case() -> dict[str, Any]:
    fixtures, document = mq._documents()
    case = fixtures["cases"][0]
    gold = document["cases"][case["case_id"]]
    bindings = [{"role": role, "columns": accepted[0], "concept_id": "concept:" + role}
                for role, accepted in gold["role_bindings"].items()]
    payloads = {
        "DesignFactSet": {"grain": gold["grains"][0], "facts": [
            {"fact_id": "assignment_mechanism", "value": gold["assignment_mechanisms"][0],
             "executable": True},
            {"fact_id": "estimand", "value": gold["estimands"][0], "executable": True}]},
        "CompiledDesign": {"method_id": gold["methods"][0], "unit": gold["units"][0],
            "comparator": gold["comparators"][0], "role_bindings": bindings},
        "RoleLedger": {"claims": [{**binding,
            "timing": gold.get("timing", {}).get(binding["role"], ["unknown"])[0],
            "column_refs": binding["columns"]} for binding in bindings]},
        "CausalContext": {"edges": [{"source_concept_id": "concept:" + source,
            "target_concept_id": "concept:" + target}
            for source, target in gold["required_role_relationships"]]},
        "PresentationContextManifest": {"approved": {"profile_id": gold["profiles"][0]}},
        "ClaimJudgment": {"status": gold["required_claim_statuses"][0], "decision": {},
            "qualifications": gold.get("required_qualification_fragments", [])},
        "FigurePlan": {"figures": [{"template_id": templates[0], "visual_evidence_ids": [evidence]}
            for evidence, templates in gold["accepted_templates"].items()]},
    }
    calls = [{"task_kind": kind, "task_id": "task:" + kind,
              "allowed_tool_count": 1 if kind == "method_design" else 0}
        for kind in ("intent", "semantic_batch", "role_evidence", "causal_context",
                     "role_ledger", "method_design", "claim_review", "figure_plan")]
    return {"case": case, "gold": gold, "payloads": payloads,
        "terminal": gold["terminal_outcomes"][0], "questions": case["expected_initial_question_ids"],
        "calls": calls, "failure_events": [], "diagnostic_events": [
            {"event_name": name} for name in ("agent.diagnostic_requested", "diagnostic.completed",
                                             "agent.design_revised")]}


@pytest.mark.parametrize("kind", sorted(mq.POST_ANALYSIS_ARTIFACTS))
def test_current_artifacts_require_rubric_migration_without_legacy_scoring(kind: str) -> None:
    _, document = mq._documents()
    gold = document["cases"][mq.LIVE_IDS[0]]
    before = mq.content_hash(gold)
    assert mq._score({}, gold, {kind: {}}, "complete", [], [], []) == [{
        "classification": "rubric_compatibility", "detail": mq.RUBRIC_MIGRATION_REQUIRED}]
    assert mq.content_hash(gold) == before


@pytest.mark.parametrize("kind", ("post_analysis_author", "post_analysis_review"))
def test_partial_post_analysis_calls_require_migration_even_before_first_artifact(kind: str) -> None:
    assert mq._score({}, {"accepted_templates": {}}, {}, "incomplete", [], [
        {"task_kind": kind}], []) == [{
            "classification": "rubric_compatibility", "detail": mq.RUBRIC_MIGRATION_REQUIRED}]


def test_historical_scoring_still_detects_claim_template_and_task_defects(
    historical_scoring_case: dict[str, Any],
) -> None:
    sample = historical_scoring_case
    assert mq._score(**sample) == []
    sample["payloads"]["ClaimJudgment"]["status"] = "not_reportable"
    sample["payloads"]["FigurePlan"]["figures"] = []
    sample["calls"] = [row for row in sample["calls"] if row["task_kind"] != "claim_review"]
    failures = mq._score(**sample)
    assert {"classification": "claim_safety", "detail": "claim status"} in failures
    assert any(row["classification"] == "curation" and row["detail"].startswith("figure:")
               for row in failures)
    assert any(row["detail"].startswith("decision_boundaries:") for row in failures)
    assert not any(row["classification"] == "rubric_compatibility" for row in failures)


def test_current_report_cannot_borrow_passing_artifacts_from_historical_scoring(
    historical_scoring_case: dict[str, Any],
) -> None:
    assert mq._score(**historical_scoring_case) == []
    historical_scoring_case["payloads"]["PostAnalysisBundle"] = {}
    assert mq._score(**historical_scoring_case) == [{
        "classification": "rubric_compatibility", "detail": mq.RUBRIC_MIGRATION_REQUIRED}]


def test_design_only_safe_terminal_retains_its_original_ask_policy_scoring(
    historical_scoring_case: dict[str, Any],
) -> None:
    sample = historical_scoring_case | {"payloads": {}, "calls": [], "terminal": "declined"}
    assert mq._score(**sample) == []


def test_matrix_result_reports_rubric_incompatibility_without_changing_real_terminal(
    historical_scoring_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample = historical_scoring_case
    monkeypatch.setattr(mq, "_payloads", lambda *_: {"PostAnalysisBundle": {}})
    monkeypatch.setattr(mq, "_failure_events", lambda *_: [])
    monkeypatch.setattr(mq, "_diagnostic_events", lambda *_: [])
    calls = [{"task_kind": "post_analysis_author", "tokens": {"total": 17}}]
    cell = {"cell_id": "dataset--rich", "case_id": "dataset", "eval_id": "evaluation-1",
            "family": "randomized_experiment", "context_variant": "rich", "analysis_id": "an-1",
            "questions": [], "task_results": calls, "hitl": []}
    result = mq._matrix_result(sample["case"], sample["gold"], {"rich": {}}, cell,
        None, SimpleNamespace(status="complete"), "run-1", "source-hash", "input-hash")
    assert result["terminal_outcome"] == "complete" and result["task_results"] == calls
    assert result["hard_failures"] == [{"classification": "rubric_compatibility",
                                        "detail": mq.RUBRIC_MIGRATION_REQUIRED}]
    assert result["execution_passed"] is False and result["positive_analysis_passed"] is False
