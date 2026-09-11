"""Exact prior review requests reach revision tasks without becoming study facts."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from causal.design import compile as compiler
from causal.design import contracts, graph
from causal.design.harness_base import DesignError, DesignState, HarnessBase
from causal.design.packs import TASK_KINDS
from causal.shared.canonical import content_hash

ROOT = Path(__file__).resolve().parents[2]
REQUESTS = ["Remove weight_change from role=time; preserve 1971–1982 as study timeframe.",
            "Retain weight_change as outcome and the nine approved adjustment columns."]
DECISION = {
    "schema_version": "design-approval-decision.v1", "decision": "changes_requested",
    "interrupt_id": "review-bundle-1", "expected_interrupt_hash": "a" * 64,
    "expected_revision": 1, "approved_artifacts": [], "change_requests": REQUESTS,
    "idempotency_key": "review-request-1"}


def deps_for(decision: dict[str, Any] | None = None) -> Any:
    body = DECISION if decision is None else decision
    envelope = SimpleNamespace(
        artifact_id="decision-1", content_hash=content_hash(body),
        artifact_type="DesignApprovalDecision", analysis_id="an-1",
        stage_run_id="dr:an-1:1", payload_locator="decision-object")

    def execute(query: str, _: Any) -> Any:
        return SimpleNamespace(fetchall=lambda: [("decision-1",)], fetchone=lambda: (1,))

    return SimpleNamespace(
        conn=SimpleNamespace(execute=execute), clock=lambda: datetime(2026, 9, 9, tzinfo=UTC),
        products=SimpleNamespace(load_envelope=lambda _: envelope,
                                 get_stage_run_state=lambda _: "running"),
        objects=SimpleNamespace(get=lambda _: json.dumps(body).encode()))


def test_new_revision_carries_exact_prior_decision_ref_and_inherits_factual_answers(
        monkeypatch: pytest.MonkeyPatch) -> None:
    deps = deps_for()
    inherited = [SimpleNamespace(source_kind="user", origin_reference_id="answer-1"),
                 SimpleNamespace(source_kind="source", origin_reference_id="source-1")]
    calls = []
    captured: dict[str, Any] = {}

    def inherit(*args: Any) -> list[Any]:
        calls.append(args)
        return inherited

    def invoke(state: DesignState, **_: Any) -> DesignState:
        captured.update(state)
        return state

    monkeypatch.setattr(graph.askgate, "PsycopgAcceptedFactStore", lambda _: SimpleNamespace(inherit=inherit))
    monkeypatch.setattr(graph, "build_graph", lambda _: SimpleNamespace(invoke=invoke))
    monkeypatch.setattr(graph, "_finish", lambda *_: SimpleNamespace(status="captured"))
    graph.run_design(deps, analysis_id="an-1", intake_outcome_artifact_id="intake-1",
                     thread_id="thread-2", design_revision=2)
    assert captured["review_feedback_id"] == "decision-1"
    assert captured["hashes"]["decision-1"] == content_hash(DECISION)
    assert captured["answer_ids"] == ["answer-1"]
    assert calls[0][:3] == ("an-1", 1, 2)
    assert json.loads(deps.objects.get("decision-object")) == DECISION


def test_every_revision_task_gets_review_feedback_separate_from_evidence_and_facts(
        monkeypatch: pytest.MonkeyPatch) -> None:
    harness = object.__new__(HarnessBase)
    harness.deps = deps_for()
    harness._cache = {"decision-1": copy.deepcopy(DECISION),
                      "intake-1": {"evidence_bundle_artifact_id": "sources"},
                      "sources": {"items": [{"evidence_id": "ev:source", "value": "Study."}]},
                      "answer-1": {"value": "one_row_per_unit"}}
    state: DesignState = {"analysis_id": "an-1", "design_revision": 2,
                          "review_feedback_id": "decision-1",
                          "hashes": {"decision-1": content_hash(DECISION)},
                          "artifacts": {"IntakeOutcome": "intake-1"}, "answer_ids": ["answer-1"]}
    facts = {"accepted_facts": [{"value": "one_row_per_unit", "accepted_fact_id": "fact-1"}]}
    before = copy.deepcopy(facts)
    monkeypatch.setattr(harness, "_open_requirements", lambda _: ())
    monkeypatch.setattr(harness, "_prerequisite_context", lambda _: facts)
    monkeypatch.setattr(harness, "_profile", lambda _: ("profile-1", {}))
    table = compiler.load_task_table(ROOT / "registries" / "design-tasks.v1.json")
    calls = []

    def run(_: Any, kind: str, model: Any, **options: Any) -> None:
        payload = options["payload"]
        calls.append((kind, payload, compiler.render_prompt(table[kind], ROOT, payload)))

    harness._runner = SimpleNamespace(run=run)  # type: ignore[assignment]
    for kind in TASK_KINDS:
        harness._run_task(state, kind, contracts.DesignIntentV1, scope_kind="design",
                          scope_ids=(), parent_kinds=(), payload={"task_input": "unchanged"})
    for _, payload, prompt in calls:
        assert payload["review_feedback"]["change_requests"] == REQUESTS
        assert payload["review_feedback"]["prior_decision"] == {
            "artifact_id": "decision-1", "content_hash": content_hash(DECISION)}
        assert "## review_feedback" in prompt
        assert json.dumps(REQUESTS[0]) in prompt
        assert payload["prerequisite_context"] == before
    assert len(calls) == len(TASK_KINDS)
    assert set(harness._evidence(state)) == {"ev:source", "ua:answer-1"}
    assert facts == before and state["answer_ids"] == ["answer-1"]


def test_feedback_refuses_another_revision_or_changed_decision() -> None:
    harness = object.__new__(HarnessBase)
    harness.deps = deps_for()
    harness._cache = {"decision-1": copy.deepcopy(DECISION)}
    state: DesignState = {"analysis_id": "an-1", "design_revision": 3,
                          "review_feedback_id": "decision-1",
                          "hashes": {"decision-1": content_hash(DECISION)}}
    with pytest.raises(DesignError, match="frozen decision"):
        harness._review_feedback(state)
    state["design_revision"] = 2
    harness._cache["decision-1"]["change_requests"] = ["silently changed"]
    with pytest.raises(DesignError, match="frozen decision"):
        harness._review_feedback(state)


def test_first_revision_does_not_load_feedback() -> None:
    assert graph._previous_review(SimpleNamespace(), "an-1", 1) is None
