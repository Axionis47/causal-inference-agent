"""Observable delivery and recovery preserve frozen inputs, revisions and spent budgets."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from causal.post_analysis import entry
from causal.post_analysis import runtime as pr
from causal.shared.canonical import canonical_bytes
from causal.shared.contracts import ArtifactRef, HandoffManifestV1
from causal.shared.events import EventEmitter
from causal.shared.persistence import (
    ILLEGAL_STATE_TRANSITION,
    RUN_STATE_TRANSITIONS,
    PersistenceError,
)
from causal.shared.tracing import ObservabilityError
from tests.shared.test_tracing import FakeTracer

NOW = datetime(2026, 9, 10, tzinfo=UTC)
ANALYSIS = "an-1"
RUN = "ps:an-1:1"


class MemoryDB:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def transaction(self) -> Any:
        return nullcontext()

    def execute(self, query: str, args: tuple[Any, ...]) -> Any:
        value: Any = None
        if query.startswith("SELECT stage_run_id"):
            if self.rows:
                key, row = max(self.rows.items(), key=lambda item: item[1]["revision"])
                value = (key, "", row["revision"], row["state"], row.get("bundle"))
        elif query.startswith("SELECT run_record"):
            value = (copy.deepcopy(self.rows[args[0]]["record"]),) if args[0] in self.rows else None
        elif query.startswith("SELECT state,bundle_artifact_id"):
            row = self.rows.get(args[0])
            value = (row["state"], row.get("bundle")) if row else None
        elif query.startswith("INSERT INTO presentation.runs"):
            self.rows[args[0]] = {"state": "running", "revision": args[2],
                                  "record": json.loads(args[3])}
        elif query.startswith("UPDATE presentation.runs SET run_record"):
            self.rows[args[-1]]["record"] = json.loads(args[0])
        elif query.startswith("UPDATE presentation.runs SET state"):
            self.rows[args[-1]].update(state=args[0], bundle=args[1], error_code=args[2],
                                        record=json.loads(args[3]))
        else:
            raise AssertionError(query)
        return SimpleNamespace(fetchone=lambda: value)


class Products:
    def __init__(self) -> None:
        self.envelopes: dict[str, Any] = {}
        self.states: dict[str, str] = {}
        self.transitions: list[tuple[str, str]] = []

    def create_stage_run(self, run: str, analysis: str, stage: str) -> None:
        assert run not in self.states
        self.states[run] = "created"

    def get_stage_run_state(self, run: str) -> str:
        if run not in self.states:
            raise PersistenceError("unknown stage run", "unknown_stage_run")
        return self.states[run]

    def transition_stage_run(self, run: str, state: str) -> None:
        current = self.states[run]
        if state not in RUN_STATE_TRANSITIONS[current]:
            raise PersistenceError(f"illegal transition {current!r} -> {state!r}",
                                   ILLEGAL_STATE_TRANSITION)
        self.transitions.append((run, state))
        self.states[run] = state

    def load_envelope(self, artifact_id: str) -> Any:
        return self.envelopes[artifact_id]


class StubGraph:
    def __init__(self, store: Any, final: dict[str, Any]) -> None:
        self.store, self.final = store, final
        self.snapshot = SimpleNamespace(next=(), values={})
        self.calls: list[Any] = []

    def get_state(self, config: Any) -> Any:
        return self.snapshot

    def invoke(self, initial: Any, config: Any) -> dict[str, Any]:
        self.store.reserve_call("author")
        self.calls.append(initial)
        return self.final


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    objects: dict[str, bytes] = {}
    deps = SimpleNamespace(conn=MemoryDB(), products=Products(),
        objects=SimpleNamespace(get=objects.__getitem__), registry=None, committer=None,
        emitter=EventEmitter(io.StringIO()), clock=lambda: NOW, gateway=None,
        checkpointer=None, tracer=FakeTracer(), render_root=tmp_path,
        require_tracing=True, max_calls=12, max_reviews=3)

    def artifact(kind: str, body: dict[str, Any], run: str = RUN) -> ArtifactRef:
        raw = canonical_bytes(body)
        digest = hashlib.sha256(raw).hexdigest()
        ref = ArtifactRef(artifact_id=kind + ":" + digest, content_hash=digest)
        objects[digest] = raw
        deps.products.envelopes[ref.artifact_id] = SimpleNamespace(
            artifact_type=kind, analysis_id=ANALYSIS, stage_run_id=run,
            content_hash=digest, payload_locator=digest)
        return ref

    compiled = artifact("CompiledDesign", {"approved": "frozen"})
    prepared = artifact("PreparedDataBundle", {"data": "frozen"})
    numerical = artifact("EstimationBundle", {
        "compiled_design": compiled.model_dump(mode="json"),
        "prepared_bundle": prepared.model_dump(mode="json")})
    outcome = artifact("EstimationOutcome", {"status": "complete",
        "estimation_bundle": numerical.model_dump(mode="json")}, "est:an-1:1")
    bundle = artifact("PostAnalysisBundle", {"export": {}, "draft": {}, "visuals": {}})

    def handoff(run: str = RUN) -> HandoffManifestV1:
        return HandoffManifestV1(handoff_id="handoff:" + run, schema_version="handoff-manifest.v1",
            analysis_id=ANALYSIS, producing_stage_run_id="est:an-1:1", receiving_stage_run_id=run,
            entries=(numerical, compiled, prepared), originating_outcome="complete", approval_ids=(),
            registry_version="registry.v1", compatibility_version="compatibility.v1",
            receiver_validation_result=None, receiver_error_codes=(), created_at_utc=NOW,
            accepted_at_utc=None)

    graph = StubGraph(entry.Store(deps, ANALYSIS, RUN),
        {"status": "complete", "bundle": bundle.model_dump(mode="json"),
         "outcome": outcome.model_dump(mode="json")})

    def build(store: Any, held: ArtifactRef) -> StubGraph:
        assert held == outcome
        graph.store = store
        return graph

    def accept(store: Any, manifest: HandoffManifestV1) -> None:
        # Shared gate/packet validation has its own tests; retain exact-byte checking here.
        assert manifest.receiving_stage_run_id == store.stage_run_id
        for ref in manifest.entries:
            store.read(ref)

    opened: list[tuple[str, str, str]] = []

    def open_handoff(est: Any, analysis: str, artifact_id: str, receiving: str) -> Any:
        opened.append((analysis, artifact_id, receiving))
        return handoff(receiving)

    monkeypatch.setattr(entry, "build_graph", build)
    monkeypatch.setattr(entry, "_accept", accept)
    monkeypatch.setattr(pr, "open_post_analysis_handoff", open_handoff)
    def call(**kwargs: Any) -> Any:
        params = {"analysis_id": ANALYSIS, "stage_run_id": RUN, "revision": 1,
                  "outcome": outcome, "handoff": handoff(), **kwargs}
        return entry.run_post_analysis(deps, **params)

    return SimpleNamespace(deps=deps, graph=graph, outcome=outcome, bundle=bundle,
        handoff=handoff, objects=objects, opened=opened, call=call, artifact=artifact)


@pytest.mark.parametrize("tracer", [None, FakeTracer(
    preflight_error=ObservabilityError("tracing unavailable", "preflight_failed"))])
def test_missing_or_failed_required_tracing_blocks_before_authoring(harness: Any, tracer: Any) -> None:
    harness.deps.tracer = tracer
    result = harness.call()
    assert result.status == "incomplete" and result.error_code == "failed_observability"
    assert result.bundle is None and result.counters == {"calls": 0, "reviews": 0}
    assert harness.graph.calls == []
    assert harness.deps.products.states[RUN] == "failed_observability"


def test_flush_failure_keeps_committed_bundle_unavailable_for_delivery(harness: Any) -> None:
    harness.deps.tracer = FakeTracer(
        flush_error=ObservabilityError("batch was not acknowledged", "flush_unacknowledged"))
    result = harness.call()
    assert result.status == "incomplete" and result.bundle is None
    assert result.error_code == "failed_observability"
    assert len(harness.graph.calls) == 1
    assert harness.bundle.artifact_id in harness.deps.products.envelopes
    with pytest.raises(PersistenceError) as failure:
        pr.bundle_view(harness.deps, harness.bundle.artifact_id, harness.bundle.content_hash)
    assert failure.value.code == "presentation_bundle_unavailable"
    assert harness.deps.conn.rows[RUN].get("bundle") is None


def test_completed_replay_returns_the_frozen_result_without_more_calls(harness: Any) -> None:
    first = harness.call()
    assert first.status == "complete"
    before = copy.deepcopy(harness.deps.conn.rows)
    repeated = harness.call()
    assert repeated == first
    assert len(harness.graph.calls) == 1
    assert harness.deps.tracer.preflights == harness.deps.tracer.flushes == 1
    assert harness.deps.conn.rows == before


@pytest.mark.parametrize("changed", [{"artifact_id": "another-outcome"}, {"content_hash": "a" * 64}])
def test_completed_run_id_rejects_a_different_outcome_identity_or_hash(
    harness: Any, changed: dict[str, str],
) -> None:
    assert harness.call().status == "complete"
    before = copy.deepcopy(harness.deps.conn.rows)
    with pytest.raises(PersistenceError) as failure:
        harness.call(outcome=harness.outcome.model_copy(update=changed))
    assert failure.value.code == "invalid_handoff_replay"
    assert len(harness.graph.calls) == 1
    assert harness.deps.conn.rows == before


def test_handoff_must_match_the_exact_receipt_before_authoring(harness: Any) -> None:
    handoff = harness.handoff()
    changed = handoff.model_copy(update={"entries": tuple(reversed(handoff.entries))})
    result = harness.call(handoff=changed)
    assert result.status == "blocked" and result.error_code == "invalid_upstream_input"
    assert result.issues[0].code == "handoff_source_mismatch"
    assert result.issues[0].owner == "analysis"
    assert result.issues[0].path == "/handoff/entries"
    assert harness.graph.calls == [] and result.bundle is None


def test_missing_numerical_prepared_reference_reports_its_exact_upstream_owner(harness: Any) -> None:
    handoff = harness.handoff()
    compiled, prepared = handoff.entries[1:]
    numerical = harness.artifact("EstimationBundle", {
        "compiled_design": compiled.model_dump(mode="json")})
    outcome = harness.artifact("EstimationOutcome", {"status": "complete",
        "estimation_bundle": numerical.model_dump(mode="json")}, "est:an-1:1")
    result = harness.call(outcome=outcome,
        handoff=handoff.model_copy(update={"entries": (numerical, compiled, prepared)}))
    assert result.status == "blocked" and result.error_code == "invalid_upstream_input"
    issue, = result.issues
    assert issue.code == "required_reference_missing" and issue.owner == "analysis"
    assert issue.source == numerical and issue.path == "/prepared_bundle"
    assert issue.expected == "an exact artifact reference" and issue.received is None
    assert issue.required_action and result.bundle is None
    assert harness.graph.calls == [] and result.counters == {"calls": 0, "reviews": 0}
    assert harness.deps.tracer.flushes == 1


def test_upstream_failure_traces_are_flushed_before_returning_blocked(harness: Any) -> None:
    handoff = harness.handoff()
    changed = handoff.model_copy(update={"entries": tuple(reversed(handoff.entries))})
    result = harness.call(handoff=changed)
    assert result.status == "blocked" and result.bundle is None
    assert harness.deps.tracer.preflights == harness.deps.tracer.flushes == 1


def test_provider_failure_traces_flush_and_reserved_calls_stay_spent(harness: Any) -> None:
    def fail(initial: Any, config: Any) -> Any:
        harness.graph.store.reserve_call("author")
        raise RuntimeError("provider connection closed after dispatch")

    harness.graph.invoke = fail
    result = harness.call()
    assert result.status == "incomplete" and result.error_code == "RuntimeError"
    assert result.bundle is None and result.counters == {"calls": 1, "reviews": 0}
    assert harness.deps.tracer.preflights == harness.deps.tracer.flushes == 1


def test_trace_flush_failure_is_preserved_on_the_upstream_failure_path(harness: Any) -> None:
    harness.deps.tracer = FakeTracer(
        flush_error=ObservabilityError("batch was not acknowledged", "flush_unacknowledged"))
    handoff = harness.handoff()
    changed = handoff.model_copy(update={"entries": tuple(reversed(handoff.entries))})
    result = harness.call(handoff=changed)
    assert result.status == "incomplete" and result.error_code == "failed_observability"
    assert result.bundle is None and result.issues[0].code == "handoff_source_mismatch"
    assert harness.deps.products.states[RUN] == "failed_observability"


@pytest.mark.parametrize("changed", [{"artifact_id": "another-outcome"}, {"content_hash": "a" * 64}])
def test_live_run_cannot_release_a_checkpoint_for_a_different_outcome(
    harness: Any, changed: dict[str, str],
) -> None:
    harness.deps.conn.rows[RUN] = {"state": "running", "revision": 1, "record": {
        "outcome": harness.outcome.model_dump(mode="json"), "handoff_id": harness.handoff().handoff_id,
        "entries": [ref.model_dump(mode="json") for ref in harness.handoff().entries],
        "thread_id": "post:" + RUN, "calls": 5, "reviews": 1}}
    harness.deps.products.states[RUN] = "running"
    harness.graph.snapshot = SimpleNamespace(next=(), values=dict(harness.graph.final))
    before = copy.deepcopy(harness.deps.conn.rows)
    with pytest.raises(PersistenceError) as failure:
        harness.call(outcome=harness.outcome.model_copy(update=changed))
    assert failure.value.code == "invalid_handoff_replay"
    assert harness.deps.conn.rows == before and harness.graph.calls == []


def test_live_run_cannot_change_its_frozen_handoff_entries(harness: Any) -> None:
    handoff = harness.handoff()
    harness.deps.conn.rows[RUN] = {"state": "running", "revision": 1, "record": {
        "outcome": harness.outcome.model_dump(mode="json"), "handoff_id": handoff.handoff_id,
        "entries": [ref.model_dump(mode="json") for ref in handoff.entries],
        "thread_id": "post:" + RUN, "calls": 5, "reviews": 1}}
    harness.deps.products.states[RUN] = "running"
    harness.graph.snapshot = SimpleNamespace(next=(), values=dict(harness.graph.final))
    with pytest.raises(PersistenceError) as failure:
        harness.call(handoff=handoff.model_copy(update={"entries": tuple(reversed(handoff.entries))}))
    assert failure.value.code == "invalid_handoff_replay"
    assert harness.graph.calls == []


def test_initial_crash_before_presentation_row_reuses_existing_shared_stage(harness: Any) -> None:
    harness.deps.products.states[RUN] = "created"
    result = harness.call()
    assert result.status == "complete" and result.counters == {"calls": 1, "reviews": 0}
    assert harness.deps.products.states[RUN] == "completed"


@pytest.mark.parametrize("pending", [(), ("agent",)])
def test_crash_recovery_uses_checkpoint_and_preserves_spent_calls(harness: Any, pending: Any) -> None:
    harness.deps.conn.rows[RUN] = {"state": "running", "revision": 1, "record": {
        "outcome": harness.outcome.model_dump(mode="json"), "handoff_id": harness.handoff().handoff_id,
        "entries": [ref.model_dump(mode="json") for ref in harness.handoff().entries],
        "thread_id": "post:" + RUN, "calls": 5, "reviews": 1}}
    harness.deps.products.states[RUN] = "running"
    harness.graph.snapshot = SimpleNamespace(next=pending, values=dict(harness.graph.final))
    result = harness.call()
    assert result.status == "complete"
    assert result.counters == {"calls": 6 if pending else 5, "reviews": 1}
    assert harness.graph.calls == ([None] if pending else [])
    assert harness.deps.tracer.flushes == 1


def test_crash_after_stage_completion_before_result_save_is_idempotent(harness: Any) -> None:
    harness.deps.conn.rows[RUN] = {"state": "running", "revision": 1, "record": {
        "outcome": harness.outcome.model_dump(mode="json"), "handoff_id": harness.handoff().handoff_id,
        "entries": [ref.model_dump(mode="json") for ref in harness.handoff().entries],
        "thread_id": "post:" + RUN, "calls": 5, "reviews": 1}}
    harness.deps.products.states[RUN] = "completed"
    harness.graph.snapshot = SimpleNamespace(next=(), values=dict(harness.graph.final))
    result = harness.call()
    assert result.status == "complete" and result.bundle == harness.bundle
    assert result.counters == {"calls": 5, "reviews": 1}
    assert harness.graph.calls == [] and harness.deps.products.transitions == []


def failed_attempt(harness: Any, *, state: str = "incomplete") -> dict[str, Any]:
    record = {"outcome": harness.outcome.model_dump(mode="json"), "thread_id": "post:" + RUN,
              "calls": 5, "reviews": 1, "error": "report revision interrupted"}
    harness.deps.conn.rows[RUN] = {"state": state, "revision": 1, "record": record}
    return copy.deepcopy(harness.deps.conn.rows[RUN])


def recover(harness: Any, failed_run: str = RUN) -> Any:
    return pr.recover_presentation(harness.deps, SimpleNamespace(), harness.deps,
        ANALYSIS, 1, failed_stage_run_id=failed_run)


def test_explicit_recovery_copies_exact_failed_outcome_and_counters_into_new_revision(harness: Any) -> None:
    original = failed_attempt(harness)
    result = recover(harness)
    assert result.status == "complete" and result.stage_run_id == "ps:an-1:2"
    assert harness.opened == [(ANALYSIS, harness.outcome.artifact_id, "ps:an-1:2")]
    recovered = harness.deps.conn.rows["ps:an-1:2"]["record"]
    assert recovered["outcome"] == original["record"]["outcome"]
    assert recovered["calls"] == 6 and recovered["reviews"] == 1
    assert harness.deps.conn.rows[RUN] == original


@pytest.mark.parametrize("status", ["complete", "complete_with_qualifications", "running", "blocked"])
def test_recovery_refuses_completed_and_nonfailed_runs(harness: Any, status: str) -> None:
    failed_attempt(harness, state=status)
    with pytest.raises(PersistenceError) as failure:
        recover(harness)
    assert failure.value.code == "presentation_recovery_unavailable"
    assert harness.opened == [] and harness.graph.calls == []


def test_recovery_refuses_an_older_failed_revision(harness: Any) -> None:
    failed_attempt(harness)
    with pytest.raises(PersistenceError):
        recover(harness, "ps:an-1:0")
    assert harness.opened == []


@pytest.mark.parametrize("counter,limit", [("calls", 12), ("reviews", 3)])
def test_recovery_never_resets_or_exceeds_the_original_budget(
    harness: Any, counter: str, limit: int,
) -> None:
    failed_attempt(harness)
    harness.deps.conn.rows[RUN]["record"][counter] = limit
    before = copy.deepcopy(harness.deps.conn.rows)
    with pytest.raises(PersistenceError) as failure:
        recover(harness)
    assert failure.value.code == "post_analysis_budget_exhausted"
    assert harness.deps.conn.rows == before and harness.opened == []


def test_recovery_refuses_changed_frozen_artifact_bytes(harness: Any) -> None:
    original = failed_attempt(harness)
    envelope = harness.deps.products.load_envelope(harness.outcome.artifact_id)
    harness.objects[envelope.payload_locator] = b'{"status":"changed"}'
    result = recover(harness)
    assert result.status == "incomplete" and result.error_code == "artifact_hash_mismatch"
    assert harness.graph.calls == []
    assert harness.deps.conn.rows[RUN] == original


def test_historical_presentation_requires_explicit_handoff_migration(harness: Any) -> None:
    failed_attempt(harness, state="failed")
    harness.deps.conn.rows[RUN]["record"] = {
        "outcome": {"context_manifest": {"artifact_id": "historical-context", "content_hash": "a" * 64}}}
    with pytest.raises(PersistenceError, match="migration") as failure:
        recover(harness)
    assert failure.value.code == "presentation_recovery_unavailable"
    assert harness.opened == [] and harness.graph.calls == []
