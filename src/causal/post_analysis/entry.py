"""Public stage entry: common handoff, durable graph, observable terminal outcome."""
from __future__ import annotations

import json
from typing import Any

from causal.post_analysis.contracts import InputError, InputIssue, PostAnalysisResult
from causal.post_analysis.graph import build_graph
from causal.post_analysis.input_sources import _ref
from causal.post_analysis.store import COMPONENT, LOGGER, PostAnalysisDeps, Store
from causal.shared.contracts import ArtifactRef, HandoffManifestV1
from causal.shared.handoff import HandoffGate, HandoffStore
from causal.shared.persistence import PersistenceError
from causal.shared.tracing import ObservabilityError, trace_span


def run_post_analysis(deps: PostAnalysisDeps, *, analysis_id: str, stage_run_id: str,
                      revision: int, outcome: ArtifactRef, handoff: HandoffManifestV1,
                      previous_counters: dict[str, int] | None = None) -> PostAnalysisResult:
    store = Store(deps, analysis_id, stage_run_id)
    if cached := _open(store, revision, outcome, handoff, previous_counters):
        return cached
    state: dict[str, Any] = {"status": "incomplete"}
    try:
        current = deps.products.get_stage_run_state(stage_run_id)
        if current == "created":
            deps.products.transition_stage_run(stage_run_id, "tracing_preflight")
        if deps.require_tracing and deps.tracer is None:
            raise ObservabilityError("LangSmith tracing is required", "preflight_failed")
        if deps.tracer is not None:
            deps.tracer.preflight()
        if deps.products.get_stage_run_state(stage_run_id) == "tracing_preflight":
            deps.products.transition_stage_run(stage_run_id, "running")
        with trace_span(deps.tracer, "post_analysis", inputs={
                "outcome": outcome.model_dump(mode="json"), "handoff": handoff.model_dump(mode="json")},
                metadata={"analysis_id": analysis_id, "stage_run_id": stage_run_id,
                          "graph_thread_id": store.thread_id}) as span:
            _accept(store, handoff)
            _bindings(store, outcome, handoff)
            deps.emitter.emit(store.event("stage.started", status="running"))
            graph = build_graph(store, outcome)
            config = {"configurable": {"thread_id": store.thread_id}, "recursion_limit": 120}
            snapshot = graph.get_state(config)
            initial = None if snapshot.next else {"outcome": outcome.model_dump(mode="json")}
            state = dict(snapshot.values if snapshot.values and not snapshot.next
                         else graph.invoke(initial, config))
            span.finish(state)
    except InputError as error:
        state = {"status": "blocked", "error_code": "invalid_upstream_input",
                 "issues": [issue.model_dump(mode="json") for issue in error.issues]}
    except Exception as error:  # noqa: BLE001 -- stage boundary persists an observable terminal
        LOGGER.exception("Post-analysis run failed: %s", stage_run_id)
        state = {**state, "status": "incomplete", "bundle": {},
                 "error_code": getattr(error, "code", type(error).__name__)}
        if isinstance(error, ObservabilityError):
            state["error_code"] = "failed_observability"
    finally:
        if deps.tracer is not None:
            try:
                deps.tracer.flush()
            except ObservabilityError:
                LOGGER.exception("Post-analysis trace delivery failed: %s", stage_run_id)
                state = {**state, "status": "incomplete", "bundle": {},
                         "error_code": "failed_observability"}
    return _close(store, state)


def _open(store: Store, revision: int, outcome: ArtifactRef, handoff: HandoffManifestV1,
          counters: dict[str, int] | None) -> PostAnalysisResult | None:
    deps = store.deps
    row = deps.conn.execute("SELECT run_record FROM presentation.runs WHERE stage_run_id=%s",
                            (store.stage_run_id,)).fetchone()
    entries = [ref.model_dump(mode="json") for ref in handoff.entries]
    if row:
        if (row[0].get("outcome") != outcome.model_dump(mode="json")
                or row[0].get("handoff_id") != handoff.handoff_id
                or row[0].get("entries", entries) != entries
                or handoff.analysis_id != store.analysis_id
                or handoff.receiving_stage_run_id != store.stage_run_id):
            raise PersistenceError("run is bound to another outcome or handoff", "invalid_handoff_replay")
        return _result(row[0]["result"]) if row[0].get("result") else None
    record = {"outcome": outcome.model_dump(mode="json"), "handoff_id": handoff.handoff_id,
              "entries": entries, "thread_id": store.thread_id, **(counters or {})}
    now = deps.clock()
    with deps.conn.transaction():
        try:
            deps.products.get_stage_run_state(store.stage_run_id)
        except PersistenceError as error:
            if error.code != "unknown_stage_run":
                raise
            deps.products.create_stage_run(store.stage_run_id, store.analysis_id, "presentation")
        deps.conn.execute("INSERT INTO presentation.runs VALUES (%s,%s,'running',%s,NULL,NULL,%s::jsonb,%s,%s)",
            (store.stage_run_id, store.analysis_id, revision, json.dumps(record), now, now))
    return None


def _bindings(store: Store, outcome: ArtifactRef, handoff: HandoffManifestV1) -> None:
    receipt = store.read(outcome)
    if receipt.get("estimation_bundle") is not None:
        numerical_ref = _ref(receipt, "estimation_bundle", outcome)
        numerical = store.read(numerical_ref)
        expected = tuple(ref.model_dump(mode="json") for ref in (
            numerical_ref, _ref(numerical, "compiled_design", numerical_ref),
            _ref(numerical, "prepared_bundle", numerical_ref)))
    else:
        context_ref = _ref(receipt, "context_manifest", outcome)
        envelope = store.deps.products.load_envelope(outcome.artifact_id)
        plans = tuple(ref.model_dump(mode="json") for ref in envelope.parent_artifacts
            if store.deps.products.load_envelope(ref.artifact_id).artifact_type == "EstimationPlan")
        expected = (outcome.model_dump(mode="json"), context_ref.model_dump(mode="json"), *plans)
    if tuple(ref.model_dump(mode="json") for ref in handoff.entries) != expected:
        raise InputError(InputIssue(code="handoff_source_mismatch", owner="analysis",
            source=outcome, path="/handoff/entries", expected=expected,
            received=[ref.model_dump(mode="json") for ref in handoff.entries],
            required_action="Assemble the handoff from this exact outcome's referenced inputs."))


def _close(store: Store, state: dict[str, Any]) -> PostAnalysisResult:
    deps, analysis_id, stage_run_id = store.deps, store.analysis_id, store.stage_run_id
    row = deps.conn.execute("SELECT run_record FROM presentation.runs WHERE stage_run_id=%s",
                            (stage_run_id,)).fetchone()
    record = dict(row[0])
    result = {"status": state["status"], "analysis_id": analysis_id, "stage_run_id": stage_run_id,
              "thread_id": store.thread_id, "bundle": state.get("bundle") or None,
              "issues": state.get("issues", []), "error_code": state.get("error_code"),
              "counters": {key: int(record.get(key, 0)) for key in ("calls", "reviews")}}
    record["result"] = result
    terminal = "completed" if result["status"] == "complete" else (
        "failed_observability" if result["error_code"] == "failed_observability" else "failed")
    bundle_id = result["bundle"]["artifact_id"] if result["bundle"] else None
    with deps.conn.transaction():
        if deps.products.get_stage_run_state(stage_run_id) != terminal:
            deps.products.transition_stage_run(stage_run_id, terminal)
        deps.conn.execute("UPDATE presentation.runs SET state=%s,bundle_artifact_id=%s,error_code=%s,"
            "run_record=%s::jsonb,updated_at=%s WHERE stage_run_id=%s",
            (result["status"], bundle_id, result["error_code"], json.dumps(record), deps.clock(), stage_run_id))
    deps.emitter.emit(store.event("stage.completed" if bundle_id else "stage.failed",
                                 status=result["status"], error_code=result["error_code"]))
    return _result(result)


def _result(result: dict[str, Any]) -> PostAnalysisResult:
    return PostAnalysisResult(**{**result,
        "bundle": ArtifactRef(**result["bundle"]) if result.get("bundle") else None,
        "issues": tuple(InputIssue.model_validate(row) for row in result.get("issues", ()))})


def _accept(store: Store, manifest: HandoffManifestV1) -> None:
    deps = store.deps
    if manifest.analysis_id != store.analysis_id or manifest.receiving_stage_run_id != store.stage_run_id:
        raise PersistenceError("handoff addresses another run", "wrong_handoff_recipient")
    handoffs = HandoffStore(deps.conn)
    try:
        prior = handoffs.load(manifest.handoff_id)
    except PersistenceError as error:
        if error.code != "unknown_handoff":
            raise
    else:
        if prior.entries != manifest.entries or prior.receiver_validation_result != "accepted":
            raise PersistenceError("handoff changed or was rejected", "invalid_handoff_replay")
        return
    result = HandoffGate(deps.objects, deps.products, handoffs, deps.registry, deps.emitter).accept(
        manifest, receiving_component=COMPONENT,
        allowed_outcomes=frozenset({"complete", "not_estimable", "invalidated", "failed"}),
        event_factory=lambda name, codes: store.event(f"handoff.{name}", error_code=codes[0] if codes else None))
    if not result.accepted:
        raise PersistenceError("; ".join(result.error_codes), "invalid_handoff")
