"""The LangGraph design graph: closing nodes, assembly, and the coordinator (T-013 §1.5).

PRD-002 §5–§7, §9, §11.1, §16.4, §19–§23; SYSTEM-CONTRACT §6.2, §7, §9. Split per D-055:
state/deps/helpers in `harness_base`, pipeline nodes in `harness_nodes`, closing + API here.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Hashable, Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from causal.design import contracts, frame, semantics
from causal.design.capacity import check_capacity
from causal.design.harness_base import (
    APPROVED_KINDS,
    COMPONENT,
    EVAL_APPROVAL,
    EVAL_METHOD,
    EVAL_STAGE,
    NEEDS_USER_INPUT,
    NODES,
    OUTCOME_REFS,
    REGISTRY_VERSIONS,
    VIEW_PARENTS,
    DesignDeps,
    DesignError,
    DesignRunResult,
    DesignState,
    GatewayProtocol,
)
from causal.design.harness_nodes import PipelineNodes
from causal.design.renderer import RendererError, render_causal_graph
from causal.shared import persistence
from causal.shared.contracts import ArtifactRef, HandoffManifestV1
from causal.shared.validation import parse_strict

__all__ = [
    "COMPONENT", "NEEDS_USER_INPUT", "REGISTRY_VERSIONS", "DesignDeps", "DesignError", "DesignRunResult",
    "DesignState", "GatewayProtocol", "build_checkpointer", "build_graph", "open_design_handoff",
    "resume_design", "run_design",
]

from typing import Final

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg import Connection

_RUN_ROW: Final = (
    "INSERT INTO design.design_runs (stage_run_id, analysis_id, graph_thread_id, design_revision,"
    " state, created_at, updated_at) VALUES (%s, %s, %s, %s, 'running', %s, %s)"
    " ON CONFLICT (stage_run_id) DO NOTHING")
_RUN_STATE: Final = (
    "UPDATE design.design_runs SET state = %s, outcome_artifact_id = %s, method_id = %s,"
    " selected_table = %s, updated_at = %s WHERE stage_run_id = %s")
_VIEW_ROW: Final = (
    "INSERT INTO design.causal_graph_views VALUES (%s, %s, %s, %s, %s)"
    " ON CONFLICT (artifact_id) DO NOTHING")


def build_checkpointer(conn: Connection[Any]) -> PostgresSaver:
    """The §19.2 checkpointer: PostgreSQL, strict msgpack allowlist, no pickle fallback."""
    saver = PostgresSaver(
        conn, serde=JsonPlusSerializer(pickle_fallback=False, allowed_msgpack_modules=None))
    saver.setup()
    return saver


def open_design_handoff(deps: DesignDeps, analysis_id: str, outcome_artifact_id: str,
                        receiving_stage_run_id: str) -> HandoffManifestV1:
    """The §23 handoff: exactly the selected CSV, design, frame contract, and capacity check."""
    env = deps.products.load_envelope(outcome_artifact_id)
    body = json.loads(deps.objects.get(env.payload_locator))
    if env.artifact_type != "DesignOutcome" or body["status"] != "approved":
        raise DesignError(f"no readable design handoff for {analysis_id!r}", "handoff_unavailable")
    design_env = deps.products.load_envelope(body["experiment_design"]["artifact_id"])
    design = json.loads(deps.objects.get(design_env.payload_locator))
    return HandoffManifestV1(
        handoff_id=f"ho:{analysis_id}:{env.content_hash[:16]}", schema_version="handoff.v1",
        analysis_id=analysis_id, producing_stage_run_id=env.stage_run_id,
        receiving_stage_run_id=receiving_stage_run_id,
        entries=tuple(ArtifactRef.model_validate(ref) for ref in (
            design["selected_csv"], body["experiment_design"], body["runnable_frame_contract"],
            body["capacity_check"])),
        originating_outcome="approved", approval_ids=(body["approval"]["artifact_id"],),
        registry_version="artifact-types.v1", compatibility_version="handoff.v1",
        receiver_validation_result=None, receiver_error_codes=(),
        created_at_utc=deps.clock(), accepted_at_utc=None)


class _Harness(PipelineNodes):
    """The closing nodes: render, capacity, approval, outcome, and handoff."""

    def render(self, state: DesignState) -> dict[str, Any]:
        """Compile the reviewable CausalGraphView; an absent renderer is a blocker (D-042)."""
        try:
            view = render_causal_graph(
                context=self._model(state, "CausalContext", semantics.CausalContextV1),
                ledger=self._model(state, "RoleLedger", semantics.RoleLedgerV1),
                measurement_map=self._model(state, "MeasurementMap", semantics.MeasurementMapV1),
                parents=tuple(self._ref(state, kind) for kind in VIEW_PARENTS),
                selected_alternative_id=None)
        except RendererError as error:
            return self._fail(state, error.code)
        built = self._commit(state, "CausalGraphView", view.canonical_payload(),
                             self._parents(state, "RunnableFrameContract"))
        self.deps.conn.execute(_VIEW_ROW, (built.artifact_id, state["analysis_id"],
                                           state["design_revision"], view.renderer_version,
                                           view.validation_status))
        return self._out(state, stage="render", graph_view_status=view.validation_status)

    def capacity_node(self, state: DesignState) -> dict[str, Any]:
        """The exact-cardinality delivery preflight; a failure refuses the revision (§13.5)."""
        design = self._model(state, "ExperimentDesign", frame.ExperimentDesignV1)
        contrasts = max(1, len(design.primary_contrasts))
        counts = dict.fromkeys(frame.CAPACITY_DIMENSIONS, 0) | {
            "arms": contrasts + 1, "contrasts": contrasts, "series": contrasts + 1,
            "evidence_items": len(design.required_visual_evidence) + len(
                design.required_postrepair_diagnostics)}
        check = check_capacity(
            self.deps.packs.get(design.method_id), counts, design.required_visual_evidence,
            registry=self.deps.capacity_registry)
        self._commit(state, "DeliveryCapacityCheck", check.canonical_payload(),
                     self._parents(state, "CausalGraphView"))
        self._emit(state, "task.completed", EVAL_METHOD, status=check.status.value)
        if check.status is frame.CapacityStatus.FAIL:
            return self._out(state, capacity_status="fail", status="refused",
                             refusal_code=check.failure_codes[0])
        return self._out(state, stage="capacity", capacity_status="pass")

    def approval(self, state: DesignState) -> dict[str, Any]:
        """The exact-hash approval interrupt and its binding (PRD-002 §22)."""
        refs = [self._ref(state, kind).model_dump(mode="json") for kind in APPROVED_KINDS]
        anchor = self._ref(state, "ExperimentDesign")
        self._emit(state, "user_interrupt.created", EVAL_APPROVAL, status="approval")
        decision = parse_strict(contracts.DesignApprovalDecisionV1, interrupt({
            "kind": contracts.InterruptKind.APPROVAL.value,
            "interrupt_artifact_id": anchor.artifact_id, "interrupt_hash": anchor.content_hash,
            "design_revision": state["design_revision"], "approved_artifacts": refs}))
        held = self._commit(state, "DesignApprovalDecision", decision.canonical_payload(),
                            self._parents(state, "ExperimentDesign"))
        self._emit(state, "user_interrupt.resumed", EVAL_APPROVAL, status=decision.decision.value)
        if decision.decision is not contracts.ApprovalDecision.APPROVED:
            return self._out(state, status=decision.decision.value,
                             approval_status=decision.decision.value)
        self._commit(state, "DesignApproval", {
            "schema_version": "design-approval.v1", "decision": "approved",
            "design_revision": state["design_revision"], "decision_artifact_id": held.artifact_id,
            "approved_artifacts": refs}, self._parents(state, *APPROVED_KINDS))
        return self._out(state, stage="approval", approval_status="approved")

    def outcome_node(self, state: DesignState) -> dict[str, Any]:
        """The one terminal DesignOutcome and the stage-run finalisation (PRD-002 §6)."""
        status = frame.DesignOutcomeStatus(state.get("status") or "approved")
        approved = status is frame.DesignOutcomeStatus.APPROVED
        payload = parse_strict(frame.DesignOutcomeV1, {
            "status": status.value, "design_revision": state["design_revision"],
            "refusal_code": state.get("refusal_code"), "error_code": state.get("error_code"),
            "open_requirement_ids": list(state["open_requirement_ids"]),
            "clarification_rounds_used": state["clarification_round"],
            "counts": {**dict.fromkeys(frame.DESIGN_COUNT_KEYS, 0), **state["counts"],
                       "corrections": sum(state["corrections"].values())},
            **{name: (self._ref(state, kind).model_dump(mode="json") if approved else None)
               for name, kind in OUTCOME_REFS.items()}})
        built = self._commit(state, "DesignOutcome", payload.canonical_payload(),
                             self._parents(state, "IntakeOutcome"))
        self.deps.products.transition_stage_run(state["stage_run_id"], "committing")
        self.deps.products.transition_stage_run(state["stage_run_id"], "completed")
        self._emit(state, "stage.completed", EVAL_STAGE, status=status.value, artifact_refs=(
            ArtifactRef(artifact_id=built.artifact_id, content_hash=built.content_hash),))
        return self._out(state, status=status.value)

    def handoff_open(self, state: DesignState) -> dict[str, Any]:
        """Open the PRD-003 handoff with exactly the four §23 entry artifacts."""
        if state.get("status") != frame.DesignOutcomeStatus.APPROVED.value:
            return self._out(state)
        opened = open_design_handoff(self.deps, state["analysis_id"],
                                     state["artifacts"]["DesignOutcome"],
                                     f"sr:{state['analysis_id']}:preparation")
        return self._out(state, handoff_id=opened.handoff_id)



def _route(state: DesignState) -> str:
    """A terminal status short-circuits to the outcome; an open ask round repeats the gate."""
    if state.get("status"):
        return "outcome_node"
    return "ask" if state.get("stage") == "gate_again" else "continue"


def build_graph(deps: DesignDeps) -> Any:
    """Compile the design StateGraph over the PostgreSQL checkpointer (PRD-002 §19)."""
    harness = _Harness(deps)
    builder = StateGraph(DesignState)
    for name in (*NODES, "handoff_open"):
        builder.add_node(name, getattr(harness, name))
    chain = (START, *NODES, "handoff_open", END)
    for name, following in itertools.pairwise(chain):
        if name in (START, "outcome_node", "handoff_open"):
            builder.add_edge(name, following)
        else:
            paths: dict[Hashable, str] = {"continue": following, "outcome_node": "outcome_node"}
            if name == "requirements_gate":
                paths["ask"] = name
            builder.add_conditional_edges(name, _route, paths)
    return builder.compile(checkpointer=deps.checkpointer)


def _config(thread_id: str) -> dict[str, Any]:
    """One design run is one graph thread; a later stage never resumes it (SC §9)."""
    return {"configurable": {"thread_id": thread_id}}


def _finish(deps: DesignDeps, final: Mapping[str, Any], thread_id: str) -> DesignRunResult:
    """Read the returned snapshot: one open interrupt, or one terminal design outcome."""
    pending = tuple(final.get("__interrupt__") or ())
    value = dict(pending[0].value) if pending else {}
    outcome_id = final.get("artifacts", {}).get("DesignOutcome")
    deps.conn.execute(_RUN_STATE, (
        "waiting_for_user" if pending else "completed", outcome_id, final.get("method_id"),
        final.get("dataset_id"), deps.clock(),
        f"dr:{final['analysis_id']}:{final['design_revision']}"))
    return DesignRunResult(
        status=NEEDS_USER_INPUT if pending else str(final.get("status") or "failed"),
        analysis_id=str(final["analysis_id"]), stage_run_id=str(final["stage_run_id"]),
        thread_id=thread_id, design_revision=int(final["design_revision"]),
        interrupt_kind=value.get("kind"), interrupt_hash=value.get("interrupt_hash"),
        interrupt_artifact_id=value.get("interrupt_artifact_id"), outcome_artifact_id=outcome_id,
        handoff_id=final.get("handoff_id"), refusal_code=final.get("refusal_code"),
        error_code=final.get("error_code"))


def run_design(deps: DesignDeps, *, analysis_id: str, intake_outcome_artifact_id: str,
               thread_id: str, design_revision: int = 1) -> DesignRunResult:
    """Start one design revision on its own graph thread (PRD-002 §5, §19.2)."""
    stage_run_id = f"dr:{analysis_id}:{design_revision}"
    now = deps.clock()
    try:
        deps.products.get_stage_run_state(stage_run_id)
    except persistence.PersistenceError:
        deps.products.create_stage_run(stage_run_id, analysis_id, "design")
        deps.products.transition_stage_run(stage_run_id, "tracing_preflight")
        deps.products.transition_stage_run(stage_run_id, "running")
    deps.conn.execute(
        _RUN_ROW, (stage_run_id, analysis_id, thread_id, design_revision, now, now))
    state: DesignState = {
        "analysis_id": analysis_id, "stage_run_id": stage_run_id, "thread_id": thread_id,
        "design_revision": design_revision, "stage": "start", "clarification_round": 0,
        "artifacts": {"IntakeOutcome": intake_outcome_artifact_id}, "hashes": {},
        "card_ids": [], "role_ids": [], "pending_task_ids": [], "open_requirement_ids": [],
        "answer_ids": [], "candidate_method_ids": [], "corrections": {}, "counts": {}}
    return _finish(deps, build_graph(deps).invoke(state, config=_config(thread_id)), thread_id)


def resume_design(deps: DesignDeps, *, thread_id: str,
                  resume_value: Mapping[str, Any]) -> DesignRunResult:
    """Resume the open interrupt on `thread_id` with one typed CLI decision (§11.1)."""
    final = build_graph(deps).invoke(
        Command(resume=dict(resume_value)), config=_config(thread_id))
    return _finish(deps, final, thread_id)
