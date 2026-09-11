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

from causal.analysis.integration import RESOURCE_ROOT as ANALYSIS_RESOURCES
from causal.design import askgate, contracts, semantics
from causal.design.capacity import compile_capacity_report
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
    TASK_RESUME_NODES,
    VIEW_PARENTS,
    DesignDeps,
    DesignError,
    DesignRunResult,
    DesignState,
    GatewayProtocol,
)
from causal.design.harness_nodes import PipelineNodes
from causal.design.renderer import RendererError, render_causal_graph
from causal.design.review_policy import approval_disclosures
from causal.design.v2 import (
    CapacityReportV2,
    CompiledDesignV2,
    DesignApprovalV2,
    DesignFactSetV2,
    DesignOutcomeV2,
    DesignReviewBundleV2,
    DiagnosticReportV2,
    GraphViewSetV2,
    ResolutionCategory,
    ResponsibleActor,
    ValidationIssueV2,
)
from causal.shared import persistence
from causal.shared.canonical import content_hash
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
    """Open the exact six-artifact Design V2 handoff; no V1 translation is attempted."""
    env = deps.products.load_envelope(outcome_artifact_id)
    body = json.loads(deps.objects.get(env.payload_locator))
    if env.artifact_type != "DesignOutcome" or body["status"] != "approved":
        raise DesignError(f"no readable design handoff for {analysis_id!r}", "handoff_unavailable")
    design_env = deps.products.load_envelope(body["compiled_design"]["artifact_id"])
    design = json.loads(deps.objects.get(design_env.payload_locator))
    return HandoffManifestV1(
        handoff_id=f"ho:{analysis_id}:{env.content_hash[:16]}", schema_version="handoff.v1",
        analysis_id=analysis_id, producing_stage_run_id=env.stage_run_id,
        receiving_stage_run_id=receiving_stage_run_id,
        entries=tuple(ArtifactRef.model_validate(ref) for ref in (
            design["selected_csv"], body["compiled_design"], body["diagnostic_report"],
            body["capacity_report"], body["review_bundle"], body["approval"])),
        originating_outcome="approved", approval_ids=(body["approval"]["artifact_id"],),
        registry_version="artifact-types.v1", compatibility_version="handoff.v1",
        receiver_validation_result=None, receiver_error_codes=(),
        created_at_utc=deps.clock(), accepted_at_utc=None)


class _Harness(PipelineNodes):
    """The closing nodes: render, capacity, approval, outcome, and handoff."""

    def render(self, state: DesignState) -> dict[str, Any]:
        """Render the base graph and every declared alternative into one typed view set."""
        context = self._model(state, "CausalContext", semantics.CausalContextV1)
        views: list[tuple[Any, Any]] = []
        try:
            for alternative_id in (None, *(row.alternative_id for row in context.alternatives)):
                view = render_causal_graph(
                    context=context,
                    ledger=self._model(state, "RoleLedger", semantics.RoleLedgerV1),
                    measurement_map=self._model(
                        state, "MeasurementMap", semantics.MeasurementMapV1),
                    parents=tuple(self._ref(state, kind) for kind in VIEW_PARENTS),
                    selected_alternative_id=alternative_id)
                built = self._commit(state, "CausalGraphView", view.canonical_payload(),
                                     self._parents(state, "CompiledDesign"))
                self.deps.conn.execute(_VIEW_ROW, (
                    built.artifact_id, state["analysis_id"], state["design_revision"],
                    view.renderer_version, view.validation_status))
                views.append((view, built))
        except RendererError as error:
            return self._fail(state, error.code)
        parents = (self.deps.products.load_envelope(
            state["artifacts"]["CompiledDesign"]), *(built for _, built in views))
        view_set = GraphViewSetV2(
            base_view=ArtifactRef(artifact_id=views[0][1].artifact_id,
                                  content_hash=views[0][1].content_hash),
            alternative_views=tuple(ArtifactRef(
                artifact_id=built.artifact_id, content_hash=built.content_hash)
                for _, built in views[1:]),
            accessible_summaries=tuple(view.accessible_summary for view, _ in views))
        self._commit(state, "GraphViewSet", view_set.canonical_payload(), parents)
        return self._out(state, stage="render")

    def capacity_node(self, state: DesignState) -> dict[str, Any]:
        """Measure applicable dimensions and compile the review bundle only when it fits."""
        design = self._model(state, "CompiledDesign", CompiledDesignV2)
        facts = self._model(state, "DesignFactSet", DesignFactSetV2)
        profile_id, _ = self._profile(state)
        report = compile_capacity_report(
            pack=self.deps.packs.get(design.method_id), facts=facts,
            profile=self._payload(profile_id), contrast_count=len(design.primary_contrasts),
            compiled_design=self._ref(state, "CompiledDesign"),
            registry=self.deps.capacity_registry)
        self._commit(state, "CapacityReport", report.canonical_payload(),
                     self._parents(state, "CompiledDesign", "GraphViewSet"))
        self._emit(state, "task.completed", EVAL_METHOD, status=report.status)
        if report.status == "fail":
            return self._route_compiler(state, report.issues)
        bundle = DesignReviewBundleV2(
            compiled_design=self._ref(state, "CompiledDesign"),
            diagnostic_report=self._ref(state, "DiagnosticReport"),
            capacity_report=self._ref(state, "CapacityReport"),
            graph_views=self._ref(state, "GraphViewSet"), rejected_methods=design.rejected_methods,
            assumptions=design.assumptions, identification_risks=design.identification_risks,
            sensitivity_requirements=design.sensitivity_requirements)
        self._commit(state, "DesignReviewBundle", bundle.canonical_payload(), self._parents(
            state, "CompiledDesign", "DiagnosticReport", "CapacityReport", "GraphViewSet"))
        return self._out(state, stage="capacity")

    def approval(self, state: DesignState) -> dict[str, Any]:
        """The exact-hash approval interrupt and its binding (PRD-002 §22)."""
        refs = [self._ref(state, kind).model_dump(mode="json") for kind in APPROVED_KINDS]
        anchor = self._ref(state, "DesignReviewBundle")
        design = self._model(state, "CompiledDesign", CompiledDesignV2)
        report = self._model(state, "DiagnosticReport", DiagnosticReportV2)
        capacity = self._model(state, "CapacityReport", CapacityReportV2)
        graphs = self._model(state, "GraphViewSet", GraphViewSetV2)
        summary = {
            "method": design.method_id, "estimand": design.estimand,
            "contrasts": ", ".join(design.primary_contrasts),
            "diagnostics": "computable" if report.computable else "not computable",
            "capacity": capacity.status,
            "graph": " | ".join(graphs.accessible_summaries)}
        question_ref = self._ref(state, "QuestionRecord")
        summary.update(approval_disclosures(
            design=design.canonical_payload(), design_ref=self._ref(state, "CompiledDesign"),
            question=self._payload(question_ref.artifact_id), question_ref=question_ref,
            registry_path=(self.deps.estimation_registry_path
                           or ANALYSIS_RESOURCES / "method-pack-estimation.v1.json")))
        self._emit(state, "user_interrupt.created", EVAL_APPROVAL, status="approval")
        decision = parse_strict(contracts.DesignApprovalDecisionV1, interrupt({
            "kind": contracts.InterruptKind.APPROVAL.value,
            "interrupt_artifact_id": anchor.artifact_id, "interrupt_hash": anchor.content_hash,
            "design_revision": state["design_revision"], "approved_artifacts": refs,
            # A person cannot consent to a design shown as four artifact ids and a hash: the
            # assumptions and identification risks they are approving must be legible (D-104).
            "design": self._payload(anchor.artifact_id), "review_summary": summary,
            "compiled_design": design.canonical_payload(),
            "diagnostic_report": report.canonical_payload(),
            "capacity_report": capacity.canonical_payload()}))
        expected_refs = tuple(ArtifactRef.model_validate(ref) for ref in refs)
        if (decision.interrupt_id != anchor.artifact_id
                or decision.expected_interrupt_hash != anchor.content_hash
                or decision.expected_revision != state["design_revision"]
                or (decision.decision is contracts.ApprovalDecision.APPROVED
                    and decision.approved_artifacts != expected_refs)):
            return self._fail(state, "stale_approval")
        held = self._commit(state, "DesignApprovalDecision", decision.canonical_payload(),
                            self._parents(state, "DesignReviewBundle"))
        self._emit(state, "user_interrupt.resumed", EVAL_APPROVAL, status=decision.decision.value)
        if decision.decision is not contracts.ApprovalDecision.APPROVED:
            return self._out(state, status=decision.decision.value)
        approval = DesignApprovalV2(
            decision="approved", design_revision=state["design_revision"], review_bundle=anchor,
            approved_bundle_hash=anchor.content_hash, change_requests=())
        self._commit(state, "DesignApproval", approval.canonical_payload(),
                     self._parents(state, "DesignReviewBundle") + (held,))
        return self._out(state, stage="approval", status="approved")

    def outcome_node(self, state: DesignState) -> dict[str, Any]:
        """Commit one explicit terminal outcome; absence of a status is a system failure."""
        status = state.get("status") or "system_failure"
        approved = status == "approved"
        issues: list[ValidationIssueV2] = []
        issues.extend(parse_strict(ValidationIssueV2, row)
                      for row in state.get("compiler_issues", ()))
        if "DiagnosticReport" in state["artifacts"]:
            issues.extend(self._model(state, "DiagnosticReport", DiagnosticReportV2).issues)
        if "CapacityReport" in state["artifacts"]:
            issues.extend(self._model(state, "CapacityReport", CapacityReportV2).issues)
        issues = list({issue.fingerprint: issue for issue in issues}.values())
        if not approved and not issues:
            category = {
                "needs_context": ResolutionCategory.HUMAN_INPUT,
                "needs_data": ResolutionCategory.NEEDS_DATA,
                "unsupported": ResolutionCategory.UNSUPPORTED,
            }.get(status, ResolutionCategory.SYSTEM_FAILURE)
            actor = {
                ResolutionCategory.HUMAN_INPUT: ResponsibleActor.USER,
                ResolutionCategory.NEEDS_DATA: ResponsibleActor.DATA_OWNER,
                ResolutionCategory.UNSUPPORTED: ResponsibleActor.PRODUCT,
                ResolutionCategory.SYSTEM_FAILURE: ResponsibleActor.SYSTEM,
            }[category]
            issues.append(ValidationIssueV2.build(
                code=state.get("error_code") or "missing_terminal_status", category=category,
                path="/status", rule_id="coordinator.terminal_status", actual=status,
                expected="an explicit recoverable or terminal state",
                why="the design did not reach an approved handoff", actor=actor,
                actions=("request_context",) if status == "needs_context" else ("inspect",),
                required=tuple(state["open_requirement_ids"])))
        payload = parse_strict(DesignOutcomeV2, {
            "status": status, "design_revision": state["design_revision"],
            "issues": [issue.model_dump(mode="json") for issue in issues],
            **{name: (self._ref(state, kind).model_dump(mode="json") if approved else None)
               for name, kind in OUTCOME_REFS.items()}})
        built = self._commit(state, "DesignOutcome", payload.canonical_payload(),
                             self._parents(state, "IntakeOutcome"))
        self.deps.products.transition_stage_run(state["stage_run_id"], "committing")
        self.deps.products.transition_stage_run(state["stage_run_id"], "completed")
        self._emit(state, "stage.completed", EVAL_STAGE, status=status, artifact_refs=(
            ArtifactRef(artifact_id=built.artifact_id, content_hash=built.content_hash),))
        return self._out(state, status=status)

    def handoff_open(self, state: DesignState) -> dict[str, Any]:
        """Open the PRD-003 handoff with exactly the four §23 entry artifacts."""
        if state.get("status") != "approved":
            return self._out(state)
        opened = open_design_handoff(self.deps, state["analysis_id"],
                                     state["artifacts"]["DesignOutcome"],
                                     f"sr:{state['analysis_id']}:preparation")
        return self._out(state, handoff_id=opened.handoff_id)



def _route(state: DesignState) -> str:
    """A terminal status exits; the shared prerequisite gate resumes its owning boundary."""
    if state.get("status"):
        return "outcome_node"
    if state.get("stage") == "gate":
        return state.get("resume_node") or "outcome_node"
    return "ask" if state.get("stage") == "gate_again" else "continue"


def build_graph(deps: DesignDeps) -> Any:
    """Compile the design StateGraph over the PostgreSQL checkpointer (PRD-002 §19)."""
    harness = _Harness(deps)
    builder = StateGraph(DesignState)
    for name in (*NODES, "requirements_gate", "handoff_open"):
        builder.add_node(name, getattr(harness, name))
    chain = (START, *NODES, "handoff_open", END)
    for name, following in itertools.pairwise(chain):
        if name in (START, "outcome_node", "handoff_open"):
            builder.add_edge(name, following)
        else:
            paths: dict[Hashable, str] = {
                "continue": following, "outcome_node": "outcome_node",
                "ask": "requirements_gate"}
            builder.add_conditional_edges(name, _route, paths)
    resumes: dict[Hashable, str] = {name: name for name in {*TASK_RESUME_NODES.values(), "method"}}
    builder.add_conditional_edges(
        "requirements_gate", _route, resumes | {"outcome_node": "outcome_node"})
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
        status=NEEDS_USER_INPUT if pending else str(final.get("status") or "system_failure"),
        analysis_id=str(final["analysis_id"]), stage_run_id=str(final["stage_run_id"]),
        thread_id=thread_id, design_revision=int(final["design_revision"]),
        interrupt_kind=value.get("kind"), interrupt_hash=value.get("interrupt_hash"),
        interrupt_artifact_id=value.get("interrupt_artifact_id"), outcome_artifact_id=outcome_id,
        handoff_id=final.get("handoff_id"), error_code=final.get("error_code"), review_summary=(
            {str(key): str(item) for key, item in value.get("review_summary", {}).items()}
            if isinstance(value.get("review_summary"), Mapping) else None))


def _previous_review(deps: DesignDeps, analysis_id: str, revision: int) -> ArtifactRef | None:
    if revision < 2:
        return None
    previous_stage = f"dr:{analysis_id}:{revision - 1}"
    rows = deps.conn.execute(
        "SELECT artifact_id FROM causal.artifacts WHERE analysis_id=%s AND stage_run_id=%s "
        "AND artifact_type='DesignApprovalDecision' ORDER BY created_at_utc DESC, artifact_id",
        (analysis_id, previous_stage)).fetchall()
    if not rows:
        return None
    if len(rows) != 1:
        raise DesignError("the previous revision has ambiguous approval decisions",
                          "review_feedback_mismatch")
    found = deps.products.load_envelope(str(rows[0][0]))
    body = json.loads(deps.objects.get(found.payload_locator))
    decision = parse_strict(contracts.DesignApprovalDecisionV1, body)
    if (found.analysis_id != analysis_id or found.stage_run_id != previous_stage
            or found.artifact_type != "DesignApprovalDecision"
            or found.content_hash != content_hash(body)
            or decision.expected_revision != revision - 1):
        raise DesignError("the previous approval decision does not match its revision",
                          "review_feedback_mismatch")
    return (ArtifactRef(artifact_id=found.artifact_id, content_hash=found.content_hash)
            if decision.decision is contracts.ApprovalDecision.CHANGES_REQUESTED else None)


def run_design(deps: DesignDeps, *, analysis_id: str, intake_outcome_artifact_id: str,
               thread_id: str, design_revision: int = 1) -> DesignRunResult:
    """Start one design revision on its own graph thread (PRD-002 §5, §19.2)."""
    inherited_answers: list[str] = []
    if design_revision > 1:
        previous = deps.conn.execute(
            "SELECT 1 FROM design.design_runs WHERE analysis_id = %s AND design_revision = %s",
            (analysis_id, design_revision - 1)).fetchone()
        if previous is None:
            raise DesignError("a design revision cannot skip its predecessor", "revision_gap")
        inherited = askgate.PsycopgAcceptedFactStore(deps.conn).inherit(
            analysis_id, design_revision - 1, design_revision, deps.clock())
        inherited_answers = sorted({fact.origin_reference_id for fact in inherited
                                    if fact.source_kind == "user"})
    feedback = _previous_review(deps, analysis_id, design_revision)
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
        "card_ids": [], "role_ids": [], "open_requirement_ids": [],
        "answer_ids": inherited_answers, "corrections": {}, "compiler_issues": []}
    if feedback is not None:
        state["review_feedback_id"] = feedback.artifact_id
        state["hashes"][feedback.artifact_id] = feedback.content_hash
    return _finish(deps, build_graph(deps).invoke(state, config=_config(thread_id)), thread_id)


def resume_design(deps: DesignDeps, *, thread_id: str,
                  resume_value: Mapping[str, Any]) -> DesignRunResult:
    """Resume the open interrupt on `thread_id` with one typed CLI decision (§11.1)."""
    final = build_graph(deps).invoke(
        Command(resume=dict(resume_value)), config=_config(thread_id))
    return _finish(deps, final, thread_id)
