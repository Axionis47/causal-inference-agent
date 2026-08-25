"""The LangGraph design harness: nodes, durable interrupts, and the coordinator (T-013 §1.5).

PRD-002 §5–§7, §9, §11.1, §16.4, §19–§23; SYSTEM-CONTRACT §6.2, §7, §9.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Protocol, TypedDict, cast

from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from psycopg import Connection
from psycopg.types.json import Json
from pydantic import BaseModel, ValidationError

from causal.design import askgate, contracts, entry, frame, semantics, validators
from causal.design import compile as compiler
from causal.design.capacity import CapacityRegistryV1, check_capacity
from causal.design.diagnostics import DIAGNOSTIC_SPECS, run_diagnostic
from causal.design.packs import MethodPackRegistry, RequirementTemplateV1, ToolRegistry
from causal.design.renderer import RendererError, render_causal_graph
from causal.design.triage import ColumnTriageRecordV1, triage
from causal.shared import envelope as agent
from causal.shared import events, handoff, persistence
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactEnvelopeV1, ArtifactRef, HandoffManifestV1
from causal.shared.gateway import GatewayResultV1
from causal.shared.readers import CatalogReader, CsvObjectFrameSource
from causal.shared.registry import ArtifactTypeRegistry
from causal.shared.validation import parse_strict

__all__ = [
    "COMPONENT", "NEEDS_USER_INPUT", "DesignDeps", "DesignError", "DesignRunResult",
    "DesignState", "GatewayProtocol", "build_checkpointer", "build_graph", "open_design_handoff",
    "resume_design", "run_design",
]

COMPONENT: Final = "design-harness"
VERSION: Final = "design-harness.v1"
NEEDS_USER_INPUT: Final = "needs_user_input"
ALLOWED_INTAKE: Final = frozenset({"usable", "partial"})
EVAL_STAGE: Final = ("EV-P2-001",)
EVAL_ASK: Final = ("EV-P2-006",)
EVAL_METHOD: Final = ("EV-P2-007",)
EVAL_APPROVAL: Final = ("EV-P2-008",)
EVAL_TASK: Final[dict[str, tuple[str, ...]]] = {
    "intent": ("EV-P2-002",), "semantic_batch": ("EV-P2-003",), "role_evidence": ("EV-P2-004",),
    "causal_synthesis": ("EV-P2-005",), "method_design": ("EV-P2-007",)}
REGISTRY_VERSIONS: Final = {
    "artifact_types": "artifact-types.v1", "field_classes": "kaggle-field-classes.v1",
    "method_packs": MethodPackRegistry.registry_version, "tools": "design-tools.v1",
    "requirements": "context-requirements.v1", "validators": validators.REGISTRY_VERSION,
    "capacity": "delivery-capacity.v1", "graph": VERSION, "schema": "design-tasks.v1"}
# The nodes in workflow order (PRD-002 §7); `handoff_open` closes the run after the outcome.
NODES: Final = (
    "entry", "selection", "manifest", "intent", "triage_node", "semantic", "measurement",
    "roles", "synthesis", "requirements_gate", "method", "render", "capacity_node", "approval",
    "outcome_node")
# The parents the renderer and the approval binding name, in §12.3 / §22 order.
VIEW_PARENTS: Final = (
    "CausalContext", "MeasurementMap", "RoleLedger", "ExperimentDesign", "RunnableFrameContract")
APPROVED_KINDS: Final = (
    "ExperimentDesign", "RunnableFrameContract", "CausalGraphView", "DeliveryCapacityCheck")
OUTCOME_REFS: Final = {
    "experiment_design": "ExperimentDesign", "runnable_frame_contract": "RunnableFrameContract",
    "causal_graph_view": "CausalGraphView", "capacity_check": "DeliveryCapacityCheck",
    "approval": "DesignApproval"}

_RUN_ROW: Final = (
    "INSERT INTO design.design_runs (stage_run_id, analysis_id, graph_thread_id, design_revision,"
    " state, created_at, updated_at) VALUES (%s, %s, %s, %s, 'running', %s, %s)"
    " ON CONFLICT (stage_run_id) DO NOTHING")
_RUN_STATE: Final = (
    "UPDATE design.design_runs SET state = %s, outcome_artifact_id = %s, method_id = %s,"
    " selected_table = %s, updated_at = %s WHERE stage_run_id = %s")
_TASK_ROW: Final = (
    "INSERT INTO design.design_tasks (task_id, analysis_id, stage_run_id, design_revision,"
    " task_kind, scope, envelope_hash, prompt_version, model_profile_version, status,"
    " output_artifact_id, attempts) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    " ON CONFLICT (task_id) DO UPDATE SET status = EXCLUDED.status,"
    " output_artifact_id = EXCLUDED.output_artifact_id, attempts = EXCLUDED.attempts")
_VIEW_ROW: Final = (
    "INSERT INTO design.causal_graph_views VALUES (%s, %s, %s, %s, %s)"
    " ON CONFLICT (artifact_id) DO NOTHING")
_OPEN_REQS: Final = (
    "SELECT requirement_id, scope_id, attempted_evidence FROM design.context_requirements"
    " WHERE analysis_id = %s AND design_revision = %s AND state = 'open'"
    " ORDER BY requirement_id, scope_id")
_REQ_STATES: Final = (
    "SELECT requirement_id, state FROM design.context_requirements"
    " WHERE analysis_id = %s AND design_revision = %s")


class DesignError(ValueError):
    """A design-harness step failed; `code` is a stable contract value."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class GatewayProtocol(Protocol):
    """The one model call the harness makes (T-010 `VertexGateway.invoke`)."""

    def invoke(self, envelope: agent.AgentTaskEnvelopeV1, prompt: str,
               response_schema: dict[str, object]) -> GatewayResultV1: ...


class DesignState(TypedDict, total=False):
    """The §19.1 allowlist: identities, hashes, statuses, counters, and pending ids only."""

    analysis_id: str
    stage_run_id: str
    thread_id: str
    design_revision: int
    dataset_id: str
    stage: str
    status: str
    error_code: str
    refusal_code: str
    artifacts: dict[str, str]
    hashes: dict[str, str]
    card_ids: list[str]
    role_ids: list[str]
    pending_task_ids: list[str]
    open_requirement_ids: list[str]
    answer_ids: list[str]
    candidate_method_ids: list[str]
    corrections: dict[str, int]
    counts: dict[str, int]
    clarification_round: int
    method_id: str
    capacity_status: str
    graph_view_status: str
    approval_status: str
    handoff_id: str


@dataclass(frozen=True)
class DesignRunResult:
    """What one `causal run` step of a design revision returns (PRD-002 §11.1)."""

    status: str
    analysis_id: str
    stage_run_id: str
    thread_id: str
    design_revision: int
    interrupt_kind: str | None = None
    interrupt_artifact_id: str | None = None
    interrupt_hash: str | None = None
    outcome_artifact_id: str | None = None
    handoff_id: str | None = None
    refusal_code: str | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class DesignDeps:
    """Everything the harness needs; nothing here is discovered at run time."""

    conn: Connection[Any]
    catalog: CatalogReader
    committer: persistence.ArtifactCommitter
    products: persistence.ProductStore
    objects: persistence.ObjectStore
    registry: ArtifactTypeRegistry
    gateway: GatewayProtocol
    emitter: events.EventEmitter
    clock: Callable[[], datetime]
    checkpointer: PostgresSaver
    packs: MethodPackRegistry
    templates: Mapping[str, RequirementTemplateV1]
    tool_registry: ToolRegistry
    task_table: Mapping[str, compiler.TaskSpecV1]
    rules: tuple[validators.ValidationRuleV1, ...]
    capacity_registry: CapacityRegistryV1
    prompts_root: Path
    repo_root: Path


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


class _Harness:
    """Node bodies over one `DesignDeps`; every node is thin and replay-safe."""

    def __init__(self, deps: DesignDeps) -> None:
        self.deps = deps
        self.requirements = askgate.PsycopgRequirementStore(deps.conn)
        self._count = 0
        self._cache: dict[str, Any] = {}

    # -- events, artifacts, and committed payloads ------------------------

    def _event(self, state: DesignState, name: str, evals: tuple[str, ...],
               **over: Any) -> events.OperationalEventV1:
        """One §20.8 event with a deterministic id and the design stage's identities."""
        self._count += 1
        return events.build_event(
            occurred_at_utc=self.deps.clock(), event_name=name, analysis_id=state["analysis_id"],
            event_id=f"evt:{state['stage_run_id']}:{self._count}", stage=events.Stage.DESIGN,
            stage_run_id=state["stage_run_id"], graph_thread_id=state["thread_id"],
            component_id=COMPONENT, component_version=VERSION, required_eval_ids=evals,
            versions={"registry": "artifact-types.v1", "prompt": VERSION}, **over)

    def _emit(self, state: DesignState, name: str, evals: tuple[str, ...], **over: Any) -> None:
        self.deps.emitter.emit(self._event(state, name, evals, **over))

    def _commit(self, state: DesignState, kind: str, payload: Mapping[str, object],
                parents: tuple[ArtifactEnvelopeV1, ...]) -> ArtifactEnvelopeV1:
        """Build and commit one immutable artifact through the flush-gated committer (§8.2)."""
        body = dict(payload)
        built = persistence.build_envelope(
            self.deps.registry, kind, body, analysis_id=state["analysis_id"],
            stage_run_id=state["stage_run_id"], producer_version=VERSION, parents=parents,
            created_at_utc=self.deps.clock())
        ref = ArtifactRef(artifact_id=built.artifact_id, content_hash=built.content_hash)
        committed = self.deps.committer.commit(built, body, self._event(
            state, "artifact.committed", EVAL_STAGE, status="committed", artifact_refs=(ref,)))
        state["artifacts"][kind] = committed.artifact_id
        state["hashes"][committed.artifact_id] = committed.content_hash
        return committed

    def _payload(self, artifact_id: str) -> dict[str, Any]:
        """One committed payload, loaded for the node that needs it and released after."""
        if artifact_id not in self._cache:
            locator = self.deps.products.load_envelope(artifact_id).payload_locator
            self._cache[artifact_id] = json.loads(self.deps.objects.get(locator))
        return cast(dict[str, Any], self._cache[artifact_id])

    def _model[M: BaseModel](self, state: DesignState, kind: str, model: type[M]) -> M:
        return parse_strict(model, self._payload(state["artifacts"][kind]))

    def _ref(self, state: DesignState, kind: str) -> ArtifactRef:
        found = state["artifacts"][kind]
        return ArtifactRef(artifact_id=found, content_hash=state["hashes"][found])

    def _parents(self, state: DesignState, *kinds: str) -> tuple[ArtifactEnvelopeV1, ...]:
        return tuple(self.deps.products.load_envelope(state["artifacts"][k]) for k in kinds)

    def _out(self, state: DesignState, **over: Any) -> dict[str, Any]:
        """Every node writes the whole allowlisted snapshot back, so replay sees one state."""
        return dict(state) | dict(over)

    def _fail(self, state: DesignState, code: str, codes: Sequence[str] = ()) -> dict[str, Any]:
        """Enter the terminal `failed` state with one stable code and a raised blocker (§7.1)."""
        self._emit(state, "blocker.raised", EVAL_STAGE, severity=events.Severity.ERROR,
                   error_code=code, safe_dimensions={"blocked_operation": state.get("stage", ""),
                                                     "detail_codes": ",".join(codes)})
        return self._out(state, status="failed", error_code=code)

    # -- validation context and requirement rows --------------------------

    def _evidence(self, state: DesignState) -> frozenset[str]:
        """The allowlisted intake evidence ids for this analysis."""
        outcome = self._payload(state["artifacts"]["IntakeOutcome"])
        bundle = outcome.get("evidence_bundle_artifact_id")
        items = self._payload(str(bundle))["items"] if bundle else []
        return frozenset(str(row["evidence_id"]) for row in items)

    def _requirement_states(self, state: DesignState) -> dict[str, str]:
        """A requirement no worker ever raised was never missing, so it reads as resolved."""
        rows = self.deps.conn.execute(
            _REQ_STATES, (state["analysis_id"], state["design_revision"])).fetchall()
        return dict.fromkeys(self.deps.templates, "resolved") | {
            str(row[0]): str(row[1]) for row in rows}

    def _open_requirements(self, state: DesignState) -> tuple[agent.ContextRequirementV1, ...]:
        """Rebuild the open requirement rows from their §10.1 templates (checkpoint-safe)."""
        rows = self.deps.conn.execute(
            _OPEN_REQS, (state["analysis_id"], state["design_revision"])).fetchall()
        return tuple(agent.ContextRequirementV1(
            requirement_id=str(row[0]), scope_id=str(row[1]), decisions_blocked=(),
            registry_version="context-requirements.v1",
            attempted_evidence=tuple(
                parse_strict(agent.AttemptedEvidenceV1, item) for item in row[2]),
            **{key: value for key, value in dict(self.deps.templates[str(row[0])]).items()
               if key != "requirement_id"})
            for row in rows if str(row[0]) in self.deps.templates)

    def _ctx(self, state: DesignState, **over: Any) -> validators.ValidationContext:
        """Everything the walls may read; a validator never loads anything itself."""
        base: dict[str, Any] = {
            "manifest": self._model(
                state, "DesignContextManifest", contracts.DesignContextManifestV1),
            "rules": self.deps.rules, "packs": self.deps.packs, "templates": self.deps.templates,
            "parents": dict(state["hashes"]), "evidence_ids": self._evidence(state),
            "user_answer_evidence_ids": frozenset(f"ua:{a}" for a in state["answer_ids"]),
            "resolved_requirements": self._requirement_states(state)}
        return validators.ValidationContext(**base | over)

    # -- the model-task loop (PRD-002 §16.1, §16.4) -----------------------

    def _invoke(self, state: DesignState, spec: compiler.TaskSpecV1, task_id: str, attempt: int,
                scope: tuple[str, Sequence[str]], refs: tuple[ArtifactRef, ...],
                payload: Mapping[str, object],
                ) -> tuple[agent.AgentTaskEnvelopeV1, agent.AgentTaskResultV1 | None]:
        """One physical attempt: envelope, prompt, gateway, strict `AgentTaskResultV1` parse."""
        built = compiler.build_task_envelope(
            spec, analysis_id=state["analysis_id"], stage_run_id=state["stage_run_id"],
            task_id=task_id, attempt_id=f"{task_id}:{attempt}", scope_kind=scope[0],
            manifest_ref=self._ref(state, "DesignContextManifest"), scope_ids=scope[1],
            parent_artifacts=refs, allowed_evidence_ids=sorted(self._evidence(state)),
            allowed_tool_ids=self.deps.tool_registry.recipient_map()[spec.task_kind],
            payload_type=f"{spec.task_kind}-context", payload=payload)
        self._emit(state, "agent.started", EVAL_TASK[spec.task_kind], task_id=task_id,
                   attempt_id=built.attempt_id, attempt_number=attempt)
        answer = self.deps.gateway.invoke(built, compiler.render_prompt(
            spec, self.deps.prompts_root, dict(payload)), {"type": "object"})
        try:
            return built, parse_strict(agent.AgentTaskResultV1, answer.parsed or {})
        except ValidationError:
            return built, None

    def _run_task(self, state: DesignState, kind: str, model: type[BaseModel], *,
                  scope_kind: str, scope_ids: Sequence[str], parent_kinds: tuple[str, ...],
                  payload: Mapping[str, object], many: bool = False, commits: str | None = None,
                  ctx: validators.ValidationContext | None = None,
                  ) -> tuple[tuple[Any, ArtifactEnvelopeV1], ...] | None:
        """One model task with the §16.4 correction loop; commits every validated payload."""
        spec, evals = self.deps.task_table[kind], EVAL_TASK[kind]
        artifact_type = commits or spec.output_artifact_type
        task_id = f"dt:{state['stage_run_id']}:{kind}:{content_hash(dict(payload))[:12]}"
        parents = self._parents(state, *parent_kinds)
        refs = tuple(ArtifactRef(artifact_id=p.artifact_id, content_hash=p.content_hash)
                     for p in parents)
        body, issues = dict(payload), cast(tuple[validators.ValidationIssueV1, ...], ())
        for attempt in range(1, spec.correction_budget + 2):
            built, result = self._invoke(
                state, spec, task_id, attempt, (scope_kind, scope_ids), refs, body)
            digest = content_hash(built.canonical_payload())
            if result is None:
                for name in ("agent.schema_failed", "agent.correction_requested"):
                    self._emit(state, name, evals, severity=events.Severity.ERROR,
                               task_id=task_id, attempt_number=attempt,
                               error_code="schema_invalid")
                state["corrections"][f"{artifact_type}:schema_invalid"] = attempt
                body = dict(payload) | {"correction": {"issues": [{"code": "schema_invalid"}]}}
                continue
            items = ([dict(row) for row in cast(list[Any], result.payload["items"])] if many
                     else [dict(result.payload)])
            issues = tuple(
                issue for row in items
                for issue in validators.validate_result(
                    spec.wall, kind, model, result.model_copy(update={"payload": row}),
                    ctx or self._ctx(state)).issues)
            self.requirements.upsert(
                result.missing_requirements, state["analysis_id"], state["design_revision"])
            state["open_requirement_ids"] = sorted({*state["open_requirement_ids"], *(
                row.requirement_id for row in result.missing_requirements)})
            if not issues:
                done = tuple((parse_strict(model, row), self._commit(
                    state, artifact_type, row, parents)) for row in items)
                self._task_row(state, task_id, kind, scope_ids, digest, built,
                               result.status.value, attempt, done[-1][1].artifact_id)
                self._emit(state, "task.completed", evals, task_id=task_id,
                           status=result.status.value)
                return done
            for issue in issues:
                state["corrections"][f"{artifact_type}:{issue.code}"] = attempt
            self._emit(state, "artifact.validation_failed", evals, task_id=task_id,
                       severity=events.Severity.WARNING, error_code=issues[0].code)
            self._emit(state, "agent.correction_requested", evals, task_id=task_id,
                       attempt_number=attempt, error_code=issues[0].code)
            body = dict(payload) | {"correction": {"failing_payload": items[0], "issues": [
                issue.model_dump(mode="json") for issue in issues]}}
        self._exhausted(state, task_id, kind, issues)
        return None

    def _task_row(self, state: DesignState, task_id: str, kind: str, scope_ids: Sequence[str],
                  digest: str, built: agent.AgentTaskEnvelopeV1, status: str, attempts: int,
                  output_id: str) -> None:
        """One §5.4 audit row per delegated model task."""
        self.deps.conn.execute(_TASK_ROW, (
            task_id, state["analysis_id"], state["stage_run_id"], state["design_revision"], kind,
            Json({"scope_kind": built.scope_kind, "scope_ids": list(scope_ids)}), digest,
            built.prompt_version, built.model_profile_version, status, output_id, attempts))

    def _exhausted(self, state: DesignState, task_id: str, kind: str,
                   issues: tuple[validators.ValidationIssueV1, ...]) -> None:
        """§16.4 routing after one response plus the two permitted targeted corrections."""
        code = issues[0].code if issues else "schema_invalid"
        status = ("needs_context" if any(issue.user_resolvable for issue in issues)
                  else "refused" if kind == "method_design" else "failed")
        self._emit(state, "retry.exhausted", EVAL_TASK[kind], severity=events.Severity.ERROR,
                   task_id=task_id, error_code=code)
        self._emit(state, "blocker.raised", EVAL_TASK[kind], severity=events.Severity.ERROR,
                   task_id=task_id, error_code="correction_exhausted",
                   safe_dimensions={"blocked_operation": kind, "validation_code": code})
        state["status"], state["error_code"] = status, "correction_exhausted"
        if status == "refused":
            state["refusal_code"] = "UNSUPPORTED_IDENTIFICATION"

    # -- nodes ------------------------------------------------------------

    def _handoff_manifest(self, state: DesignState) -> HandoffManifestV1:
        """The intake handoff this design revision opens, rebuilt from the committed outcome."""
        found = self.deps.products.load_envelope(state["artifacts"]["IntakeOutcome"])
        payload = self._payload(found.artifact_id)
        return HandoffManifestV1(
            handoff_id=f"ho:{state['analysis_id']}:{found.content_hash[:16]}",
            schema_version="handoff.v1", analysis_id=state["analysis_id"],
            producing_stage_run_id=found.stage_run_id,
            receiving_stage_run_id=state["stage_run_id"], approval_ids=(),
            entries=(ArtifactRef(artifact_id=found.artifact_id,
                                 content_hash=found.content_hash),),
            originating_outcome=str(payload["status"]), registry_version="artifact-types.v1",
            compatibility_version="handoff.v1", receiver_validation_result=None,
            receiver_error_codes=(), created_at_utc=self.deps.clock(), accepted_at_utc=None)

    def entry(self, state: DesignState) -> dict[str, Any]:
        """Open the intake handoff through the shared T-006 gate (PRD-002 §5 conditions 2–3)."""
        self._emit(state, "stage.started", EVAL_STAGE)
        manifest = self._handoff_manifest(state)
        payload = self._payload(state["artifacts"]["IntakeOutcome"])
        gate = handoff.HandoffGate(self.deps.objects, self.deps.products,
                                   handoff.HandoffStore(self.deps.conn), self.deps.registry,
                                   self.deps.emitter)
        opened = gate.accept(manifest, COMPONENT, ALLOWED_INTAKE, lambda verdict, codes: self._event(
            state, f"handoff.{verdict}", EVAL_STAGE, status=verdict,
            error_code=codes[0] if codes else None))
        for found in (manifest.entries[0].artifact_id, str(payload["question_artifact_id"])):
            state["hashes"][found] = self.deps.products.load_envelope(found).content_hash
        state["artifacts"]["QuestionRecord"] = str(payload["question_artifact_id"])
        if not opened.accepted:
            return self._fail(state, "handoff_unavailable", opened.error_codes)
        return self._out(state, stage="entry", dataset_id=str(payload["dataset_id"]))

    def selection(self, state: DesignState) -> dict[str, Any]:
        """One admitted CSV, or the durable table-selection interrupt (PRD-002 §5, §11.1)."""
        dataset = state["dataset_id"]
        candidates = entry.list_csv_candidates(self.deps.catalog, dataset)
        try:
            routed = entry.resolve_selection(candidates, None, dataset)
            if isinstance(routed, entry.SelectionRequired):
                anchor = self._ref(state, "IntakeOutcome")
                self._emit(state, "user_interrupt.created", EVAL_STAGE, status="table_selection")
                decision = parse_strict(contracts.TableSelectionDecisionV1, interrupt({
                    "kind": contracts.InterruptKind.TABLE_SELECTION.value,
                    "interrupt_artifact_id": anchor.artifact_id,
                    "interrupt_hash": anchor.content_hash,
                    "design_revision": state["design_revision"],
                    "candidates": [row.logical_name for row in routed.candidates]}))
                chosen = self._commit(state, "TableSelectionDecision", decision.canonical_payload(),
                                      self._parents(state, "IntakeOutcome"))
                self._emit(state, "user_interrupt.resumed", EVAL_STAGE, status="table_selection")
                routed = entry.resolve_selection(
                    candidates, decision, dataset, decision_artifact_id=chosen.artifact_id,
                    other_admitted=entry.list_admitted_non_csv(self.deps.catalog, dataset))
        except entry.EntryError as error:
            return self._out(state, status="refused", refusal_code=error.code)
        assert isinstance(routed, contracts.TableSelectionV1)
        self._commit(state, "TableSelection", routed.canonical_payload(),
                     self._parents(state, "IntakeOutcome", "QuestionRecord"))
        return self._out(state, stage="selection")

    def manifest(self, state: DesignState) -> dict[str, Any]:
        """Check §5 conditions 1–5, then compile the one immutable context surface."""
        selection = self._model(state, "TableSelection", contracts.TableSelectionV1)
        try:
            entry.validate_entry(self._handoff_manifest(state),
                                 self._payload(state["artifacts"]["IntakeOutcome"]),
                                 self.deps.products, selection=selection)
        except entry.EntryError as error:
            return self._fail(state, error.code, error.detail_codes)
        compiled = entry.compile_manifest(
            self.deps.catalog, selection, question_ref=self._ref(state, "QuestionRecord"),
            outcome_ref=self._ref(state, "IntakeOutcome"),
            selection_ref=self._ref(state, "TableSelection"),
            design_revision=state["design_revision"], registry_versions=REGISTRY_VERSIONS,
            recipient_map=self.deps.tool_registry.recipient_map())
        self._commit(state, "DesignContextManifest", compiled.canonical_payload(),
                     self._parents(state, "TableSelection"))
        return self._out(state, stage="manifest")

    def intent(self, state: DesignState) -> dict[str, Any]:
        """The single bounded intent task (PRD-002 §8.1)."""
        book = self._model(state, "DesignContextManifest", contracts.DesignContextManifestV1)
        done = self._run_task(
            state, "intent", contracts.DesignIntentV1, scope_kind="design",
            scope_ids=(book.selected_table,), parent_kinds=("DesignContextManifest",),
            payload={"question": self._payload(state["artifacts"]["QuestionRecord"]),
                     "selected_table": book.selected_table, "columns": [
                         row.column_name for row in book.structural_inventory]})
        return self._out(state) if done is None else self._out(state, stage="intent")

    def triage_node(self, state: DesignState) -> dict[str, Any]:
        """Deterministic `triage.v1` over the committed table profile; no model runs here."""
        book = self._model(state, "DesignContextManifest", contracts.DesignContextManifestV1)
        hypotheses: dict[str, tuple[str, ...]] = {}
        for found in self._payload(state["artifacts"]["IntakeOutcome"])["table_profile_artifact_ids"]:
            profile = self._payload(str(found))
            if profile.get("logical_name") == book.selected_table:
                hypotheses = {name: tuple(str(item["kind"]) for item in col.get("hypotheses") or ())
                              for name, col in (profile.get("columns") or {}).items()}
        record = triage(
            self._model(state, "DesignIntent", contracts.DesignIntentV1), book, hypotheses)
        self._commit(state, "ColumnTriageRecord", record.canonical_payload(),
                     self._parents(state, "DesignIntent"))
        return self._out(state, stage="triage", pending_task_ids=[
            batch.batch_id for batch in record.batches], counts={
                **state["counts"], "columns_triaged": len(record.match_trace),
                "deferred_columns": len(record.deferred)})

    def semantic(self, state: DesignState) -> dict[str, Any]:
        """One sequential task per frozen batch, one card per assigned column (D-050)."""
        record = self._model(state, "ColumnTriageRecord", ColumnTriageRecordV1)
        cards = list(state["card_ids"])
        for batch in record.batches:
            done = self._run_task(
                state, "semantic_batch", semantics.ColumnSemanticCardV1, scope_kind="column",
                scope_ids=batch.column_names, parent_kinds=("DesignIntent",), many=True,
                ctx=self._ctx(state, triage=record),
                payload={"batch_id": batch.batch_id, "table_name": record.table_name,
                         "columns": list(batch.column_names)})
            if done is None:
                return self._out(state)
            cards.extend(found.artifact_id for _, found in done)
        return self._out(state, stage="semantic", card_ids=cards, counts={
            **state["counts"], "columns_carded": len(cards)})

    def measurement(self, state: DesignState) -> dict[str, Any]:
        """The deterministic MeasurementMap compiler over the validated cards (§9.5)."""
        built = compiler.compile_measurement_map(
            self._model(state, "DesignIntent", contracts.DesignIntentV1),
            [parse_strict(semantics.ColumnSemanticCardV1, self._payload(found))
             for found in state["card_ids"]])
        self._commit(state, "MeasurementMap", built.canonical_payload(),
                     self._parents(state, "DesignIntent"))
        return self._out(state, stage="measurement")

    def roles(self, state: DesignState) -> dict[str, Any]:
        """One role task per frozen batch scope; a worker never widens it (SC §7)."""
        record = self._model(state, "ColumnTriageRecord", ColumnTriageRecordV1)
        mapped = self._model(state, "MeasurementMap", semantics.MeasurementMapV1)
        found_ids = list(state["role_ids"])
        for batch in record.batches:
            concepts = sorted({link.concept_id for link in mapped.links
                               if link.column_name in batch.column_names})
            done = self._run_task(
                state, "role_evidence", semantics.RoleEvidenceV1, scope_kind="relationship",
                scope_ids=concepts or [record.table_name], parent_kinds=("MeasurementMap",),
                payload={"batch_id": batch.batch_id, "concept_ids": concepts,
                         "columns": list(batch.column_names)})
            if done is None:
                return self._out(state)
            found_ids.extend(found.artifact_id for _, found in done)
        return self._out(state, stage="roles", role_ids=found_ids, counts={
            **state["counts"], "role_tasks": len(found_ids)})

    def synthesis(self, state: DesignState) -> dict[str, Any]:
        """CausalContext then RoleLedger, both behind the wall-5 causal validator (§8.4)."""
        payload: dict[str, object] = {
            "measurement_map": self._payload(state["artifacts"]["MeasurementMap"]),
            "hypotheses": [self._payload(found) for found in state["role_ids"]]}
        done = self._run_task(
            state, "causal_synthesis", semantics.CausalContextV1, scope_kind="design",
            scope_ids=("causal_context",), parent_kinds=("MeasurementMap",), payload=payload)
        if done is None:
            return self._out(state)
        ledger = self._run_task(
            state, "causal_synthesis", semantics.RoleLedgerV1, scope_kind="design",
            scope_ids=("role_ledger",), parent_kinds=("CausalContext",), commits="RoleLedger",
            ctx=self._ctx(state, causal_context=done[0][0]),
            payload=payload | {"causal_context": self._ref(state, "CausalContext").model_dump(
                mode="json")})
        return self._out(state) if ledger is None else self._out(state, stage="synthesis")

    def requirements_gate(self, state: DesignState) -> dict[str, Any]:
        """Freeze, route, and ask at most twice per revision (SC §6.2; PRD-002 §11)."""
        frozen = askgate.freeze_requirements(self._open_requirements(state))
        round_number = state["clarification_round"] + 1
        decisions = askgate.gate(frozen, {}, round_number, self.deps.templates)
        settled = {askgate.GateRoute.RESOLVED: askgate.RequirementState.RESOLVED,
                   askgate.GateRoute.RECORD_SENSITIVITY: askgate.RequirementState.UNKNOWN_ACCEPTED}
        for row in decisions:
            if row.route in settled:
                self.requirements.set_state(state["analysis_id"], state["design_revision"],
                                            row.requirement_id, row.scope_id, settled[row.route],
                                            None)
        asking = {row.requirement_id for row in decisions if row.route is askgate.GateRoute.ASK}
        if not asking:
            terminal = [row for row in decisions if row.route.value.startswith("terminal")]
            if not terminal:
                return self._out(state, stage="gate")
            refused = terminal[0].route is askgate.GateRoute.TERMINAL_REFUSED
            return self._out(state, status="refused" if refused else "needs_context",
                             refusal_code="UNSUPPORTED_IDENTIFICATION" if refused else None)
        return self._ask(state, frozen, asking, round_number)

    def _ask(self, state: DesignState, frozen: tuple[agent.ContextRequirementV1, ...],
             asking: set[str], round_number: int) -> dict[str, Any]:
        """One clarification round: packet, durable interrupt, typed answers (PRD-002 §11.1)."""
        book = self._model(state, "DesignContextManifest", contracts.DesignContextManifestV1)
        columns = [row.column_name for row in book.structural_inventory]
        packet = askgate.build_packet([row for row in frozen if row.requirement_id in asking],
                                      state["design_revision"], round_number, columns)
        opened = self._commit(state, "UserQuestionPacket", packet.canonical_payload(),
                              self._parents(state, "DesignContextManifest"))
        self._emit(state, "user_interrupt.created", EVAL_ASK, status="clarification",
                   artifact_refs=(self._ref(state, "UserQuestionPacket"),))
        answer = parse_strict(contracts.UserContextAnswerV1, interrupt({
            "kind": contracts.InterruptKind.CLARIFICATION.value,
            "interrupt_artifact_id": opened.artifact_id, "interrupt_hash": opened.content_hash,
            "design_revision": state["design_revision"], "round_number": round_number,
            "packet": packet.canonical_payload()}))
        stored = self._commit(state, "UserContextAnswer", answer.canonical_payload(), (opened,))
        index = {row.requirement_id: row for row in frozen}
        for outcome in askgate.validate_answers(packet, answer, index, columns):
            self.requirements.set_state(
                state["analysis_id"], state["design_revision"], outcome.requirement_id,
                index[outcome.requirement_id].scope_id, outcome.state, stored.artifact_id)
        self._emit(state, "user_interrupt.resumed", EVAL_ASK, status="clarification")
        return self._out(state, stage="gate_again", clarification_round=round_number,
                         answer_ids=[*state["answer_ids"], stored.artifact_id])

    def method(self, state: DesignState) -> dict[str, Any]:
        """Pack eligibility, the method-design task, prerepair diagnostics, and the frame (§13)."""
        ledger = self._model(state, "RoleLedger", semantics.RoleLedgerV1)
        held = {row.role.value for row in ledger.claims
                if row.status is not agent.EpistemicStatus.UNKNOWN}
        eligible = [row.method_id for row in self.deps.packs.all()
                    if set(row.required_roles) <= held]
        if not eligible:
            return self._out(state, status="refused", refusal_code="UNSUPPORTED_METHOD",
                             candidate_method_ids=[])
        payload: dict[str, object] = {
            "eligible_methods": eligible, "parents": {
                kind: self._ref(state, kind).model_dump(mode="json") for kind in
                ("TableSelection", "MeasurementMap", "CausalContext", "RoleLedger")},
            "role_ledger": self._payload(state["artifacts"]["RoleLedger"]),
            "method_contracts": [self.deps.packs.get(name).model_dump(mode="json")
                                 for name in eligible]}
        common = {"role_ledger": ledger, "causal_context": self._model(
            state, "CausalContext", semantics.CausalContextV1)}
        done = self._run_task(
            state, "method_design", frame.ExperimentDesignV1, scope_kind="design",
            scope_ids=("experiment_design",), parent_kinds=("RoleLedger",), payload=payload,
            ctx=self._ctx(state, method_id=eligible[0], **common))
        if done is None:
            return self._out(state, candidate_method_ids=eligible)
        design = cast(frame.ExperimentDesignV1, done[0][0])
        self._prerepair(state, design, ledger)
        contract = self._run_task(
            state, "method_design", frame.RunnableFrameContractV1, scope_kind="design",
            scope_ids=("runnable_frame_contract",), parent_kinds=("ExperimentDesign",),
            commits="RunnableFrameContract",
            ctx=self._ctx(state, method_id=design.method_id, design=design, **common),
            payload=payload | {"experiment_design": self._ref(
                state, "ExperimentDesign").model_dump(mode="json")})
        if contract is None:
            return self._out(state, candidate_method_ids=eligible)
        return self._out(state, stage="method", method_id=design.method_id,
                         candidate_method_ids=eligible)

    def _prerepair(self, state: DesignState, design: frame.ExperimentDesignV1,
                   ledger: semantics.RoleLedgerV1) -> None:
        """The design's required read-only diagnostics over the selected CSV (PRD-002 §14)."""
        selection = self._model(state, "TableSelection", contracts.TableSelectionV1)
        source = CsvObjectFrameSource(self.deps.objects, selection.resource_object_locator,
                                      self._ref(state, "TableSelection"))
        columns = {row.role.value: list(row.column_refs) for row in ledger.claims}
        params = {
            "columns": columns.get("treatment", []), "by": columns.get("group", []),
            "key_columns": columns.get("unit_identifier", []),
            "target": next(iter(columns.get("outcome", [])), None),
            "column": next(iter(columns.get("treatment", [])), None),
            "running_column": next(iter(columns.get("running_variable", [])), None)}
        results = tuple(run_diagnostic(name, source, params)
                        for name in design.required_prerepair_diagnostics
                        if name in DIAGNOSTIC_SPECS)
        if results:
            self._commit(state, "PreRepairFeasibilityReport", frame.PreRepairFeasibilityReportV1(
                method_id=design.method_id, results=results).canonical_payload(),
                self._parents(state, "RoleLedger"))

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
