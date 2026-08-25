"""Design-harness state, dependencies, and shared node helpers (T-013 §1.5 split, D-055)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Protocol, TypedDict, cast

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.types.json import Json
from pydantic import BaseModel, ValidationError

from causal.design import askgate, contracts, validators
from causal.design import compile as compiler
from causal.design.capacity import CapacityRegistryV1
from causal.design.packs import MethodPackRegistry, RequirementTemplateV1, ToolRegistry
from causal.shared import envelope as agent
from causal.shared import events, persistence
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactEnvelopeV1, ArtifactRef, HandoffManifestV1
from causal.shared.gateway import GatewayResultV1
from causal.shared.readers import CatalogReader
from causal.shared.registry import ArtifactTypeRegistry
from causal.shared.validation import parse_strict

__all__ = [
    "COMPONENT", "NEEDS_USER_INPUT", "DesignDeps", "DesignError", "DesignRunResult",
    "DesignState", "GatewayProtocol", "HarnessBase",
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

_TASK_ROW: Final = (
    "INSERT INTO design.design_tasks (task_id, analysis_id, stage_run_id, design_revision,"
    " task_kind, scope, envelope_hash, prompt_version, model_profile_version, status,"
    " output_artifact_id, attempts) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    " ON CONFLICT (task_id) DO UPDATE SET status = EXCLUDED.status,"
    " output_artifact_id = EXCLUDED.output_artifact_id, attempts = EXCLUDED.attempts")
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


class HarnessBase:
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

