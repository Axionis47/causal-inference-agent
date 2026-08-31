"""Design-harness state, dependencies, and shared node helpers (T-013 §1.5 split, D-055)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, TypedDict, cast

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg import Connection
from psycopg.types.json import Json
from pydantic import BaseModel

from causal.design import askgate, contracts, validators
from causal.design import compile as compiler
from causal.design.capacity import CapacityRegistryV1
from causal.design.packs import MethodPackRegistry, RequirementTemplateV1, ToolRegistry
from causal.shared import agenttask, events, persistence
from causal.shared import envelope as agent
from causal.shared.agenttask import GatewayProtocol
from causal.shared.contracts import ArtifactEnvelopeV1, ArtifactRef, HandoffManifestV1
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

# The bounded per-column profile facts a model task may see (PRD-002 §15). The harness
# measured these; a card slot citing one is a `measured_observation`, not an inference.
MEASURED_FACT_KEYS: Final = ("dtype", "cardinality", "null_count", "null_rate", "all_null",
                             "constant", "numeric", "hypotheses")
GRAIN_KEYS: Final = ("row_count", "column_count", "duplicate_row_count", "unique_single_columns")

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
        self._runner = agenttask.TaskRunner(
            gateway=deps.gateway, tasks=deps.task_table, evals=EVAL_TASK,
            tools=deps.tool_registry.recipient_map(), requirements=sorted(deps.templates),
            prompts_root=deps.prompts_root,
            envelope=compiler.build_task_envelope, prompt=compiler.render_prompt,
            validate=validators.validate_result, context=self._ctx, evidence=self._evidence,
            manifest=lambda state: self._ref(state, "DesignContextManifest"),
            parents=self._parents, commit=self._commit, emit=self._emit,
            record=self._task_row, exhausted=self._exhausted,
            upsert=self._upsert)

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

    def _evidence(self, state: DesignState) -> dict[str, str]:
        """The allowlisted evidence, id to the text it carries: intake sources, then measurements.

        The bundle holds a Kaggle description or data dictionary that answers many registered
        requirements outright. Returning ids alone left every worker reporting the document as
        `not_offered`, so SC §6.2 condition 2 held trivially and the gate asked the user for
        facts already committed to this analysis (D-102). The committed profile is the same
        loss: the harness measured every column and asked the model what the column meant while
        showing it the column's name (D-103).
        """
        outcome = self._payload(state["artifacts"]["IntakeOutcome"])
        bundle = outcome.get("evidence_bundle_artifact_id")
        items = self._payload(str(bundle))["items"] if bundle else []
        found, columns = self._profile(state)
        return {str(row["evidence_id"]): str(row.get("value") or "") for row in items} | {
            f"{found}#/columns/{name}": json.dumps(
                {key: value for key, value in facts.items() if key in MEASURED_FACT_KEYS},
                ensure_ascii=False, sort_keys=True)
            for name, facts in columns.items()}

    def _profile(self, state: DesignState) -> tuple[str, dict[str, Any]]:
        """The committed profile of the selected table: its artifact id and its column facts."""
        table = self._model(
            state, "DesignContextManifest", contracts.DesignContextManifestV1).selected_table
        for found in self._payload(state["artifacts"]["IntakeOutcome"])["table_profile_artifact_ids"]:
            payload = self._payload(str(found))
            if payload.get("logical_name") == table:
                return str(found), dict(payload.get("columns") or {})
        return "", {}

    def _grain(self, state: DesignState) -> dict[str, Any]:
        """Table-level measurements: what the harness counted, for grain and unit identity."""
        found, _ = self._profile(state)
        payload = self._payload(found) if found else {}
        return {key: payload.get(key) for key in GRAIN_KEYS}

    def _upsert(self, raised: Sequence[agent.ContextRequirementV1], analysis_id: str,
                design_revision: int) -> None:
        """Only registered ids reach the gate, at one scope convention per column (D-100).

        Nothing in production mints a `requirement_id` or a `scope_id`: both arrive verbatim from
        a result's `missing_requirements`, and a column is named as either `column` or
        `table::column`, so one column opens twice. Fold that, and drop the invented ids.
        """
        self.requirements.upsert([
            row if row.scope_kind is not agent.RequirementScopeKind.COLUMN
            else row.model_copy(update={"scope_id": row.scope_id.rsplit("::", 1)[-1]})
            for row in raised if row.requirement_id in self.deps.templates],
            analysis_id, design_revision)

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
            "parents": dict(state["hashes"]), "evidence_ids": frozenset(self._evidence(state)),
            "user_answer_evidence_ids": frozenset(f"ua:{a}" for a in state["answer_ids"]),
            "resolved_requirements": self._requirement_states(state)}
        return validators.ValidationContext(**base | over)

    # -- the model-task loop (PRD-002 §16.1, §16.4) -----------------------

    def _run_task(self, state: DesignState, kind: str, model: type[BaseModel], *,
                  scope_kind: str, scope_ids: Sequence[str], parent_kinds: tuple[str, ...],
                  payload: Mapping[str, object], many: bool = False, commits: str | None = None,
                  ctx: validators.ValidationContext | None = None,
                  ) -> tuple[tuple[Any, ArtifactEnvelopeV1], ...] | None:
        """One model task with the §16.4 correction loop; the loop itself is shared (D-065)."""
        return self._runner.run(
            state, kind, model, scope_kind=scope_kind, scope_ids=scope_ids,
            parent_kinds=parent_kinds, payload=payload, many=many, commits=commits, ctx=ctx)

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

