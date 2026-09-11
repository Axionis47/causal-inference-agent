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

from causal.design import askgate, contracts, diagnostics, semantics, validators
from causal.design import compile as compiler
from causal.design.capacity import CapacityRegistryV1
from causal.design.packs import TASK_KINDS, MethodPackRegistry, RequirementTemplateV1
from causal.shared import agenttask, events, persistence
from causal.shared import envelope as agent
from causal.shared.agenttask import GatewayProtocol
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactEnvelopeV1, ArtifactRef, HandoffManifestV1
from causal.shared.readers import CatalogReader
from causal.shared.registry import ArtifactTypeRegistry
from causal.shared.validation import parse_strict

__all__ = [
    "COMPONENT", "NEEDS_USER_INPUT", "DesignDeps", "DesignError", "DesignRunResult",
    "DesignState", "GatewayProtocol", "HarnessBase",
]

COMPONENT: Final = "design-harness"
VERSION: Final = "design-harness.v2"
NEEDS_USER_INPUT: Final = "needs_user_input"
ALLOWED_INTAKE: Final = frozenset({"usable", "partial"})
EVAL_STAGE: Final = ("EV-P2-001",)
EVAL_ASK: Final = ("EV-P2-006",)
EVAL_METHOD: Final = ("EV-P2-007",)
EVAL_APPROVAL: Final = ("EV-P2-008",)
EVAL_TASK: Final[dict[str, tuple[str, ...]]] = {
    "intent": ("EV-P2-002",), "semantic_batch": ("EV-P2-003",), "role_evidence": ("EV-P2-004",),
    "causal_context": ("EV-P2-005",), "role_ledger": ("EV-P2-005",),
    "method_design": ("EV-P2-007",)}
REGISTRY_VERSIONS: Final = {
    "artifact_types": "artifact-types.v1", "field_classes": "kaggle-field-classes.v1",
    "method_packs": MethodPackRegistry.registry_version,
    "requirements": "context-requirements.v1", "validators": validators.REGISTRY_VERSION,
    "capacity": "delivery-capacity.v1", "graph": VERSION, "schema": "design-tasks.v1"}
TASK_RESUME_NODES: Final = {
    "intent": "intent", "semantic_batch": "semantic", "role_evidence": "roles",
    "causal_context": "synthesis", "role_ledger": "synthesis",
    "method_design": "method_proposal"}
# The nodes in workflow order (PRD-002 §7); `handoff_open` closes the run after the outcome.
NODES: Final = (
    "entry", "selection", "manifest", "intent", "triage_node", "semantic", "measurement",
    "roles", "synthesis", "method_proposal", "method", "render",
    "capacity_node", "approval", "outcome_node")
# The parents the renderer and the approval binding name, in §12.3 / §22 order.
VIEW_PARENTS: Final = (
    "CausalContext", "MeasurementMap", "RoleLedger", "CompiledDesign")
APPROVED_KINDS: Final = ("DesignReviewBundle",)
OUTCOME_REFS: Final = {
    "compiled_design": "CompiledDesign", "diagnostic_report": "DiagnosticReport",
    "capacity_report": "CapacityReport", "review_bundle": "DesignReviewBundle",
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
    "SELECT requirement_id, scope_id, state, resolving_answer_artifact_id, resolving_fact_id"
    " FROM design.context_requirements"
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
    artifacts: dict[str, str]
    hashes: dict[str, str]
    card_ids: list[str]
    role_ids: list[str]
    open_requirement_ids: list[str]
    answer_ids: list[str]
    review_feedback_id: str
    corrections: dict[str, int]
    compiler_issues: list[dict[str, object]]
    clarification_round: int
    resume_node: str
    method_id: str
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
    error_code: str | None = None
    review_summary: dict[str, str] | None = None


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
    task_table: Mapping[str, compiler.TaskSpecV1]
    rules: tuple[validators.ValidationRuleV1, ...]
    capacity_registry: CapacityRegistryV1
    prompts_root: Path
    estimation_registry_path: Path | None = None
    tracer: Any = None


class HarnessBase:
    """Node bodies over one `DesignDeps`; every node is thin and replay-safe."""

    def __init__(self, deps: DesignDeps) -> None:
        self.deps = deps
        self.requirements = askgate.PsycopgRequirementStore(deps.conn)
        self.accepted_facts = askgate.PsycopgAcceptedFactStore(deps.conn)
        self._count = 0
        self._cache: dict[str, Any] = {}
        self._runner = agenttask.TaskRunner(
            gateway=deps.gateway, tasks=deps.task_table, evals=EVAL_TASK,
            tools=dict.fromkeys(TASK_KINDS, ()) | {
                "method_design": (diagnostics.DIAGNOSTIC_TOOL_ID,)},
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
        """Enter explicit system failure with one stable code and a raised blocker."""
        self._emit(state, "blocker.raised", EVAL_STAGE, severity=events.Severity.ERROR,
                   error_code=code, safe_dimensions={"blocked_operation": state.get("stage", ""),
                                                     "detail_codes": ",".join(codes)})
        return self._out(state, status="system_failure", error_code=code)

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
        answers = {f"ua:{found}": json.dumps(self._payload(found), ensure_ascii=False,
                                              sort_keys=True)
                   for found in state["answer_ids"]}
        return {str(row["evidence_id"]): str(row.get("value") or "") for row in items} | {
            f"{found}#/columns/{name}": json.dumps(
                {key: value for key, value in facts.items() if key in MEASURED_FACT_KEYS},
                ensure_ascii=False, sort_keys=True)
            for name, facts in columns.items()} | answers

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
                design_revision: int, state: DesignState,
                payloads: Sequence[Mapping[str, object]] = ()) -> None:
        """Only registered ids reach the gate, at one scope convention per column (D-100).

        Nothing in production mints a `requirement_id` or a `scope_id`: both arrive verbatim from
        a result's `missing_requirements`, and a column is named as either `column` or
        `table::column`, so one column opens twice. Fold that, and drop the invented ids.
        """
        admitted = []
        ctx = self._ctx(state)
        for row in raised:
            template = self.deps.templates.get(row.requirement_id)
            if template is None:
                continue
            kind = template.scope_kind
            scope_id = next((scope for payload in (*payloads, None)
                             if (scope := validators.canonical_requirement_scope(
                                 row, ctx, payload)) is not None), None)
            if scope_id is None:
                continue
            admitted.append(row.model_copy(update={"scope_kind": kind, "scope_id": scope_id}))
        self.requirements.upsert(admitted, analysis_id, design_revision)

    def _requirement_states(self, state: DesignState) -> dict[str, str]:
        """Persisted requirement states keyed by their exact registered id and scope."""
        rows = self.deps.conn.execute(
            _REQ_STATES, (state["analysis_id"], state["design_revision"])).fetchall()
        return {f"{row[0]}::{row[1]}": str(row[2]) for row in rows}

    def _accepted_facts(self, state: DesignState) -> dict[tuple[str, str], askgate.AcceptedFactV1]:
        """The only persisted values allowed to settle an exact requirement instance."""
        return self.accepted_facts.current(state["analysis_id"], state["design_revision"])

    def _prerequisite_context(self, state: DesignState) -> dict[str, object]:
        rows = self.deps.conn.execute(
            _REQ_STATES, (state["analysis_id"], state["design_revision"])).fetchall()
        facts = self._accepted_facts(state)
        settled = [{"requirement_id": str(row[0]), "scope_id": str(row[1]),
                    "state": str(row[2]), "accepted_fact_id": (
                        str(row[4]) if row[4] else None), "answer_evidence_id": (
                            f"ua:{row[3]}" if row[3] else None)}
                   for row in rows if str(row[2]) != "open"]
        answer_ids = {str(row["answer_evidence_id"]) for row in settled
                      if row["answer_evidence_id"]} | {
            evidence_id for fact in facts.values() if fact.source_kind == "user"
            for evidence_id in fact.evidence_ids if evidence_id.startswith("ua:")}
        availability = agenttask.evidence_availability(self._evidence(state)) if answer_ids else {}
        return {"settled_requirements": settled,
                "accepted_facts": [{
                    "accepted_fact_id": fact.accepted_fact_id,
                    "requirement_id": fact.requirement_id, "scope_id": fact.scope_id,
                    "value": fact.value, "value_schema": fact.value_schema,
                    "evidence_class": fact.evidence_class.value,
                    "support_relation": fact.relation,
                    "provenance": {"source_kind": fact.source_kind,
                        "origin_revision": fact.origin_revision,
                        "origin_reference_id": fact.origin_reference_id,
                        "origin_reference_hash": fact.origin_reference_hash,
                        "evidence_ids": list(fact.evidence_ids)}}
                    for fact in facts.values()],
                "accepted_answer_evidence_ids": sorted(
                    item for item in answer_ids if availability.get(item) == "evidenced"),
                "contract": "Accepted facts are authoritative for their exact requirement and "
                            "scope; reuse their values and do not ask again. accepted_fact_id "
                            "(af:) identifies a fact, not citable evidence. Provenance preserves "
                            "origin references and evidence IDs without granting citation "
                            "authority: cite only this task's reference_catalog.source_evidence_ids "
                            "with matching allowed_evidence text."}

    def _open_requirements(self, state: DesignState) -> tuple[agent.ContextRequirementV1, ...]:
        """Rebuild the open requirement rows from their §10.1 templates (checkpoint-safe)."""
        rows = self.deps.conn.execute(
            _OPEN_REQS, (state["analysis_id"], state["design_revision"])).fetchall()
        return tuple(agent.ContextRequirementV1(
            requirement_id=str(row[0]), scope_id=str(row[1]), decisions_blocked=(),
            registry_version="context-requirements.v1",
            attempted_evidence=tuple(
                parse_strict(agent.AttemptedEvidenceV1, item) for item in row[2]),
            **self.deps.templates[str(row[0])].model_dump(
                exclude={"requirement_id", "accepted_fact"}))
            for row in rows if str(row[0]) in self.deps.templates)

    def _ctx(self, state: DesignState, **over: Any) -> validators.ValidationContext:
        """Everything the walls may read; a validator never loads anything itself."""
        base: dict[str, Any] = {
            "manifest": self._model(
                state, "DesignContextManifest", contracts.DesignContextManifestV1),
            "dataset_id": state["dataset_id"],
            "rules": self.deps.rules, "packs": self.deps.packs, "templates": self.deps.templates,
            "parents": dict(state["hashes"]), "evidence_ids": frozenset(self._evidence(state)),
            "user_answer_evidence_ids": frozenset(f"ua:{a}" for a in state["answer_ids"]), "concept_ids": frozenset(row.concept_id for row in self._model(state, "MeasurementMap", semantics.MeasurementMapV1).concepts) if "MeasurementMap" in state["artifacts"] else frozenset(),
            "resolved_requirements": self._requirement_states(state), "intent": self._model(state, "DesignIntent", contracts.DesignIntentV1) if "DesignIntent" in state["artifacts"] else None, "measurement_map": self._model(state, "MeasurementMap", semantics.MeasurementMapV1) if "MeasurementMap" in state["artifacts"] else None,
            "relationship_ids": frozenset(edge.edge_id for edge in self._model(
                state, "CausalContext", semantics.CausalContextV1).edges)
                if "CausalContext" in state["artifacts"] else frozenset()}
        return validators.ValidationContext(**base | over)

    # -- the model-task loop (PRD-002 §16.1, §16.4) -----------------------

    def _review_feedback(self, state: DesignState) -> dict[str, Any] | None:
        """Prior review instructions are immutable task context, never study evidence."""
        found = state.get("review_feedback_id")
        if not found:
            return None
        envelope = self.deps.products.load_envelope(found)
        body = self._payload(found)
        decision = parse_strict(contracts.DesignApprovalDecisionV1, body)
        if (envelope.artifact_type != "DesignApprovalDecision"
                or envelope.analysis_id != state["analysis_id"]
                or envelope.content_hash != state["hashes"][found]
                or content_hash(body) != envelope.content_hash
                or decision.expected_revision != state["design_revision"] - 1
                or decision.decision is not contracts.ApprovalDecision.CHANGES_REQUESTED):
            raise DesignError("prior review feedback does not match its frozen decision",
                              "review_feedback_mismatch")
        return {"prior_decision": {"artifact_id": found, "content_hash": envelope.content_hash},
                "reviewed_design_revision": decision.expected_revision,
                "change_requests": list(decision.change_requests),
                "instruction": "Address these exact prior review requests in this revision. "
                "They are design-revision instructions, not new source evidence or accepted "
                "study facts. Preserve accepted facts and admitted source evidence. Substantiate "
                "revised bindings using the supplied evidence; if a factual resolution conflicts "
                "with the request, surface that conflict instead of silently replacing the fact."}

    def _run_task(self, state: DesignState, kind: str, model: type[BaseModel], *,
                  scope_kind: str, scope_ids: Sequence[str], parent_kinds: tuple[str, ...],
                  payload: Mapping[str, object], many: bool = False, commits: str | None = None,
                  ctx: validators.ValidationContext | None = None,
                  evidence_scope_ids: Sequence[str] | None = None,
                  precommit_admission: Callable[
                      [tuple[BaseModel, ...], agent.AgentTaskResultV1],
                      tuple[validators.ValidationIssueV1, ...]
                  ] | None = None,
                  ) -> tuple[tuple[Any, ArtifactEnvelopeV1], ...] | None:
        """One model task with the §16.4 correction loop; the loop itself is shared (D-065)."""
        resume_node = TASK_RESUME_NODES[kind]
        if self._open_requirements(state):
            state["stage"], state["resume_node"] = "gate_again", resume_node
            return None
        task_payload = dict(payload) | {
            "prerequisite_context": self._prerequisite_context(state)}
        if feedback := self._review_feedback(state):
            task_payload["review_feedback"] = feedback
        completed = self._runner.run(
            state, kind, model, scope_kind=scope_kind, scope_ids=scope_ids,
            parent_kinds=parent_kinds, payload=task_payload, many=many, commits=commits, ctx=ctx,
            evidence_scope_ids=evidence_scope_ids,
            precommit_admission=precommit_admission)
        if completed is not None:
            # Recording an unchanged supporting unknown needs no new model decision.
            # Keep factual resolutions and blocking questions on the existing gate/resume path.
            supporting = tuple(row for row in self._open_requirements(state)
                               if row.criticality is not agent.Criticality.BLOCKING)
            for decision in askgate.gate(
                    supporting, self._accepted_facts(state), state["clarification_round"] + 1,
                    self.deps.templates):
                if decision.route is askgate.GateRoute.RECORD_SENSITIVITY:
                    self.requirements.set_state(
                        state["analysis_id"], state["design_revision"], decision.requirement_id,
                        decision.scope_id, askgate.RequirementState.UNKNOWN_ACCEPTED, None)
            state["open_requirement_ids"] = sorted({
                row.requirement_id for row in self._open_requirements(state)})
        if completed is not None and self._open_requirements(state):
            state["stage"], state["resume_node"] = "gate_again", resume_node
            return None
        if completed is not None:
            state["resume_node"] = ""
        return completed

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
        self._emit(state, "retry.exhausted", EVAL_TASK[kind], severity=events.Severity.ERROR,
                   task_id=task_id, error_code=code)
        self._emit(state, "blocker.raised", EVAL_TASK[kind], severity=events.Severity.ERROR,
                   task_id=task_id, error_code="correction_exhausted",
                   safe_dimensions={"blocked_operation": kind, "validation_code": code})
        state["status"], state["error_code"] = "system_failure", "agent_output_invalid"

    # -- nodes ------------------------------------------------------------

    def _handoff_manifest(self, state: DesignState) -> HandoffManifestV1:
        """The intake handoff this design revision opens, rebuilt from the committed outcome."""
        found = self.deps.products.load_envelope(state["artifacts"]["IntakeOutcome"])
        payload = self._payload(found.artifact_id)
        return HandoffManifestV1(
            handoff_id=(f"ho:{state['analysis_id']}:{state['design_revision']}:"
                        f"{found.content_hash[:16]}"),
            schema_version="handoff.v1", analysis_id=state["analysis_id"],
            producing_stage_run_id=found.stage_run_id,
            receiving_stage_run_id=state["stage_run_id"], approval_ids=(),
            entries=(ArtifactRef(artifact_id=found.artifact_id,
                                 content_hash=found.content_hash),),
            originating_outcome=str(payload["status"]), registry_version="artifact-types.v1",
            compatibility_version="handoff.v1", receiver_validation_result=None,
            receiver_error_codes=(), created_at_utc=self.deps.clock(), accepted_at_utc=None)
