# Estimation state, injected dependencies, and the plumbing every estimation node shares
# (PRD-004 §19.1, §20.5, §21, §26.1). Amendment 1 makes this stage a plain sequential
# coordinator: no checkpointer, no StateGraph, no interrupt. Every upstream artifact is read
# as data — this module never imports `causal.design` or `causal.preparation`.

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Final, TypedDict, cast

import polars as pl
from psycopg import Connection

from causal.estimation import contracts as ec
from causal.estimation import engine, plancompile
from causal.estimation import walls as ew
from causal.estimation.packs import EstimationPackRegistry, EstimationPackV1
from causal.shared import events, frames, gateway, handoff, persistence
from causal.shared.contracts import ArtifactEnvelopeV1, ArtifactRef, HandoffManifestV1
from causal.shared.registry import ArtifactTypeRegistry
from causal.shared.validation import ValidationReport, ValidationRuleV1, parse_strict

COMPONENT, VERSION = "estimation-harness", "estimation-harness.v1"
BUILD_IDENTIFIER: Final = "causal-0.1.0"
# The §7 sequential flow: entry, plan and capacity, estimate, evidence, judgment, close.
PHASES: Final = ("entry", "plan", "estimate", "evidence", "judgment", "close")
# §20.6 evaluation surfaces: the entry/plan/capacity gate, the evidence fan-in, the frozen
# figure boundary, and the coordinator's own restart, conflict, and handoff legs.
EVAL_STAGE, EVAL_EVIDENCE = ("EV-P4-001",), ("EV-P4-007",)
EVAL_FIGURE, EVAL_CLOSE = ("EV-P4-009",), ("EV-P4-010",)
ERROR: Final = events.Severity.ERROR
CONFLICT, FAILED = "design_conflict", "failed"
PREPARED: Final = "prepared"
# §14.2: this stage approves no non-computed terminal status for any severity.
APPROVED_HANDLING: Final[frozenset[str]] = frozenset()
# D-091: §4 condition 7 alone admits an uncomputable PRD-003 pre-check. Every pack post-repair
# diagnostic id has a rigorous counterpart this stage computes and gates itself, so a rough
# prepared-stage pre-check that could not run costs nothing the diagnostic wall does not recover.
PREPARED_APPROVED_HANDLING: Final[frozenset[str]] = frozenset({"not_computable"})
# The estimator-input class of an approved role's prepared column (§4 condition 6).
IDENTIFIER_ROLES: Final = ("unit_identifier", "cluster")
CATEGORICAL_ROLES: Final = ("treatment", "stratum", "group", "assignment_variable")
TIME_ROLES: Final = ("time", "adoption_time")
# The roles a pack schema may read as binary; a two-valued numeric column is what it means.
BINARY_ROLES: Final = ("treatment", "missingness_indicator")
# `estimation.estimation_runs.state` is a closed vocabulary; the outcome artifact carries the
# §5.2 status, so a conflict, an invalidation, and a not-estimable run are still completed.
ROW_STATE: Final[dict[str, str]] = {
    "complete": "completed", "not_estimable": "completed", "invalidated": "completed",
    CONFLICT: "completed", "failed_observability": "failed_observability"}
RUN_ROW: Final = (
    "INSERT INTO estimation.estimation_runs (stage_run_id, analysis_id, graph_thread_id, state,"
    " preparation_stage_run_id, estimation_revision, created_at, updated_at)"
    " VALUES (%s, %s, %s, 'running', %s, %s, %s, %s) ON CONFLICT (stage_run_id) DO UPDATE SET"
    " state = 'running', updated_at = EXCLUDED.updated_at")
RUN_STATE: Final = (
    "UPDATE estimation.estimation_runs SET state = %s, outcome_artifact_id = %s,"
    " context_manifest_id = %s, plan_artifact_id = %s, row_set_hash = %s, overall_ceiling = %s,"
    " conflict_code = %s, error_code = %s, updated_at = %s WHERE stage_run_id = %s")
REF_ROW: Final = (
    "INSERT INTO estimation.estimation_artifact_refs (stage_run_id, kind, artifact_id,"
    " content_hash, schema_version) VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING")
MASK_ROW: Final = (
    "INSERT INTO estimation.contribution_mask_index (mask_artifact_id, stage_run_id,"
    " calculation_id, mask_rule_id, parent_row_set_hash, included_rows, noncontributing_rows,"
    " created_at) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING")


# The §19.1 state allowlist: ids, hashes, statuses, codes, counts, and phase only. Prepared
# rows, predictions, weights, replicates, and figure data are object payloads, never state.
EstimationState = TypedDict("EstimationState", {
    "analysis_id": str, "stage_run_id": str, "thread_id": str, "estimation_revision": int,
    "upstream": dict[str, str], "phase": str, "wall": int, "status": str, "error_code": str,
    "conflict_code": str, "artifacts": dict[str, str], "hashes": dict[str, str],
    "row_set_hash": str, "seed": int, "capacity_status": str, "overall_ceiling": str,
    "mask_ids": list[str], "assignment_ids": list[str], "diagnostic_ids": list[str],
    "sensitivity_ids": list[str], "figure_ids": list[str], "task_ids": list[str],
    "counts": dict[str, int], "handoff_id": str}, total=False)


@dataclass(frozen=True)
class EstimationRunResult:
    # What one `causal run` step of an estimation revision returns (PRD-004 §5.2).

    status: str
    analysis_id: str
    stage_run_id: str
    thread_id: str
    estimation_revision: int
    outcome_artifact_id: str | None = None
    conflict_code: str | None = None
    overall_ceiling: str | None = None
    row_set_hash: str | None = None
    handoff_id: str | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class EstimationDeps:
    # Everything the stage needs; nothing here is discovered at run time.

    conn: Connection[Any]
    products: persistence.ProductStore
    objects: persistence.ObjectStore
    committer: persistence.ArtifactCommitter
    registry: ArtifactTypeRegistry
    emitter: events.EventEmitter
    clock: Callable[[], datetime]
    packs: EstimationPackRegistry
    rules: tuple[ValidationRuleV1, ...]
    frames: frames.FrameStore
    registries: Path
    repo_root: Path
    # One registered adapter factory per method id, called with the run's contribution mask;
    # §19 forbids a generic estimator tool and any runtime adapter search.
    adapters: Mapping[str, Callable[[ArtifactRef], engine.EstimatorAdapter]] = field(
        default_factory=dict)
    # The §6.4 distributions whose exact versions the environment manifest records.
    packages: tuple[str, ...] = ("numpy", "polars", "scipy", "pyfixest", "scikit-learn")
    # §19: the ONE model receiver in this stage is the §16.2 claim-review call (T-025).
    gateway: gateway.VertexGateway | None = None


def role_columns(ledger: Mapping[str, Any], columns: Sequence[str]) -> dict[str, str]:
    # The approved role ledger read as data: one prepared column per approved role.
    named = [(str(row["role"]), [str(name) for name in row.get("column_refs") or ()])
             for row in ledger.get("claims") or ()]
    return {role: found[0] for role, refs in named
            if (found := [name for name in refs if name in columns])}


def input_types(frame: pl.DataFrame, roles: Mapping[str, str]) -> dict[str, str]:
    # What the estimator would receive for each approved role, in the pack's schema vocabulary.
    return {role: "identifier" if role in IDENTIFIER_ROLES else "time" if role in TIME_ROLES
            else "binary" if role in BINARY_ROLES and frame[column].n_unique() <= 2
            else "categorical"
            if role in CATEGORICAL_ROLES or not frame.schema[column].is_numeric() else "numeric"
            for role, column in roles.items() if column in frame.columns}


def build_handoff(analysis_id: str, outcome: ArtifactEnvelopeV1, receiving_stage_run_id: str,
                  entries: Sequence[Mapping[str, Any]], originating: str,
                  now: datetime) -> HandoffManifestV1:
    # One cross-stage manifest, rebuilt from committed payloads. The id is the shared
    # `ho:{analysis_id}:{outcome_hash16}`, recomputed here and never imported (D-037).
    return HandoffManifestV1(
        handoff_id=f"ho:{analysis_id}:{outcome.content_hash[:16]}", schema_version="handoff.v1",
        analysis_id=analysis_id, producing_stage_run_id=outcome.stage_run_id,
        receiving_stage_run_id=receiving_stage_run_id,
        entries=tuple(ArtifactRef.model_validate(dict(ref)) for ref in entries),
        originating_outcome=originating, approval_ids=(), registry_version="artifact-types.v1",
        compatibility_version="handoff.v1", receiver_validation_result=None,
        receiver_error_codes=(), created_at_utc=now, accepted_at_utc=None)


class HarnessBase:
    # Events, flush-gated commits, committed reads, walls, conflict routing, and the run-row
    # lifecycle every estimation node shares.

    def __init__(self, deps: EstimationDeps) -> None:
        self.deps = deps
        self._count = 0
        self._cache: dict[str, Any] = {}
        # Everything this run has frozen so far, in the exact shape the fifteen walls read.
        self.frozen: dict[str, Any] = {}
        self.harvest: dict[str, ec.ValueMap] = {}
        # Evidence id to its committed artifact id: the closed §16.2 citation allowlist.
        self.evidence: dict[str, str] = {}
        self.bundles: dict[str, ArtifactRef] = {}
        # The §10.2 fold assignments this run committed; empty for a method that cross-fits none.
        self.dealt: tuple[ArtifactRef, ...] = ()
        self.denominators: dict[str, int] = {}
        self.claim_status: ec.JudgmentStatus = "reportable"
        self.adapter: engine.MethodAdapter | None = None
        self.view_cache: pl.DataFrame | None = None

    # -- events, commits, and committed reads ------------------------------

    def event(self, state: EstimationState, name: str, evals: tuple[str, ...],
              **over: Any) -> events.OperationalEventV1:
        # One §20.5 event with a deterministic id and this stage's own identities.
        self._count += 1
        return events.build_event(
            occurred_at_utc=self.deps.clock(), event_name=name, analysis_id=state["analysis_id"],
            event_id=f"evt:{state['stage_run_id']}:{self._count}",
            stage=events.Stage.ESTIMATION, stage_run_id=state["stage_run_id"],
            graph_thread_id=state["thread_id"], component_id=COMPONENT,
            component_version=VERSION, required_eval_ids=evals,
            versions={"registry": "artifact-types.v1", "prompt": VERSION}, **over)

    def emit(self, state: EstimationState, name: str, evals: tuple[str, ...],
             **over: Any) -> None:
        self.deps.emitter.emit(self.event(state, name, evals, **over))

    def commit(self, state: EstimationState, kind: str, payload: Mapping[str, object],
               parents: tuple[ArtifactEnvelopeV1, ...]) -> ArtifactEnvelopeV1:
        # Build and commit one immutable artifact through the flush-gated committer (SC §8.2):
        # the commit completes only after the trace span it names is acknowledged.
        body = dict(payload)
        built = persistence.build_envelope(
            self.deps.registry, kind, body, analysis_id=state["analysis_id"],
            stage_run_id=state["stage_run_id"], producer_version=VERSION, parents=parents,
            created_at_utc=self.deps.clock(), producer_component=COMPONENT)
        committed = self.deps.committer.commit(built, body, self.event(
            state, "artifact.committed", EVAL_STAGE, status="committed", artifact_refs=(
                ArtifactRef(artifact_id=built.artifact_id, content_hash=built.content_hash),)))
        state["artifacts"][kind] = committed.artifact_id
        state["hashes"][committed.artifact_id] = committed.content_hash
        self.deps.conn.execute(REF_ROW, (state["stage_run_id"], kind, committed.artifact_id,
                                         committed.content_hash, committed.schema_version))
        return committed

    def payload(self, artifact_id: str) -> dict[str, Any]:
        # One committed payload, loaded for the node that needs it and released after.
        if artifact_id not in self._cache:
            locator = self.deps.products.load_envelope(artifact_id).payload_locator
            self._cache[artifact_id] = json.loads(self.deps.objects.get(locator))
        return cast(dict[str, Any], self._cache[artifact_id])

    def ref(self, state: EstimationState, kind: str) -> ArtifactRef:
        found = state["artifacts"][kind]
        return ArtifactRef(artifact_id=found, content_hash=state["hashes"][found])

    def parents(self, state: EstimationState, *kinds: str) -> tuple[ArtifactEnvelopeV1, ...]:
        return tuple(self.deps.products.load_envelope(state["artifacts"][k]) for k in kinds)

    def out(self, state: EstimationState, **over: Any) -> dict[str, Any]:
        # Every node writes the whole allowlisted snapshot back, so a replay sees one state.
        return dict(state) | dict(over)

    def fail(self, state: EstimationState, code: str, status: str = FAILED) -> dict[str, Any]:
        # Enter a terminal status with one stable code and a raised blocker (§20.5).
        self.emit(state, "blocker.raised", EVAL_STAGE, severity=ERROR, error_code=code,
                  safe_dimensions={"blocked_operation": state.get("phase", "")})
        return self.out(state, status=status, error_code=code)

    def put(self, state: EstimationState, kind: str, payload: ec._Payload,
            *parents: str) -> ArtifactRef:
        # Commit one estimation payload under its registered parents and hold its reference.
        self.commit(state, kind, payload.canonical_payload(), self.parents(state, *parents))
        return self.ref(state, kind)

    def put_object(self, payload: Mapping[str, object]) -> ec.ObjectRefV1:
        # One restricted object: it reaches the object store and no envelope, event, or payload.
        locator = self.deps.frames.put_object(payload)
        return ec.ObjectRefV1(object_locator=locator, content_hash=locator.split("/")[-1])

    def record_fit(self, state: EstimationState, plan: ec.EstimationPlanV1,
                   fitted: engine.AdapterResult) -> dict[str, ec.ValueMap]:
        # Hold what one adapter call produced. A cross-fitted adapter hands its own run back, so
        # the coordinator commits what was dealt — the §10.2 fold assignment beside its restricted
        # mapping and prediction objects — and holds the receipts wall 6 measures. A fit that
        # dealt no folds commits nothing here (§10.2, §19.1).
        run = fitted.fit
        if isinstance(run, engine.CrossFitFit):
            held = self.ref(state, "EstimationPlan")
            mapping = self.put_object(run.mapping_payload())
            self.put_object(run.prediction_payload())
            dealt = run.assignment(plan, mapping, plan_ref=held, parents=(held,))
            self.dealt += (self.put(state, "CrossFitAssignment", dealt, "EstimationPlan"),)
            state.setdefault("assignment_ids", []).append(self.dealt[-1].artifact_id)
            self.frozen |= {"assignments": (dealt,), "fold_fit_counts": run.receipts}
        return dict(fitted.harvest)

    def refs_of(self, state: EstimationState) -> dict[str, ArtifactRef]:
        return {name: ArtifactRef(artifact_id=found, content_hash=state["hashes"][found])
                for name, found in self.evidence.items()}

    # -- walls, conflicts, and the frozen frame ----------------------------

    def context(self, state: EstimationState, **over: Any) -> ew.WallContext:
        # One frozen wall context per node; a wall never loads or recomputes anything itself.
        book = self.manifest(state) if "EstimationContextManifest" in state["artifacts"] else None
        found: dict[str, Any] = {
            "rules": self.deps.rules, "handoff_accepted": True, "manifest": book,
            "plan": self.plan(state) if "EstimationPlan" in state["artifacts"] else None,
            "pack": None if book is None else self.pack(book),
            "approved_handling": APPROVED_HANDLING}
        return ew.WallContext(**(found | self.frozen | over))

    def gate(self, state: EstimationState, highest: int, status: str = FAILED,
             **over: Any) -> dict[str, Any] | None:
        # Walls 1..`highest` over the frozen context; the first failure is that node's exit.
        report = self.wall(state, highest, self.context(state, **over))
        return None if report.passed else self.fail(state, report.issues[0].code, status)

    def wall(self, state: EstimationState, highest: int, ctx: ew.WallContext) -> ValidationReport:
        # Walls 1..`highest` in order (§18); the first failure stops the run and no later wall
        # ever waives it. A failure is reported once, with every offending code.
        report = ew.validate(highest, replace(ctx, rules=ctx.rules or self.deps.rules))
        state["wall"] = report.wall
        if not report.passed:
            self.emit(state, "artifact.validation_failed", EVAL_STAGE, severity=ERROR,
                      error_code=report.issues[0].code,
                      safe_dimensions={"wall": report.wall, "detail_codes": ",".join(sorted(
                          {found for issue in report.issues for found in issue.artifact_ids}))})
        return report

    def conflict(self, state: EstimationState,
                 draft: plancompile.DesignConflictDraftV1) -> dict[str, Any]:
        # §19: a required semantic, contrast, or capacity change becomes a DesignConflict and
        # returns to PRD-002. PRD-004 never interrupts the user and resolves nothing itself.
        held = self.ref(state, "EstimationContextManifest")
        body = draft.model_dump(mode="json") | {
            "schema_version": "design-conflict.v1", "context_manifest": held.model_dump(),
            "conflict_id": f"dc:{state['stage_run_id']}:{draft.conflict_code}",
            "evidence_artifact_ids": [held.artifact_id], "evidence": [held.model_dump()],
            "preparation_revision": state["estimation_revision"]}
        self.commit(state, "DesignConflict", body,
                    self.parents(state, "EstimationContextManifest"))
        return self.out(state, status=CONFLICT, conflict_code=draft.conflict_code)

    def accept_handoff(self, state: EstimationState, manifest: HandoffManifestV1,
                       outcome: str) -> bool:
        # T-006: load first, then check — a recorded handoff is never re-recorded, so a rerun
        # replays the same acceptance instead of raising `duplicate_handoff` (D-037).
        deps, store = self.deps, handoff.HandoffStore(self.deps.conn)
        try:
            recorded = store.load(manifest.handoff_id)
        except persistence.PersistenceError:
            gate = handoff.HandoffGate(deps.objects, deps.products, store, deps.registry,
                                       deps.emitter)
            return gate.accept(manifest, COMPONENT, frozenset({outcome}),
                               lambda verdict, codes: self.event(
                                   state, f"handoff.{verdict}", EVAL_STAGE, status=verdict,
                                   error_code=codes[0] if codes else None)).accepted
        return (recorded.receiver_validation_result == "accepted"
                and recorded.entries == manifest.entries)

    def frame(self, artifact_id: str) -> pl.DataFrame:
        # The frozen prepared frame, reopened under its own recorded dtypes through the shared
        # frame store. The payload is read as data; PRD-003's models are never imported.
        body = self.payload(artifact_id)
        columns = tuple(frames.FrameColumnV1(column_name=str(row["column_name"]),
                                             dtype=str(row["dtype"]).split("(")[0])
                        for row in body["columns"])
        return self.deps.frames.read_frame(str(body["frame_object"]["object_locator"]), columns)

    def prepared_frame(self, state: EstimationState) -> pl.DataFrame:
        bundle = self.payload(state["upstream"]["prepared_bundle"])
        return self.frame(str(bundle["prepared_frame"]["artifact_id"]))

    def view(self, state: EstimationState,
             manifest: ec.EstimationContextManifestV1) -> pl.DataFrame:
        # §19.1: the adapter reads the declared estimator-input view, not the prepared frame.
        if self.view_cache is None:
            self.view_cache = engine.estimator_view(self.prepared_frame(state),
                                                    manifest.role_columns)
        return self.view_cache

    def entry_inputs(self, state: EstimationState, outcome: ArtifactEnvelopeV1
                     ) -> tuple[plancompile.EntryInputs, tuple[ArtifactRef, ...], bool]:
        # The PRD-003 handoff rebuilt and accepted, then its four §4 entry payloads, the PRD-003
        # stabilization record, and the prepared frame's own roles and dtypes — all read as data.
        body = self.payload(outcome.artifact_id)
        bundle = self.payload(str(body["prepared_bundle"]["artifact_id"]))
        held = (body["prepared_bundle"], bundle["experiment_design"],
                bundle["runnable_frame_contract"], bundle["capacity_check"])
        entries = tuple(ArtifactRef.model_validate(dict(ref)) for ref in held)
        accepted = self.accept_handoff(state, build_handoff(
            state["analysis_id"], outcome, state["stage_run_id"], held, str(body["status"]),
            self.deps.clock()), PREPARED)
        design, record = self.payload(entries[1].artifact_id), self.payload(
            str(bundle["stabilization_record"]["artifact_id"]))
        frame = self.frame(str(bundle["prepared_frame"]["artifact_id"]))
        roles = role_columns(self.payload(str(design["role_ledger"]["artifact_id"])),
                             frame.columns)
        found = self.deps.products.find_artifact_hash
        return plancompile.EntryInputs(
            outcome=body, bundle=bundle, design=design, record=record,
            contract=self.payload(entries[2].artifact_id),
            capacity=self.payload(entries[3].artifact_id),
            declared=dict(zip(plancompile.ENTRY_KEYS, entries, strict=True)),
            committed={key: None if (digest := found(ref.artifact_id)) is None else ArtifactRef(
                artifact_id=ref.artifact_id, content_hash=digest)
                for key, ref in zip(plancompile.ENTRY_KEYS, entries, strict=True)},
            estimator_input_types=input_types(frame, roles), role_columns=roles,
            # The prepared-stage reports the PRD-003 bundle carries are the §4 truth; the
            # stabilized-stage generics in the record stand only where the bundle is silent.
            postrepair_statuses={str(row["diagnostic_id"]): str(row["status"]) for row
                                 in record.get("post_stabilization_diagnostics") or ()}
            | {str(row["diagnostic_id"]): str(row["status"]) for row
               in bundle.get("postrepair_diagnostics") or ()}
            ), entries, accepted

    # -- committed estimation payloads and the run row ---------------------

    def manifest(self, state: EstimationState) -> ec.EstimationContextManifestV1:
        return parse_strict(ec.EstimationContextManifestV1,
                            self.payload(state["artifacts"]["EstimationContextManifest"]))

    def plan(self, state: EstimationState) -> ec.EstimationPlanV1:
        return parse_strict(ec.EstimationPlanV1, self.payload(state["artifacts"]["EstimationPlan"]))

    def pack(self, manifest: ec.EstimationContextManifestV1) -> EstimationPackV1:
        return self.deps.packs.get(manifest.method_id, manifest.method_pack_version)

    def environment(self, plan: ec.EstimationPlanV1) -> ec.NumericalEnvironmentManifestV1:
        return engine.numerical_environment(plan, self.deps.packages,
                                            build_identifier=BUILD_IDENTIFIER)

    def evidence_base(self, state: EstimationState, plan: ec.EstimationPlanV1,
                      denominators: Mapping[str, int],
                      mask_hash: str | None = None) -> dict[str, Any]:
        # The §14.1 lineage every diagnostic and sensitivity result repeats: plan, primary
        # result, mask hash, denominators, and the exact numerical environment.
        return {"parents": (self.ref(state, "EstimationPlan"),
                            self.ref(state, "PrimaryAnalysisResult")),
                "versions": dict(plan.versions), "plan": self.ref(state, "EstimationPlan"),
                "primary_result": self.ref(state, "PrimaryAnalysisResult"),
                "denominators": dict(denominators), "contribution_mask_hash": mask_hash,
                "numerical_environment": self.ref(state, "NumericalEnvironmentManifest")}

    def index_mask(self, state: EstimationState, envelope: ArtifactEnvelopeV1,
                   mask: ec.AnalysisContributionMaskV1) -> None:
        # The §21 contribution-mask index: identities, counts, and the frozen row set only.
        self.deps.conn.execute(MASK_ROW, (
            envelope.artifact_id, state["stage_run_id"], mask.calculation_id, mask.mask_rule_id,
            mask.parent_row_set_hash, mask.included_counts.get("row", 0),
            mask.noncontributing_counts.get("row", 0), self.deps.clock()))
        state.setdefault("mask_ids", []).append(envelope.artifact_id)

    def open_run(self, state: EstimationState, preparation_stage_run_id: str) -> None:
        # SC §4: the stage run and its estimation row exist before any artifact is committed.
        # A rerun is artifact replay (D-035): a NEW stage_run_id at the next revision.
        stage_run_id, now = state["stage_run_id"], self.deps.clock()
        try:
            self.deps.products.get_stage_run_state(stage_run_id)
        except persistence.PersistenceError:
            self.deps.products.create_stage_run(stage_run_id, state["analysis_id"], "estimation")
            self.deps.products.transition_stage_run(stage_run_id, "tracing_preflight")
            self.deps.products.transition_stage_run(stage_run_id, "running")
        self.deps.conn.execute(RUN_ROW, (
            stage_run_id, state["analysis_id"], state["thread_id"], preparation_stage_run_id,
            state["estimation_revision"], now, now))

    def close_run(self, state: EstimationState) -> EstimationRunResult:
        # SC §4: the run row reaches a terminal state on every exit path, success or not.
        artifacts, status = state.get("artifacts", {}), state.get("status") or FAILED
        outcome = artifacts.get("EstimationOutcome")
        self.deps.conn.execute(RUN_STATE, (
            ROW_STATE.get(status, FAILED), outcome, artifacts.get("EstimationContextManifest"),
            artifacts.get("EstimationPlan"), state.get("row_set_hash"),
            state.get("overall_ceiling"), state.get("conflict_code"), state.get("error_code"),
            self.deps.clock(), state["stage_run_id"]))
        return EstimationRunResult(
            status=status, analysis_id=state["analysis_id"],
            stage_run_id=state["stage_run_id"], thread_id=state["thread_id"],
            estimation_revision=state["estimation_revision"], outcome_artifact_id=outcome,
            conflict_code=state.get("conflict_code"), overall_ceiling=state.get("overall_ceiling"),
            row_set_hash=state.get("row_set_hash"), handoff_id=state.get("handoff_id"),
            error_code=state.get("error_code") or None)


def new_state(analysis_id: str, stage_run_id: str, revision: int,
              upstream: Mapping[str, str]) -> EstimationState:
    # §19.1: PRD-004 opens a new graph thread and inherits approved artifact references only —
    # never PRD-003's thread, messages, scratch context, tool results, or model memory.
    return EstimationState(
        analysis_id=analysis_id, stage_run_id=stage_run_id, thread_id=f"et:{analysis_id}:{revision}",
        estimation_revision=revision, upstream=dict(upstream), phase=PHASES[0], wall=0,
        artifacts={}, hashes={}, mask_ids=[], assignment_ids=[], diagnostic_ids=[],
        sensitivity_ids=[], figure_ids=[], task_ids=[], counts={})
