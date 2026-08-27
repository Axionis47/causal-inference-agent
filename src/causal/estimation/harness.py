# Estimation state, injected dependencies, and the plumbing every estimation node shares
# (PRD-004 §19.1, §20.5, §21, §26.1). Amendment 1 makes this stage a plain sequential
# coordinator: no checkpointer, no StateGraph, no interrupt. Every upstream artifact is read
# as data — this module never imports `causal.design` or `causal.preparation`.

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Final, TypedDict, cast

import polars as pl
from psycopg import Connection

from causal.estimation import contracts as ec
from causal.estimation import engine
from causal.estimation import walls as ew
from causal.estimation.packs import EstimationPackRegistry, EstimationPackV1
from causal.estimation.plancompile import DesignConflictDraftV1
from causal.shared import events, frames, gateway, persistence
from causal.shared.contracts import ArtifactEnvelopeV1, ArtifactRef
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
    # One registered adapter per method id; §19 forbids a generic estimator tool.
    adapters: Mapping[str, engine.EstimatorAdapter] = field(default_factory=dict)
    # The §6.4 distributions whose exact versions the environment manifest records.
    packages: tuple[str, ...] = ("numpy", "polars", "scipy", "pyfixest", "scikit-learn")
    # §19: the ONE model receiver in this stage is the §16.2 claim-review call (T-025).
    gateway: gateway.VertexGateway | None = None


class HarnessBase:
    # Events, flush-gated commits, committed reads, walls, conflict routing, and the run-row
    # lifecycle every estimation node shares.

    def __init__(self, deps: EstimationDeps) -> None:
        self.deps = deps
        self._count = 0
        self._cache: dict[str, Any] = {}

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
            created_at_utc=self.deps.clock())
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

    # -- walls, conflicts, and the frozen frame ----------------------------

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
                 draft: DesignConflictDraftV1) -> dict[str, Any]:
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
        return engine.estimator_view(self.prepared_frame(state), manifest.role_columns)

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
