"""Preparation state, dependencies, and the plumbing every node shares (PRD-003 §7, §17.7)."""

# T-019 §1.3 leaves the coordinator two modules: the state allowlist, the injected dependencies,
# and `HarnessBase` here; the node bodies, the sequential caller, and the §20 handoff in `nodes`.
# Amendment 2 removed every model call, so this scope holds no task table, gateway, or
# checkpointer, and a rerun replays artifacts rather than resuming a checkpoint.

from __future__ import annotations

import io
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Final, TypedDict, cast

import polars as pl
from psycopg import Connection

from causal.preparation import contracts as pc
from causal.preparation import diagnostics as diag
from causal.preparation import entry, impact, operations, plancompile, stabilize
from causal.preparation import plans as pp
from causal.preparation import validators as walls
from causal.shared import events, frames, persistence
from causal.shared.contracts import ArtifactEnvelopeV1, ArtifactRef, HandoffManifestV1
from causal.shared.registry import ArtifactTypeRegistry, RegistryError
from causal.shared.validation import ValidationReport, parse_strict

__all__ = [
    "COMPONENT", "NODES", "REGISTRY_VERSIONS", "VERSION", "HarnessBase", "PreparationDeps",
    "PreparationRunResult", "PreparationState", "build_handoff"]

COMPONENT, VERSION = "preparation-harness", "preparation-harness.v1"
PARSER_PROFILE, UNRESOLVED = "polars-csv.v1", "unresolved_conflict"
ENTRY_WALL, ROWS_WALL, FREEZE_WALL, PLAN_WALL, EXEC_WALL, FINAL_WALL = 1, 2, 3, 4, 5, 6
EVAL_STAGE, EVAL_ROWS = ("EV-P3-001",), ("EV-P3-002",)
EVAL_EXEC, EVAL_FINAL = ("EV-P3-005",), ("EV-P3-007",)
ERROR: Final = events.Severity.ERROR
# The §7 flow, lite: entry + manifest, rows + freeze, plan, mutation, close.
NODES: Final = ("entry_node", "stabilize_node", "plan_node", "execute_node", "outcome_node")
REGISTRY_VERSIONS: Final[dict[str, str]] = {
    "artifact_types": "artifact-types.v1", "method_packs": "method-packs.v1",
    "preparation_packs": "method-pack-preparation.v1", "operations": "repair-operations.v1",
    "diagnostics": "preparation-diagnostics.v1", "parser": PARSER_PROFILE,
    "schema": "preparation-plan.v1", "validators": walls.REGISTRY_VERSION}
APPROVED_HANDLING: Final = frozenset({pc.DiagnosticStatus.NOT_COMPUTABLE.value})
STABILIZED_DIAGNOSTICS: Final = ("row_disposition_reconciliation", "key_uniqueness_grain")
PREPARED_DIAGNOSTICS: Final = ("schema_type_validation", "missingness_before_after",
                               "row_set_invariance", "changed_cells_by_operation_column",
                               "contract_completeness")
PLAN_ROW: Final = (
    "INSERT INTO preparation.plan_items (plan_artifact_id, plan_item_id, stage_run_id, phase,"
    " state, created_at, updated_at) VALUES (%s, %s, %s, %s, %s, %s, %s)"
    " ON CONFLICT (plan_artifact_id, plan_item_id) DO UPDATE SET state = EXCLUDED.state,"
    " updated_at = EXCLUDED.updated_at")
# `preparation.preparation_runs.state` is a closed vocabulary; the outcome artifact carries the
# §5.2 status, so a conflict or a not-runnable frame is still a completed run.
ROW_STATE: Final[dict[str, str]] = {
    "prepared": "completed", "design_conflict": "completed", "not_runnable": "completed",
    "failed_observability": "failed_observability"}
RUN_ROW: Final = (
    "INSERT INTO preparation.preparation_runs (stage_run_id, analysis_id, graph_thread_id,"
    " state, design_stage_run_id, preparation_revision, created_at, updated_at)"
    " VALUES (%s, %s, %s, 'running', %s, %s, %s, %s) ON CONFLICT (stage_run_id) DO UPDATE SET"
    " state = 'running', updated_at = EXCLUDED.updated_at")
RUN_STATE: Final = (
    "UPDATE preparation.preparation_runs SET state = %s, outcome_artifact_id = %s,"
    " row_set_hash = %s, updated_at = %s WHERE stage_run_id = %s")


# The §17.7 state allowlist: ids, hashes, statuses, gap codes, counts, and phase only.
PreparationState = TypedDict("PreparationState", {
    "analysis_id": str, "stage_run_id": str, "thread_id": str, "preparation_revision": int,
    "design_outcome_artifact_id": str, "phase": str, "status": str, "error_code": str,
    "conflict_code": str, "artifacts": dict[str, str], "hashes": dict[str, str],
    "row_set_hash": str, "gap_codes": list[str], "group_ids": list[str],
    "plan_item_ids": list[str], "counts": dict[str, int], "handoff_id": str}, total=False)


@dataclass(frozen=True)
class PreparationRunResult:
    """What one `causal run` step of a preparation revision returns (PRD-003 §5.2)."""

    status: str
    analysis_id: str
    stage_run_id: str
    thread_id: str
    preparation_revision: int
    outcome_artifact_id: str | None = None
    conflict_code: str | None = None
    row_set_hash: str | None = None
    handoff_id: str | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class PreparationDeps:
    """Everything the harness needs; nothing here is discovered at run time."""

    conn: Connection[Any]
    products: persistence.ProductStore
    objects: persistence.ObjectStore
    committer: persistence.ArtifactCommitter
    registry: ArtifactTypeRegistry
    emitter: events.EventEmitter
    clock: Callable[[], datetime]
    packs: pp.PreparationPackRegistry
    operations: operations.OperationRegistry
    diagnostic_registry: Mapping[str, diag.PreparationDiagnosticRowV1]
    rules: tuple[walls.ValidationRuleV1, ...]
    frames: frames.FrameStore
    method_packs_path: Path
    repo_root: Path
    # The §15 extra-implementation hook: a method pack may register real V1 diagnostics here.
    diagnostic_impls: Mapping[str, diag.DiagnosticFn] = field(default_factory=dict)


def build_handoff(analysis_id: str, outcome: ArtifactEnvelopeV1, receiving_stage_run_id: str,
                  entries: Sequence[Mapping[str, Any]], originating_outcome: str,
                  approval_ids: Sequence[str], now: datetime) -> HandoffManifestV1:
    """The one §20/§23 handoff manifold: opened on entry and reopened for PRD-004 (D-037)."""
    return HandoffManifestV1(
        handoff_id=entry.handoff_id(analysis_id, outcome.content_hash),
        schema_version="handoff.v1", analysis_id=analysis_id,
        producing_stage_run_id=outcome.stage_run_id,
        receiving_stage_run_id=receiving_stage_run_id,
        entries=tuple(ArtifactRef.model_validate(ref) for ref in entries),
        originating_outcome=originating_outcome, approval_ids=tuple(approval_ids),
        registry_version="artifact-types.v1", compatibility_version="handoff.v1",
        receiver_validation_result=None, receiver_error_codes=(), created_at_utc=now,
        accepted_at_utc=None)


class HarnessBase:
    """Events, commits, committed reads, walls, and the registry compilers nodes share."""

    def __init__(self, deps: PreparationDeps) -> None:
        self.deps = deps
        self.reports: dict[str, pc.PreparationDiagnosticV1] = {}
        self._count = 0
        self._cache: dict[str, Any] = {}

    # -- events, commits, and committed reads ------------------------------

    def event(self, state: PreparationState, name: str, evals: tuple[str, ...],
              **over: Any) -> events.OperationalEventV1:
        """One §18.6 event with a deterministic id and the preparation stage's identities."""
        self._count += 1
        return events.build_event(
            occurred_at_utc=self.deps.clock(), event_name=name, analysis_id=state["analysis_id"],
            event_id=f"evt:{state['stage_run_id']}:{self._count}",
            stage=events.Stage.PREPARATION, stage_run_id=state["stage_run_id"],
            graph_thread_id=state["thread_id"], component_id=COMPONENT,
            component_version=VERSION, required_eval_ids=evals,
            versions={"registry": "artifact-types.v1", "prompt": VERSION}, **over)

    def emit(self, state: PreparationState, name: str, evals: tuple[str, ...],
             **over: Any) -> None:
        self.deps.emitter.emit(self.event(state, name, evals, **over))

    def commit(self, state: PreparationState, kind: str, payload: Mapping[str, object],
               parents: tuple[ArtifactEnvelopeV1, ...]) -> ArtifactEnvelopeV1:
        """Build and commit one immutable artifact through the flush-gated committer (SC §8.2)."""
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
        return committed

    def payload(self, artifact_id: str) -> dict[str, Any]:
        """One committed payload, loaded for the node that needs it and released after."""
        if artifact_id not in self._cache:
            locator = self.deps.products.load_envelope(artifact_id).payload_locator
            self._cache[artifact_id] = json.loads(self.deps.objects.get(locator))
        return cast(dict[str, Any], self._cache[artifact_id])

    def ref(self, state: PreparationState, kind: str) -> ArtifactRef:
        found = state["artifacts"][kind]
        return ArtifactRef(artifact_id=found, content_hash=state["hashes"][found])

    def parents(self, state: PreparationState, *kinds: str) -> tuple[ArtifactEnvelopeV1, ...]:
        return tuple(self.deps.products.load_envelope(state["artifacts"][k]) for k in kinds)

    def entry_ref(self, state: PreparationState, kind: str) -> ArtifactRef:
        """The one entry reader: one of the four §4 entries as the manifest's own parent ref."""
        parents = self.deps.products.load_envelope(
            state["artifacts"]["PreparationContextManifest"]).parent_artifacts
        return parents[entry.ENTRY_TYPES.index(kind)]

    def entry_body(self, state: PreparationState, kind: str) -> dict[str, Any]:
        return self.payload(self.entry_ref(state, kind).artifact_id)

    def out(self, state: PreparationState, **over: Any) -> dict[str, Any]:
        """Every node writes the whole allowlisted snapshot back, so replay sees one state."""
        return dict(state) | dict(over)

    def fail(self, state: PreparationState, code: str, codes: Sequence[str] = (),
             status: str = pc.PreparationOutcomeStatus.FAILED.value) -> dict[str, Any]:
        """Enter a terminal status with one stable code and a raised blocker (§18.6)."""
        self.emit(state, "blocker.raised", EVAL_STAGE, severity=ERROR, error_code=code,
                  safe_dimensions={"blocked_operation": state.get("phase", ""),
                                   "detail_codes": ",".join(codes)})
        return self.out(state, status=status, error_code=code)

    def wall(self, state: PreparationState, number: int,
             ctx: walls.WallContext) -> ValidationReport:
        """One §24.3 wall; a failure is reported once and never waived by a later wall."""
        report = walls.wall(number, ctx)
        if not report.passed:
            self.emit(state, "artifact.validation_failed", EVAL_STAGE, severity=ERROR,
                      error_code=report.issues[0].code,
                      safe_dimensions={"wall": number, "detail_codes": ",".join(sorted(
                          {found for issue in report.issues for found in issue.artifact_ids}))})
        return report

    def conflict(self, state: PreparationState,
                 draft: pc.DesignConflictDraftV1) -> dict[str, Any]:
        """Commit the §16 conflict PRD-003 returns to PRD-002 and stop; nothing is resolved."""
        held = self.ref(state, "PreparationContextManifest")
        self.commit(state, "DesignConflict", pc.DesignConflictV1(
            **dict(draft) | {"evidence_artifact_ids": (held.artifact_id,)},
            conflict_id=f"dc:{state['stage_run_id']}:{draft.conflict_code}",
            context_manifest=held, evidence=(held,),
            preparation_revision=state["preparation_revision"]).canonical_payload(),
            self.parents(state, "PreparationContextManifest"))
        return self.out(state, status=pc.PreparationOutcomeStatus.DESIGN_CONFLICT.value,
                        conflict_code=draft.conflict_code)

    def design_handoff(self, state: PreparationState,
                       outcome: ArtifactEnvelopeV1) -> HandoffManifestV1:
        """Rebuild the PRD-002 §23 handoff from committed payloads; never import PRD-002."""
        body = self.payload(outcome.artifact_id)
        design = self.payload(str(body["experiment_design"]["artifact_id"]))
        return build_handoff(
            state["analysis_id"], outcome, state["stage_run_id"],
            (design["selected_csv"], body["experiment_design"], body["runnable_frame_contract"],
             body["capacity_check"]), str(body["status"]),
            (str(body["approval"]["artifact_id"]),), self.deps.clock())

    # -- frames, diagnostics, and the registry compilers -------------------

    def book(self, state: PreparationState) -> pc.PreparationContextManifestV1:
        return parse_strict(pc.PreparationContextManifestV1,
                            self.payload(state["artifacts"]["PreparationContextManifest"]))

    def pack(self, book: pc.PreparationContextManifestV1) -> pp.PreparationPackV1:
        return self.deps.packs.get(book.method_id, book.method_pack_version)

    def schema_of(self, frame: pl.DataFrame) -> tuple[pc.ColumnSchemaFieldV1, ...]:
        return tuple(pc.ColumnSchemaFieldV1(column_name=name, dtype=str(dtype), prepared_from=())
                     for name, dtype in frame.schema.items())

    def frame(self, state: PreparationState, kind: str) -> pl.DataFrame:
        payload = self.payload(state["artifacts"][kind])
        schema = tuple(parse_strict(pc.ColumnSchemaFieldV1, row) for row in payload["columns"])
        return self.deps.frames.read_frame(str(payload["frame_object"]["object_locator"]), schema)

    def dtypes(self, frame: pl.DataFrame) -> dict[str, str]:
        return {name: str(dtype) for name, dtype in frame.schema.items()}

    def commit_frame(self, state: PreparationState, kind: str, frame: pl.DataFrame,
                     model: type[Any], parents: tuple[ArtifactEnvelopeV1, ...],
                     schema: tuple[pc.ColumnSchemaFieldV1, ...] | None = None,
                     **over: Any) -> ArtifactEnvelopeV1:
        """Write one immutable frame object, then commit the artifact that points at it."""
        columns = schema or self.schema_of(frame)
        locator, digest = self.deps.frames.write_frame(frame, columns)
        return self.commit(state, kind, model(
            columns=columns, row_count=frame.height, writer_version=VERSION,
            frame_object=pc.ObjectRefV1(object_locator=locator, content_hash=digest),
            **over).canonical_payload(), parents)

    def diagnose(self, diagnostic_id: str,
                 request: diag.DiagnosticRequest) -> pc.PreparationDiagnosticV1:
        """Run one registered diagnostic; an unimplemented pack row is `not_computable` (§15)."""
        try:
            return diag.run_diagnostic(diagnostic_id, request, self.deps.diagnostic_registry,
                                       self.deps.diagnostic_impls)
        except RegistryError as error:
            if error.code != diag.DIAGNOSTIC_NOT_IMPLEMENTED:
                raise
            return pc.PreparationDiagnosticV1(
                diagnostic_id=diagnostic_id, diagnostic_version="v1",
                frame_stage=request.frame_stage, inputs=request.inputs, columns_read=(),
                total_rows=request.frame.height, used_rows=0, unused_reason_counts={},
                row_set_hash=request.row_set_hash, values={}, warnings=(),
                status=pc.DiagnosticStatus.NOT_COMPUTABLE,
                implementation_version=diag.IMPLEMENTATION_VERSION)

    def readiness(self, state: PreparationState, book: pc.PreparationContextManifestV1,
                   pack: pp.PreparationPackV1, bundle: pc.PreparedFrameBundleV1,
                   diagnostics_pass: bool) -> dict[str, bool]:
        """PRD-003 §20's thirteen conditions, built from real state before the bundle commit."""
        record = parse_strict(pc.StabilizationRecordV1,
                              self.payload(state["artifacts"]["StabilizationRecord"]))
        receipts = self.payload(state["artifacts"]["ExecutionReceiptBundle"])
        plan = parse_strict(pp.PreparationPlanV1,
                            self.payload(state["artifacts"]["PreparationPlan"]))
        entries = self.entry_body(state, "ExperimentDesign")
        return {
            "outcome_prepared": True,
            "artifacts_and_parents_match": all(kind in state["artifacts"] for kind in (
                "StabilizationRecord", "StabilizedFrame", "PreparedFrame")),
            "bundle_binds_design": bundle.experiment_design.content_hash == self.entry_ref(
                state, "ExperimentDesign").content_hash and bool(entries),
            "row_set_hash_shared": bundle.stabilized_frame_row_set_hash == (
                bundle.prepared_frame_row_set_hash),
            "dispositions_terminal": sum(
                row.row_count for row in record.dispositions.counts
            ) == record.source_row_index.row_count,
            "lineage_recoverable": set(receipts["changed_counts_by_column"]) <= {
                item.output_column for item in plan.items} | {None},
            "receipts_complete": len(receipts["receipts"]) == len(plan.items),
            "postrepair_diagnostics_handled": diagnostics_pass,
            "prepared_schema_matches_pack": (
                book.prepared_frame_schema_id == pack.prepared_frame_schema_id),
            "recipes_permitted": all(recipe.operation_id in book.permitted_operation_ids
                                     for recipe in plan.recipes),
            "no_estimate_content": all(item.phase is not pp.ItemPhase.DIAGNOSTIC
                                       for item in plan.items),
            "capacity_still_pass": self.entry_body(
                state, "DeliveryCapacityCheck").get("status") == "pass",
            # SC §8.2 completes a commit only after an acknowledged flush, so every artifact
            # above is proof its own span was delivered (an outage raises before this node).
            "spans_acknowledged": True}

    def postcondition(self, state: PreparationState, item: pp.PlanItemV1, before: pl.DataFrame,
                      after: pl.DataFrame) -> pc.PreparationDiagnosticV1:
        """One §17.6 postcondition report per mutation; the executor gates on it."""
        report = self.diagnose("row_set_invariance", diag.DiagnosticRequest(
            frame=after, inputs=(self.ref(state, "StabilizedFrame"),),
            frame_stage=pc.FrameStage.PREPARED, baseline=before,
            row_set_hash=state["row_set_hash"]))
        self.reports[item.plan_item_id] = report
        return report

    def reconcile_inputs(self, state: PreparationState, book: pc.PreparationContextManifestV1,
                         pack: pp.PreparationPackV1, phase: pp.PlanPhase,
                         gaps: tuple[plancompile.FrameGapV1, ...] = (),
                         groups: tuple[plancompile.TaskGroupPlanV1, ...] = (),
                         ) -> plancompile.ReconcileInputs:
        return plancompile.ReconcileInputs(
            manifest=book, operations=self.deps.operations, pack=pack, gaps=gaps, groups=groups,
            context_manifest=self.ref(state, "PreparationContextManifest"), phase=phase,
            plan_revision=state["preparation_revision"], versions=REGISTRY_VERSIONS,
            stabilized_frame=self.ref(state, "StabilizedFrame")
            if "StabilizedFrame" in state["artifacts"] else None)

    def policy(self, pack: pp.PreparationPackV1, design: ArtifactEnvelopeV1) -> entry.EntryPolicy:
        """The §4 policy the composition supplies: vocabulary, strategies, statuses (D-075)."""
        computed = {str(row["diagnostic_id"]): "computed"
                    for parent in design.parent_artifacts
                    for row in self.payload(parent.artifact_id).get("results") or ()}
        return entry.EntryPolicy(
            pack=pack, registry_versions=REGISTRY_VERSIONS, prerepair_statuses=computed,
            eligibility_vocabulary=pp.eligibility_vocabulary(
                self.deps.method_packs_path, pack.method_id),
            imputation_strategy_ids=tuple(sorted(
                target.strategy_id for target in pack.permitted_imputation_targets)),
            approved_handling=APPROVED_HANDLING)

    def structure(self, book: pc.PreparationContextManifestV1,
                  contract: Mapping[str, Any]) -> impact.MethodStructureSpecV1:
        """The role columns and the boundary the §12 structure gates read."""
        roles = operations.by_role(book.column_roles)
        held = {str(key): str(value)
                for key, value in (contract.get("method_structure") or {}).items()}
        units = tuple(column for column in (roles.get("unit_identifier"), roles.get("time"))
                      if column is not None)
        return impact.MethodStructureSpecV1(
            unit_columns=units if book.method_id == "did" else units[:1],
            treatment_column=roles.get("treatment"), outcome_column=roles.get("outcome"),
            group_column=roles.get("group"), time_column=roles.get("time"),
            running_variable_column=roles.get("running_variable"),
            threshold=held.get("cutoff") or held.get("adoption_time"), minimum_cell_rows=1)

    def parse_specs(self, data: bytes,
                    book: pc.PreparationContextManifestV1) -> tuple[Any, ...]:
        """The pinned profile: inferred dtypes, with identity and protected cells critical."""
        critical = set(book.key_columns) | set(book.protected_columns)
        return tuple(stabilize.ColumnParseSpecV1(
            column_name=name, dtype=dtype_name(dtype), parse_critical=name in critical)
            for name, dtype in pl.read_csv(io.BytesIO(data), n_rows=256).schema.items())

    def row_rules(self, book: pc.PreparationContextManifestV1, pack: pp.PreparationPackV1,
                  columns: Sequence[str]) -> tuple[Any, Any]:
        """§9.3 required-role rules and the §9.4 duplicate authorisations, registered only."""
        known = sorted(set(pack.permitted_disposition_rule_ids) & set(book.unusable_row_rule_ids))
        roles = operations.by_role(book.column_roles)
        rules = tuple(stabilize.RequiredRoleRuleV1(
            role=role, column=column, observed_rule_id=f"{role}_observed",
            disposition_rule_id=next((found for found in known if role in found), None))
            for role, column in sorted(roles.items())
            if role in pack.protected_roles and column in columns)
        return rules, stabilize.DuplicatePolicyV1(conflict_resolution_rule_id=next(
            (found for found in known if "collision" in found), None))


def dtype_name(dtype: pl.DataType) -> str:
    """The V1 frame dtype name, or `String` for anything the vocabulary does not carry."""
    try:
        return frames.dtype_name(dtype)
    except frames.FrameError:
        return "String"
