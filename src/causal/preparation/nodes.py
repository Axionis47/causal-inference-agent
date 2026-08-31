"""The five deterministic nodes, the sequential caller, and the §20 handoff (T-019 §1.3)."""

# PRD-003 §7 flow, §9, §10, §15, §18.6, §20, §25; SC §4, §9. Amendment 2 removed every model
# call from this stage: each node is a pure function of committed artifacts, so a rerun is
# artifact replay under a new `stage_run_id` and needs no checkpoint (D-035, D-069).

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

import polars as pl

from causal.preparation import contracts as pc
from causal.preparation import diagnostics as diag
from causal.preparation import entry, executor, impact, operations, plancompile, stabilize
from causal.preparation import harness as hb
from causal.preparation import plans as pp
from causal.preparation import validators as walls
from causal.preparation.harness import HarnessBase, PreparationDeps, PreparationState
from causal.shared import events, handoff, persistence
from causal.shared.contracts import ArtifactRef, HandoffManifestV1
from causal.shared.validation import parse_strict

__all__ = ["PreparationNodes", "open_preparation_handoff", "run_preparation"]

ERROR, PREPARED = events.Severity.ERROR, pc.PreparationOutcomeStatus.PREPARED.value
NOT_RUNNABLE = pc.PreparationOutcomeStatus.NOT_RUNNABLE.value
CONFLICT = pc.PreparationOutcomeStatus.DESIGN_CONFLICT.value


def open_preparation_handoff(deps: PreparationDeps, analysis_id: str, outcome_artifact_id: str,
                             receiving_stage_run_id: str) -> HandoffManifestV1:
    """The §20 handoff, rebuilt from committed artifacts and never recorded here (D-037)."""
    found = deps.products.load_envelope(outcome_artifact_id)
    body = json.loads(deps.objects.get(found.payload_locator))
    if found.artifact_type != "PreparationOutcome" or body["status"] != PREPARED:
        raise persistence.PersistenceError(
            f"no readable preparation handoff for {analysis_id!r}", "handoff_unavailable")
    bundle = json.loads(deps.objects.get(deps.products.load_envelope(
        str(body["prepared_bundle"]["artifact_id"])).payload_locator))
    return hb.build_handoff(
        analysis_id, found, receiving_stage_run_id,
        (body["prepared_bundle"], bundle["experiment_design"], bundle["runnable_frame_contract"],
         bundle["capacity_check"]), PREPARED, (), deps.clock())


class PreparationNodes(HarnessBase):
    """Entry and manifest, rows and freeze, plan, mutation, and the terminal outcome."""

    # -- entry ------------------------------------------------------------

    # Open the design handoff, run the nine §4 checks, freeze the §7.2 manifest.
    def entry_node(self, state: PreparationState) -> dict[str, Any]:
        deps = self.deps
        self.emit(state, "stage.started", hb.EVAL_STAGE)
        self.emit(state, "task.started", hb.EVAL_STAGE, status="entry")
        outcome = deps.products.load_envelope(state["design_outcome_artifact_id"])
        store = handoff.HandoffStore(deps.conn)
        gate = handoff.HandoffGate(deps.objects, deps.products, store, deps.registry, deps.emitter)
        try:
            opened = entry.accept_handoff(
                gate, store, self.design_handoff(state, outcome),
                lambda verdict, codes: self.event(
                    state, f"handoff.{verdict}", hb.EVAL_STAGE, status=verdict,
                    error_code=codes[0] if codes else None))
            inputs = entry.read_entries(opened.manifest, outcome.artifact_id, deps.products,
                                        deps.objects)
            pack = deps.packs.get(str(inputs.design["method_id"]),
                                  str(inputs.design["method_pack_version"]))
            entry.validate_entry(
                opened.manifest, inputs,
                self.policy(pack, deps.products.load_envelope(
                    opened.manifest.entries[1].artifact_id)),
                products=deps.products, objects=deps.objects)
        except (entry.PreparationEntryError, KeyError, TypeError) as error:
            code = getattr(error, "code", entry.ENTRY_UNREADABLE)
            codes = getattr(error, "detail_codes", ())
            self.wall(state, hb.ENTRY_WALL,
                      walls.WallContext(rules=deps.rules, entry_codes=codes or (code,)))
            self.emit(state, "task.failed", hb.EVAL_STAGE, severity=ERROR, error_code=code)
            return self.fail(state, code, codes)
        report = self.wall(state, hb.ENTRY_WALL,
                           walls.WallContext(rules=deps.rules, handoff_accepted=True))
        if not report.passed:
            return self.fail(state, report.issues[0].code)
        # The one authoritative approved-context surface, compiled and frozen here (§7.2).
        entries = opened.manifest.entries
        types = self.deps.products.artifact_types_of(
            self.deps.products.load_envelope(entries[0].artifact_id).parent_artifacts)
        book = entry.compile_context_manifest(
            inputs, entries, pack=pack,
            role_ledger=self.payload(str(inputs.design["role_ledger"]["artifact_id"])),
            measurement_map=self.payload(str(inputs.design["measurement_map"]["artifact_id"])),
            causal_context=ArtifactRef.model_validate(inputs.design["causal_context"]),
            question_id=next((found for found, kind in sorted(types.items())
                              if kind == "QuestionRecord"), state["analysis_id"]),
            parser_profile_id=hb.PARSER_PROFILE, registry_versions=hb.REGISTRY_VERSIONS,
            # Amendment 2: no preparation task is delegated, so no recipient may hold a tool.
            recipient_map={})
        self.commit(state, "PreparationContextManifest", book.canonical_payload(),
                    tuple(self.deps.products.load_envelope(ref.artifact_id) for ref in entries))
        self.emit(state, "task.completed", hb.EVAL_STAGE, status="entry")
        return self.out(state, phase=pp.PlanPhase.STABILIZATION.value)

    # -- rows, impact, structure, and the row-set freeze --------------------

    # Parse, identify rows, run the §9.1 engine, and freeze the row set (§9.6).
    def stabilize_node(self, state: PreparationState) -> dict[str, Any]:
        book, contract = self.book(state), self.entry_body(state, "RunnableFrameContract")
        pack, data = self.pack(book), self.deps.objects.get(book.source_object_locator)
        self.emit(state, "task.started", hb.EVAL_ROWS, status="stabilize")
        try:
            parsed = stabilize.with_row_unit(
                stabilize.parse_source_csv(data, self.parse_specs(data, book)), book.key_columns)
            role_rules, duplicates = self.row_rules(book, pack, parsed.frame.columns)
            result = stabilize.stabilize(
                parsed, csv_hash=hashlib.sha256(data).hexdigest(), manifest=book, pack=pack,
                rules=(), role_rules=role_rules, duplicates=duplicates, vocabulary=())
        except stabilize.StabilizationError as error:
            return self.fail(state, error.code)
        unresolved = sum(row.row_count for row in result.counts()
                         if row.disposition is pc.RowDisposition.UNRESOLVED_CONFLICT)
        if unresolved:  # §9.3: no registered V1 operation clears a mismatched required role
            return self.conflict(state, pc.conflict_draft(hb.UNRESOLVED, unresolved))
        return self._freeze(state, book, pack, contract, parsed, result)

    # Impact, method structure, walls 2 and 3, then the frozen `StabilizedFrame`.
    def _freeze(self, state: PreparationState, book: pc.PreparationContextManifestV1,
                pack: pp.PreparationPackV1, contract: Mapping[str, Any],
                parsed: stabilize.ParsedSource,
                result: stabilize.StabilizationResult) -> dict[str, Any]:
        spec, keep = self.structure(book, contract), result.retained_mask()
        retained = parsed.frame.filter(pl.Series(values=keep, dtype=pl.Boolean))
        units = (retained.select(spec.unit_columns).unique().height if spec.unit_columns
                 else retained.height)
        try:
            structure = impact.validate_structure(retained, spec, pack)
            impacts = impact.dimension_impact(parsed.frame, keep, impact.dimension_specs(
                operations.by_role(book.column_roles), book.deletion_impact_dimensions,
                None if spec.threshold is None else str(spec.threshold)))
            summaries = stabilize.stabilization_summaries(self.deps.objects, parsed, result,
                                                          (), units)
        except stabilize.StabilizationError as error:
            return self.fail(state, error.code)
        if structure.verdict is impact.StructureVerdict.CONFLICT:
            return self.conflict(state, pc.conflict_draft(structure.codes[0], retained.height,
                                                          "method_structure"))
        record = pc.StabilizationRecordV1(
            context_manifest=self.ref(state, "PreparationContextManifest"),
            source_row_index=summaries[0], eligibility=summaries[1], dispositions=summaries[2],
            impact=impacts, method_structure_status=structure.status,
            method_structure_codes=structure.codes, freeze=summaries[3],
            pre_stabilization_diagnostics=(), versions=hb.REGISTRY_VERSIONS,
            post_stabilization_diagnostics=tuple(
                self.diagnose(name, diag.DiagnosticRequest(
                    frame=retained, inputs=(book.selected_csv,),
                    frame_stage=pc.FrameStage.STABILIZED, source_row_count=len(result.rows),
                    row_set_hash=summaries[3].row_set_hash, key_columns=tuple(book.key_columns),
                    disposition_counts={row.disposition.value: row.row_count
                                        for row in summaries[2].counts}))
                for name in hb.STABILIZED_DIAGNOSTICS))
        context = walls.WallContext(
            rules=self.deps.rules, manifest=book, record=record, structure=structure,
            evaluated_rule_ids=tuple(result.rule_counts), row_set_hash=summaries[3].row_set_hash)
        for number in (hb.ROWS_WALL, hb.FREEZE_WALL):
            report = self.wall(state, number, context)
            if not report.passed:
                return self.fail(state, report.issues[0].code, status=PREPARED if (
                    structure.verdict is impact.StructureVerdict.RUNNABLE) else NOT_RUNNABLE)
        held = self.commit(state, "StabilizationRecord", record.canonical_payload(),
                           self.parents(state, "PreparationContextManifest"))
        self.commit_frame(state, "StabilizedFrame", retained, pc.StabilizedFrameV1, (held,),
                          row_set_hash=summaries[3].row_set_hash,
                          stabilization_record=ArtifactRef(artifact_id=held.artifact_id,
                                                           content_hash=held.content_hash),
                          source_csv=book.selected_csv)
        self.emit(state, "task.completed", hb.EVAL_ROWS, status="frozen")
        return self.out(state, phase=pp.PlanPhase.PREPARATION.value,
                        row_set_hash=summaries[3].row_set_hash,
                        counts={**state["counts"], "source_rows": len(result.rows),
                                "retained_rows": retained.height})

    # -- the post-freeze task graph and the one reconciled plan -------------

    # The §25.1 deterministic gap compilation over the §7.4 task graph, then the fan-in.
    def plan_node(self, state: PreparationState) -> dict[str, Any]:
        book, frame = self.book(state), self.frame(state, "StabilizedFrame")
        pack, schema = self.pack(book), self.dtypes(frame)
        surface = plancompile.contract_surface(
            book, pack, self.entry_body(state, "RunnableFrameContract"), schema)
        gaps = plancompile.contract_gaps(schema, surface, dict(zip(
            frame.columns, (int(found) for found in frame.null_count().row(0)), strict=True)))
        groups = plancompile.group_gaps(gaps, surface)
        by_id = {group.group_id: group for group in groups.groups}
        drafts: list[pp.PreparationTaskDraftV1] = []
        for group in (by_id[found] for wave in groups.waves for found in wave):
            drafted = plancompile.compile_drafts(
                group, tuple(gap for gap in gaps if gap.column in group.columns), surface)
            if isinstance(drafted, pc.DesignConflictDraftV1):
                return self.conflict(state, drafted)
            drafts.append(drafted)
        # The §7.5 fan-in: one plan out of every group's proposals, then wall 4 (§24.3).
        try:
            routed = plancompile.reconcile(drafts, self.reconcile_inputs(
                state, book, pack, pp.PlanPhase.PREPARATION, gaps, groups.groups))
        except plancompile.PlanCompileError as error:
            return self.fail(state, error.code, error.detail_codes)
        if isinstance(routed, plancompile.ConflictRoute):
            return self.conflict(state, routed.draft)
        held = self.commit(state, "PreparationPlan", routed.canonical_payload(), self.parents(
            state, "PreparationContextManifest", "StabilizationRecord"))
        now, run = self.deps.clock(), state["stage_run_id"]
        for item in routed.items:
            self.deps.conn.execute(hb.PLAN_ROW, (held.artifact_id, item.plan_item_id, run,
                                                 item.phase.value, "committed", now, now))
        report = self.wall(state, hb.PLAN_WALL, walls.WallContext(
            rules=self.deps.rules, manifest=book, plan=routed, pack=pack, gaps=gaps,
            operations=self.deps.operations))
        if not report.passed:
            return self.fail(state, report.issues[0].code)
        return self.out(state, gap_codes=sorted({gap.gap_code for gap in gaps}),
                        group_ids=[group.group_id for group in groups.groups],
                        plan_item_ids=[item.plan_item_id for item in routed.items],
                        counts={**state["counts"], "plan_items": len(routed.items),
                                "task_groups": len(groups.groups)})

    # -- sequential mutation under the §17.6 gate ---------------------------

    # Preview, execute in dependency order, then wall 5 over the receipt bundle.
    def execute_node(self, state: PreparationState) -> dict[str, Any]:
        book, frame = self.book(state), self.frame(state, "StabilizedFrame")
        plan_ref = self.ref(state, "PreparationPlan")
        plan = parse_strict(pp.PreparationPlanV1, self.payload(plan_ref.artifact_id))
        stored = self.payload(state["artifacts"]["StabilizedFrame"])["frame_object"]
        source = executor.FrameArtifact(ArtifactRef(
            artifact_id=state["artifacts"]["StabilizedFrame"],
            content_hash=str(stored["content_hash"])), str(stored["object_locator"]),
            self.schema_of(frame))
        if plan.items:
            diag.preview(plan, plan_ref, frame)  # §17.2: no mutation unlocks without a preview
            first = next((item for item in plan.items if not item.depends_on), None)
            self.emit(state, "tool.started", hb.EVAL_EXEC, status="execute")
            try:
                outcome = executor.PlanExecutor(
                    registry=self.deps.operations, context=operations.operation_context(book),
                    store=self.deps.frames,
                    postcondition=lambda item, before, after: self.postcondition(
                        state, item, before, after),
                    stage_run_id=state["stage_run_id"], clock=self.deps.clock).run(
                    plan, plan_ref, frame, source, row_set_hash=state["row_set_hash"],
                    parents=(plan_ref,),
                    expected_inputs={} if first is None else {first.plan_item_id: source.ref})
            except executor.ExecutionError as error:
                self.emit(state, "tool.failed", hb.EVAL_EXEC, severity=ERROR, error_code=error.code)
                return self.fail(state, error.code, error.codes)
            self.emit(state, "tool.completed", hb.EVAL_EXEC, status="executed")
        # §25: the contract is already satisfied, so nothing is previewed or mutated and the
        # prepared frame is the stabilized frame under an honestly empty receipt bundle.
        else:
            outcome = executor.ExecutionOutcome(pp.ExecutionReceiptBundleV1(
                plan=plan_ref, receipts=(), changed_counts_by_column={}, imputed_cell_mask=None,
                missingness_before=diag.nulls(frame), missingness_after=diag.nulls(frame),
                parents=(plan_ref,), row_set_hash=state["row_set_hash"]), frame, source, {})
        report = self.wall(state, hb.EXEC_WALL, walls.WallContext(
            rules=self.deps.rules, source_ref=source.ref, row_set_hash=state["row_set_hash"],
            receipts=tuple(walls.ReceiptCheck(
                receipt=receipt, output=receipt.output_ref,
                postcondition=self.reports[receipt.plan_item_id])
                for receipt in outcome.bundle.receipts)))
        if not report.passed:
            return self.fail(state, report.issues[0].code)
        bundle = self.commit(state, "ExecutionReceiptBundle", outcome.bundle.canonical_payload(),
                             self.parents(state, "PreparationPlan"))
        self.commit_frame(
            state, "PreparedFrame", outcome.frame, pc.PreparedFrameV1,
            self.parents(state, "StabilizedFrame") + (bundle,), outcome.output.schema,
            row_set_hash=state["row_set_hash"], stabilized_frame=self.ref(state, "StabilizedFrame"),
            execution_receipt_bundle=ArtifactRef(artifact_id=bundle.artifact_id,
                                                 content_hash=bundle.content_hash),
            prepared_frame_schema_id=book.prepared_frame_schema_id)
        return self.out(state, counts={**state["counts"], "receipts": len(outcome.bundle.receipts)})

    # -- diagnostics, the §20 bundle, and the terminal outcome --------------

    # The §15 prepared-stage diagnostics, wall 6, and the §20 bundle commit.
    def _finalise(self, state: PreparationState) -> tuple[str, str | None]:
        book, prepared = self.book(state), self.frame(state, "PreparedFrame")
        pack, baseline = self.pack(book), self.frame(state, "StabilizedFrame")
        surface = plancompile.contract_surface(
            book, pack, self.entry_body(state, "RunnableFrameContract"), self.dtypes(baseline))
        required_schema = {column: "Boolean" for column in surface.required_derivations}
        receipts = self.payload(state["artifacts"]["ExecutionReceiptBundle"])
        request = diag.DiagnosticRequest(
            frame=prepared, inputs=(self.ref(state, "PreparedFrame"),),
            frame_stage=pc.FrameStage.PREPARED, baseline=baseline, expected_dtypes=required_schema,
            row_set_hash=state["row_set_hash"], key_columns=tuple(book.key_columns),
            required_columns=tuple(book.key_columns) + tuple(required_schema),
            changed_by_column={str(name): int(count) for name, count
                               in receipts["changed_counts_by_column"].items()})
        required = tuple(pack.required_postrepair_diagnostic_ids) + hb.PREPARED_DIAGNOSTICS
        reports = tuple(self.diagnose(name, request) for name in required)
        refs = {kind: self.ref(state, kind) for kind in (
            "StabilizationRecord", "StabilizedFrame", "PreparedFrame", "ExecutionReceiptBundle")}
        bundle = pc.PreparedFrameBundleV1(
            selected_table=book.selected_csv,
            experiment_design=self.entry_ref(state, "ExperimentDesign"),
            runnable_frame_contract=self.entry_ref(state, "RunnableFrameContract"),
            capacity_check=self.entry_ref(state, "DeliveryCapacityCheck"),
            stabilization_record=refs["StabilizationRecord"],
            stabilized_frame=refs["StabilizedFrame"], prepared_frame=refs["PreparedFrame"],
            execution_receipt_bundle=refs["ExecutionReceiptBundle"], row_set_hash=state[
                "row_set_hash"], stabilized_frame_row_set_hash=state["row_set_hash"],
            prepared_frame_row_set_hash=state["row_set_hash"],
            postrepair_diagnostics=reports, versions=hb.REGISTRY_VERSIONS)
        healthy = all(report.status is pc.DiagnosticStatus.PASS
                      or report.status.value in hb.APPROVED_HANDLING for report in reports)
        report = self.wall(state, hb.FINAL_WALL, walls.WallContext(
            rules=self.deps.rules, diagnostics=reports, required_diagnostic_ids=required,
            diagnostic_registry=self.deps.diagnostic_registry,
            implemented_diagnostic_ids=frozenset(self.deps.diagnostic_impls),
            approved_handling=hb.APPROVED_HANDLING, required_schema=required_schema,
            prepared_schema=self.dtypes(prepared),
            readability=self.readiness(state, book, pack, bundle, healthy)))
        if not report.passed:
            return NOT_RUNNABLE, report.issues[0].code
        opened = self.deps.products.load_envelope(
            state["artifacts"]["PreparationContextManifest"]).parent_artifacts
        self.commit(state, "PreparedFrameBundle", bundle.canonical_payload(), tuple(
            self.deps.products.load_envelope(ref.artifact_id) for ref in opened)
            + self.parents(state, "StabilizationRecord", "StabilizedFrame", "PreparedFrame",
                           "ExecutionReceiptBundle"))
        return PREPARED, None

    # The one terminal `PreparationOutcome`, its stage-run close, and the §20 handoff.
    def outcome_node(self, state: PreparationState) -> dict[str, Any]:
        status, code = state.get("status") or PREPARED, state.get("error_code")
        if "PreparationContextManifest" not in state["artifacts"]:
            self.deps.products.transition_stage_run(state["stage_run_id"], "failed")
            self.emit(state, "stage.failed", hb.EVAL_FINAL, severity=ERROR, error_code=code)
            return self.out(state, status=status)
        if status == PREPARED:
            status, code = self._finalise(state)
        built = self.commit(state, "PreparationOutcome", pc.PreparationOutcomeV1(
            status=pc.PreparationOutcomeStatus(status),
            context_manifest=self.ref(state, "PreparationContextManifest"),
            prepared_bundle=self.ref(state, "PreparedFrameBundle") if status == PREPARED else None,
            design_conflict=self.ref(state, "DesignConflict") if status == CONFLICT else None,
            stage_run_id=state["stage_run_id"], graph_thread_id=state["thread_id"],
            error_code=code).canonical_payload(), self.parents(state, "PreparationContextManifest"))
        self.deps.products.transition_stage_run(state["stage_run_id"], "committing")
        self.deps.products.transition_stage_run(state["stage_run_id"], "completed")
        self.emit(state, "stage.completed", hb.EVAL_FINAL, status=status, artifact_refs=(
            ArtifactRef(artifact_id=built.artifact_id, content_hash=built.content_hash),))
        if status != PREPARED:
            return self.out(state, status=status, error_code=code or "")
        opened = open_preparation_handoff(self.deps, state["analysis_id"], built.artifact_id,
                                          f"sr:{state['analysis_id']}:estimation")
        return self.out(state, status=status, handoff_id=opened.handoff_id)


def run_preparation(deps: PreparationDeps, *, analysis_id: str, design_outcome_artifact_id: str,
                    stage_run_id: str, preparation_revision: int = 1) -> hb.PreparationRunResult:
    """One revision, run straight through; a terminal status short-circuits to the close (§7.1)."""
    # A rerun is artifact replay (D-035): a NEW `stage_run_id`, deterministic artifact ids that
    # make every recommit a no-op, no checkpoint, and no committed-artifact skip logic.
    # §7.1: a new graph thread per revision; PRD-003 never continues the design stage's.
    thread_id, now = f"pt:{analysis_id}:{preparation_revision}", deps.clock()
    design_run = deps.products.load_envelope(design_outcome_artifact_id).stage_run_id
    try:
        deps.products.get_stage_run_state(stage_run_id)
    except persistence.PersistenceError:
        deps.products.create_stage_run(stage_run_id, analysis_id, "preparation")
        deps.products.transition_stage_run(stage_run_id, "tracing_preflight")
        deps.products.transition_stage_run(stage_run_id, "running")
    deps.conn.execute(hb.RUN_ROW, (stage_run_id, analysis_id, thread_id, design_run,
                                 preparation_revision, now, now))
    state: PreparationState = {
        "analysis_id": analysis_id, "stage_run_id": stage_run_id, "thread_id": thread_id,
        "preparation_revision": preparation_revision, "phase": "entry",
        "design_outcome_artifact_id": design_outcome_artifact_id, "artifacts": {}, "hashes": {},
        "gap_codes": [], "group_ids": [], "plan_item_ids": [], "counts": {}}
    nodes = PreparationNodes(deps)
    for name in hb.NODES[:-1]:
        state = cast(PreparationState, getattr(nodes, name)(state))
        if state.get("status"):
            break
    final = cast(PreparationState, nodes.outcome_node(state))
    # SC §4: the run row reaches a terminal state on every exit path, success or not.
    status, outcome = final.get("status") or "failed", final["artifacts"].get("PreparationOutcome")
    deps.conn.execute(hb.RUN_STATE, (hb.ROW_STATE.get(status, "failed"), outcome,
                                   final.get("row_set_hash"), deps.clock(), stage_run_id))
    return hb.PreparationRunResult(
        status=status, analysis_id=analysis_id, stage_run_id=stage_run_id, thread_id=thread_id,
        preparation_revision=preparation_revision, outcome_artifact_id=outcome,
        conflict_code=final.get("conflict_code"), row_set_hash=final.get("row_set_hash"),
        handoff_id=final.get("handoff_id"), error_code=final.get("error_code") or None)
