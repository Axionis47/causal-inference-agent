"""Numerical execution and evidence handoff; report reasoning belongs to post-analysis."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any, Final, cast

import polars as pl

from causal.analysis.common import legacy_diagnostics as diagnostics
from causal.analysis.common import legacy_engine as engine
from causal.analysis.integration import contracts as ec
from causal.analysis.integration import harness as eh
from causal.analysis.integration import plancompile
from causal.analysis.integration import walls as ew
from causal.shared import persistence
from causal.shared.contracts import ArtifactRef, HandoffManifestV1

COMPLETE: Final = "complete"
NOT_ESTIMABLE: Final = "not_estimable"
ADAPTER_UNREGISTERED: Final = "estimator_adapter_unregistered"
PRIMARY_NOT_ESTIMABLE: Final = "primary_result_incomplete"
# Numerical computation is complete before reporting begins.
NODES: Final = ("entry_node", "plan_node", "estimate_node", "evidence_node")
# The closed §6.1/§19.1 registry vector this stage pins on every manifest, plan, and result.
REGISTRY_VERSIONS: Final[dict[str, str]] = {
    "artifact_types": "artifact-types.v1", "method_packs": "method-packs.v1",
    "estimation_packs": "method-pack-estimation.v1", "diagnostics": "estimation-diagnostics.v1",
    "sensitivities": "estimation-sensitivities.v1", "figure_builders": "estimation-figures.v1",
    "capacity": "delivery-capacity.v1", "visualization_catalog": "visualization-catalog.v1",
    "schema": "estimation-schemas.v1", "validators": ew.REGISTRY_VERSION}
# §6.1: the comparison tolerances, fixed in the plan before any result is read.
TOLERANCES: Final[dict[str, float]] = {"sensitivity_magnitude": 0.25, "interval_width": 0.5}
# Scientific result bundles have no figure or claim contract.
EVIDENCE_KINDS: Final[tuple[ec.EvidenceKind, ...]] = ("diagnostic", "sensitivity")


def open_post_analysis_handoff(deps: eh.EstimationDeps, analysis_id: str,
                               outcome_artifact_id: str,
                               receiving_stage_run_id: str) -> HandoffManifestV1:
    """Bind post-analysis to the numerical bundle and its frozen design sources."""
    found = deps.products.load_envelope(outcome_artifact_id)
    body = json.loads(deps.objects.get(found.payload_locator))
    if found.analysis_id != analysis_id or found.artifact_type != "EstimationOutcome":
        raise persistence.PersistenceError(
            f"no readable numerical handoff for {analysis_id!r}", "handoff_unavailable")
    if body["status"] != COMPLETE:
        attempts = tuple(ref.model_dump(mode="json") for ref in found.parent_artifacts
                         if deps.products.load_envelope(ref.artifact_id).artifact_type == "EstimationPlan")
        return eh.build_handoff(
            analysis_id, found, receiving_stage_run_id,
            ({"artifact_id": found.artifact_id, "content_hash": found.content_hash},
             body["context_manifest"], *attempts), body["status"], deps.clock())
    bundle_ref = ArtifactRef.model_validate(body["estimation_bundle"])
    envelope = deps.products.load_envelope(bundle_ref.artifact_id)
    if (envelope.analysis_id != analysis_id or envelope.artifact_type != "NumericalBundle"
            or envelope.content_hash != bundle_ref.content_hash):
        raise persistence.PersistenceError("numerical bundle binding differs", "handoff_unavailable")
    bundle = json.loads(deps.objects.get(envelope.payload_locator))
    return eh.build_handoff(
        analysis_id, found, receiving_stage_run_id,
        (body["estimation_bundle"], bundle["compiled_design"], bundle["prepared_bundle"]),
        COMPLETE, deps.clock())


class EstimationNodes(eh.HarnessBase):
    # Frozen context, numerical plan, primary fit, evidence, and terminal outcome.

    # -- entry: the PRD-003 handoff and the approved-context manifest -------

    def entry_node(self, state: eh.EstimationState) -> dict[str, Any]:
        # Open the prepared handoff through the T-006 gate, verify the twelve §4 conditions, and
        # freeze the one authoritative approved-context surface (§4, §19.1).
        self.emit(state, "stage.started", eh.EVAL_STAGE)
        self.emit(state, "task.started", eh.EVAL_STAGE, status="entry")
        accepted, policy = False, None
        entries: tuple[ArtifactRef, ...] = ()
        try:
            inputs, entries, accepted = self.entry_inputs(state, self.deps.products.load_envelope(
                state["upstream"]["preparation_outcome"]))
            policy = plancompile.EntryPolicy(
                pack=self.deps.packs.get(str(inputs.design["method_id"]),
                                         str(inputs.design["method_pack_version"])),
                registry_versions=REGISTRY_VERSIONS, numerical_tolerances=TOLERANCES,
                approved_handling=eh.PREPARED_APPROVED_HANDLING)
            codes = plancompile.entry_codes(inputs, policy)
        except (persistence.PersistenceError, ec.EstimationError, KeyError, TypeError) as error:
            codes = (str(getattr(error, "code", plancompile.ENTRY_VALIDATION_FAILED)),)
        if (stop := self.gate(state, 1, handoff_accepted=accepted, entry_codes=codes)) is not None:
            self.emit(state, "task.failed", eh.EVAL_STAGE, severity=eh.ERROR,
                      error_code=str(stop.get("error_code")))
            return stop
        book = plancompile.compile_context_manifest(inputs, cast(plancompile.EntryPolicy, policy))
        self.commit(state, "EstimationContextManifest", book.canonical_payload(), tuple(
            self.deps.products.load_envelope(ref.artifact_id) for ref in entries))
        self.emit(state, "task.completed", eh.EVAL_STAGE, status="entry")
        return self.out(state, phase=eh.PHASES[1], row_set_hash=book.row_set_hash,
                        upstream=dict(state["upstream"]) | {
                            "prepared_bundle": entries[0].artifact_id})

    # -- the frozen plan and the exact §6.5 capacity recheck ----------------

    def plan_node(self, state: eh.EstimationState) -> dict[str, Any]:
        # The §6.1 plan, compiled deterministically and committed before any outcome is read,
        # then the exact §6.5 recheck against the frozen structure and planned cardinality.
        book = self.manifest(state)
        self.emit(state, "task.started", eh.EVAL_STAGE, status="plan")
        plan = plancompile.compile_plan(book, self.pack(book),
                                        self.ref(state, "EstimationContextManifest"))
        self.put(state, "EstimationPlan", plan, "EstimationContextManifest")
        if (stop := self.gate(state, 2)) is not None:
            return stop
        self.emit(state, "task.completed", eh.EVAL_STAGE, status="plan")
        return self.out(state, phase=eh.PHASES[2])

    # -- the one primary estimator (§9.1) -----------------------------------

    def estimate_node(self, state: eh.EstimationState) -> dict[str, Any]:
        # The registered contribution mask, the frozen numerical environment, and exactly one
        # adapter call. The estimator reads the outcome only after walls 1..5 pass (§19).
        book, plan = self.manifest(state), self.plan(state)
        if plan.method_id not in self.deps.adapters:
            return self.fail(state, ADAPTER_UNREGISTERED)
        view = self.view(state, book)
        self.frozen["estimator_input_types"] = eh.input_types(view, book.role_columns)
        if (stop := self.gate(state, 4)) is not None:
            return stop
        self.put(state, "NumericalEnvironmentManifest", self.environment(plan), "EstimationPlan")
        mask = self._mask(state, plan, view)
        if (stop := self.gate(state, 5)) is not None:
            return stop
        self.adapter = cast(engine.MethodAdapter, self.deps.adapters[plan.method_id](mask))
        self.emit(state, "tool.started", eh.EVAL_STAGE, status="estimate")
        try:
            fitted = self.adapter.fit(view, plan, self.pack(book))
        except (ec.EstimationError, ValueError, KeyError) as error:
            code = str(getattr(error, "code", PRIMARY_NOT_ESTIMABLE))
            self.emit(state, "tool.failed", eh.EVAL_STAGE, severity=eh.ERROR, error_code=code)
            return self.fail(state, code, NOT_ESTIMABLE)
        self.harvest = self.record_fit(state, plan, fitted)
        if not fitted.items:
            return self.fail(state, PRIMARY_NOT_ESTIMABLE, NOT_ESTIMABLE)
        return self._freeze_result(state, plan, fitted.items)

    def _mask(self, state: eh.EstimationState, plan: ec.EstimationPlanV1,
              view: pl.DataFrame) -> ArtifactRef:
        # §6.2: the primary mask declares which frozen rows contribute. It deletes nothing, the
        # denominators below stay the stabilized population, and the bit vector itself is a
        # restricted object that no envelope or event carries.
        rule, unit = plan.primary_mask_rule_id, plan.role_columns["unit_identifier"]
        bits = engine.mask_bits(rule, view, plan.role_columns, plan.estimator_parameters)
        locator = self.deps.frames.put_object(engine.mask_object_payload(rule, bits))
        mask = engine.contribution_mask(
            plan, rule, bits, ec.ObjectRefV1(object_locator=locator,
                                             content_hash=locator.split("/")[-1]),
            frame_row_set_hash=state["row_set_hash"], calculation_id="primary_analysis",
            parents=(self.ref(state, "EstimationPlan"),), unit_ids=view[unit])
        ref = self.put(state, "AnalysisContributionMask", mask, "EstimationPlan")
        self.index_mask(state, self.deps.products.load_envelope(ref.artifact_id), mask)
        self.denominators = {"row": view.height, "assigned": view.height,
                             "unit": view[unit].n_unique()}
        self.frozen |= {"masks": (mask,), "mask_refs": (ref,), "frozen_row_count": view.height}
        return ref

    def _freeze_result(self, state: eh.EstimationState, plan: ec.EstimationPlanV1,
                       items: tuple[ec.PrimaryContrastResultV1, ...]) -> dict[str, Any]:
        # One `PrimaryAnalysisResult` for the whole approved contrast family, with the mandatory
        # multiplicity result when more than one contrast is confirmatory (§6.3, §9.1). The
        # result is atomic: a contrast the estimator could not produce leaves it incomplete.
        plan_ref, order = self.ref(state, "EstimationPlan"), tuple(r.contrast_id for r in items)
        adjusted = None
        if plan.multiplicity_policy_id is not None and len(items) > 1:
            adjusted = self.put(state, "MultiplicityResult", ec.MultiplicityResultV1(
                parents=(plan_ref,), versions=dict(plan.versions), plan=plan_ref,
                policy_id=plan.multiplicity_policy_id,
                adjusted_by_contrast=cast(engine.MethodAdapter, self.adapter).multiplicity(
                    items, plan.confidence_level)), "EstimationPlan")
        result = ec.PrimaryAnalysisResultV1(
            parents=(plan_ref, self.ref(state, "AnalysisContributionMask")),
            versions=dict(plan.versions), plan=plan_ref, method_id=plan.method_id,
            estimator_id=plan.estimator_id, outcome_id=plan.outcome_id,
            estimand_family=plan.estimand_id, primary_items=items, contrast_order=order,
            multiplicity_result=adjusted, complete=order == plan.contrast_ids)
        found = self.put(state, "PrimaryAnalysisResult", result, "EstimationPlan",
                         "AnalysisContributionMask")
        self.frozen |= {"primary_result": result, "numerical_environment": self.ref(
            state, "NumericalEnvironmentManifest")}
        self.emit(state, "tool.completed", eh.EVAL_STAGE, status="estimate", artifact_refs=(found,))
        stop = self.gate(state, 8, eh.FAILED if result.complete else NOT_ESTIMABLE)
        return stop if stop is not None else self.out(
            state, phase=eh.PHASES[3],
            counts=dict(state["counts"]) | {"primary_items": len(items)})

    # -- diagnostics, sensitivities, and their two evidence bundles ---------

    def evidence_node(self, state: eh.EstimationState) -> dict[str, Any]:
        # §14 and §15 in registered order at concurrency one: every required diagnostic and
        # every prespecified branch reaches a visible terminal result, favorable or not.
        book, plan = self.manifest(state), self.plan(state)
        pack, view = self.pack(book), self.view(state, book)
        base = self.evidence_base(state, plan, self.denominators,
                                  self.frozen["masks"][0].mask_object.content_hash)
        self.emit(state, "task.started", eh.EVAL_EVIDENCE, status="diagnostics")
        rows = diagnostics.run_diagnostics(pack, self.harvest, base)
        state["diagnostic_ids"] = self._record(state, "DiagnosticResult", rows,
                                               [row.diagnostic_id for row in rows])
        self.frozen["diagnostics"] = rows
        if (stop := self.gate(state, 9)) is not None:
            return stop
        branches = engine.run_sensitivities(
            plan, pack, cast(engine.MethodAdapter, self.adapter), view,
            self.frozen["primary_result"].primary_items[0], base)
        state["sensitivity_ids"] = self._record(state, "SensitivityResult", branches,
                                                [row.branch_id for row in branches])
        self.frozen["sensitivities"] = branches
        self.emit(state, "task.completed", eh.EVAL_EVIDENCE, status="sensitivities")
        if (stop := self.gate(state, 11)) is not None:
            return stop
        self._bundle(state, "diagnostic", rows, [row.diagnostic_id for row in rows])
        self._bundle(state, "sensitivity", branches, [row.branch_id for row in branches])
        return self.out(state, phase=eh.PHASES[4], counts=dict(state["counts"]) | {
            "diagnostics": len(rows), "sensitivities": len(branches)})

    def _record(self, state: eh.EstimationState, kind: str, rows: Sequence[Any],
                names: Sequence[str]) -> list[str]:
        # Every evidence result is committed under its own artifact id and stays citable (§14.1).
        for row, name in zip(rows, names, strict=True):
            self.evidence[name] = self.put(
                state, kind, row, "EstimationPlan", "PrimaryAnalysisResult").artifact_id
        return [self.evidence[name] for name in names]

    def _bundle(self, state: eh.EstimationState, kind: ec.EvidenceKind, rows: Sequence[Any],
                names: Sequence[str]) -> ArtifactRef:
        # §26.2: one `EvidenceBundleV1` per kind, counting every result under one terminal status.
        plan_ref, refs = self.ref(state, "EstimationPlan"), self.refs_of(state)
        self.bundles[kind] = self.put(state, "EstimationEvidenceBundle", engine.evidence_bundle(
            self.plan(state), kind,
            [(refs[name], getattr(row, "execution_status", "computed"))
             for row, name in zip(rows, names, strict=True)],
            plan_ref=plan_ref, parents=(plan_ref,)), "EstimationPlan")
        return self.bundles[kind]

    # -- scientific support, numerical bundle, and terminal outcome --------

    def close_node(self, state: eh.EstimationState) -> dict[str, Any]:
        # The one terminal `EstimationOutcome`, its stage-run close, and — only for `complete` —
        # the §22 handoff. Every other status stops here with its typed reason (§5.2).
        status, code = state.get("status") or COMPLETE, state.get("error_code")
        if "EstimationContextManifest" not in state["artifacts"]:
            self.deps.products.transition_stage_run(state["stage_run_id"], "failed")
            self.emit(state, "stage.failed", eh.EVAL_CLOSE, severity=eh.ERROR, error_code=code)
            return self.out(state, status=status)
        if status == COMPLETE:
            status, code = self._finalise(state)
        built = self.put(state, "EstimationOutcome", ec.EstimationOutcomeV1(
            status=cast(ec.EstimationOutcomeStatus, status),
            context_manifest=self.ref(state, "EstimationContextManifest"),
            estimation_bundle=self.ref(state, "NumericalBundle") if status == COMPLETE else None,
            design_conflict=self.ref(state, "DesignConflict") if status == eh.CONFLICT else None,
            stage_run_id=state["stage_run_id"], graph_thread_id=state["thread_id"],
            error_code=code or None), "EstimationContextManifest",
            *(("NumericalBundle",) if status == COMPLETE else
              ("EstimationPlan",) if "EstimationPlan" in state["artifacts"] else ()))
        self.deps.products.transition_stage_run(state["stage_run_id"], "committing")
        self.deps.products.transition_stage_run(state["stage_run_id"], "completed")
        self.emit(state, "stage.completed", eh.EVAL_CLOSE, status=status, artifact_refs=(built,))
        if status != COMPLETE:
            return self.out(state, status=status, error_code=code or "")
        opened = open_post_analysis_handoff(self.deps, state["analysis_id"], built.artifact_id,
                                           f"sr:{state['analysis_id']}:presentation")
        return self.out(state, status=status, handoff_id=opened.handoff_id)

    def _finalise(self, state: eh.EstimationState) -> tuple[str, str | None]:
        """Seal the existing numerical results without claim or figure production."""
        book, plan = self.manifest(state), self.plan(state)
        plan_ref, primary_ref = self.ref(state, "EstimationPlan"), self.ref(
            state, "PrimaryAnalysisResult")
        unavailable: list[str] = []
        measurements: dict[str, ec.ValueMap] = {}
        for name, values in self.harvest.items():
            measurements[name] = {}
            for key, value in values.items():
                if isinstance(value, float) and not math.isfinite(value):
                    unavailable.append(f"{name}/{key}")
                    value = None
                measurements[name][key] = value
        supporting = self.put(state, "AnalysisSupportingData", ec.AnalysisSupportingDataV1(
            parents=(plan_ref, primary_ref), versions=dict(plan.versions),
            plan=plan_ref, primary_result=primary_ref, measurements=measurements,
            unavailable_values=tuple(sorted(unavailable)), contributing_counts=self.denominators,
            contribution_mask_hash=self.frozen["masks"][0].mask_object.content_hash),
            "EstimationPlan", "PrimaryAnalysisResult")
        environment = self.ref(state, "NumericalEnvironmentManifest")
        multiplicity = self.ref(state, "MultiplicityResult") if (
            "MultiplicityResult" in state["artifacts"]) else None
        parents = (self.ref(state, "EstimationContextManifest"), plan_ref, primary_ref,
                   *self.bundles.values(), supporting, book.compiled_design,
                   book.prepared_bundle, environment, *self.frozen["mask_refs"], *self.dealt,
                   *((multiplicity,) if multiplicity is not None else ()))
        bundle = ec.NumericalBundleV1(
            parents=parents, versions=dict(plan.versions),
            compiled_design=book.compiled_design, prepared_bundle=book.prepared_bundle,
            row_set_hash=book.row_set_hash, plan=plan_ref,
            context_manifest=self.ref(state, "EstimationContextManifest"),
            contribution_masks=self.frozen["mask_refs"], cross_fit_assignments=self.dealt,
            primary_result=primary_ref,
            multiplicity_result=multiplicity,
            evidence_bundles=tuple(self.bundles[kind] for kind in EVIDENCE_KINDS),
            supporting_data=supporting,
            numerical_environment=environment)
        self.commit(state, "NumericalBundle", bundle.canonical_payload(),
                    tuple(self.deps.products.load_envelope(ref.artifact_id) for ref in parents))
        return COMPLETE, None


def run_estimation(deps: eh.EstimationDeps, *, analysis_id: str,
                   preparation_outcome_artifact_id: str, stage_run_id: str,
                   estimation_revision: int = 1) -> eh.EstimationRunResult:
    # One estimation revision, run straight through; a terminal status short-circuits to the
    # close (§7, §26.1). A rerun is artifact replay (D-035): a NEW `stage_run_id`, deterministic
    # artifact ids that make every recommit a no-op, no checkpoint, and no skip logic.
    state = eh.new_state(analysis_id, stage_run_id, estimation_revision,
                         {"preparation_outcome": preparation_outcome_artifact_id})
    nodes = EstimationNodes(deps)
    nodes.open_run(state, deps.products.load_envelope(
        preparation_outcome_artifact_id).stage_run_id)
    for name in NODES:
        state = cast(eh.EstimationState, getattr(nodes, name)(state))
        if state.get("status"):
            break
    return nodes.close_run(cast(eh.EstimationState, nodes.close_node(state)))
