# The six §7 estimation nodes, the plain sequential coordinator, and the §22 PRD-005 handoff
# (PRD-004 §5.2, §7, §14–§17, §19.1, §20.5, §22, §26.1). Amendment 1 makes this stage a
# sequential coordinator: no checkpointer, no StateGraph, no interrupt, and one model call —
# the §16.2 claim review. Every upstream artifact is read as data: this module never imports
# `causal.design` or `causal.preparation`, and it knows no method pack by name.

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Final, cast

import polars as pl

from causal.estimation import contracts as ec
from causal.estimation import engine, plancompile
from causal.estimation import harness as eh
from causal.estimation import judge as ej
from causal.estimation import walls as ew
from causal.shared import persistence
from causal.shared.agenttask import TaskRunner
from causal.shared.contracts import ArtifactRef, HandoffManifestV1
from causal.shared.validation import ValidationReport

COMPLETE: Final = "complete"
NOT_ESTIMABLE: Final = "not_estimable"
ADAPTER_UNREGISTERED: Final = "estimator_adapter_unregistered"
PRIMARY_NOT_ESTIMABLE: Final = "primary_result_incomplete"
CLAIM_UNAVAILABLE: Final = "claim_judgment_unavailable"
EVAL_CLAIM: Final = ("EV-P4-008",)
# The five §7 nodes before the close; a terminal status short-circuits straight to it.
NODES: Final = ("entry_node", "plan_node", "estimate_node", "evidence_node", "judgment_node")
# The closed §6.1/§19.1 registry vector this stage pins on every manifest, plan, and result.
REGISTRY_VERSIONS: Final[dict[str, str]] = {
    "artifact_types": "artifact-types.v1", "method_packs": "method-packs.v1",
    "estimation_packs": "method-pack-estimation.v1", "diagnostics": "estimation-diagnostics.v1",
    "sensitivities": "estimation-sensitivities.v1", "figure_builders": "estimation-figures.v1",
    "capacity": "delivery-capacity.v1", "visualization_catalog": "visualization-catalog.v1",
    "schema": "estimation-schemas.v1", "validators": ew.REGISTRY_VERSION}
# §6.1: the comparison tolerances, fixed in the plan before any result is read.
TOLERANCES: Final[dict[str, float]] = {"sensitivity_magnitude": 0.25, "interval_width": 0.5}
# §26.2: one evidence bundle per kind, committed and referenced in exactly this order.
EVIDENCE_KINDS: Final[tuple[ec.EvidenceKind, ...]] = ("diagnostic", "sensitivity", "figure_data")
# The §5.1 bundle's committed parents, beside the four approved upstream ones it also names.
BUNDLE_PARENTS: Final = ("EstimationContextManifest", "EstimationPlan", "PrimaryAnalysisResult",
                         "EstimationEvidenceBundle", "JudgmentCeiling", "ClaimJudgment",
                         "NumericalEnvironmentManifest")


def open_presentation_handoff(deps: eh.EstimationDeps, analysis_id: str, outcome_artifact_id: str,
                              receiving_stage_run_id: str) -> HandoffManifestV1:
    # §22: the five ids PRD-005 opens with — the estimation bundle, the claim judgment, the
    # figure-data bundle, the approved design, and the exact passing capacity recheck — rebuilt
    # from committed artifacts and never recorded here. Only a `complete` outcome is readable.
    found = deps.products.load_envelope(outcome_artifact_id)
    body = json.loads(deps.objects.get(found.payload_locator))
    if found.artifact_type != "EstimationOutcome" or body["status"] != COMPLETE:
        raise persistence.PersistenceError(
            f"no readable estimation handoff for {analysis_id!r}", "handoff_unavailable")
    bundle = json.loads(deps.objects.get(deps.products.load_envelope(
        str(body["estimation_bundle"]["artifact_id"])).payload_locator))
    return eh.build_handoff(
        analysis_id, found, receiving_stage_run_id,
        (body["estimation_bundle"], bundle["claim_judgment"],
         bundle["evidence_bundles"][EVIDENCE_KINDS.index("figure_data")],
         bundle["experiment_design"], bundle["capacity_check"]), COMPLETE, deps.clock())


def _unbound(*args: Any, **over: Any) -> ValidationReport:
    # §16.3: the judge binds its own claim validator; nothing else may answer wall 13.
    raise ec.EstimationError("wall 13 has no claim validator", ew.CLAIM_VALIDATOR_MISSING)


class EstimationNodes(eh.HarnessBase):
    # Entry and the approved-context manifest, the frozen plan and its capacity recheck, the one
    # primary fit, the evidence fan-in, the §16 judgment, and the terminal outcome.

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
                approved_handling=eh.PREPARED_APPROVED_HANDLING,
                recipient_map={ej.TASK_KIND: (ej.SCOPE_KIND,)})
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
        return self.out(state, phase=eh.PHASES[1], row_set_hash=book.row_set_hash, seed=book.seed,
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
        self.frozen["capacity_conflict"] = plancompile.recheck_capacity(
            plan, plancompile.PreparedStructureV1(
                cardinalities=dict(book.result_cardinalities) | {
                    "contrasts": len(plan.contrast_ids)},
                required_visual_evidence=self._visual_evidence(book),
                registry_path=self.deps.registries / "delivery-capacity.v1.json"))
        if (stop := self.gate(state, 3)) is None:
            self.emit(state, "task.completed", eh.EVAL_STAGE, status="plan")
            return self.out(state, phase=eh.PHASES[2])
        conflict = self.frozen["capacity_conflict"]
        return stop if conflict is None else self.conflict(state, conflict)

    def _visual_evidence(self, book: ec.EstimationContextManifestV1) -> tuple[str, ...]:
        # The approved method pack's required visual-evidence ids, read from the frozen design
        # registry as raw data; PRD-002's models are never imported.
        document = json.loads(
            (self.deps.registries / "method-packs.v1.json").read_text(encoding="utf-8"))
        return next((tuple(str(name) for name in row.get("required_visual_evidence_ids") or ())
                     for row in document["packs"] if row["method_id"] == book.method_id), ())

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
        rows = engine.run_diagnostics(pack, self.harvest, base)
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

    # -- the deterministic ceiling and the one claim-review call ------------

    def judgment_node(self, state: eh.EstimationState) -> dict[str, Any]:
        # §16.1 first: the ceiling is fixed before any model-authored interpretation and nobody
        # may raise it. Then the ONE §16.2 call, rechecked by the deterministic §16.3 wall 13.
        book, plan = self.manifest(state), self.plan(state)
        plan_ref, result_ref = self.ref(state, "EstimationPlan"), self.ref(
            state, "PrimaryAnalysisResult")
        self.frozen["ceiling"] = ceiling = engine.judgment_ceiling(
            plan, self.frozen["primary_result"], self.frozen["diagnostics"], self.refs_of(state),
            plan_ref=plan_ref, primary_ref=result_ref, parents=(plan_ref,),
            approved_handling=eh.APPROVED_HANDLING)
        ceiling_ref = self.put(state, "JudgmentCeiling", ceiling, "EstimationPlan")
        if (stop := self.gate(state, 12)) is not None:
            return stop
        seen = ej.review_context(
            book, ceiling, self.frozen["primary_result"],
            self.payload(book.experiment_design.artifact_id), self.frozen["diagnostics"],
            self.frozen["sensitivities"], population_summary=self.denominators,
            refs=dict(self.evidence) | {"plan": plan_ref.artifact_id, "ceiling":
                                        ceiling_ref.artifact_id, "result": result_ref.artifact_id})
        self.emit(state, "task.started", EVAL_CLAIM, status="claim_review")
        found = self._judge(state, seen, (plan_ref, ceiling_ref, result_ref))
        if found.judgment is None:
            return self.fail(state, found.error_code or CLAIM_UNAVAILABLE)
        self.emit(state, "task.completed", EVAL_CLAIM, status=found.judgment.status)
        # Walls 14 and 15 replay wall 13, so the committed judgment keeps answering it (§18).
        self.frozen["claim_validator"] = ej.claim_validator(
            {name: value for name, value in found.judgment.model_dump(mode="json").items()
             if name in ej.ClaimJudgmentDraftV1.model_fields},
            frozenset(seen.allowed_artifact_ids), seen.required_qualifications)
        self.claim_status, status = found.judgment.status, found.judgment.outcome_status
        return self.out(state, phase=eh.PHASES[5], overall_ceiling=ceiling.overall_ceiling,
                        status="" if status == COMPLETE else status,
                        error_code="" if status == COMPLETE else found.judgment.status)

    def _judge(self, state: eh.EstimationState, seen: ej.ClaimReviewContextV1,
               refs: tuple[ArtifactRef, ArtifactRef, ArtifactRef]) -> ej.ClaimJudgmentOutcome:
        # SC §5.4: the one estimation model receiver, wired to this harness's own hooks. The
        # lineage below is the harness's; nothing the model returns can decide it (D-071).
        plan_ref, ceiling_ref, result_ref = refs
        task_state: dict[str, Any] = {
            "analysis_id": state["analysis_id"], "stage_run_id": state["stage_run_id"],
            "design_revision": state["estimation_revision"], "corrections": {},
            "open_requirement_ids": []}
        runner = TaskRunner(
            gateway=cast(Any, self.deps.gateway), tasks={ej.TASK_KIND: ej.CLAIM_TASK},
            tools={ej.TASK_KIND: ()}, evals={ej.TASK_KIND: EVAL_CLAIM}, validate=_unbound,
            prompts_root=self.deps.repo_root, envelope=ej.build_claim_envelope,
            prompt=ej.render_claim_prompt, upsert=lambda *args: None,
            context=lambda inner: self.context(state),
            manifest=lambda inner: self.ref(state, "EstimationContextManifest"),
            evidence=lambda inner: frozenset(seen.allowed_artifact_ids),
            parents=lambda inner, *kinds: self.parents(state, *kinds),
            commit=lambda inner, kind, body, parents: self.commit(state, kind, body, parents),
            emit=lambda inner, name, evals, **over: self.emit(state, name, evals, **over),
            record=lambda inner, task_id, *rest: state["task_ids"].append(task_id),
            exhausted=lambda inner, task_id, kind, issues: self.emit(
                state, "blocker.raised", EVAL_CLAIM, severity=eh.ERROR,
                error_code=ej.CORRECTION_EXHAUSTED))
        return ej.judge(runner, task_state, seen, ctx=self.context(state),
                        run_walls=lambda highest, walls: self.wall(state, highest, walls),
                        lineage={"parents": [ref.model_dump(mode="json")
                                             for ref in (plan_ref, ceiling_ref)],
                                 "judgment_ceiling": ceiling_ref.model_dump(mode="json"),
                                 "primary_result": result_ref.model_dump(mode="json"),
                                 "supporting_refs": [ref.model_dump(mode="json") for ref
                                                     in self.refs_of(state).values()]})

    # -- figure data, the §5.1 bundle, and the terminal outcome -------------

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
            estimation_bundle=self.ref(state, "EstimationBundle") if status == COMPLETE else None,
            design_conflict=self.ref(state, "DesignConflict") if status == eh.CONFLICT else None,
            stage_run_id=state["stage_run_id"], graph_thread_id=state["thread_id"],
            error_code=code or None), "EstimationContextManifest")
        self.deps.products.transition_stage_run(state["stage_run_id"], "committing")
        self.deps.products.transition_stage_run(state["stage_run_id"], "completed")
        self.emit(state, "stage.completed", eh.EVAL_CLOSE, status=status, artifact_refs=(built,))
        if status != COMPLETE:
            return self.out(state, status=status, error_code=code or "")
        opened = open_presentation_handoff(self.deps, state["analysis_id"], built.artifact_id,
                                           f"sr:{state['analysis_id']}:presentation")
        return self.out(state, status=status, handoff_id=opened.handoff_id)

    def _finalise(self, state: eh.EstimationState) -> tuple[str, str | None]:
        # §17 then §5.1: the frozen figure-ready data, its bundle, and the one estimation bundle
        # every PRD-005 identity hangs off. Nothing here estimates or discloses above the claim.
        book, plan = self.manifest(state), self.plan(state)
        figures = self._figures(state, plan)
        if (stop := self.gate(state, 14)) is not None:
            return eh.FAILED, str(stop.get("error_code"))
        self._bundle(state, "figure_data", figures, [row.builder_id for row in figures])
        upstream = (book.experiment_design, book.runnable_frame_contract, book.prepared_bundle,
                    book.capacity_check)
        self.frozen["bundle"] = bundle = ec.EstimationBundleV1(
            parents=(self.ref(state, "EstimationContextManifest"),
                     self.ref(state, "EstimationPlan")), versions=dict(plan.versions),
            experiment_design=book.experiment_design, prepared_bundle=book.prepared_bundle,
            runnable_frame_contract=book.runnable_frame_contract, row_set_hash=book.row_set_hash,
            capacity_check=book.capacity_check, plan=self.ref(state, "EstimationPlan"),
            context_manifest=self.ref(state, "EstimationContextManifest"),
            contribution_masks=self.frozen["mask_refs"], cross_fit_assignments=self.dealt,
            primary_result=self.ref(state, "PrimaryAnalysisResult"),
            multiplicity_result=self.ref(state, "MultiplicityResult")
            if "MultiplicityResult" in state["artifacts"] else None,
            evidence_bundles=tuple(self.bundles[kind] for kind in EVIDENCE_KINDS),
            judgment_ceiling=self.ref(state, "JudgmentCeiling"),
            claim_judgment=self.ref(state, "ClaimJudgment"),
            numerical_environment=self.ref(state, "NumericalEnvironmentManifest"))
        self.commit(state, "EstimationBundle", bundle.canonical_payload(),
                    self.parents(state, *BUNDLE_PARENTS) + tuple(
                        self.deps.products.load_envelope(ref.artifact_id) for ref in upstream))
        stop = self.gate(state, 15)
        return (COMPLETE, None) if stop is None else (eh.FAILED, str(stop.get("error_code")))

    def _figures(self, state: eh.EstimationState,
                 plan: ec.EstimationPlanV1) -> tuple[ec.FigureDataArtifactV1, ...]:
        # §17: one frozen dataset per required builder, disclosed at the claim's own status and
        # built out of values this run already committed.
        adapter = cast(engine.MethodAdapter, self.adapter)
        builders, rows = adapter.figures(self.harvest), []
        parents = (self.ref(state, "EstimationPlan"), self.ref(state, "PrimaryAnalysisResult"))
        for builder_id in plan.figure_builder_ids:
            self.emit(state, "tool.started", eh.EVAL_FIGURE, status=builder_id)
            found = engine.figure_data(
                plan, builder_id, builders[builder_id], self.frozen["primary_result"],
                visual_evidence_id=adapter.visual_evidence(builder_id),
                disclosure=self.claim_status, parents=parents, counts=self.denominators,
                mask_hash=self.frozen["masks"][0].mask_object.content_hash)
            self.evidence[builder_id] = self.put(
                state, "FigureDataArtifact", found, "EstimationPlan",
                "PrimaryAnalysisResult").artifact_id
            state["figure_ids"].append(self.evidence[builder_id])
            self.emit(state, "tool.completed", eh.EVAL_FIGURE, status=builder_id)
            rows.append(found)
        self.frozen["figures"] = tuple(rows)
        return tuple(rows)


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
