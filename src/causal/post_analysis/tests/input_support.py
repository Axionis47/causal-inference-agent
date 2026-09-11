"""Content-addressed, in-memory scientific handoffs for receiver boundary tests."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from causal.analysis.integration import contracts as ac
from causal.analysis.integration.tests.support.plans import manifest, payloads, plan
from causal.analysis.integration.tests.support.walls import (
    diagnostic,
    item,
    mask,
    result,
    sensitivity,
)
from causal.post_analysis.input import load_packet
from causal.shared.canonical import canonical_bytes, content_hash
from causal.shared.contracts import ArtifactRef
from causal.shared.registry import load_artifact_type_registry

REGISTRY = load_artifact_type_registry(Path(__file__).resolve().parents[4] / "registries/artifact-types.v1.json")


class Sources:
    def __init__(self) -> None:
        self.envelopes: dict[str, Any] = {}
        self.objects: dict[str, bytes] = {}
        self.refs: dict[str, ArtifactRef] = {}

    def put(self, kind: str, body: Any, *, parents: tuple[ArtifactRef, ...] = (),
            key: str | None = None) -> ArtifactRef:
        row = REGISTRY.lookup(kind)
        payload = body.canonical_payload() if hasattr(body, "canonical_payload") else {
            "schema_version": row.schema_version, **body}
        digest = content_hash(payload)
        identity = key or kind
        artifact_id = f"{identity}:{digest[:16]}"
        ref = ArtifactRef(artifact_id=artifact_id, content_hash=digest)
        self.envelopes[artifact_id] = SimpleNamespace(
            artifact_id=artifact_id, artifact_type=kind, schema_version=payload["schema_version"],
            analysis_id="study", content_hash=digest, payload_locator=artifact_id,
            producer_component=row.producer_component, parent_artifacts=parents)
        self.objects[artifact_id] = canonical_bytes(payload)
        self.refs[identity] = ref
        return ref

    def packet(self) -> Any:
        return load_packet(SimpleNamespace(load_envelope=self.envelopes.__getitem__),
                           SimpleNamespace(get=self.objects.__getitem__), REGISTRY,
                           "study", self.refs["EstimationOutcome"])


def handoff(method: str = "randomized_experiment", *, coverage: str = "complete",
            failed: bool = False, attempted_plan: bool = True,
            wrong_role: bool = False, incompatible_branch: bool = False,
            concept_count: int = 2) -> Sources:
    store = Sources()
    pack_version = f"{method.replace('_', '-')}-pack.v1"
    roles = {"outcome": "finished", "treatment": "arm", "unit_identifier": "person"}
    approved = dict(roles)
    structure: dict[str, Any] = {}
    grain = "one_row_per_unit"
    if method == "aipw":
        roles["adjustment_covariate"] = "age"
        approved["confounder_candidate"] = "age"
    elif method == "did":
        roles.update(group="cohort", unit_identifier="cohort", time="wave")
        approved.pop("unit_identifier")
        approved.update(group="cohort", time="wave")
        grain = "one_row_per_group_time"
        structure = {"adoption_time": 2}
    elif method == "sharp_rdd":
        roles.update(running_variable="score", predetermined_covariate="age")
        approved.update(running_variable="score", precision_covariate="age")
        structure = {"cutoff": 0.0, "assignment_direction": "above"}
    card = store.put("ColumnSemanticCard", {"column_name": "finished", "label": "Finished"})
    measures = {"finished": {"label": "Finished", "units": "proportion", "scale": "binary",
                             "source_card": card.model_dump(), "supporting_evidence_ids": []}}
    design = payloads()["design"] | {
        "method_id": method, "method_pack_version": pack_version, "role_bindings": [
            {"role": role, "columns": [column]} for role, column in approved.items()],
        "column_measurements": measures, "causal_question": "Does the offer change completion?",
        "preparation": {"output_grain": grain, "method_structure": structure},
        "identification_risks": ["Unmeasured causes may remain."]}
    design_ref = store.put("CompiledDesign", design)
    concepts = ["offer", "completion", *(f"context_{index}" for index in range(concept_count - 2))]
    dag = store.put("CausalContext", {"frame": design["frame"], "concept_ids": concepts,
                                      "edges": [{"edge_id": "offer-completion", "source_concept_id": "offer",
                                                 "target_concept_id": "completion", "status": "hypothesis",
                                                 "supporting_evidence_ids": [], "contrary_evidence_ids": []}]})
    measurement = store.put("MeasurementMap", {"concepts": [
        {"concept_id": concept, "name": concept.title()} for concept in concepts]})
    view = store.put("CausalGraphView", {"parents": [source.model_dump() for source in (
        design_ref, dag, measurement)], "svg": "upstream layout must not reach the author"})
    views = store.put("GraphViewSet", {"base_view": view.model_dump(), "alternative_views": []})
    review = store.put("DesignReviewBundle", {"compiled_design": design_ref.model_dump(),
                                               "graph_views": views.model_dump()})
    approval = store.put("DesignApproval", {"decision": "approved", "review_bundle": review.model_dump(),
                                            "approved_bundle_hash": review.content_hash})
    frame = store.put("PreparedFrame", {"columns": [{"column_name": name} for name in sorted(set(roles.values()))]})
    prepared = store.put("PreparedFrameBundle", {"compiled_design": design_ref.model_dump(),
                                                 "design_approval": approval.model_dump(),
                                                 "prepared_frame": frame.model_dump(),
                                                 "row_set_hash": manifest().row_set_hash})
    wanted = "future_unknown_diagnostic"
    measurements = {name: ac.ColumnMeasurementV1.model_validate_json(json.dumps(row))
                    for name, row in measures.items()}
    selected = {"method_id": method, "method_pack_version": pack_version, "role_columns": roles,
                "column_measurements": measurements, "required_sensitivity_ids": ("alternative",)}
    context = manifest().model_copy(update=selected | {"compiled_design": design_ref,
        "prepared_bundle": prepared, "method_structure": structure})
    context_ref = store.put("EstimationContextManifest", context)
    actual_roles = roles | ({"outcome": "arm"} if wrong_role else {})
    request = plan().model_copy(update=selected | {"context_manifest": context_ref,
        "role_columns": actual_roles, "required_diagnostics": {wanted: "qualification_guard"},
        "estimator_parameters": plan().estimator_parameters | structure})
    plan_ref = store.put("EstimationPlan", request, parents=(context_ref,))
    if failed:
        store.put("EstimationOutcome", ac.EstimationOutcomeV1(
            status="not_estimable", context_manifest=context_ref, estimation_bundle=None,
            design_conflict=None, stage_run_id="stage", graph_thread_id="thread",
            error_code="approved_contrast_without_support"),
            parents=(context_ref, plan_ref) if attempted_plan else (context_ref,))
        return store
    mask_ref = store.put("AnalysisContributionMask", mask(request.primary_mask_rule_id))
    primary_row = item().model_copy(update={"contribution_mask": mask_ref})
    primary = result((primary_row,)).model_copy(update={"plan": plan_ref, "method_id": method})
    primary_ref = store.put("PrimaryAnalysisResult", primary)
    env = store.put("NumericalEnvironmentManifest", ac.NumericalEnvironmentManifestV1(
        python_version="3.12", package_versions={}, platform="test", seeds={"plan": request.seed},
        parallelism={"threads": 1}, float_dtype="float64", serialization_policy="canonical-json.v1",
        numerical_tolerances=request.numerical_tolerances, build_identifier="test", runtime_image_digest=None))
    shared = {"plan": plan_ref, "primary_result": primary_ref, "numerical_environment": env}
    check = diagnostic(next(iter(plan().required_diagnostics))).model_copy(update=shared | {
        "diagnostic_id": wanted, "severity": "qualification_guard", "values": {"recorded_metric": 0.3},
        "execution_status": "failed" if coverage == "failed" else "computed"})
    diag = store.put("DiagnosticResult", check)
    refs = () if coverage == "missing" else ((diag, diag) if coverage == "duplicate" else (diag,))
    diagnostics = store.put("EstimationEvidenceBundle", ac.EvidenceBundleV1(
        parents=(plan_ref,), versions={}, kind="diagnostic", plan=plan_ref, results=refs,
        terminal_status_counts={check.execution_status: len(refs)}), key="diagnostics")
    branch_row = primary_row.model_copy(update={"estimate": 0.25,
                                               **({"estimate_units": "other"} if incompatible_branch else {})})
    branch = store.put("SensitivityResult", sensitivity("alternative").model_copy(
        update=shared | {"result": branch_row, "values": {}}))
    sensitivities = store.put("EstimationEvidenceBundle", ac.EvidenceBundleV1(
        parents=(plan_ref,), versions={}, kind="sensitivity", plan=plan_ref, results=(branch,),
        terminal_status_counts={"computed": 1}), key="sensitivities")
    support = store.put("AnalysisSupportingData", ac.AnalysisSupportingDataV1(
        parents=(plan_ref, primary_ref), versions={}, plan=plan_ref, primary_result=primary_ref,
        measurements={"balance": {"smd/age": 0.125}}, contributing_counts={"row": 90},
        contribution_mask_hash=None))
    bundle = store.put("NumericalBundle", ac.NumericalBundleV1(
        parents=(plan_ref,), versions={}, compiled_design=design_ref, prepared_bundle=prepared,
        row_set_hash=request.row_set_hash, context_manifest=context_ref, plan=plan_ref,
        contribution_masks=(mask_ref,), cross_fit_assignments=(), primary_result=primary_ref,
        multiplicity_result=None, evidence_bundles=(diagnostics, sensitivities),
        supporting_data=support, numerical_environment=env))
    store.put("EstimationOutcome", ac.EstimationOutcomeV1(
        status="complete", context_manifest=context_ref, estimation_bundle=bundle, design_conflict=None,
        stage_run_id="stage", graph_thread_id="thread", error_code=None), parents=(context_ref, bundle))
    return store
