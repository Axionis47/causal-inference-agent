"""Validate the exact numerical handoff and project evidence for post-analysis."""
from __future__ import annotations

from collections import Counter
from typing import Any

from causal.analysis.integration import contracts as ac
from causal.post_analysis.contracts import EvidencePacket
from causal.post_analysis.input_sources import _bind_request, _design, _equal, _Reader, _reject
from causal.post_analysis.input_tables import EvidenceRows, _evidence_tables
from causal.shared.contracts import ArtifactRef


def _plan_evidence(plan: ac.EstimationPlanV1) -> dict[str, Any]:
    return {key: value for key, value in plan.canonical_payload().items() if key not in {
        "figure_builder_ids", "capacity_report", "versions"}}


def _pin_sources(reader: _Reader, sources: dict[str, ArtifactRef]) -> None:
    for held in reader.cache:
        sources.setdefault(f"source:{held.artifact_id}", held)


def _numerical_lineage(reader: _Reader, bundle: ac.NumericalBundleV1,
                       plan: ac.EstimationPlanV1) -> None:
    environment = reader.model(bundle.numerical_environment, "NumericalEnvironmentManifest",
                               ac.NumericalEnvironmentManifestV1)
    _equal(plan.seed, environment.seeds.get("plan"), bundle.numerical_environment, "/seeds/plan")
    for ref in bundle.contribution_masks:
        mask = reader.model(ref, "AnalysisContributionMask", ac.AnalysisContributionMaskV1)
        _equal(bundle.row_set_hash, mask.parent_row_set_hash, ref, "/parent_row_set_hash")
        _equal(plan.outcome_id, mask.outcome_id, ref, "/outcome_id")
    for ref in bundle.cross_fit_assignments:
        assignment = reader.model(ref, "CrossFitAssignment", ac.CrossFitAssignmentV1)
        _equal(bundle.plan, assignment.plan, ref, "/plan")
        _equal(plan.fold_count, assignment.fold_count, ref, "/fold_count")


def _failure_packet(reader: _Reader, outcome: ArtifactRef, receipt: ac.EstimationOutcomeV1,
                    manifest: ac.EstimationContextManifestV1) -> EvidencePacket:
    envelope, _ = reader.read(outcome, "EstimationOutcome")
    attempts = [ref for ref in envelope.parent_artifacts
                if reader.read(ref, None)[0].artifact_type == "EstimationPlan"]
    if len(attempts) != 1:
        _reject("attempt_request_unavailable", "analysis", outcome, "/parent_artifacts",
                "one exact attempted plan reference", [ref.model_dump() for ref in attempts],
                "Supply the attempted request with its failure receipt; do not invent outcomes.")
    plan_ref = attempts[0]
    plan = reader.model(plan_ref, "EstimationPlan", ac.EstimationPlanV1)
    sources = {"failure": outcome, "plan": plan_ref}
    evidence = {"failure": receipt.canonical_payload(), "plan": _plan_evidence(plan)}
    design, prepared, diagram = _design(reader, manifest, sources, evidence)
    _bind_request(reader, receipt.context_manifest, plan_ref, manifest, plan, design, prepared)
    _pin_sources(reader, sources)
    limitation = (f"The numerical attempt ended with status {receipt.status} "
                  f"({receipt.error_code or 'no error code supplied'}). No numerical response was "
                  "supplied; requested computations have no reported outcomes.")
    return EvidencePacket(reader.analysis_id, sources, evidence, {}, ("design", "plan", "failure"),
                          diagram, (limitation,))


def _primary(reader: _Reader, bundle: ac.NumericalBundleV1,
             plan: ac.EstimationPlanV1) -> ac.PrimaryAnalysisResultV1:
    primary: ac.PrimaryAnalysisResultV1 = reader.model(
        bundle.primary_result, "PrimaryAnalysisResult", ac.PrimaryAnalysisResultV1)
    for name, expected in (("plan", bundle.plan), ("method_id", plan.method_id),
                           ("estimator_id", plan.estimator_id),
                           ("outcome_id", plan.outcome_id), ("estimand_family", plan.estimand_id),
                           ("contrast_order", plan.contrast_ids), ("complete", True)):
        _equal(expected, getattr(primary, name), bundle.primary_result, f"/{name}")
    for index, row in enumerate(primary.primary_items):
        for name, expected in (("estimand_id", plan.estimand_id), ("comparator_id", plan.comparator_id),
                               ("estimator_id", plan.estimator_id), ("estimator_version", plan.estimator_version)):
            _equal(expected, getattr(row, name), bundle.primary_result, f"/primary_items/{index}/{name}")
        if row.contribution_mask not in bundle.contribution_masks:
            _reject("source_binding_mismatch", "analysis", bundle.primary_result,
                    f"/primary_items/{index}/contribution_mask", bundle.contribution_masks,
                    row.contribution_mask, "Supply the exact contributing-mask lineage.")
        reader.read(row.contribution_mask, "AnalysisContributionMask")
    return primary


def _collections(reader: _Reader, bundle: ac.NumericalBundleV1,
                 plan: ac.EstimationPlanV1) -> EvidenceRows:
    rows: EvidenceRows = {}
    for index, kind in enumerate(("diagnostic", "sensitivity")):
        collection_ref = bundle.evidence_bundles[index]
        collection = reader.model(collection_ref, "EstimationEvidenceBundle", ac.EvidenceBundleV1)
        _equal(kind, collection.kind, collection_ref, "/kind")
        _equal(bundle.plan, collection.plan, collection_ref, "/plan")
        expected = set(plan.required_diagnostics if kind == "diagnostic" else plan.required_sensitivity_ids)
        received: list[str] = []
        statuses: Counter[str] = Counter()
        for ref in collection.results:
            model = ac.DiagnosticResultV1 if kind == "diagnostic" else ac.SensitivityResultV1
            row = reader.model(ref, "DiagnosticResult" if kind == "diagnostic" else "SensitivityResult", model)
            name = row.diagnostic_id if kind == "diagnostic" else row.branch_id
            received.append(name)
            statuses[row.execution_status] += 1
            _equal(bundle.plan, row.plan, ref, "/plan")
            _equal(bundle.primary_result, row.primary_result, ref, "/primary_result")
            _equal(bundle.numerical_environment, row.numerical_environment, ref, "/numerical_environment")
            key = f"{kind}:{name}"
            rows[key] = ref, row
            if kind == "diagnostic":
                _equal(plan.required_diagnostics.get(name), row.severity, ref, "/severity")
            if isinstance(row, ac.SensitivityResultV1) and row.result is not None:
                reader.read(row.result.contribution_mask, "AnalysisContributionMask")
        _equal(sorted(expected), sorted(received), collection_ref, "/results")
        _equal(dict(statuses), {key: value for key, value in collection.terminal_status_counts.items() if value},
               collection_ref, "/terminal_status_counts")
    return rows


def load_packet(products: Any, objects: Any, registry: Any, analysis_id: str,
                outcome: ArtifactRef) -> EvidencePacket:
    """Load the frozen numerical-only handoff; never query a latest design or rerun analysis."""
    reader = _Reader(products, objects, registry, analysis_id)
    receipt = reader.model(outcome, "EstimationOutcome", ac.EstimationOutcomeV1)
    manifest = reader.model(receipt.context_manifest, "EstimationContextManifest",
                            ac.EstimationContextManifestV1)
    if receipt.estimation_bundle is None:
        return _failure_packet(reader, outcome, receipt, manifest)
    bundle = reader.model(receipt.estimation_bundle, "NumericalBundle", ac.NumericalBundleV1)
    _equal(receipt.context_manifest, bundle.context_manifest, receipt.estimation_bundle,
           "/context_manifest")
    plan = reader.model(bundle.plan, "EstimationPlan", ac.EstimationPlanV1)
    primary = _primary(reader, bundle, plan)
    sources = {"outcome": outcome, "plan": bundle.plan, "primary": bundle.primary_result,
               "support": bundle.supporting_data}
    evidence = {"outcome": receipt.canonical_payload(), "primary": primary.canonical_payload(),
                "plan": _plan_evidence(plan)}
    for name in ("compiled_design", "prepared_bundle", "row_set_hash"):
        _equal(getattr(manifest, name), getattr(bundle, name), receipt.estimation_bundle, f"/{name}")
    design, prepared, diagram = _design(reader, manifest, sources, evidence)
    _bind_request(reader, receipt.context_manifest, bundle.plan, manifest, plan, design, prepared)
    required = ["design", "plan", "primary"]
    rows = _collections(reader, bundle, plan)
    sources.update({key: ref for key, (ref, _) in rows.items()})
    evidence.update({key: row.canonical_payload() for key, (_, row) in rows.items()})
    required.extend(rows)
    support = reader.model(bundle.supporting_data, "AnalysisSupportingData", ac.AnalysisSupportingDataV1)
    _equal(bundle.plan, support.plan, bundle.supporting_data, "/plan")
    _equal(bundle.primary_result, support.primary_result, bundle.supporting_data, "/primary_result")
    evidence["support"] = support.canonical_payload()
    tables, limitations = _evidence_tables(bundle, receipt.estimation_bundle, primary, rows, support)
    limitations = list(design.get("identification_risks") or ()) + limitations
    _numerical_lineage(reader, bundle, plan)
    if bundle.multiplicity_result is not None:
        multiplicity = reader.model(bundle.multiplicity_result, "MultiplicityResult", ac.MultiplicityResultV1)
        _equal(bundle.plan, multiplicity.plan, bundle.multiplicity_result, "/plan")
        evidence["multiplicity"] = multiplicity.canonical_payload()
        sources["multiplicity"] = bundle.multiplicity_result
        required.append("multiplicity")
    _pin_sources(reader, sources)
    return EvidencePacket(analysis_id, sources, evidence, tables, tuple(required), diagram,
                          tuple(limitations))
