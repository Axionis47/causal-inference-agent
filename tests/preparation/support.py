"""Approved preparation handoffs for cross-stage integration tests; no collected tests are imported."""

from __future__ import annotations

import hashlib
import io
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from causal.preparation.harness import PreparationDeps
from causal.runtime.failures import preparation_deps
from causal.shared import events, persistence
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactEnvelopeV1
from causal.shared.registry import load_artifact_type_registry

ROOT = Path(__file__).resolve().parents[2]


REGISTRIES = ROOT / "registries"


REGISTRY = load_artifact_type_registry(REGISTRIES / "artifact-types.v1.json")


NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)


METHODS = ("randomized_experiment", "aipw", "did", "sharp_rdd")


VERSIONS = {method: f"{method.replace('_', '-')}-pack.v1" for method in METHODS}


CATALOG, CAPACITY = "visualization-catalog.v1", "delivery-capacity.v1"


PREREPAIR = "frame_completeness"


COLUMNS: dict[str, tuple[str, ...]] = {
    "randomized_experiment": ("unit_id", "treat", "earn", "age"),
    "aipw": ("unit_id", "treat", "earn", "age"),
    "did": ("unit_id", "period", "grp", "treat", "earn", "age"),
    "sharp_rdd": ("unit_id", "score", "treat", "earn", "age")}


ROLES: dict[str, dict[str, str]] = {
    "randomized_experiment": {"unit_id": "unit_identifier", "treat": "treatment",
                              "earn": "outcome", "age": "precision_covariate"},
    "aipw": {"unit_id": "unit_identifier", "treat": "treatment", "earn": "outcome",
             "age": "precision_covariate"},
    "did": {"unit_id": "unit_identifier", "period": "time", "grp": "group",
            "treat": "treatment", "earn": "outcome", "age": "confounder_candidate"},
    "sharp_rdd": {"unit_id": "unit_identifier", "score": "running_variable",
                  "treat": "treatment", "earn": "outcome", "age": "precision_covariate"}}


KEYS: dict[str, tuple[str, ...]] = {"did": ("unit_id", "period")}


CARDINALITIES: dict[str, int] = {"arms": 2, "contrasts": 1, "subgroups": 0, "cohorts": 0,
                                 "periods": 0, "event_times": 0, "cutoff_sides": 0, "series": 2,
                                 "evidence_items": 20}


STRUCTURE: dict[str, dict[str, str]] = {
    "did": {"adoption_time": "2", "adoption_profile_id": "simultaneous"},
    "sharp_rdd": {"cutoff": "0", "assignment_direction": "above",
                  "treated_value": "1", "comparator_value": "0"}}


ESTIMANDS = {"randomized_experiment": "itt", "aipw": "ate",
             "did": "att_group_time_aggregate", "sharp_rdd": "late_at_cutoff"}


def csv_for(method: str, blank: str = "", limit: int = 0) -> bytes:
    """One frozen source table per pack: the smallest frame its §12 structure gates accept."""
    if method == "did":
        rows = [[f"u{u}", str(p), str(u % 2), str(int(p > 1)), str(u * p), str(u + p)]
                for u in range(1, 5) for p in (1, 2)]
    elif method == "randomized_experiment":
        rows = [[f"u{u}", str(u % 2), str(u * 3), str(u + 1)] for u in range(1, 5)]
    elif method == "aipw":
        rows = [[f"u{u}", str(int(u > 10)), str(u * 3), str(u + 1)] for u in range(1, 21)]
    else:
        rows = [[f"u{u}", str(u - 10), str(int(u >= 10)), str(u * 3), str(u + 1)]
                for u in range(1, 21)]
    columns = COLUMNS[method]
    if blank:
        rows[0][columns.index(blank)] = ""
    rows = rows[:limit] if limit else rows
    return ("\n".join([",".join(columns), *(",".join(row) for row in rows)]) + "\n").encode()


def event(analysis_id: str, name: str = "artifact.committed") -> events.OperationalEventV1:
    return events.build_event(
        occurred_at_utc=NOW, event_name=name, analysis_id=analysis_id,
        event_id=f"evt:{uuid.uuid4().hex}", stage=events.Stage.DESIGN,
        stage_run_id=f"dr:{analysis_id}:1", component_id="design-harness",
        component_version="0.1.0", required_eval_ids=("EV-P2-001",))


def make_deps(conn: Any, objects: Any, sink: io.StringIO) -> PreparationDeps:
    """The production wiring, over the docker stores and the frozen registries (T-019 §1.4)."""
    emitter = events.EventEmitter(sink)
    products = persistence.ProductStore(conn)
    held = SimpleNamespace(
        conn=conn, products=products, objects=objects, registry=REGISTRY, emitter=emitter,
        clock=lambda: NOW,
        committer=persistence.ArtifactCommitter(objects, products, REGISTRY, emitter))
    return preparation_deps(cast(Any, held), REGISTRIES, ROOT)


def ident(kind: str, analysis_id: str, payload: Mapping[str, Any]) -> dict[str, str]:
    """One artifact's deterministic identity, computed before its parents exist (D-031)."""
    digest = content_hash(dict(payload))
    return {"artifact_id": f"{kind.lower()}:{analysis_id}:{digest[:16]}", "content_hash": digest}


def ref(envelope: ArtifactEnvelopeV1) -> dict[str, str]:
    return {"artifact_id": envelope.artifact_id, "content_hash": envelope.content_hash}


def commit(deps: PreparationDeps, analysis_id: str, kind: str, payload: dict[str, Any],
           parents: tuple[ArtifactEnvelopeV1, ...] = ()) -> ArtifactEnvelopeV1:
    built = persistence.build_envelope(
        deps.registry, kind, payload, analysis_id=analysis_id,
        stage_run_id=f"dr:{analysis_id}:1", producer_version="0.1.0", parents=parents,
        created_at_utc=NOW)
    return deps.committer.commit(built, payload, event(analysis_id))


def approved_design(deps: PreparationDeps, analysis_id: str, method: str, data: bytes, *,
                    impute: Sequence[str] = (), drop_rules: Sequence[str] = (),
                    indicators: Sequence[str] = ("outcome_observed",),
                    roles: Mapping[str, str] | None = None,
                    contrasts: Sequence[str] = ("treated_vs_control",),
                    cardinalities: Mapping[str, int] | None = None,
                    structure: Mapping[str, str] | None = None) -> str:
    """Commit an approved V2 compiler chain and return its DesignOutcome id."""
    pack = deps.packs.get(method, VERSIONS[method])
    deps.products.create_stage_run(f"dr:{analysis_id}:1", analysis_id, "design")
    locator = deps.objects.put_if_absent(hashlib.sha256(data).hexdigest(), data)
    roles, keys = roles or ROLES[method], KEYS.get(method, ("unit_id",))
    made = commit(deps, analysis_id, "QuestionRecord", {"question_text": "does it work?"})
    intake = commit(deps, analysis_id, "IntakeOutcome", {"status": "usable"}, (made,))
    selection = commit(deps, analysis_id, "TableSelection", {
        "resource_object_locator": locator, "logical_name": "frame.csv",
        "resource_sha256": hashlib.sha256(data).hexdigest()}, (intake, made))
    book = commit(deps, analysis_id, "DesignContextManifest", {"table": "frame.csv"}, (selection,))
    intent = commit(deps, analysis_id, "DesignIntent", {"question_kind": "causal"}, (book,))
    mapped = commit(deps, analysis_id, "MeasurementMap", {"links": [
        {"column_name": name, "concept_id": f"c:{name}"} for name in sorted(roles)]}, (intent,))
    context = commit(deps, analysis_id, "CausalContext", {"concept_ids": sorted(roles)}, (mapped,))
    ledger = commit(deps, analysis_id, "RoleLedger", {"claims": [
        {"role": role, "column_refs": [name]} for name, role in sorted(roles.items())]}, (context,))
    proposal = commit(deps, analysis_id, "AgentDesignProposal", {
        "assignment_mechanism": method, "requested_estimand": ESTIMANDS[method],
        "ranked_method_ids": list(METHODS)}, (ledger,))
    facts = commit(deps, analysis_id, "DesignFactSet", {
        "selected_csv": ref(selection), "grain": "one_row_per_unit",
        "facts": [], "role_bindings": [], "conflicts": []}, (proposal, ledger))
    diagnostic_plan = commit(deps, analysis_id, "DiagnosticPlan", {
        "selected_csv": ref(selection), "candidate_method_id": method,
        "items": [{"diagnostic_id": PREREPAIR}], "issues": []}, (facts,))
    diagnostics = commit(deps, analysis_id, "DiagnosticReport", {
        "selected_csv": ref(selection), "candidate_method_id": method,
        "results": [{"diagnostic_id": PREREPAIR, "status": "computed"}],
        "issues": [], "computable": True}, (diagnostic_plan,))
    dimensions = dict(cardinalities or CARDINALITIES) | {"contrasts": len(contrasts)}
    preparation = {
        "estimator_input_schema_id": pack.prepared_frame_schema_id,
        "output_grain": "one_row_per_unit", "key_columns": list(keys),
        "required_roles": sorted(set(roles.values())),
        "protected_columns": [name for name, role in roles.items()
                              if role in pack.protected_roles],
        "eligibility_rule_ids": [],
        "unusable_row_rule_ids": [rule for rule in pack.permitted_disposition_rule_ids
                                    if rule not in drop_rules],
        "imputation_permitted": list(impute),
        "required_missingness_indicators": list(indicators),
        "required_final_diagnostic_ids": list(pack.required_postrepair_diagnostic_ids),
        "deletion_impact_dimensions": list(pack.dimension_impact_dimensions),
        "method_structure": dict(structure or STRUCTURE.get(method, {}))}
    design = commit(deps, analysis_id, "CompiledDesign", {
        "selected_csv": ref(selection), "method_id": method, "unit": "participant",
        "method_pack_version": VERSIONS[method], "comparator": "the untreated",
        "estimand": ESTIMANDS[method],
        "primary_contrasts": list(contrasts),
        "frame": {"treatment": "tr", "outcome": "out", "population": "pop", "timeframe": "tf"},
        "measurement_map": ref(mapped), "role_ledger": ref(ledger),
        "role_bindings": [
            {"role": role, "columns": [name], "concept_id": f"c:{name}"}
            for name, role in sorted(roles.items())],
        "causal_context": ref(context), "diagnostic_report": ref(diagnostics),
        "preparation": preparation, "estimator": {
            "estimand": ESTIMANDS[method], "parameters": {"estimand": ESTIMANDS[method]}},
        "registry_versions": {"schema": "compiled-design.v2",
                              "visualization_catalog": CATALOG,
                              "capacity": CAPACITY}},
        (facts, diagnostic_plan, diagnostics, context, mapped, ledger))
    view = commit(deps, analysis_id, "CausalGraphView", {"nodes": sorted(roles)},
                  (design, context, mapped, ledger))
    views = commit(deps, analysis_id, "GraphViewSet", {
        "base_view": ref(view), "alternative_views": []}, (design, view))
    capacity = {"status": "pass", "compiled_design": ref(design),
                "dimensions": [{"dimension": name, "value": value,
                                "applicability": "applicable", "source": "fixture"}
                               for name, value in dimensions.items()]}
    checked = commit(deps, analysis_id, "CapacityReport", capacity,
                     (design, views, diagnostics))
    review = commit(deps, analysis_id, "DesignReviewBundle", {
        "compiled_design": ref(design), "diagnostic_report": ref(diagnostics),
        "capacity_report": ref(checked), "graph_views": ref(views)},
        (design, diagnostics, checked, views))
    approval = commit(deps, analysis_id, "DesignApproval", {
        "decision": "approved", "review_bundle": ref(review),
        "approved_bundle_hash": review.content_hash}, (review,))
    return commit(deps, analysis_id, "DesignOutcome", {
        "status": "approved", "approval": ref(approval), "compiled_design": ref(design),
        "diagnostic_report": ref(diagnostics), "capacity_report": ref(checked),
        "review_bundle": ref(review)},
        (intake, selection, intent, design, diagnostics, checked, review, approval)).artifact_id

