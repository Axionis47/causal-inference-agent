"""Read frozen numerical and approved design artifacts through the common store."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Literal, NoReturn

from pydantic import ValidationError

from causal.analysis.integration import contracts as ac
from causal.post_analysis.contracts import InputError, InputIssue
from causal.post_analysis.visualization.contracts import (
    CausalDiagram,
    DiagramEdge,
    DiagramNode,
)
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef
from causal.shared.registry import RegistryError

Owner = Literal["analysis", "design", "post_analysis", "storage"]
COMPONENT = "post-analysis"


def _reject(code: str, owner: Owner, source: ArtifactRef | None, path: str,
            expected: Any, received: Any, action: str) -> NoReturn:
    raise InputError(InputIssue(code=code, owner=owner, source=source, path=path,
                                expected=expected, received=received, required_action=action))


def _equal(expected: Any, received: Any, source: ArtifactRef, path: str,
           owner: Owner = "analysis") -> None:
    if expected != received:
        _reject("source_binding_mismatch", owner, source, path, expected, received,
                "Reconcile the frozen source bindings and provide a new handoff.")


def _ref(body: Mapping[str, Any], key: str, source: ArtifactRef,
         owner: Owner = "analysis") -> ArtifactRef:
    try:
        return ArtifactRef.model_validate(body[key])
    except (KeyError, TypeError, ValidationError):
        _reject("required_reference_missing", owner, source, f"/{key}",
                "an exact artifact reference", body.get(key),
                "Supply the original referenced artifact through a corrected handoff.")
    raise AssertionError("unreachable")


class _Reader:
    def __init__(self, products: Any, objects: Any, registry: Any, analysis_id: str) -> None:
        self.products, self.objects, self.registry = products, objects, registry
        self.analysis_id = analysis_id
        self.cache: dict[ArtifactRef, tuple[Any, dict[str, Any]]] = {}

    def read(self, ref: ArtifactRef, kind: str | None,
             owner: Owner = "analysis") -> tuple[Any, dict[str, Any]]:
        if ref in self.cache:
            envelope, body = self.cache[ref]
        else:
            try:
                envelope = self.products.load_envelope(ref.artifact_id)
                raw = self.objects.get(envelope.payload_locator)
            except Exception as error:  # noqa: BLE001 -- adapters expose diverse operational failures
                _reject("source_unreachable", "storage", ref, "/", "readable frozen source",
                        type(error).__name__, "Restore source access and retry the same handoff.")
            try:
                body = json.loads(raw)
                actual_hash = content_hash(body)
                raw_hash = hashlib.sha256(raw).hexdigest()
            except (TypeError, ValueError):
                _reject("source_invalid", owner, ref, "/", "canonical JSON object", None,
                        "Restore the exact committed source; do not repair it in the report.")
            if not isinstance(body, dict):
                _reject("source_invalid", owner, ref, "/", "JSON object", type(body).__name__,
                        "Provide a schema-conforming source artifact.")
            _equal(ref.artifact_id, envelope.artifact_id, ref, "/artifact_id", owner)
            _equal(ref.content_hash, envelope.content_hash, ref, "/content_hash", owner)
            _equal(ref.content_hash, actual_hash, ref, "/payload_hash", owner)
            _equal(ref.content_hash, raw_hash, ref, "/payload_bytes", owner)
            _equal(self.analysis_id, envelope.analysis_id, ref, "/analysis_id", owner)
            try:
                registration = self.registry.lookup(envelope.artifact_type)
            except RegistryError:
                _reject("reader_unsupported", "post_analysis", ref, "/artifact_type",
                        "registered source type", envelope.artifact_type,
                        "Add validated reader support; do not change the numerical results.")
            if envelope.producer_component not in {registration.producer_component,
                                                    *registration.also_produced_by}:
                _reject("producer_not_allowed", owner, ref, "/producer_component",
                        registration.producer_component, envelope.producer_component,
                        "Restore the original registered producer binding.")
            if COMPONENT not in registration.allowed_reader_components:
                _reject("reader_not_allowed", "post_analysis", ref, "/allowed_reader_components",
                        COMPONENT, list(registration.allowed_reader_components),
                        "Correct the common artifact reader registration.")
            _equal(envelope.schema_version, body.get("schema_version"), ref,
                   "/schema_version", owner)
            if envelope.schema_version != registration.schema_version:
                _reject("reader_unsupported", "post_analysis", ref, "/schema_version",
                        registration.schema_version, envelope.schema_version,
                        "Add validated support for this source version; preserve its original data.")
            self.cache[ref] = envelope, body
        if kind is not None:
            _equal(kind, envelope.artifact_type, ref, "/artifact_type", owner)
        return envelope, body

    def model(self, ref: ArtifactRef, kind: str, model: Any) -> Any:
        _, body = self.read(ref, kind)
        try:
            return model.model_validate_json(json.dumps(body))
        except ValidationError as error:
            _reject("scientific_result_invalid", "analysis", ref, "/",
                    model.__name__, error.errors(include_input=False),
                    "Supply a valid result from the numerical producer.")


def _design(reader: _Reader, bundle: ac.EstimationContextManifestV1,
            sources: dict[str, ArtifactRef], evidence: dict[str, dict[str, Any]],
            ) -> tuple[dict[str, Any], dict[str, Any], CausalDiagram]:
    design_ref, prepared_ref = bundle.compiled_design, bundle.prepared_bundle
    _, design = reader.read(design_ref, "CompiledDesign", "design")
    _, prepared = reader.read(prepared_ref, "PreparedFrameBundle")
    _equal(design_ref.model_dump(), prepared.get("compiled_design"), prepared_ref,
           "/compiled_design")
    approval_ref = _ref(prepared, "design_approval", prepared_ref)
    _, approval = reader.read(approval_ref, "DesignApproval", "design")
    _equal("approved", approval.get("decision"), approval_ref, "/decision", "design")
    review_ref = _ref(approval, "review_bundle", approval_ref, "design")
    _equal(review_ref.content_hash, approval.get("approved_bundle_hash"), approval_ref,
           "/approved_bundle_hash", "design")
    _, review = reader.read(review_ref, "DesignReviewBundle", "design")
    _equal(design_ref.model_dump(), review.get("compiled_design"), review_ref,
           "/compiled_design", "design")
    views_ref = _ref(review, "graph_views", review_ref, "design")
    _, views = reader.read(views_ref, "GraphViewSet", "design")
    view_ref = _ref(views, "base_view", views_ref, "design")
    _, view = reader.read(view_ref, "CausalGraphView", "design")
    parents: dict[str, tuple[ArtifactRef, dict[str, Any]]] = {}
    for raw in view.get("parents", ()):
        try:
            parent = ArtifactRef.model_validate(raw)
        except ValidationError:
            _reject("source_invalid", "design", view_ref, "/parents", "exact references", raw,
                    "Supply the original approved graph source references.")
        envelope, body = reader.read(parent, None, "design")
        if envelope.artifact_type in parents:
            _reject("source_binding_ambiguous", "design", view_ref, "/parents",
                    "one source per scientific graph input", envelope.artifact_type,
                    "Resolve the ambiguous approved graph source binding.")
        parents[envelope.artifact_type] = parent, body
    for kind in ("CompiledDesign", "CausalContext", "MeasurementMap"):
        if kind not in parents:
            _reject("required_reference_missing", "design", view_ref, "/parents", kind,
                    list(parents), "Provide the exact scientific parents of the approved graph.")
    _equal(design_ref, parents["CompiledDesign"][0], view_ref, "/parents", "design")
    dag_ref, dag = parents["CausalContext"]
    measurement_ref, measurement = parents["MeasurementMap"]
    _equal(design.get("frame"), dag.get("frame"), dag_ref, "/frame", "design")
    try:
        names = {row["concept_id"]: row["name"] for row in measurement["concepts"]}
        diagram = CausalDiagram(source=dag_ref, selector="", nodes=tuple(
            DiagramNode(node_id=key, label=names[key]) for key in dag["concept_ids"]),
            edges=tuple(DiagramEdge(
                edge_id=row["edge_id"], source_node=row["source_concept_id"],
                target_node=row["target_concept_id"], status=row["status"],
                evidence_ids=tuple(row["supporting_evidence_ids"]),
                contrary_evidence_ids=tuple(row["contrary_evidence_ids"])) for row in dag["edges"]))
    except ValidationError as error:
        if any(issue["type"] in {"too_long", "string_too_long"} for issue in error.errors()):
            _reject("reader_unsupported", "post_analysis", dag_ref, "/", "supported display size",
                    error.errors(include_input=False),
                    "Extend the receiver's bounded graph display support; preserve the approved DAG.")
        _reject("causal_structure_invalid", "design", dag_ref, "/", "resolved causal semantics",
                error.errors(include_input=False), "Resolve the approved concepts and edge bindings.")
    except (KeyError, TypeError) as error:
        _reject("causal_structure_invalid", "design", dag_ref, "/", "resolved causal semantics",
                type(error).__name__, "Resolve the approved concepts, edges and variable mapping.")
    sources.update(design=design_ref, dag=dag_ref, measurement_map=measurement_ref)
    evidence["design"] = {key: value for key, value in design.items() if key not in {
        "required_visual_evidence", "registry_versions"}}
    evidence.update(dag=dag, measurement_map=measurement)
    for column, row in design.get("column_measurements", {}).items():
        source = _ref(row, "source_card", design_ref, "design")
        _, card = reader.read(source, "ColumnSemanticCard", "design")
        _equal(column, card.get("column_name"), source, "/column_name", "design")
        sources[f"column:{column}"] = source
        evidence[f"column:{column}"] = card
    return design, prepared, diagram


def _bind_request(reader: _Reader, context_ref: ArtifactRef, plan_ref: ArtifactRef,
                  manifest: ac.EstimationContextManifestV1, plan: ac.EstimationPlanV1,
                  design: dict[str, Any], prepared: dict[str, Any]) -> None:
    pairs = {
        "context_manifest": (context_ref, plan.context_manifest),
        "row_set_hash": (manifest.row_set_hash, plan.row_set_hash),
        "prepared_row_set_hash": (manifest.row_set_hash, prepared.get("row_set_hash")),
        "role_columns": (manifest.role_columns, plan.role_columns),
        "column_measurements": (manifest.column_measurements, plan.column_measurements),
        "required_sensitivity_ids": (manifest.required_sensitivity_ids, plan.required_sensitivity_ids),
        "method_id": (design.get("method_id"), plan.method_id),
        "method_pack_version": (design.get("method_pack_version"), plan.method_pack_version),
        "estimand_id": (design.get("estimand"), plan.estimand_id),
        "contrast_ids": (tuple(design.get("primary_contrasts", ())), plan.contrast_ids),
        "outcome_id": ((design.get("frame") or {}).get("outcome"), plan.outcome_id),
        "comparator_id": (design.get("comparator"), plan.comparator_id),
        "population_id": ((design.get("frame") or {}).get("population"), plan.population_id),
        "timeframe_id": ((design.get("frame") or {}).get("timeframe"), plan.timeframe_id),
        "unit_id": (design.get("unit"), plan.unit_id),
    }
    for path, (expected, received) in pairs.items():
        _equal(expected, received, plan_ref, f"/{path}")
    for column in plan.role_columns.values():
        actual = plan.column_measurements.get(column)
        _equal((design.get("column_measurements") or {}).get(column),
               actual.model_dump(mode="json") if actual is not None else None,
               plan_ref, f"/column_measurements/{column}")
    frame_ref = _ref(prepared, "prepared_frame", manifest.prepared_bundle)
    _, frame = reader.read(frame_ref, "PreparedFrame")
    try:
        columns = {row["column_name"] for row in frame["columns"]}
        approved: dict[str, set[str]] = {}
        for binding in design["role_bindings"]:
            approved.setdefault(binding["role"], set()).update(binding["columns"])
    except (KeyError, TypeError):
        _reject("scientific_context_unresolved", "design", manifest.compiled_design,
                "/role_bindings", "approved roles and prepared column definitions", None,
                "Supply the original approved role and column metadata.")
    # These are the historical estimation-plan.v1 aliases, not a mutable capability lookup.
    aliases = ({"adjustment_covariate": {"confounder_candidate", "precision_covariate"}}
               if plan.method_id == "aipw" else
               {"predetermined_covariate": {"precision_covariate", "confounder_candidate"}}
               if plan.method_id == "sharp_rdd" else {})
    for role, column in plan.role_columns.items():
        base = role.partition("__")[0]
        permitted = set(approved.get(base, ()))
        for alias in aliases.get(base, ()):
            permitted.update(approved.get(alias, ()))
        if (base == "unit_identifier" and
                (design.get("preparation") or {}).get("output_grain") == "one_row_per_group_time"):
            permitted.update(approved.get("group", ()))
        if column not in permitted or column not in columns:
            _reject("role_binding_mismatch", "analysis", plan_ref, f"/role_columns/{role}",
                    sorted(permitted & columns), column,
                    "Reconcile this numerical field with its approved role and prepared source.")
    for role in ("outcome", "treatment", "unit_identifier"):
        if role not in plan.role_columns:
            _reject("scientific_context_unresolved", "analysis", plan_ref, "/role_columns",
                    role, plan.role_columns, "Supply the exact indispensable numerical role binding.")
    for name, value in ((design.get("preparation") or {}).get("method_structure") or {}).items():
        _equal(value, plan.estimator_parameters.get(name), plan_ref,
               f"/estimator_parameters/{name}")
