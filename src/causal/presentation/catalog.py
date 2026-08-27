# The fail-closed visualization-catalog loader, its typed accessors, and the thirteen-condition
# PRD-005 §5 entry gate feeding gate 1. Every upstream artifact is read as data: this module
# imports no estimation code (`entry_codes` idiom from estimation/plancompile.py).

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from causal.presentation import contracts as pc
from causal.shared.contracts import ArtifactRef

INVALID_CATALOG_FILE, ENTRY_VALIDATION_FAILED = "invalid_catalog_file", "entry_validation_failed"
UNKNOWN_PROFILE, NO_COMPATIBLE_TEMPLATE = "unknown_method_profile", "no_compatible_template"
# The thirteen §5 conditions, one stable code family each; the gate reports every failing one.
ESTIMATION_NOT_COMPLETE, CLAIM_NOT_REPORTABLE = "estimation_not_complete", "claim_not_reportable"
MISSING_ARTIFACT, ENTRY_HASH_MISMATCH = "missing_artifact", "entry_hash_mismatch"
AMBIGUOUS_SELECTION, INCOMPLETE_PRIMARY = "ambiguous_method_selection", "incomplete_primary_result"
CLAIM_EXCEEDS_CEILING, UNRESOLVED_EVIDENCE = "claim_exceeds_ceiling", "unresolved_claim_evidence"
MISSING_FIGURE_DATA, INCOMPLETE_FIGURE_DATA = "missing_figure_data", "incomplete_figure_data"
UNAPPROVED_FIGURE_CONTENT, GRAPH_VIEW_MISMATCH = ("unapproved_figure_content",
                                                  "approved_graph_view_mismatch")
UNSUPPORTED_CATALOG_VERSION, OBSERVABILITY_PREFLIGHT = ("unsupported_catalog_version",
                                                        "observability_preflight_failed")
CAPACITY_NOT_PASS, CAPACITY_BINDING = "capacity_check_not_pass", "capacity_binding_mismatch"
# §16.1 judgment ceilings, most restrictive first; a claim ranked above its ceiling exceeds it.
RANK: Final = {name: index for index, name in enumerate(
    ("failed", "not_estimable", "not_reportable", "reportable_with_qualifications", "reportable"))}
# §5 condition 8: what a figure-data payload must carry. Uncertainty and denominators travel
# inside `points`; `builder_id` and `rule_ids` carry provenance.
FIGURE_DATA_FIELDS: Final = ("points", "units", "labels", "contributing_counts", "rule_ids",
                             "builder_id", "disclosure_status")
# §5 condition 9: any key that would carry a raw observation or an unapproved statistic.
FORBIDDEN_FIGURE_KEYS: Final = ("rows", "raw_rows", "observations", "frame", "computed_statistics")


# The five §5 handoff payloads plus the approved graph view, all read as raw mappings.
@dataclass(frozen=True)
class EntryInputs:
    outcome: Mapping[str, Any]
    bundle: Mapping[str, Any]
    judgment: Mapping[str, Any]
    design: Mapping[str, Any]
    capacity: Mapping[str, Any]
    graph_view: Mapping[str, Any]
    # What the handoff declared and what the store actually holds, both keyed by ENTRY_KEYS.
    declared: Mapping[str, ArtifactRef]
    committed: Mapping[str, ArtifactRef | None]
    # Visual-evidence id to the committed FigureDataArtifact payload behind it.
    figure_data: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


# The catalog selection and the approved facts the §5 checks measure against.
@dataclass(frozen=True)
class EntryPolicy:
    catalog: pc.VisualizationCatalogV1
    profile: pc.MethodProfileV1
    approved_graph_view: ArtifactRef
    # Condition 13: the caller confirms the PRD-004 trace acknowledgement and runs the PRD-005
    # LangSmith health and authorization preflight, reporting one result here; the gate itself
    # never calls a service.
    observability_ready: bool = False


# Fail closed: a catalog that does not parse and validate is never partially loaded (§8).
def load_visualization_catalog(path: Path) -> pc.VisualizationCatalogV1:
    try:
        return pc.VisualizationCatalogV1.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise pc.PresentationError(f"invalid catalog file {path}: {error}",
                                   INVALID_CATALOG_FILE) from error


# The one profile registered for an approved method; zero or many resolve to nothing (§8).
def profile_for_method(catalog: pc.VisualizationCatalogV1, method_id: str) -> pc.MethodProfileV1:
    found = [row for row in catalog.profiles if row.method_id == method_id]
    if len(found) != 1:
        raise pc.PresentationError(f"no single catalog profile for {method_id!r}", UNKNOWN_PROFILE)
    return found[0]


# Every template the profile registers for one evidence question, in catalog order (§8).
def templates_for_evidence(catalog: pc.VisualizationCatalogV1, profile: pc.MethodProfileV1,
                           evidence_id: str) -> tuple[pc.TemplateV1, ...]:
    ids = profile.templates_by_evidence.get(evidence_id, ())
    if not (rows := tuple(row for row in catalog.templates if row.template_id in ids)):
        raise pc.PresentationError(f"no compatible template for {evidence_id!r}",
                                   NO_COMPATIBLE_TEMPLATE)
    return rows


# One `over_limit` code per method-level or catalog-wide ceiling the frozen counts exceed (§8).
def capacity_codes(catalog: pc.VisualizationCatalogV1, profile: pc.MethodProfileV1,
                   counts: Mapping[str, int]) -> tuple[str, ...]:
    limits = dict(profile.capacity_limits) | {"figures": catalog.limits.max_figures,
                                              "panels": catalog.limits.max_panels_per_figure}
    return tuple(sorted(f"over_limit:{profile.profile_id}:{name}"
                        for name, limit in limits.items() if int(counts.get(name, 0)) > limit))


# §5 conditions 1-3: a complete estimation outcome, a presentable claim, and five declared
# entries that exist in the store and hash true.
def _handoff_codes(inputs: EntryInputs) -> set[str]:
    codes: set[str] = set()
    if inputs.outcome.get("status") != "complete":
        codes.add(ESTIMATION_NOT_COMPLETE)
    if inputs.judgment.get("status") not in ("reportable", "reportable_with_qualifications"):
        codes.add(CLAIM_NOT_REPORTABLE)
    for key in pc.ENTRY_KEYS:
        declared, held = inputs.declared.get(key), inputs.committed.get(key)
        if declared is None or held is None:
            codes.add(MISSING_ARTIFACT)
        elif declared != held:
            codes.add(ENTRY_HASH_MISMATCH)
    return codes


# §5 conditions 4-6: one approved method and estimand, one atomic primary result with its
# complete evidence, a claim inside its ceiling, and every claim item citing frozen evidence.
def _claim_codes(inputs: EntryInputs, policy: EntryPolicy) -> set[str]:
    codes: set[str] = set()
    estimands = {str(inputs.design.get("estimand")), str(inputs.judgment.get("estimand_id"))}
    if str(inputs.design.get("method_id")) != policy.profile.method_id or len(estimands) != 1:
        codes.add(AMBIGUOUS_SELECTION)
    items = tuple(inputs.judgment.get("items") or ())
    if (not inputs.bundle.get("primary_result") or not items
            or len(tuple(inputs.bundle.get("evidence_bundles") or ())) != 3
            or any(item.get("confidence_level") is None for item in items)):
        codes.add(INCOMPLETE_PRIMARY)
    if RANK.get(str(inputs.judgment.get("status")), -1) > RANK.get(
            str(inputs.judgment.get("overall_ceiling")), -1):
        codes.add(CLAIM_EXCEEDS_CEILING)
    frozen = {str(row.get("artifact_id")) for row in inputs.bundle.values()
              if isinstance(row, Mapping)}
    for item in items:
        cited = tuple(str(name) for name in item.get("cited_artifact_ids") or ())
        if not cited or set(cited) - frozen:
            codes.add(UNRESOLVED_EVIDENCE)
        if RANK.get(str(item.get("status")), -1) > RANK.get(str(item.get("ceiling")), -1):
            codes.add(CLAIM_EXCEEDS_CEILING)
    return codes


# §5 conditions 7-9: every required visual-evidence question resolves to a typed figure-data
# artifact that carries its bounded fields and no unrestricted observation.
def _evidence_codes(inputs: EntryInputs, policy: EntryPolicy) -> set[str]:
    codes: set[str] = set()
    for evidence_id in policy.profile.required_evidence_ids:
        payload = inputs.figure_data.get(evidence_id)
        if payload is None or payload.get("schema_version") != "figure-data-artifact.v1":
            codes.add(MISSING_FIGURE_DATA)
        elif any(not payload.get(name) for name in FIGURE_DATA_FIELDS):
            codes.add(INCOMPLETE_FIGURE_DATA)
        if payload is not None and any(name in payload for name in FORBIDDEN_FIGURE_KEYS):
            codes.add(UNAPPROVED_FIGURE_CONTENT)
    return codes


# §5 conditions 10-13: the approved graph view, one supported catalog and display profile, the
# exact passing pre-estimation capacity check, and the observability preflight.
def _binding_codes(inputs: EntryInputs, policy: EntryPolicy) -> set[str]:
    codes, catalog, view = set(), policy.catalog, inputs.graph_view
    if (view.get("artifact_id") != policy.approved_graph_view.artifact_id
            or view.get("content_hash") != policy.approved_graph_view.content_hash):
        codes.add(GRAPH_VIEW_MISMATCH)
    if (inputs.design.get("visualization_catalog_version") != catalog.catalog_version
            or catalog.display_profile.display_profile_id != pc.DISPLAY_PROFILE_ID):
        codes.add(UNSUPPORTED_CATALOG_VERSION)
    if inputs.capacity.get("status") != "pass":
        codes.add(CAPACITY_NOT_PASS)
    if (inputs.capacity.get("visualization_catalog_version") != catalog.catalog_version
            or inputs.capacity.get("method_id") != policy.profile.method_id
            or inputs.declared.get("capacity_check") != inputs.committed.get("capacity_check")
            or capacity_codes(catalog, policy.profile,
                              inputs.capacity.get("cardinalities") or {})):
        codes.add(CAPACITY_BINDING)
    if not policy.observability_ready:
        codes.add(OBSERVABILITY_PREFLIGHT)
    return codes


def entry_codes(inputs: EntryInputs, policy: EntryPolicy) -> tuple[str, ...]:
    # Every failing §5 condition at once; the gate never stops at the first (fail closed).
    return tuple(sorted(_handoff_codes(inputs) | _claim_codes(inputs, policy)
                        | _evidence_codes(inputs, policy) | _binding_codes(inputs, policy)))
