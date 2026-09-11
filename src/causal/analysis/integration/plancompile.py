# The estimation entry gate, deterministic plan compilation, and the exact delivery-capacity
# recheck (PRD-004 §4, §6.1, §6.5, §26; D-061, D-083). Every upstream artifact is read as data:
# this module never imports `causal.design` or `causal.preparation`.

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from causal.analysis.integration import contracts as ec
from causal.analysis.integration.packs import EstimationPackV1
from causal.shared.contracts import ArtifactRef, Identity
from causal.shared.gateway import derive_seed

PREPARED, ENTRY_VALIDATION_FAILED = "prepared", "entry_validation_failed"
# The twelve §4 entry conditions, one stable code family each; wall 1 reports every failing one.
PREPARATION_NOT_PREPARED, MISSING_ARTIFACT = "preparation_not_prepared", "missing_artifact"
ENTRY_HASH_MISMATCH, BUNDLE_BINDING_MISMATCH = "entry_hash_mismatch", "bundle_binding_mismatch"
ROW_SET_HASH_MISMATCH = "row_set_hash_mismatch"
ROW_ACCOUNTING_INCOMPLETE = "row_accounting_incomplete"
ESTIMATOR_SCHEMA_MISMATCH = "estimator_schema_mismatch"
POSTREPAIR_DIAGNOSTIC_UNHANDLED = "postrepair_diagnostic_unhandled"
UNREGISTERED_PREPROCESSING = "unregistered_preprocessing_recipe"
AMBIGUOUS_METHOD_SELECTION = "ambiguous_method_selection"
UNSUPPORTED_REGISTRY_VERSION = "unsupported_registry_version"
UPSTREAM_ESTIMATE_MARKER = "upstream_estimate_marker"
CAPACITY_REPORT_NOT_PASS, CAPACITY_REPORT_BINDING_MISMATCH = (
    "capacity_report_not_pass", "capacity_report_binding_mismatch")
MISSING_PLAN_FIELD = "missing_plan_field"
# The four §4 handoff entries, keyed by the manifest field each one fills.
ENTRY_KEYS: Final = ("prepared_bundle", "compiled_design", "capacity_report")
# The three row-set hashes a prepared bundle carries; §4 condition 4 needs one value.
ROW_SET_KEYS: Final = ("row_set_hash", "stabilized_frame_row_set_hash",
                       "prepared_frame_row_set_hash")
# Any upstream payload key that would declare a result already read (§4 condition 11).
ESTIMATE_MARKERS: Final = ("estimate", "point_estimate", "primary_result", "results_influenced")
# The approved-selection surface the plan copies verbatim from the manifest (§6.1, §19.1).
_APPROVED: Final = tuple(ec._ApprovedSelection.model_fields)
# The approved design fact each role makes mandatory: a running variable is estimable only at
# the cutoff the design approved, and no pack default may stand in for one (§12.2).
REQUIRED_FACTS: Final = {"running_variable": ("cutoff", "assignment_direction", "treated_value", "comparator_value")}


# The four §4 handoff payloads plus the PRD-003 record, read as raw mappings.
@dataclass(frozen=True)
class EntryInputs:
    outcome: Mapping[str, Any]
    bundle: Mapping[str, Any]
    design: Mapping[str, Any]
    capacity: Mapping[str, Any]
    record: Mapping[str, Any]
    # What the handoff declared and what the store actually holds, both keyed by ENTRY_KEYS.
    declared: Mapping[str, ArtifactRef]
    committed: Mapping[str, ArtifactRef | None]
    # Estimator input role to the prepared frame's dtype for it (§4 condition 6).
    estimator_input_types: Mapping[str, str] = field(default_factory=dict)
    # Approved role to the prepared column carrying it, from the design-side role ledger.
    role_columns: Mapping[str, str] = field(default_factory=dict)
    postrepair_statuses: Mapping[str, str] = field(default_factory=dict)
    preprocessing_recipe_ids: tuple[str, ...] = ()


# The registered vocabularies and approved handling the §4 checks measure against.
@dataclass(frozen=True)
class EntryPolicy:
    pack: EstimationPackV1
    registry_versions: Mapping[str, str]
    registered_recipe_ids: frozenset[str] = frozenset()
    approved_handling: frozenset[str] = frozenset()
    numerical_tolerances: Mapping[str, float] = field(default_factory=dict)


# The frozen prepared structure and exact planned result cardinality the §6.5 recheck measures.
@dataclass(frozen=True)
class PreparedStructureV1:
    cardinalities: Mapping[str, int]
    required_visual_evidence: tuple[str, ...]
    registry_path: Path


# The conflict PRD-004 returns to PRD-002; field-compatible with the `design-conflict.v1` type.
class DesignConflictDraftV1(ec._Row):
    conflict_code: Identity
    failed_rule_id: Identity
    affected_row_count: ec._NonNegInt
    affected_unit_count: ec._NonNegInt
    affected_dimension_counts: dict[str, int]
    evidence_artifact_ids: tuple[Identity, ...]
    why_no_permitted_operation: str
    material_design_fields: ec._Ids
    recommended_action: Literal["revise_design", "refuse"]


def _ref(payload: Mapping[str, Any], key: str) -> ArtifactRef | None:
    held = payload.get(key)
    return ArtifactRef.model_validate(held) if isinstance(held, Mapping) else None


# §4 conditions 1, 2, and 3: a prepared outcome whose entries hash true and bind one design.
def _handoff_codes(inputs: EntryInputs) -> set[str]:
    codes: set[str] = set()
    if inputs.outcome.get("status") != PREPARED:
        codes.add(PREPARATION_NOT_PREPARED)
    for key in ENTRY_KEYS:
        declared, held = inputs.declared.get(key), inputs.committed.get(key)
        if declared is None or held is None:
            codes.add(MISSING_ARTIFACT)
        elif held != declared:
            codes.add(ENTRY_HASH_MISMATCH)
    bound = ((_ref(inputs.outcome, "prepared_bundle"), inputs.declared.get("prepared_bundle")),
             (_ref(inputs.bundle, "compiled_design"), inputs.declared.get("compiled_design")),
             (_ref(inputs.bundle, "capacity_report"), inputs.declared.get("capacity_report")))
    if any(found is None or found != want for found, want in bound):
        codes.add(BUNDLE_BINDING_MISMATCH)
    return codes


# §4 conditions 4 and 5: one row-set hash across the frames, every source row terminal.
def _frame_codes(inputs: EntryInputs) -> set[str]:
    hashes = {str(inputs.bundle.get(name) or "") for name in ROW_SET_KEYS}
    frozen = str((inputs.record.get("freeze") or {}).get("row_set_hash") or "")
    codes = set() if hashes == {frozen} and frozen else {ROW_SET_HASH_MISMATCH}
    ledger = inputs.record.get("dispositions") or {}
    total = sum(int(row.get("row_count", 0)) for row in (ledger.get("counts") or ()))
    indexed = int((inputs.record.get("source_row_index") or {}).get("row_count", 0))
    if not indexed or total != indexed:
        codes.add(ROW_ACCOUNTING_INCOMPLETE)
    return codes


# §4 conditions 6, 7, and 8: the estimator's schema, its diagnostics, its preprocessing.
def aliased(roles: Mapping[str, Any], pack: EstimationPackV1) -> dict[str, Any]:
    # The approved design's roles under the names this pack's estimator inputs carry (§8).
    renamed = {}
    for role, value in roles.items():
        base, separator, suffix = str(role).partition("__")
        alias = str(pack.estimator_role_aliases.get(base, base))
        renamed[alias + (separator + suffix if separator else "")] = value
    return renamed


def _structure_codes(inputs: EntryInputs, policy: EntryPolicy) -> set[str]:
    schema = policy.pack.estimator_input_schema
    found = {role: dtype for role, dtype in aliased(inputs.estimator_input_types, policy.pack).items() if role.partition("__")[0] in schema}
    preparation = inputs.design.get("preparation") or {}
    required = set(aliased({str(role): str(role) for role in
                            preparation.get("required_roles") or ()}, policy.pack))
    bases = {role.partition("__")[0] for role in found}
    offending = (required - bases) | {
        role for role, dtype in found.items()
        if (base := role.partition("__")[0]) not in schema or dtype not in schema[base]}
    codes = {ESTIMATOR_SCHEMA_MISMATCH} if offending else set()
    for diagnostic_id in preparation.get("required_final_diagnostic_ids") or ():
        status = inputs.postrepair_statuses.get(str(diagnostic_id))
        if status != "pass" and (status is None or status not in policy.approved_handling):
            codes.add(POSTREPAIR_DIAGNOSTIC_UNHANDLED)
    if set(inputs.preprocessing_recipe_ids) - policy.registered_recipe_ids:
        codes.add(UNREGISTERED_PREPROCESSING)
    # The approved design facts the estimator cannot be run without; a missing one is refused
    # here rather than defaulted downstream (D-089b).
    structure = preparation.get("method_structure") or {}
    required = {key for role, keys in REQUIRED_FACTS.items() if role in inputs.role_columns for key in keys}
    if "adoption_profile_id" in policy.pack.parameter_defaults:
        required |= {"adoption_time", "adoption_profile_id"}
    if required - set(structure):
        codes.add(MISSING_PLAN_FIELD)
    return codes


# §4 conditions 9, 10, and 11: one resolved selection, supported versions, no upstream estimate.
def _selection_codes(inputs: EntryInputs, policy: EntryPolicy) -> set[str]:
    design, pack = inputs.design, policy.pack
    frame = design.get("frame") or {}
    singles = (design.get("method_id"), design.get("method_pack_version"), design.get("estimand"),
               design.get("comparator"), design.get("unit"), frame.get("outcome"),
               frame.get("population"), frame.get("timeframe"))
    codes: set[str] = set()
    if not all(isinstance(value, str) and value for value in singles):
        codes.add(AMBIGUOUS_METHOD_SELECTION)
    if (design.get("method_id"), design.get("method_pack_version")) != (
            pack.method_id, pack.pack_version) or not (design.get("primary_contrasts") or ()):
        codes.add(AMBIGUOUS_METHOD_SELECTION)
    versions = policy.registry_versions
    if set(versions) != set(ec.ESTIMATION_REGISTRY_KEYS) or not all(versions.values()) or not (
            isinstance(design.get("registry_versions"), Mapping)):
        codes.add(UNSUPPORTED_REGISTRY_VERSION)
    payloads = (inputs.outcome, inputs.bundle, design, inputs.record)
    if any(row.get(marker) for row in payloads for marker in ESTIMATE_MARKERS):
        codes.add(UPSTREAM_ESTIMATE_MARKER)
    return codes


# A historical capacity reference must bind the same design; its layout verdict is not a numerical gate.
def _capacity_codes(inputs: EntryInputs) -> set[str]:
    capacity = inputs.capacity
    codes: set[str] = set()
    pinned = ((_ref(capacity, "compiled_design"), inputs.declared.get("compiled_design")),)
    return codes | ({CAPACITY_REPORT_BINDING_MISMATCH}
                    if any(a is None or a != b for a, b in pinned) else set())


def entry_codes(inputs: EntryInputs, policy: EntryPolicy) -> tuple[str, ...]:
    # Every failing §4 condition at once; the gate never stops at the first (fail closed).
    return tuple(sorted(_handoff_codes(inputs) | _frame_codes(inputs)
                        | _structure_codes(inputs, policy) | _selection_codes(inputs, policy)
                        | _capacity_codes(inputs)))


def compile_context_manifest(inputs: EntryInputs,
                             policy: EntryPolicy) -> ec.EstimationContextManifestV1:
    # Verify the twelve §4 conditions, then hydrate the §19.1 closed approved-context surface.
    if codes := entry_codes(inputs, policy):
        raise ec.EstimationError("the estimation entry gate refused the preparation handoff",
                              ENTRY_VALIDATION_FAILED, codes)
    design, pack = inputs.design, policy.pack
    preparation = design.get("preparation") or {}
    frame = design.get("frame") or {}
    refs = {key: inputs.declared[key] for key in ENTRY_KEYS}
    # The contract's `estimator_input_schema` names both the prepared-frame schema the estimator
    # reads and the view it reads it through; §4 condition 6 compares against exactly this id.
    roles = {role: column for role, column in aliased(inputs.role_columns, pack).items() if role.partition("__")[0] in pack.estimator_input_schema}
    if (preparation.get("output_grain") == "one_row_per_group_time"
            and "unit_identifier" not in roles and "group" in roles):
        roles["unit_identifier"] = roles["group"]
    return ec.EstimationContextManifestV1(
        method_id=pack.method_id, method_pack_version=pack.pack_version,
        estimand_id=str(design["estimand"]), population_id=str(frame["population"]),
        timeframe_id=str(frame["timeframe"]), comparator_id=str(design["comparator"]),
        outcome_id=str(frame["outcome"]), unit_id=str(design["unit"]),
        role_columns=roles,
        column_measurements={column: ec.ColumnMeasurementV1.model_validate_json(json.dumps(row))
            for column, row in (design.get("column_measurements") or {}).items()
            if column in roles.values()},
        row_set_hash=str(inputs.bundle["row_set_hash"]),
        contrast_ids=tuple(str(item) for item in design["primary_contrasts"]),
        required_sensitivity_ids=tuple(row.branch_id for row in pack.sensitivity_branches),
        figure_builder_ids=pack.figure_builder_ids, capacity_report=refs["capacity_report"],
        numerical_tolerances=dict(policy.numerical_tolerances),
        compiled_design=refs["compiled_design"],
        prepared_bundle=refs["prepared_bundle"],
        method_structure=dict(preparation.get("method_structure") or {}),
        preprocessing_rule_ids=inputs.preprocessing_recipe_ids,
        result_cardinalities={str(row["dimension"]): int(row["value"] or 0)
                              for row in inputs.capacity["dimensions"]},
        required_visual_evidence_ids=tuple(str(item) for item in design.get("required_visual_evidence") or ()),
        registry_versions=dict(policy.registry_versions),
    )


def _required(pack: EstimationPackV1, key: str) -> str:
    value = pack.parameter_defaults.get(key)
    if value is None:
        raise ec.EstimationError(f"{pack.method_id} registers no {key}", MISSING_PLAN_FIELD)
    return str(value)


def compile_plan(manifest: ec.EstimationContextManifestV1, pack: EstimationPackV1,
                 context_manifest: ArtifactRef, *, plan_revision: int = 1) -> ec.EstimationPlanV1:
    # The §6.1 plan, compiled deterministically from approved artifacts and the registry: no
    # choice point, no default absent from the pack, and one 31-bit seed per plan identity.
    folded = pack.fold_count_default is not None
    profile = pack.primary_nuisance_profile()
    parents = (context_manifest, manifest.prepared_bundle, manifest.compiled_design,
               manifest.capacity_report)
    identity = ":".join((context_manifest.content_hash, manifest.method_id, pack.estimator_id,
                         pack.estimator_version, str(plan_revision)))
    policy_id = pack.parameter_defaults.get("multiplicity_policy_id")
    approved = {name: getattr(manifest, name) for name in _APPROVED}
    return ec.EstimationPlanV1(
        **{**approved, "seed": derive_seed(identity)}, parents=parents,
        versions=dict(manifest.registry_versions), context_manifest=context_manifest,
        plan_revision=plan_revision, estimator_id=pack.estimator_id,
        estimator_version=pack.estimator_version,
        # Physical measurement units belong to the approved data, not the estimand vocabulary.
        outcome_scale=(measurement.units if (measurement := manifest.column_measurements.get(
            manifest.role_columns.get("outcome", ""))) and measurement.units else "outcome units"),
        multiplicity_policy_id=(str(policy_id) if policy_id is not None
                                and len(manifest.contrast_ids) > 1 else None),
        primary_mask_rule_id=_required(pack, "mask_rule_id"),
        confidence_level=pack.confidence_level, uncertainty_method=pack.uncertainty_method,
        finite_sample_correction=pack.finite_sample_correction,
        # The registered defaults, then the approved design facts — the cutoff and its
        # direction are the design's to state and the pack's to know nothing about (§12.2).
        estimator_parameters=dict(pack.parameter_defaults) | dict(manifest.method_structure)
        | {"estimand": manifest.estimand_id},
        nuisance_profile_id=profile.profile_id if profile is not None else None,
        fold_count=pack.fold_count_default,
        fold_assignment_rule_id=_required(pack, "fold_assignment_rule_id") if folded else None,
        preprocessing_recipe_ids=manifest.preprocessing_rule_ids,
        required_diagnostics=pack.severities(),
        numerical_failure_rule_ids=pack.not_estimable_rule_ids + pack.invalidation_rule_ids)




# One `over_limit` code per dimension the template cannot carry (PRD-002 §13.5 semantics).


