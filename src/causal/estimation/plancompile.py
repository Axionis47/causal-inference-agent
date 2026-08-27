# The estimation entry gate, deterministic plan compilation, and the exact delivery-capacity
# recheck (PRD-004 §4, §6.1, §6.5, §26; D-061, D-083). Every upstream artifact is read as data:
# this module never imports `causal.design` or `causal.preparation`.

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from causal.estimation import contracts as ec
from causal.estimation.packs import EstimationPackV1
from causal.shared.canonical import content_hash
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
CAPACITY_CHECK_NOT_PASS, CAPACITY_BINDING_MISMATCH = ("capacity_check_not_pass",
                                                      "capacity_binding_mismatch")
INVALID_CAPACITY_REGISTRY, CAPACITY_EXCEEDED = "invalid_capacity_registry", "capacity_exceeded"
MISSING_PLAN_FIELD, CAPACITY_VERSION_DRIFT = "missing_plan_field", "capacity_version_drift"
# The four §4 handoff entries, keyed by the manifest field each one fills.
ENTRY_KEYS: Final = ("prepared_bundle", "experiment_design", "runnable_frame_contract",
                     "capacity_check")
# The three row-set hashes a prepared bundle carries; §4 condition 4 needs one value.
ROW_SET_KEYS: Final = ("row_set_hash", "stabilized_frame_row_set_hash",
                       "prepared_frame_row_set_hash")
# Any upstream payload key that would declare a result already read (§4 condition 11).
ESTIMATE_MARKERS: Final = ("estimate", "point_estimate", "primary_result", "results_influenced")
# The closed capacity dimension vector and which template limit each must fit (SC §11; PRD-002
# §13.5), read here as data so the recheck never imports the design harness.
CAPACITY_DIMENSIONS: Final = ("arms", "contrasts", "subgroups", "cohorts", "periods",
                              "event_times", "cutoff_sides", "series", "evidence_items")
_DIMENSION_FIT: Final[dict[str, tuple[str, ...]]] = {
    "arms": ("max_series", "max_panels"), "contrasts": ("max_labels", "max_annotations"),
    "subgroups": ("max_panels",), "cohorts": ("max_panels",), "periods": ("max_series",),
    "event_times": ("max_series",), "cutoff_sides": ("max_panels",), "series": ("max_series",),
    "evidence_items": ()}
# The approved-selection surface the plan copies verbatim from the manifest (§6.1, §19.1).
_APPROVED: Final = tuple(ec._ApprovedSelection.model_fields)
# The approved design fact each role makes mandatory: a running variable is estimable only at
# the cutoff the design approved, and no pack default may stand in for one (§12.2).
REQUIRED_FACTS: Final[dict[str, str]] = {"running_variable": "cutoff"}


# The four §4 handoff payloads plus the PRD-003 record, read as raw mappings.
@dataclass(frozen=True)
class EntryInputs:
    outcome: Mapping[str, Any]
    bundle: Mapping[str, Any]
    design: Mapping[str, Any]
    contract: Mapping[str, Any]
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
    recipient_map: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


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
             (_ref(inputs.bundle, "experiment_design"), inputs.declared.get("experiment_design")),
             (_ref(inputs.bundle, "runnable_frame_contract"),
              inputs.declared.get("runnable_frame_contract")),
             (_ref(inputs.bundle, "capacity_check"), inputs.declared.get("capacity_check")))
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
    return {str(pack.estimator_role_aliases.get(role, role)): value
            for role, value in roles.items()}


def _structure_codes(inputs: EntryInputs, policy: EntryPolicy) -> set[str]:
    schema = policy.pack.estimator_input_schema
    found = aliased(inputs.estimator_input_types, policy.pack)
    offending = (set(inputs.contract.get("required_roles") or ()) - set(found)) | (
        set(found) - set(schema)) | {
        role for role, dtype in found.items() if role in schema and dtype not in schema[role]}
    codes = {ESTIMATOR_SCHEMA_MISMATCH} if offending else set()
    for diagnostic_id in inputs.design.get("required_postrepair_diagnostics") or ():
        status = inputs.postrepair_statuses.get(str(diagnostic_id))
        if status != "pass" and (status is None or status not in policy.approved_handling):
            codes.add(POSTREPAIR_DIAGNOSTIC_UNHANDLED)
    if set(inputs.preprocessing_recipe_ids) - policy.registered_recipe_ids:
        codes.add(UNREGISTERED_PREPROCESSING)
    # The approved design facts the estimator cannot be run without; a missing one is refused
    # here rather than defaulted downstream (D-089b).
    structure = inputs.contract.get("method_structure") or {}
    if any(role in inputs.role_columns and key not in structure
           for role, key in REQUIRED_FACTS.items()):
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
    payloads = (inputs.outcome, inputs.bundle, design, inputs.contract, inputs.record)
    if any(row.get(marker) for row in payloads for marker in ESTIMATE_MARKERS):
        codes.add(UPSTREAM_ESTIMATE_MARKER)
    return codes


# §4 condition 12: the explicit design-time capacity artifact passed and binds this design.
def _capacity_codes(inputs: EntryInputs) -> set[str]:
    capacity, design = inputs.capacity, inputs.design
    codes = set() if capacity.get("status") == "pass" else {CAPACITY_CHECK_NOT_PASS}
    pinned = ((capacity.get("method_id"), design.get("method_id")),
              (capacity.get("visualization_catalog_version"),
               design.get("visualization_catalog_version")),
              (capacity.get("capacity_registry_version"),
               design.get("capacity_registry_version")),
              (_ref(design, "capacity_check"), inputs.declared.get("capacity_check")))
    return codes | ({CAPACITY_BINDING_MISMATCH}
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
    design, contract, pack = inputs.design, inputs.contract, policy.pack
    frame = design.get("frame") or {}
    refs = {key: inputs.declared[key] for key in ENTRY_KEYS}
    # The contract's `estimator_input_schema` names both the prepared-frame schema the estimator
    # reads and the view it reads it through; §4 condition 6 compares against exactly this id.
    view_id = str(contract["estimator_input_schema"])
    upstream: dict[str, Any] = {key: ref.model_dump(mode="json") for key, ref in refs.items()}
    upstream |= {"pack": [pack.method_id, pack.pack_version], "view": view_id}
    manifest_hash = content_hash(upstream)
    return ec.EstimationContextManifestV1(
        method_id=pack.method_id, method_pack_version=pack.pack_version,
        estimand_id=str(design["estimand"]), population_id=str(frame["population"]),
        timeframe_id=str(frame["timeframe"]), comparator_id=str(design["comparator"]),
        outcome_id=str(frame["outcome"]), unit_id=str(design["unit"]),
        role_columns=aliased(inputs.role_columns, pack), prepared_frame_schema_id=view_id,
        row_set_hash=str(inputs.bundle["row_set_hash"]),
        contrast_ids=tuple(str(item) for item in design["primary_contrasts"]),
        required_sensitivity_ids=tuple(row.branch_id for row in pack.sensitivity_branches),
        figure_builder_ids=pack.figure_builder_ids, capacity_check=refs["capacity_check"],
        seed=derive_seed(manifest_hash), numerical_tolerances=dict(policy.numerical_tolerances),
        experiment_design=refs["experiment_design"],
        runnable_frame_contract=refs["runnable_frame_contract"],
        prepared_bundle=refs["prepared_bundle"], estimator_input_view_id=view_id,
        method_structure=dict(contract.get("method_structure") or {}),
        contribution_mask_rule_ids=pack.allowed_mask_rule_ids,
        preprocessing_rule_ids=inputs.preprocessing_recipe_ids,
        uncertainty_rule_id=pack.uncertainty_method,
        required_diagnostic_ids=tuple(pack.severities()),
        judgment_rule_ids=pack.not_estimable_rule_ids + pack.invalidation_rule_ids,
        result_cardinalities={str(name): int(count)
                              for name, count in (inputs.capacity["cardinalities"]).items()},
        registry_versions=dict(policy.registry_versions),
        recipient_map={key: tuple(value) for key, value in policy.recipient_map.items()},
        manifest_hash=manifest_hash)


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
    parents = (context_manifest, manifest.prepared_bundle, manifest.experiment_design,
               manifest.runnable_frame_contract, manifest.capacity_check)
    identity = ":".join((context_manifest.content_hash, manifest.method_id, pack.estimator_id,
                         pack.estimator_version, str(plan_revision)))
    policy_id = pack.parameter_defaults.get("multiplicity_policy_id")
    approved = {name: getattr(manifest, name) for name in _APPROVED}
    return ec.EstimationPlanV1(
        **{**approved, "seed": derive_seed(identity)}, parents=parents,
        versions=dict(manifest.registry_versions), context_manifest=context_manifest,
        plan_revision=plan_revision, estimator_id=pack.estimator_id,
        estimator_version=pack.estimator_version,
        # The pack's registered scale, or the approved estimand's own when it registers none.
        outcome_scale=str(pack.parameter_defaults.get("outcome_scale") or manifest.estimand_id),
        multiplicity_policy_id=(str(policy_id) if policy_id is not None
                                and len(manifest.contrast_ids) > 1 else None),
        primary_mask_rule_id=_required(pack, "mask_rule_id"),
        confidence_level=pack.confidence_level, uncertainty_method=pack.uncertainty_method,
        finite_sample_correction=pack.finite_sample_correction,
        # The registered defaults, then the approved design facts — the cutoff and its
        # direction are the design's to state and the pack's to know nothing about (§12.2).
        estimator_parameters=dict(pack.parameter_defaults) | dict(manifest.method_structure),
        nuisance_profile_id=profile.profile_id if profile is not None else None,
        fold_count=pack.fold_count_default,
        fold_assignment_rule_id=_required(pack, "fold_assignment_rule_id") if folded else None,
        preprocessing_recipe_ids=manifest.preprocessing_rule_ids,
        required_diagnostics=pack.severities(),
        numerical_failure_rule_ids=pack.not_estimable_rule_ids + pack.invalidation_rule_ids)


def _capacity_registry(path: Path) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, Mapping) or not document.get("templates"):
            raise ValueError("no registered templates")
    except (OSError, TypeError, ValueError) as error:
        raise ec.EstimationError(f"invalid capacity registry {path}: {error}",
                              INVALID_CAPACITY_REGISTRY) from error
    return document


# One `over_limit` code per dimension the template cannot carry (PRD-002 §13.5 semantics).
def _violations(template: Mapping[str, Any], dimensions: Mapping[str, int]) -> list[str]:
    return [f"over_limit:{template['template_id']}:{name}"
            for name, limits in _DIMENSION_FIT.items()
            if any(dimensions[name] > int(template[limit]) for limit in limits)]


def recheck_capacity(plan: ec.EstimationPlanV1,
                     structure: PreparedStructureV1) -> DesignConflictDraftV1 | None:
    # §6.5: rerun the capacity check against the frozen structure and exact planned cardinality,
    # measured against the versions PRD-002 approval bound. None means pass; a draft is returned
    # before any primary outcome is read, and the result set is never weakened instead.
    registry = _capacity_registry(structure.registry_path)
    if unknown := sorted(set(structure.cardinalities) - set(CAPACITY_DIMENSIONS)):
        raise ec.EstimationError(f"unknown cardinalities: {unknown}", INVALID_CAPACITY_REGISTRY)
    dimensions = {name: int(structure.cardinalities.get(name, 0)) for name in CAPACITY_DIMENSIONS}
    bound = (plan.versions.get("visualization_catalog"), plan.versions.get("capacity"))
    failures: list[str] = []
    if bound != (registry.get("visualization_catalog_version"),
                 registry.get("capacity_registry_version")):
        failures.append(CAPACITY_VERSION_DRIFT)
    templates = tuple(registry["templates"])
    for evidence_id in structure.required_visual_evidence:
        carriers = [row for row in templates
                    if evidence_id in (row.get("visual_evidence_ids") or ())]
        if not carriers:
            failures.append(f"no_template:{evidence_id}")
            continue
        broken = [_violations(row, dimensions) for row in carriers]
        if all(broken):
            failures.extend(code for codes in broken for code in codes)
    if dimensions["evidence_items"] > int(registry.get("accessible_table_max_rows") or 0):
        failures.append("over_limit:accessible_table:evidence_items")
    if not failures:
        return None
    return DesignConflictDraftV1(
        conflict_code=CAPACITY_EXCEEDED, failed_rule_id=failures[0],
        affected_row_count=0, affected_unit_count=0, affected_dimension_counts=dimensions,
        evidence_artifact_ids=(plan.capacity_check.artifact_id, plan.context_manifest.artifact_id),
        why_no_permitted_operation=(
            "the frozen result cardinality no longer fits the approved delivery capacity"),
        material_design_fields=("primary_contrasts", "required_visual_evidence"),
        recommended_action="revise_design")
