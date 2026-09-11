"""Method packs and §10.1 requirement templates (PRD-002 §10, §13)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, Field, ValidationError, model_validator

from causal.design.contracts import _Row
from causal.design.semantics import RoleName
from causal.shared.contracts import Identity
from causal.shared.envelope import (
    Criticality,
    EvidenceClass,
    MissingAction,
    RequirementScopeKind,
    SupportRequirement,
)

__all__ = [
    "METHOD_IDS", "PREREPAIR_DIAGNOSTIC_IDS", "TASK_KINDS", "AcceptedFactContractV1",
    "MethodPackRegistry", "MethodPackV1", "PackRegistryError", "RequirementTemplateV1",
    "load_method_packs",
    "load_requirement_templates", "verify_requirement_references",
]

INVALID_REGISTRY_FILE: Final = "invalid_registry_file"
DUPLICATE_METHOD: Final = "duplicate_method"
WRONG_PACK_COUNT: Final = "wrong_pack_count"
UNKNOWN_ROLE: Final = "unknown_role"
UNKNOWN_DIAGNOSTIC: Final = "unknown_diagnostic"
UNKNOWN_REQUIREMENT: Final = "unknown_requirement"
REQUIREMENT_MISMATCH: Final = "requirement_mismatch"
UNSUPPORTED_METHOD: Final = "unsupported_method"

METHOD_IDS: Final = ("randomized_experiment", "aipw", "did", "sharp_rdd")
TASK_KINDS: Final = ("intent", "semantic_batch", "role_evidence", "causal_context",
                     "role_ledger", "method_design")
# The closed pre-repair diagnostic vocabulary: one id per inspection named in PRD-002 §13.1–§13.4.
PREREPAIR_DIAGNOSTIC_IDS: Final = (
    "arm_counts", "assignment_unit_uniqueness", "cluster_sizes", "baseline_availability",
    "outcome_missingness", "compliance_availability", "power_precision_feasibility",
    "treatment_prevalence", "covariate_availability", "missingness", "rough_overlap",
    "level_sparsity", "effective_sample_feasibility", "cross_fitting_feasibility",
    "unit_period_uniqueness", "group_time_counts", "panel_completeness", "adoption_cohorts",
    "missingness_by_group_time", "pre_period_availability", "composition",
    "clustering_feasibility", "cutoff_side_counts", "distance_to_cutoff_support",
    "missingness_by_side_and_distance", "mass_points", "duplicates",
    "density_manipulation_warnings", "bandwidth_feasibility",
)
_ROLE_FIELDS: Final = ("required_roles", "optional_roles", "forbidden_adjustment_roles",
                       "imputation_forbidden_roles", "imputation_eligible_roles")
_KNOWN_ROLES: Final = frozenset(role.value for role in RoleName)
AssignmentMechanism = Literal["randomized", "self_selected", "policy_cutoff", "time_of_adoption"]
Estimand = Literal["itt", "ate", "att", "att_group_time_aggregate",
                   "late_at_cutoff"]
StructuralRequirement = Literal[
    "one_row_per_unit", "one_row_per_randomization_unit", "unit_time_or_group_time_rows",
    "running_variable_with_cutoff", "fixed_cutoff", "sharp_assignment_at_cutoff",
    "support_on_both_sides_of_cutoff", "treatment_binary", "arms_ge_2", "defined_comparator",
    "treatment_and_outcome_observed", "treated_and_comparison_groups", "adoption_time_defined",
    "pre_and_post_periods",
]
TaskKind = Literal["intent", "semantic_batch", "role_evidence", "causal_context", "role_ledger",
                   "method_design"]
AcceptedFactType = Literal[
    "text", "choice", "boolean", "number", "date", "duration", "column", "column_list",
    "mapping"]
AcceptedFactConsumer = Literal[
    "design_intent", "semantic_compiler", "causal_context", "role_ledger",
    "method_compiler", "preparation_compiler", "claim_compiler", "presentation_compiler"]

_Ids = Annotated[tuple[Identity, ...], Field(min_length=1)]


class PackRegistryError(ValueError):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class MethodPackV1(_Row):
    method_id: Identity
    pack_version: Identity
    display_name: str
    compatible_assignment_mechanisms: Annotated[tuple[AssignmentMechanism, ...],
                                                Field(min_length=1)]
    required_roles: _Ids
    optional_roles: tuple[Identity, ...]
    forbidden_adjustment_roles: _Ids
    supported_estimands: Annotated[tuple[Estimand, ...], Field(min_length=1)]
    required_context_requirement_ids: _Ids
    structural_requirements: Annotated[tuple[StructuralRequirement, ...], Field(min_length=1)]
    allowed_prerepair_diagnostic_ids: _Ids
    eligibility_rule_vocabulary: _Ids
    unusable_row_rule_ids: _Ids
    imputation_forbidden_roles: _Ids
    imputation_eligible_roles: tuple[Identity, ...]
    deletion_impact_dimensions: _Ids
    statements: dict[Identity, str]
    mandatory_assumption_ids: _Ids
    optional_assumption_ids: tuple[Identity, ...]
    mandatory_risk_ids: tuple[Identity, ...]
    optional_risk_ids: tuple[Identity, ...]
    mandatory_sensitivity_ids: tuple[Identity, ...]
    optional_sensitivity_ids: tuple[Identity, ...]
    invalidation_rule_ids: _Ids
    required_postrepair_diagnostic_ids: _Ids
    required_visual_evidence_ids: _Ids
    reserved_estimator_id: Identity

    @model_validator(mode="after")
    def _treatment_and_outcome_never_imputed(self) -> Self:
        forbidden = set(self.imputation_forbidden_roles)
        if not {RoleName.TREATMENT.value, RoleName.OUTCOME.value} <= forbidden:
            raise ValueError("imputation_forbidden_roles must include treatment and outcome")
        if forbidden & set(self.imputation_eligible_roles):
            raise ValueError("a role cannot be both forbidden and eligible for imputation")
        ids = {name for field in (
            "mandatory_assumption_ids", "optional_assumption_ids", "mandatory_risk_ids",
            "optional_risk_ids", "mandatory_sensitivity_ids", "optional_sensitivity_ids",
            "invalidation_rule_ids") for name in getattr(self, field)}
        if ids != set(self.statements):
            raise ValueError("every registered method statement must be classified exactly once")
        return self


class AcceptedFactContractV1(_Row):
    """How one validated human answer becomes a typed downstream fact."""

    fact_key: Identity
    value_type: AcceptedFactType
    consumer_ids: Annotated[tuple[AcceptedFactConsumer, ...], Field(min_length=1)]


class RequirementTemplateV1(_Row):
    requirement_id: Identity
    scope_kind: RequirementScopeKind
    fact_required: str
    why_required: str
    criticality: Criticality
    acceptable_evidence_types: Annotated[tuple[EvidenceClass, ...], Field(min_length=1)]
    required_support: SupportRequirement
    methods_required_for: _Ids
    missing_action: MissingAction
    expected_answer_schema: Identity
    user_may_know: bool
    accepted_fact: AcceptedFactContractV1

    @model_validator(mode="after")
    def _answer_schema_matches_fact_type(self) -> Self:
        schema = self.expected_answer_schema
        expected = ("choice" if schema.startswith("choice:") else {
            "free_text": "text", "free-text.v1": "text", "iso_date": "date",
            "boolean": "boolean", "number": "number", "column_name": "column",
            "column-list.v1": "column_list", "duration-window.v1": "duration",
            "level-map.v1": "mapping", "mapping-list.v1": "mapping",
            "timing-class.v1": "choice"}.get(schema))
        if expected != self.accepted_fact.value_type:
            raise ValueError(
                f"{schema!r} cannot produce {self.accepted_fact.value_type!r}")
        return self


class _PackFileV1(_Row):
    registry_version: Literal["method-packs.v1"]
    packs: tuple[MethodPackV1, ...]


class _RequirementFileV1(_Row):
    registry_version: Literal["context-requirements.v1"]
    requirements: tuple[RequirementTemplateV1, ...]


def _check_pack(pack: MethodPackV1) -> None:
    cited = {role for field in _ROLE_FIELDS for role in getattr(pack, field)}
    if unknown_roles := sorted(cited - _KNOWN_ROLES):
        raise PackRegistryError(f"{pack.method_id} unknown roles: {unknown_roles}", UNKNOWN_ROLE)
    allowed = set(pack.allowed_prerepair_diagnostic_ids)
    if unknown := sorted(allowed - set(PREREPAIR_DIAGNOSTIC_IDS)):
        raise PackRegistryError(f"{pack.method_id} unknown diagnostics: {unknown}",
                                UNKNOWN_DIAGNOSTIC)
    if outside := sorted(set(pack.required_postrepair_diagnostic_ids) - allowed):
        raise PackRegistryError(f"{pack.method_id} requires unallowed diagnostics: {outside}",
                                UNKNOWN_DIAGNOSTIC)


class MethodPackRegistry:
    registry_version: Final = "method-packs.v1"

    def __init__(self, packs: tuple[MethodPackV1, ...]) -> None:
        self._by_id: dict[str, MethodPackV1] = {}
        for pack in packs:
            if pack.method_id in self._by_id:
                raise PackRegistryError(f"duplicate pack {pack.method_id!r}", DUPLICATE_METHOD)
            _check_pack(pack)
            self._by_id[pack.method_id] = pack
        if sorted(self._by_id) != sorted(METHOD_IDS):
            raise PackRegistryError(f"expected packs {sorted(METHOD_IDS)}, got "
                                    f"{sorted(self._by_id)}", WRONG_PACK_COUNT)

    def get(self, method_id: str) -> MethodPackV1:
        pack = self._by_id.get(method_id)
        if pack is None:
            raise PackRegistryError(f"no method pack for {method_id!r}", UNSUPPORTED_METHOD)
        return pack

    def all(self) -> tuple[MethodPackV1, ...]:
        return tuple(self._by_id[method_id] for method_id in METHOD_IDS)

    def __len__(self) -> int:
        return len(self._by_id)


def _parse[Model: BaseModel](path: Path, model: type[Model]) -> Model:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise PackRegistryError(f"invalid registry file {path}: {error}",
                                INVALID_REGISTRY_FILE) from error


def load_method_packs(path: Path) -> MethodPackRegistry:
    return MethodPackRegistry(_parse(path, _PackFileV1).packs)


def load_requirement_templates(path: Path) -> dict[str, RequirementTemplateV1]:
    templates = _parse(path, _RequirementFileV1).requirements
    by_id = {template.requirement_id: template for template in templates}
    if len(by_id) != len(templates):
        raise PackRegistryError("duplicate requirement template", INVALID_REGISTRY_FILE)
    fact_keys = [template.accepted_fact.fact_key for template in templates]
    if len(fact_keys) != len(set(fact_keys)):
        raise PackRegistryError("duplicate accepted fact key", INVALID_REGISTRY_FILE)
    return by_id


def verify_requirement_references(
    packs: MethodPackRegistry, templates: dict[str, RequirementTemplateV1]
) -> None:
    for pack in packs.all():
        actual = set(pack.required_context_requirement_ids)
        if unknown := sorted(actual - set(templates)):
            raise PackRegistryError(f"{pack.method_id} unknown requirement ids: {unknown}",
                                    UNKNOWN_REQUIREMENT)
        expected = {requirement_id for requirement_id, template in templates.items()
                    if pack.method_id in template.methods_required_for}
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise PackRegistryError(
                f"{pack.method_id} requirement ids do not match the requirement registry; "
                f"missing: {missing}; extra: {extra}", REQUIREMENT_MISMATCH)
