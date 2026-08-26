"""Preparation plans, previews, receipt bundles, and the method-pack overlay (PRD-003 §7.3–§7.5, §10–§13, §24.2)."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import Field, ValidationError, model_validator

from causal.preparation.contracts import DesignConflictDraftV1, ObjectRefV1, _Payload, _Row
from causal.shared.contracts import ArtifactRef, Identity, Sha256Hex
from causal.shared.receipts import ExecutionReceiptV1
from causal.shared.registry import INVALID_REGISTRY_FILE, RegistryError

UNKNOWN_METHOD_PACK: Final = "unknown_method_pack"

_NonNegInt = Annotated[int, Field(ge=0)]
_PositiveInt = Annotated[int, Field(ge=1)]
_Ids = Annotated[tuple[Identity, ...], Field(min_length=1)]
ParameterValue = str | int | float | bool | None


class FitScope(StrEnum):
    """Where an imputation may learn its values (§11.2)."""

    NONE = "none"
    FROZEN_FRAME_BLINDED = "frozen_frame_blinded"
    PRE_TREATMENT_ONLY = "pre_treatment_only"
    CROSS_FIT_TRAINING_FOLD = "cross_fit_training_fold"


# The registered imputation strategy id, as a method pack states it, to the operation that runs
# it; the one map the manifest's permitted operations and the compiler both read (§25.1).
STRATEGY_OPERATIONS: Final[dict[str, str]] = {
    "numeric_median_with_indicator": "numeric_median_imputation",
    "categorical_explicit_missing_level": "categorical_missing_encoding"}
# A fold-scoped target is never fitted here: PRD-003 registers the recipe PRD-004 fits (§11.2).
CROSS_FIT_OPERATION: Final = "estimator_scoped_recipe"


# The operation one permitted imputation target compiles to, or None when V1 has none.
def strategy_operation(strategy_id: str, fit_scope: FitScope) -> str | None:
    operation = STRATEGY_OPERATIONS.get(strategy_id)
    if fit_scope is FitScope.CROSS_FIT_TRAINING_FOLD:
        return CROSS_FIT_OPERATION if operation is not None else None
    # A pre-treatment median fit needs an approved boolean pre-period mask, and no V1 contract
    # pins one, so that target has no registered resolution and becomes a §16 conflict (§25.1).
    if operation == "numeric_median_imputation" and fit_scope is FitScope.PRE_TREATMENT_ONLY:
        return None
    return operation


class ItemPhase(StrEnum):
    STABILIZATION = "stabilization"
    REPAIR = "repair"
    DERIVATION = "derivation"
    IMPUTATION = "imputation"
    DIAGNOSTIC = "diagnostic"


class PlanPhase(StrEnum):
    STABILIZATION = "stabilization"
    PREPARATION = "preparation"


class GroupKind(StrEnum):
    """The §7.4 fan-out scopes; a group is one task, never one task per column."""

    SINGLE_COLUMN = "single_column"
    COUPLED_COLUMNS = "coupled_columns"
    RECIPE_GROUP = "recipe_group"
    TABLE_WIDE = "table_wide"


class TaskStopState(StrEnum):
    """The explicit stopping states one preparation task may end in (§7.3)."""

    PROPOSED = "proposed"
    NEEDS_DEPENDENCY = "needs_dependency"
    DESIGN_CONFLICT = "design_conflict"
    FAILED = "failed"


# One registered operation, frozen with its parameters before any tool may run it (§10.1).
class PlanItemV1(_Row):
    plan_item_id: Identity
    phase: ItemPhase
    operation_id: Identity
    operation_version: Identity
    target_columns: tuple[Identity, ...]
    output_column: Identity | None
    parameters: dict[str, ParameterValue]
    fit_scope: FitScope
    predicted_missingness_change: dict[str, int]
    depends_on: tuple[Identity, ...]
    postcondition_ids: tuple[Identity, ...]
    rationale_evidence: tuple[ArtifactRef, ...]

    @model_validator(mode="after")
    def _no_self_dependency(self) -> Self:
        if self.plan_item_id in self.depends_on:
            raise ValueError(f"plan item {self.plan_item_id} cannot depend on itself")
        return self


# True when the item dependency edges do not form a DAG (§7.4 topological execution).
def has_cycle(items: tuple[PlanItemV1, ...]) -> bool:
    pending = {item.plan_item_id: set(item.depends_on) for item in items}
    while pending:
        ready = [item_id for item_id, deps in pending.items() if not deps & set(pending)]
        if not ready:
            return True
        for item_id in ready:
            del pending[item_id]
    return False


# A fold-scoped recipe recorded for PRD-004; PRD-003 never fits it (§11.2).
class EstimatorScopedRecipeV1(_Row):
    recipe_id: Identity
    target_columns: _Ids
    fit_scope: Literal[FitScope.CROSS_FIT_TRAINING_FOLD] = FitScope.CROSS_FIT_TRAINING_FOLD
    method_id: Identity
    pack_version: Identity
    operation_id: Identity


# One §7.4 task group and the plan items it produced.
class TaskGroupV1(_Row):
    group_id: Identity
    group_kind: GroupKind
    plan_item_ids: _Ids


# Stabilization, repair, derivation, imputation, and recipes in one plan (§24.2).
class PreparationPlanV1(_Payload):
    schema_version: Literal["preparation-plan.v1"] = "preparation-plan.v1"
    plan_revision: _PositiveInt
    phase: PlanPhase
    items: tuple[PlanItemV1, ...]
    eligibility_rule_ids: tuple[Identity, ...]
    unusable_row_rule_ids: tuple[Identity, ...]
    recipes: tuple[EstimatorScopedRecipeV1, ...]
    groups: tuple[TaskGroupV1, ...]
    context_manifest: ArtifactRef
    stabilized_frame: ArtifactRef | None
    versions: dict[str, Identity]

    @model_validator(mode="after")
    def _items_form_a_closed_dag(self) -> Self:
        declared = [item.plan_item_id for item in self.items]
        if len(set(declared)) != len(declared):
            raise ValueError("plan_item_id must be unique inside one plan")
        known = set(declared)
        for item in self.items:
            if unknown := sorted(set(item.depends_on) - known):
                raise ValueError(f"{item.plan_item_id} depends on undeclared items {unknown}")
        for group in self.groups:
            if unknown := sorted(set(group.plan_item_ids) - known):
                raise ValueError(f"group {group.group_id} names undeclared items {unknown}")
        if has_cycle(self.items):
            raise ValueError("plan item dependencies must form a DAG")
        return self

    @model_validator(mode="after")
    def _phase_matches_its_items(self) -> Self:
        stabilization = self.phase is PlanPhase.STABILIZATION
        for item in self.items:
            if (item.phase is ItemPhase.STABILIZATION) is not stabilization:
                raise ValueError(f"{item.plan_item_id} does not belong to a {self.phase} plan")
        rules = self.eligibility_rule_ids + self.unusable_row_rule_ids
        if rules and not stabilization:
            raise ValueError("row rules belong to the stabilization phase only")
        if self.recipes and stabilization:
            raise ValueError("estimator-scoped recipes belong to the preparation phase only")
        return self


# What one hydrated single-shot preparation task returns (§24.1).
class PreparationTaskDraftV1(_Row):
    task_id: Identity
    stop_state: TaskStopState
    plan_items: tuple[PlanItemV1, ...]
    design_conflict: DesignConflictDraftV1 | None
    notes: tuple[str, ...]

    @model_validator(mode="after")
    def _payload_matches_the_stop_state(self) -> Self:
        proposed = self.stop_state is TaskStopState.PROPOSED
        conflict = self.stop_state is TaskStopState.DESIGN_CONFLICT
        if bool(self.plan_items) is not proposed:
            raise ValueError("plan_items is non-empty if and only if stop_state is proposed")
        if (self.design_conflict is not None) is not conflict:
            raise ValueError("design_conflict is present iff stop_state is design_conflict")
        return self


# Every plan-item receipt plus the operation-level lineage that replaces cell rows (§24.2).
class ExecutionReceiptBundleV1(_Payload):
    schema_version: Literal["execution-receipt-bundle.v1"] = "execution-receipt-bundle.v1"
    plan: ArtifactRef
    # §25: a satisfied contract compiles no plan item, so an honest bundle may be empty.
    receipts: tuple[ExecutionReceiptV1, ...]
    changed_counts_by_column: dict[str, int]
    missingness_before: dict[str, int]
    missingness_after: dict[str, int]
    imputed_cell_mask: ObjectRefV1 | None
    row_set_hash: Sha256Hex
    parents: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def _every_receipt_asserts_the_bundle_row_set(self) -> Self:
        for receipt in self.receipts:
            observed = (receipt.row_set_hash_before, receipt.row_set_hash_after)
            if any(value != self.row_set_hash for value in observed):
                raise ValueError(
                    f"receipt {receipt.plan_item_id} does not assert the frozen row_set_hash"
                )
        return self


# Expected effect of one plan item; producing it writes nothing (§17.2).
class PlanItemPreviewV1(_Row):
    plan_item_id: Identity
    expected_row_count: _NonNegInt
    expected_columns: _Ids
    missingness_delta: dict[str, int]
    affected_cell_count: _NonNegInt
    warnings: tuple[Identity, ...]


# The ordered per-item preview a plan must have before any mutation tool unlocks.
class PlanPreviewV1(_Row):
    plan: ArtifactRef
    plan_revision: _PositiveInt
    items: Annotated[tuple[PlanItemPreviewV1, ...], Field(min_length=1)]


# -- the §12/§13 method-pack preparation overlay (D-057) ----------------------


# One permitted imputation target: the role, its fit scope, and the registered strategy.
class ImputationTargetV1(_Row):
    role: Identity
    fit_scope: FitScope
    strategy_id: Identity
    requires_missingness_indicator: bool


# The §13 preparation contract one method pack adds, keyed by (method_id, pack_version).
class PreparationPackV1(_Row):
    method_id: Identity
    pack_version: Identity
    permitted_disposition_rule_ids: _Ids
    protected_roles: _Ids
    required_observed_role_rules: tuple[Identity, ...]
    minimum_rows: _PositiveInt
    minimum_unique_units: _PositiveInt
    required_structure_gates: _Ids
    cell_support_gates: tuple[Identity, ...]
    dimension_impact_dimensions: _Ids
    invalidation_rule_ids: _Ids
    permitted_repair_operation_ids: _Ids
    permitted_imputation_targets: tuple[ImputationTargetV1, ...]
    required_missingness_indicators: tuple[Identity, ...]
    required_poststabilization_diagnostic_ids: _Ids
    required_postrepair_diagnostic_ids: _Ids
    prepared_frame_schema_id: Identity
    estimator_input_contract_id: Identity

    @model_validator(mode="after")
    def _protected_roles_are_never_imputed(self) -> Self:
        targets = {target.role for target in self.permitted_imputation_targets}
        if overlap := sorted(targets & set(self.protected_roles)):
            raise ValueError(f"{self.method_id} would impute protected roles: {overlap}")
        return self


class _OverlayFileV1(_Row):
    registry_version: Literal["method-pack-preparation.v1"]
    packs: tuple[PreparationPackV1, ...]


# The (method_id, pack_version) pairs of the frozen design registry, read as raw data.
def _design_pack_keys(method_packs_path: Path) -> set[tuple[str, str]]:
    try:
        document = json.loads(method_packs_path.read_text(encoding="utf-8"))
        return {(row["method_id"], row["pack_version"]) for row in document["packs"]}
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise RegistryError(f"invalid registry file {method_packs_path}: {error}",
                            INVALID_REGISTRY_FILE) from error


class PreparationPackRegistry:
    """The preparation overlay; a row that no design pack backs fails closed."""

    registry_version: Final = "method-pack-preparation.v1"

    def __init__(self, packs: tuple[PreparationPackV1, ...], known: set[tuple[str, str]]) -> None:
        self._by_key: dict[tuple[str, str], PreparationPackV1] = {}
        for pack in packs:
            key = (pack.method_id, pack.pack_version)
            if key in self._by_key:
                raise RegistryError(f"duplicate overlay row {key}", INVALID_REGISTRY_FILE)
            if key not in known:
                raise RegistryError(f"no method pack for {key}", UNKNOWN_METHOD_PACK)
            self._by_key[key] = pack
        if missing := sorted(known - set(self._by_key)):
            raise RegistryError(f"overlay rows missing for {missing}", UNKNOWN_METHOD_PACK)

    def get(self, method_id: str, pack_version: str) -> PreparationPackV1:
        pack = self._by_key.get((method_id, pack_version))
        if pack is None:
            raise RegistryError(f"no preparation overlay for {(method_id, pack_version)}",
                                UNKNOWN_METHOD_PACK)
        return pack

    def all(self) -> tuple[PreparationPackV1, ...]:
        return tuple(self._by_key.values())

    def __len__(self) -> int:
        return len(self._by_key)


# The T-011 `eligibility_rule_vocabulary` of one design pack, read as raw registry data.
def eligibility_vocabulary(method_packs_path: Path, method_id: str) -> tuple[str, ...]:
    try:
        document = json.loads(method_packs_path.read_text(encoding="utf-8"))
        for pack in document["packs"]:
            if pack["method_id"] == method_id:
                return tuple(pack["eligibility_rule_vocabulary"])
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise RegistryError(f"invalid registry file {method_packs_path}: {error}",
                            INVALID_REGISTRY_FILE) from error
    raise RegistryError(f"no method pack for {method_id}", UNKNOWN_METHOD_PACK)


# Load the overlay and bind every row to the frozen design method packs (§13).
def load_preparation_packs(path: Path, method_packs_path: Path) -> PreparationPackRegistry:
    try:
        parsed = _OverlayFileV1.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise RegistryError(f"invalid registry file {path}: {error}",
                            INVALID_REGISTRY_FILE) from error
    return PreparationPackRegistry(parsed.packs, _design_pack_keys(method_packs_path))
