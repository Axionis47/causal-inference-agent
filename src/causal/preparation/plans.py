"""Preparation plans, task contexts, previews, and receipt bundles (PRD-003 §7.3–§7.5, §10, §11.2, §24.2)."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from causal.preparation.contracts import (
    DesignConflictDraftV1,
    ObjectRefV1,
    _Payload,
    _Row,
)
from causal.shared.contracts import ArtifactRef, Identity, Sha256Hex
from causal.shared.envelope import TaskBudgets
from causal.shared.receipts import ExecutionReceiptV1

__all__ = [
    "EstimatorScopedRecipeV1", "ExecutionReceiptBundleV1", "FitScope", "GroupKind", "ItemPhase",
    "PlanItemPreviewV1", "PlanItemV1", "PlanPhase", "PlanPreviewV1", "PreparationPlanV1",
    "PreparationTaskContextV1", "PreparationTaskDraftV1", "TaskGroupV1", "TaskStopState",
    "has_cycle",
]

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


class PlanItemV1(_Row):
    """One registered operation, frozen with its parameters before any tool may run it (§10.1)."""

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


def has_cycle(items: tuple[PlanItemV1, ...]) -> bool:
    """True when the item dependency edges do not form a DAG (§7.4 topological execution)."""
    pending = {item.plan_item_id: set(item.depends_on) for item in items}
    while pending:
        ready = [item_id for item_id, deps in pending.items() if not deps & set(pending)]
        if not ready:
            return True
        for item_id in ready:
            del pending[item_id]
    return False


class EstimatorScopedRecipeV1(_Row):
    """A fold-scoped recipe recorded for PRD-004; PRD-003 never fits it (§11.2)."""

    recipe_id: Identity
    target_columns: _Ids
    fit_scope: Literal[FitScope.CROSS_FIT_TRAINING_FOLD] = FitScope.CROSS_FIT_TRAINING_FOLD
    method_id: Identity
    pack_version: Identity
    operation_id: Identity


class TaskGroupV1(_Row):
    """One §7.4 task group and the plan items it produced."""

    group_id: Identity
    group_kind: GroupKind
    plan_item_ids: _Ids


class PreparationPlanV1(_Payload):
    """Stabilization, repair, derivation, imputation, and recipes in one plan (§24.2)."""

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


class PreparationTaskContextV1(_Row):
    """The typed payload of one isolated preparation task envelope (§7.3)."""

    task_id: Identity
    phase: PlanPhase
    scope_kind: GroupKind
    scope_ids: tuple[Identity, ...]
    context_manifest: ArtifactRef
    frame: ArtifactRef | None
    gap_codes: _Ids
    column_roles: dict[str, Identity]
    column_concepts: dict[str, Identity]
    measurement_timing: dict[str, Identity]
    protected_columns: tuple[Identity, ...]
    evidence_refs: tuple[ArtifactRef, ...]
    permitted_operation_ids: _Ids
    dependency_task_ids: tuple[Identity, ...]
    dependency_plan_item_ids: tuple[Identity, ...]
    output_schema_version: Identity
    postcondition_ids: tuple[Identity, ...]
    budgets: TaskBudgets
    allowed_stopping_states: Annotated[tuple[TaskStopState, ...], Field(min_length=1)]


class PreparationTaskDraftV1(_Row):
    """What one hydrated single-shot preparation task returns (§24.1)."""

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


class ExecutionReceiptBundleV1(_Payload):
    """Every plan-item receipt plus the operation-level lineage that replaces cell rows (§24.2)."""

    schema_version: Literal["execution-receipt-bundle.v1"] = "execution-receipt-bundle.v1"
    plan: ArtifactRef
    receipts: Annotated[tuple[ExecutionReceiptV1, ...], Field(min_length=1)]
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


class PlanItemPreviewV1(_Row):
    """Expected effect of one plan item; producing it writes nothing (§17.2)."""

    plan_item_id: Identity
    expected_row_count: _NonNegInt
    expected_columns: _Ids
    missingness_delta: dict[str, int]
    affected_cell_count: _NonNegInt
    warnings: tuple[Identity, ...]


class PlanPreviewV1(_Row):
    """The ordered per-item preview a plan must have before any mutation tool unlocks."""

    plan: ArtifactRef
    plan_revision: _PositiveInt
    items: Annotated[tuple[PlanItemPreviewV1, ...], Field(min_length=1)]
