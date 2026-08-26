"""Preparation plans, task payloads, receipt bundles, and previews (T-015 §1.2; PRD-003 §7, §11)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from causal.preparation.contracts import ConflictAction, DesignConflictDraftV1, ObjectRefV1
from causal.preparation.plans import (
    EstimatorScopedRecipeV1,
    ExecutionReceiptBundleV1,
    FitScope,
    GroupKind,
    ItemPhase,
    PlanItemPreviewV1,
    PlanItemV1,
    PlanPhase,
    PlanPreviewV1,
    PreparationPlanV1,
    PreparationTaskDraftV1,
    TaskGroupV1,
    TaskStopState,
    has_cycle,
)
from causal.shared.contracts import ArtifactRef
from causal.shared.receipts import ExecutionReceiptV1, FrameShapeV1, ReceiptStatus

HASH = "a" * 64
OTHER_HASH = "b" * 64
REF = ArtifactRef(artifact_id="art-1", content_hash=HASH)
NOW = datetime(2026, 8, 26, 12, 0, 0, tzinfo=UTC)


def item(item_id: str, *, depends_on: tuple[str, ...] = (),
         phase: ItemPhase = ItemPhase.REPAIR) -> PlanItemV1:
    return PlanItemV1(
        plan_item_id=item_id, phase=phase, operation_id="type_conversion",
        operation_version="v1", target_columns=("age",), output_column="age_prepared",
        parameters={"target_dtype": "float64"}, fit_scope=FitScope.NONE,
        predicted_missingness_change={"age_prepared": 0}, depends_on=depends_on,
        postcondition_ids=("schema_matches",), rationale_evidence=(REF,),
    )


PLAN: dict[str, Any] = {
    "plan_revision": 1, "phase": PlanPhase.PREPARATION,
    "items": (item("pi-1"), item("pi-2", depends_on=("pi-1",))),
    "eligibility_rule_ids": (), "unusable_row_rule_ids": (),
    "recipes": (EstimatorScopedRecipeV1(
        recipe_id="rec-1", target_columns=("age",), method_id="aipw",
        pack_version="aipw-pack.v1", operation_id="numeric_median_with_indicator",
    ),),
    "groups": (TaskGroupV1(group_id="g-1", group_kind=GroupKind.SINGLE_COLUMN,
                           plan_item_ids=("pi-1", "pi-2")),),
    "context_manifest": REF, "stabilized_frame": REF, "versions": {"operations": "v1"},
}


def receipt(**overrides: Any) -> ExecutionReceiptV1:
    base: dict[str, Any] = {
        "stage_run_id": "run-1", "plan_artifact_id": "art-1", "plan_item_id": "pi-1",
        "operation_id": "type_conversion", "operation_version": "v1",
        "implementation_version": "v1", "input_ref": REF, "output_ref": REF,
        "parameters_hash": HASH,
        "shape_before": FrameShapeV1(row_count=100, column_count=4),
        "shape_after": FrameShapeV1(row_count=100, column_count=5),
        "row_set_hash_before": HASH, "row_set_hash_after": HASH, "examined_count": 100,
        "changed_count": 3, "derived_count": 0, "imputed_count": 0, "warning_codes": (),
        "error_codes": (), "attempt_id": "att-1", "idempotency_key": "idem-1",
        "status": ReceiptStatus.SUCCEEDED, "started_at_utc": NOW, "finished_at_utc": NOW,
    }
    return ExecutionReceiptV1(**{**base, **overrides})


BUNDLE: dict[str, Any] = {
    "plan": REF, "receipts": (receipt(),), "changed_counts_by_column": {"age_prepared": 3},
    "missingness_before": {"age": 7}, "missingness_after": {"age_prepared": 0},
    "imputed_cell_mask": ObjectRefV1(object_locator="objects/" + HASH, content_hash=HASH),
    "row_set_hash": HASH, "parents": (REF,),
}


class TestPlanShape:
    def test_plan_roundtrips(self) -> None:
        plan = PreparationPlanV1(**PLAN)
        assert PreparationPlanV1.model_validate(plan.model_dump()) == plan

    def test_dependency_on_an_undeclared_item_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="depends on undeclared items"):
            PreparationPlanV1(**{**PLAN, "items": (item("pi-1", depends_on=("pi-9",)),)})

    def test_duplicate_plan_item_ids_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="must be unique"):
            PreparationPlanV1(**{**PLAN, "items": (item("pi-1"), item("pi-1"))})

    def test_a_group_naming_an_undeclared_item_is_rejected(self) -> None:
        group = TaskGroupV1(group_id="g-2", group_kind=GroupKind.TABLE_WIDE,
                            plan_item_ids=("pi-9",))
        with pytest.raises(ValidationError, match="names undeclared items"):
            PreparationPlanV1(**{**PLAN, "groups": (group,)})

    def test_a_dependency_cycle_is_rejected(self) -> None:
        items = (item("pi-1", depends_on=("pi-2",)), item("pi-2", depends_on=("pi-1",)))
        with pytest.raises(ValidationError, match="must form a DAG"):
            PreparationPlanV1(**{**PLAN, "items": items})

    def test_a_self_dependency_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cannot depend on itself"):
            item("pi-1", depends_on=("pi-1",))

    def test_stabilization_items_cannot_sit_in_a_preparation_plan(self) -> None:
        with pytest.raises(ValidationError, match="does not belong to a"):
            PreparationPlanV1(**{**PLAN, "items": (item("pi-1", phase=ItemPhase.STABILIZATION),),
                                 "groups": ()})

    def test_row_rules_and_recipes_belong_to_their_own_phase(self) -> None:
        with pytest.raises(ValidationError, match="stabilization phase only"):
            PreparationPlanV1(**{**PLAN, "eligibility_rule_ids": ("target_population_filter",)})
        stabilization = {
            **PLAN, "phase": PlanPhase.STABILIZATION,
            "items": (item("pi-1", phase=ItemPhase.STABILIZATION),),
            "groups": (TaskGroupV1(group_id="g-1", group_kind=GroupKind.TABLE_WIDE,
                                   plan_item_ids=("pi-1",)),),
        }
        with pytest.raises(ValidationError, match="preparation phase only"):
            PreparationPlanV1(**stabilization)


class TestHasCycle:
    def test_empty_and_linear_chains_are_acyclic(self) -> None:
        assert has_cycle(()) is False
        assert has_cycle((item("pi-1"), item("pi-2", depends_on=("pi-1",)))) is False

    def test_a_diamond_is_acyclic(self) -> None:
        items = (
            item("pi-1"), item("pi-2", depends_on=("pi-1",)), item("pi-3", depends_on=("pi-1",)),
            item("pi-4", depends_on=("pi-2", "pi-3")),
        )
        assert has_cycle(items) is False

    def test_a_three_item_cycle_is_detected(self) -> None:
        items = (
            item("pi-1", depends_on=("pi-3",)), item("pi-2", depends_on=("pi-1",)),
            item("pi-3", depends_on=("pi-2",)),
        )
        assert has_cycle(items) is True


def test_the_four_fit_scopes_match_the_prd() -> None:
    assert tuple(scope.value for scope in FitScope) == (
        "none", "frozen_frame_blinded", "pre_treatment_only", "cross_fit_training_fold",
    )


def test_a_recipe_is_recorded_never_fitted() -> None:
    recipe = PLAN["recipes"][0]
    assert recipe.fit_scope is FitScope.CROSS_FIT_TRAINING_FOLD
    # No field can carry a learned value out of PRD-003 (§11.2), and none can be added.
    assert not {"fitted_values", "learned_values", "fitted_parameters"} & set(
        EstimatorScopedRecipeV1.model_fields
    )
    with pytest.raises(ValidationError):
        EstimatorScopedRecipeV1(**{**recipe.model_dump(), "fitted_values": {"age": 31.0}})


class TestTaskPayloads:
    def test_a_proposing_draft_must_carry_plan_items(self) -> None:
        with pytest.raises(ValidationError, match="plan_items is non-empty"):
            PreparationTaskDraftV1(task_id="task-1", stop_state=TaskStopState.PROPOSED,
                                   plan_items=(), design_conflict=None, notes=())

    def test_a_conflict_draft_carries_the_conflict_and_no_items(self) -> None:
        draft = DesignConflictDraftV1(
            conflict_code="grain_unresolved", failed_rule_id="one_row_per_unit",
            affected_row_count=4, affected_unit_count=2, affected_dimension_counts={},
            evidence_artifact_ids=("art-1",), why_no_permitted_operation="aggregation is forbidden",
            material_design_fields=("approved_grain",), recommended_action=ConflictAction.ASK_USER,
        )
        built = PreparationTaskDraftV1(task_id="task-1", stop_state=TaskStopState.DESIGN_CONFLICT,
                                       plan_items=(), design_conflict=draft, notes=("n",))
        assert built.design_conflict == draft
        with pytest.raises(ValidationError, match="design_conflict is present iff"):
            PreparationTaskDraftV1(task_id="task-1", stop_state=TaskStopState.FAILED,
                                   plan_items=(), design_conflict=draft, notes=())


class TestReceiptBundle:
    def test_bundle_roundtrips(self) -> None:
        built = ExecutionReceiptBundleV1(**BUNDLE)
        assert ExecutionReceiptBundleV1.model_validate(built.model_dump()) == built

    def test_a_receipt_that_drifts_from_the_frozen_row_set_is_rejected(self) -> None:
        drifted = receipt(row_set_hash_after=OTHER_HASH)
        with pytest.raises(ValidationError, match="does not assert the frozen row_set_hash"):
            ExecutionReceiptBundleV1(**{**BUNDLE, "receipts": (drifted,)})


def test_a_preview_describes_every_item_and_writes_nothing() -> None:
    preview = PlanPreviewV1(plan=REF, plan_revision=1, items=(
        PlanItemPreviewV1(
            plan_item_id="pi-1", expected_row_count=100, expected_columns=("age_prepared",),
            missingness_delta={"age_prepared": -7}, affected_cell_count=7, warnings=(),
        ),
    ))
    assert PlanPreviewV1.model_validate(preview.model_dump()) == preview
