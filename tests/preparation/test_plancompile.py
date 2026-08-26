"""Contract gaps, §7.4 grouping, and the §7.5 fan-in rejection matrix (T-018 §1.2)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from causal.preparation import contracts as ct
from causal.preparation import plancompile as pc
from causal.preparation import plans as pl
from causal.preparation.operations import load_operation_registry
from causal.preparation.plans import load_preparation_packs
from tests.preparation.test_contracts import MANIFEST, REF

REGISTRIES = Path(__file__).resolve().parents[2] / "registries"
AIPW = load_preparation_packs(
    REGISTRIES / "method-pack-preparation.v1.json", REGISTRIES / "method-packs.v1.json"
).get("aipw", "aipw-pack.v1")
OPERATIONS = load_operation_registry(REGISTRIES / "repair-operations.v1.json")
CONTEXT = ct.PreparationContextManifestV1(**{
    **MANIFEST,
    "column_roles": {"age": "confounder_candidate", "treat": "treatment", "signup": "time"},
    "protected_columns": ("treat",), "permitted_imputation_columns": ("age",),
    "permitted_operation_ids": ("missing_sentinel_normalization", "type_conversion",
                                "registered_derivation", "numeric_median_imputation",
                                "estimator_scoped_recipe")})
SURFACE = pc.ContractSurfaceV1(
    required_dtypes={"age": "Float64", "treat": "Int64", "post": "Boolean"},
    sentinel_columns={"age": "ev:sentinel/age"}, required_derivations={"post": "post_period"},
    derivation_sources={"post": ("signup",)},
    imputation_targets={"age": "numeric_median_with_indicator"},
    coupled_columns=(("unit_id", "signup"),), recipe_groups={"r1": ("x1", "x2")},
    table_wide_codes=("grain_unvalidated",))
SCHEMA = {"age": "String", "treat": "Int64"}
AGE_GAP = (pc.FrameGapV1(gap_code=pc.TYPE_MISMATCH, column="age"),)


def item(plan_item_id: str, operation_id: str, targets: tuple[str, ...], *,
         phase: pl.ItemPhase = pl.ItemPhase.REPAIR, output: str | None = None,
         parameters: dict[str, Any] | None = None, fit_scope: pl.FitScope = pl.FitScope.NONE,
         depends_on: tuple[str, ...] = ()) -> pl.PlanItemV1:
    return pl.PlanItemV1(
        plan_item_id=plan_item_id, phase=phase, operation_id=operation_id, operation_version="v1",
        target_columns=targets, output_column=output, parameters=parameters or {},
        fit_scope=fit_scope, predicted_missingness_change={}, depends_on=depends_on,
        postcondition_ids=(), rationale_evidence=())


def draft(task_id: str, *items: pl.PlanItemV1,
          stop_state: pl.TaskStopState = pl.TaskStopState.PROPOSED,
          conflict: ct.DesignConflictDraftV1 | None = None) -> pl.PreparationTaskDraftV1:
    return pl.PreparationTaskDraftV1(task_id=task_id, stop_state=stop_state, plan_items=items,
                                     design_conflict=conflict, notes=())


def run(*drafts: pl.PreparationTaskDraftV1, gaps: tuple[pc.FrameGapV1, ...] = AGE_GAP
        ) -> pl.PreparationPlanV1 | pc.ConflictRoute:
    return pc.reconcile(drafts, pc.ReconcileInputs(
        manifest=CONTEXT, operations=OPERATIONS, pack=AIPW, context_manifest=REF, gaps=gaps,
        groups=pc.group_gaps(gaps, SURFACE).groups, versions={"schema": "v1"}))


def rejected(*drafts: pl.PreparationTaskDraftV1, **kwargs: Any) -> tuple[str, ...]:
    with pytest.raises(pc.PlanCompileError) as error:
        run(*drafts, **kwargs)
    assert error.value.code == pc.FAN_IN_REJECTED
    return error.value.detail_codes


def test_every_registered_mismatch_is_computed_once_and_a_satisfied_column_yields_none() -> None:
    # `treat` already satisfies the contract, so it produces no task (EV-P3-001).
    assert [(gap.gap_code, gap.column) for gap in pc.contract_gaps(SCHEMA, SURFACE, {"age": 3})] == [
        ("grain_unvalidated", pc.TABLE_WIDE_SCOPE), (pc.TYPE_MISMATCH, "age"),
        (pc.SENTINEL_EVIDENCE, "age"), (pc.DERIVATION_REQUIRED, "post"),
        (pc.IMPUTATION_REQUIRED, "age")]
    assert pc.contract_gaps(SCHEMA, pc.ContractSurfaceV1(imputation_targets={"age": "m"})) == ()
    assert pc.contract_gaps(SCHEMA, pc.ContractSurfaceV1(
        required_dtypes={"weight": "Float64"}))[0].gap_code == pc.MISSING_PREPARED_COLUMN


def test_each_scope_lands_in_its_own_group_with_its_gap_codes_and_dependencies() -> None:
    gaps = pc.contract_gaps(SCHEMA, SURFACE, {"age": 3}) + (
        pc.FrameGapV1(gap_code=pc.MISSING_PREPARED_COLUMN, column="signup"),
        pc.FrameGapV1(gap_code=pc.IMPUTATION_REQUIRED, column="x1"))
    plan = pc.group_gaps(gaps, SURFACE)
    by_id = {group.group_id: group for group in plan.groups}
    assert {name: group.group_kind for name, group in by_id.items()} == {
        "g:coupled:signup+unit_id": pl.GroupKind.COUPLED_COLUMNS,
        "g:recipe:r1": pl.GroupKind.RECIPE_GROUP, "g:single:age": pl.GroupKind.SINGLE_COLUMN,
        "g:single:post": pl.GroupKind.SINGLE_COLUMN,
        pc.TABLE_WIDE_GROUP: pl.GroupKind.TABLE_WIDE}
    assert by_id["g:single:post"].depends_on == ("g:coupled:signup+unit_id", pc.TABLE_WIDE_GROUP)
    assert by_id["g:single:age"].gap_codes == (
        pc.IMPUTATION_REQUIRED, pc.SENTINEL_EVIDENCE, pc.TYPE_MISMATCH)
    assert (plan.waves[0], plan.waves[-1]) == ((pc.TABLE_WIDE_GROUP,), ("g:single:post",))


def test_the_queue_dispatches_at_most_eight_groups_per_wave_in_a_stable_order() -> None:
    gaps = tuple(pc.FrameGapV1(gap_code=pc.TYPE_MISMATCH, column=f"c{index}")
                 for index in range(10))
    plan = pc.group_gaps(gaps, pc.ContractSurfaceV1())
    assert [len(wave) for wave in plan.waves] == [pc.MAX_CONCURRENT_GROUPS, 2]
    assert plan.waves[0] == tuple(sorted(group.group_id for group in plan.groups))[:8]


class TestFanIn:
    def test_one_ordered_plan_is_compiled_from_the_proposals(self) -> None:
        plan = run(draft("t-2", item("i-1", "missing_sentinel_normalization", ("age",),
                                     output="age_clean"),
                         item("i-2", "type_conversion", ("age",), output="age_num",
                              depends_on=("i-1",))),
                   draft("t-1", item("i-3", "registered_derivation", ("signup",), output="post")))
        assert isinstance(plan, pl.PreparationPlanV1)
        assert [found.plan_item_id for found in plan.items] == ["i-3", "i-1", "i-2"]
        assert {group.group_id: group.plan_item_ids for group in plan.groups} == {
            "g:single:age": ("i-1", "i-2")}
        assert plan.versions == {"schema": "v1"}

    def test_a_cross_fit_item_is_recorded_as_an_estimator_scoped_recipe(self) -> None:
        plan = run(draft("t-1", item(
            "i-1", "estimator_scoped_recipe", ("age",), phase=pl.ItemPhase.IMPUTATION,
            fit_scope=pl.FitScope.CROSS_FIT_TRAINING_FOLD, parameters={"recipe_id": "r-1"})))
        assert isinstance(plan, pl.PreparationPlanV1)
        assert [recipe.recipe_id for recipe in plan.recipes] == ["r-1"]

    def test_a_design_conflict_draft_routes_to_the_conflict_path(self) -> None:
        conflict = ct.DesignConflictDraftV1(
            conflict_code="grain_unestablished", failed_rule_id="one_row_per_unit",
            affected_row_count=4, affected_unit_count=2, affected_dimension_counts={},
            evidence_artifact_ids=(), why_no_permitted_operation="no registered repair",
            material_design_fields=("output_grain",),
            recommended_action=ct.ConflictAction.ASK_USER)
        route = run(draft("t-1", stop_state=pl.TaskStopState.DESIGN_CONFLICT, conflict=conflict))
        assert isinstance(route, pc.ConflictRoute)
        assert (route.task_id, route.draft.conflict_code) == ("t-1", "grain_unestablished")

    @pytest.mark.parametrize(("drafts", "expected"), [
        ((draft("t-1", item("i-1", "invented_operation", ("age",))),), pc.UNREGISTERED_OPERATION),
        ((draft("t-1", item("i-1", "category_normalization", ("age",))),),
         pc.OPERATION_NOT_PERMITTED),
        ((draft("t-1", item("i-1", "type_conversion", ("age", "treat"))),),
         pc.PROTECTED_ROLE_TARGET),
        ((draft("t-1", item("i-1", "numeric_median_imputation", ("age",),
                            phase=pl.ItemPhase.IMPUTATION)),), pc.ILLEGAL_FIT_SCOPE),
        ((draft("t-1", item("i-1", "type_conversion", ("age",),
                            phase=pl.ItemPhase.STABILIZATION)),), pc.ROW_MEMBERSHIP_CHANGE),
        ((draft("t-1", item("i-1", "type_conversion", ("age",), parameters={"target_dtype": "x"})),
          draft("t-2", item("i-2", "type_conversion", ("age",),
                            parameters={"target_dtype": "x"}))), pc.DUPLICATE_MAPPING),
        ((draft("t-1", item("i-1", "type_conversion", ("age",), parameters={"target_dtype": "x"})),
          draft("t-2", item("i-2", "type_conversion", ("age",),
                            parameters={"target_dtype": "y"}))), pc.INCONSISTENT_MAPPING),
        ((draft("t-1", item("i-1", "type_conversion", ("age",), output="a"),
                item("i-2", "missing_sentinel_normalization", ("age",), output="b")),),
         pc.CONFLICTING_OPERATION_ON_COLUMN),
        ((draft("t-1", item("i-1", "type_conversion", ("age",), depends_on=("gone",))),),
         pc.UNKNOWN_DEPENDENCY),
        ((draft("t-1", stop_state=pl.TaskStopState.FAILED),), pc.TASK_NOT_PROPOSED),
    ])
    def test_the_rejection_matrix_reports_stable_codes(
        self, drafts: Sequence[pl.PreparationTaskDraftV1], expected: str
    ) -> None:
        assert expected in rejected(*drafts)

    def test_an_uncovered_gap_stops_the_plan(self) -> None:
        gaps = AGE_GAP + (pc.FrameGapV1(gap_code=pc.MISSING_PREPARED_COLUMN, column="weight"),)
        assert rejected(draft("t-1", item("i-1", "type_conversion", ("age",))),
                        gaps=gaps) == (pc.GAP_NOT_COVERED,)
