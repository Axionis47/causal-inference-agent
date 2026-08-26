"""The seven registered operation families and their guards (T-017 §1.1; PRD-003 §10, §11)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from causal.preparation import operations as ops
from causal.preparation.plans import FitScope, ItemPhase, PlanItemV1
from causal.shared.contracts import ArtifactRef
from causal.shared.registry import RegistryError

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "registries" / "repair-operations.v1.json"
REGISTRY = ops.load_operation_registry(REGISTRY_PATH)
REF = ArtifactRef(artifact_id="ev-1", content_hash="a" * 64)
CONTEXT = ops.OperationContext(
    column_roles={"arm": "treatment", "y": "outcome", "age": "precision_covariate",
                  "city": "precision_covariate", "score": "confounder_candidate"},
    measurement_timing={"age": "pre_treatment", "y": "post_treatment", "pre": "pre_treatment"},
    permitted_repair_columns=("age", "age_text", "city", "arm", "joined", "score", "frac",
                              "ghost"),
    permitted_imputation_columns=("age", "city", "score"))
IMPUTE: dict[str, Any] = {"phase": ItemPhase.IMPUTATION, "fit_scope": FitScope.FROZEN_FRAME_BLINDED}
DERIVE: dict[str, Any] = {"phase": ItemPhase.DERIVATION}


def frame() -> pl.DataFrame:
    return pl.DataFrame({
        "source_row_id": ["r1", "r2", "r3", "r4"],
        "age": [30.0, None, 50.0, -999.0],
        "age_text": ["30", "41", "50", "x"],
        "frac": [1.5, 2.0, None, 3.25],
        "city": ["NY", None, "sf", "NY"],
        "arm": ["a", "b", "a", "b"],
        "y": [1.0, None, 3.0, 4.0],
        "joined": ["2020-01-05", "2021-06-30", "2019-03-02", "2022-12-31"],
        "pre": [True, True, False, False],
        "score": [1.0, 2.0, None, 4.0]})


def item(operation_id: str, *, targets: tuple[str, ...] = ("age",), output: str | None = "out",
         phase: ItemPhase = ItemPhase.REPAIR, fit_scope: FitScope = FitScope.NONE,
         **parameters: Any) -> PlanItemV1:
    return PlanItemV1(
        plan_item_id="pi-1", phase=phase, operation_id=operation_id, operation_version="v1",
        target_columns=targets, output_column=output, parameters=parameters, fit_scope=fit_scope,
        predicted_missingness_change={}, depends_on=(), postcondition_ids=("row_set_invariance",),
        rationale_evidence=(REF,))


def run(plan_item: PlanItemV1, *, data: pl.DataFrame | None = None,
        context: ops.OperationContext = CONTEXT) -> ops.OperationResult:
    source = frame() if data is None else data
    return ops.run_operation(source, plan_item, REGISTRY.get(plan_item.operation_id), context)


GOLDENS = (
    item("missing_sentinel_normalization", sentinels='["-999.0"]'),
    item("type_conversion", targets=("age_text",), target_dtype="string"),
    item("category_normalization", targets=("city",),
         mapping='{"NY": "new_york", "sf": "san_francisco"}'),
    item("registered_derivation", targets=("y",), derivation_id="outcome_observed", **DERIVE),
    item("numeric_median_imputation", indicator_column="out_missing", **IMPUTE),
    item("categorical_missing_encoding", targets=("city",), missing_level="__missing__",
         phase=ItemPhase.IMPUTATION),
    item("estimator_scoped_recipe", targets=("score",), output=None, recipe_id="rec-1",
         strategy_id="numeric_median_with_indicator", phase=ItemPhase.IMPUTATION,
         fit_scope=FitScope.CROSS_FIT_TRAINING_FOLD),
)


@pytest.mark.parametrize("plan_item", GOLDENS, ids=lambda plan_item: plan_item.operation_id)
def test_every_family_adds_columns_and_leaves_the_frozen_row_set_alone(
        plan_item: PlanItemV1) -> None:
    source, result = frame(), run(plan_item)
    assert result.frame.height == source.height
    assert result.frame["source_row_id"].equals(source["source_row_id"])
    assert all(result.frame[column].equals(source[column]) for column in source.columns)


def test_missing_sentinel_normalization_nulls_the_approved_encoding_only() -> None:
    result = run(GOLDENS[0])
    assert result.frame["out"].to_list() == [30.0, None, 50.0, None]
    assert result.change_counts == {"out": 1}


def test_type_conversion_casts_strictly_and_refuses_a_lossy_cast() -> None:
    dated = run(item("type_conversion", targets=("joined",), target_dtype="date",
                     date_format="%Y-%m-%d"))
    assert dated.frame.schema["out"] == pl.Date
    with pytest.raises(ops.OperationError) as lossy:
        run(item("type_conversion", targets=("frac",), target_dtype="int64"))
    assert lossy.value.code == "lossy_conversion"
    with pytest.raises(ops.OperationError) as unparsed:
        run(item("type_conversion", targets=("age_text",), target_dtype="int64"))
    assert unparsed.value.code == "lossy_conversion"
    assert "'x'" not in str(unparsed.value)


def test_category_normalization_maps_one_to_one_and_refuses_an_unmapped_level() -> None:
    result = run(GOLDENS[2])
    assert result.frame["out"].to_list() == ["new_york", None, "san_francisco", "new_york"]
    assert result.change_counts == {"out": 3}
    with pytest.raises(ops.OperationError) as unmapped:
        run(item("category_normalization", targets=("city",), mapping='{"NY": "new_york"}'))
    assert unmapped.value.code == "unmapped_category"
    assert "sf" not in str(unmapped.value)


@pytest.mark.parametrize(("plan_item", "expected"), [
    (item("registered_derivation", targets=("y",), derivation_id="outcome_observed", **DERIVE),
     [True, False, True, True]),
    (item("registered_derivation", targets=("score",), derivation_id="cutoff_side", boundary=2.0,
          **DERIVE), [False, True, None, True]),
    (item("registered_derivation", targets=("joined",), derivation_id="post_period",
          boundary="2020-06-01", **DERIVE), [False, True, False, True]),
])
def test_registered_derivation_writes_the_closed_flag_set(plan_item: PlanItemV1,
                                                          expected: list[bool | None]) -> None:
    result = run(plan_item)
    assert result.frame["out"].to_list() == expected
    assert result.derived == sum(value is not None for value in expected)


def test_registered_derivation_takes_date_components_and_refuses_free_form_ids() -> None:
    dated = run(item("type_conversion", targets=("joined",), target_dtype="date",
                     date_format="%Y-%m-%d")).frame.rename({"out": "joined_date"})
    result = run(item("registered_derivation", targets=("joined_date",),
                      derivation_id="date_component", component="year", **DERIVE), data=dated)
    assert result.frame["out"].to_list() == [2020, 2021, 2019, 2022]
    with pytest.raises(ops.OperationError) as unknown:
        run(item("registered_derivation", targets=("age",), derivation_id="log_transform",
                 **DERIVE))
    assert unknown.value.code == "unknown_derivation"


def test_numeric_median_imputation_fills_with_the_fitted_median_and_an_indicator() -> None:
    result = run(GOLDENS[4])
    assert result.frame["out"].to_list() == [30.0, 30.0, 50.0, -999.0]
    assert result.frame["out_missing"].to_list() == [False, True, False, False]
    assert result.imputed_mask_delta == {"out": (1,)}
    assert result.fitted_params == {"median": 30.0, "fit_row_count": 4}


def test_numeric_median_imputation_fits_only_the_pre_treatment_rows() -> None:
    result = run(item("numeric_median_imputation", targets=("score",),
                      indicator_column="out_missing", pre_period_column="pre",
                      phase=ItemPhase.IMPUTATION, fit_scope=FitScope.PRE_TREATMENT_ONLY))
    assert result.fitted_params == {"median": 1.5, "fit_row_count": 2}
    assert result.frame["out"].to_list() == [1.0, 2.0, 1.5, 4.0]


@pytest.mark.parametrize("fit_column", ["y", "arm"])
def test_no_fit_may_read_a_treatment_or_outcome_column(fit_column: str) -> None:
    with pytest.raises(ops.OperationError) as leak:
        run(item("numeric_median_imputation", targets=("age",), indicator_column="out_missing",
                 pre_period_column=fit_column, phase=ItemPhase.IMPUTATION,
                 fit_scope=FitScope.PRE_TREATMENT_ONLY))
    assert leak.value.code == "leakage_guard"


def test_no_fit_may_read_a_post_treatment_covariate() -> None:
    context = replace(CONTEXT, measurement_timing={"age": "post_treatment"})
    with pytest.raises(ops.OperationError) as leak:
        run(GOLDENS[4], context=context)
    assert leak.value.code == "leakage_guard"


def test_categorical_missing_encoding_reserves_one_explicit_level() -> None:
    result = run(GOLDENS[5])
    assert result.frame["out"].to_list() == ["NY", "__missing__", "sf", "NY"]
    assert result.imputed_mask_delta == {"out": (1,)}
    with pytest.raises(ops.OperationError) as collision:
        run(item("categorical_missing_encoding", targets=("city",), missing_level="NY",
                 phase=ItemPhase.IMPUTATION))
    assert collision.value.code == "reserved_level_collides"


def test_estimator_scoped_recipe_records_without_fitting() -> None:
    source = frame()
    result = run(GOLDENS[6], data=source)
    assert result.frame is source and result.change_counts == {}
    assert result.fitted_params is None


REFUSALS = (
    (item("missing_sentinel_normalization", sentinels="[]", output="age"), "output_column_exists"),
    (item("missing_sentinel_normalization", targets=("y",), sentinels="[]"),
     "column_not_permitted"),
    (item("numeric_median_imputation", targets=("arm",), indicator_column="m", **IMPUTE),
     "forbidden_target_role"),
    (item("missing_sentinel_normalization", sentinels="[]",
          fit_scope=FitScope.FROZEN_FRAME_BLINDED), "fit_scope_not_allowed"),
    (item("missing_sentinel_normalization", sentinels="[]", phase=ItemPhase.DERIVATION),
     "phase_mismatch"),
    (item("missing_sentinel_normalization", sentinels="[]", extra="1"), "parameter_invalid"),
    (item("missing_sentinel_normalization", targets=("age", "city"), sentinels="[]"),
     "target_arity_invalid"),
    (item("missing_sentinel_normalization", sentinels="[1]"), "parameter_invalid"),
    (item("missing_sentinel_normalization", targets=("ghost",), sentinels="[]"),
     "unknown_column"),
    (item("numeric_median_imputation", targets=("city",), indicator_column="m", **IMPUTE),
     "non_numeric_target"),
)


@pytest.mark.parametrize(("plan_item", "code"), REFUSALS, ids=[code for _, code in REFUSALS])
def test_a_refused_plan_item_names_its_stable_code(plan_item: PlanItemV1, code: str) -> None:
    with pytest.raises(ops.OperationError) as refused:
        run(plan_item)
    assert refused.value.code == code


def test_the_registry_covers_exactly_the_seven_families() -> None:
    assert len(REGISTRY.rows) == 7
    with pytest.raises(RegistryError) as unknown:
        REGISTRY.get("winsorize")
    assert unknown.value.code == "unknown_operation"
