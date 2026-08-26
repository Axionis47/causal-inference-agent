"""The §15 common diagnostics and the write-free plan preview (T-017 §1.3; PRD-003 §15, §17.2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from causal.preparation import diagnostics as dx
from causal.preparation.contracts import DiagnosticStatus, FrameStage
from causal.shared.registry import RegistryError
from tests.preparation.test_executor import PLAN_REF, ROW_SET_HASH, plan, sentinel_item
from tests.preparation.test_operations import REGISTRY, frame

DIAGNOSTICS = dx.load_preparation_diagnostics(
    Path(__file__).resolve().parents[2] / "registries" / "preparation-diagnostics.v1.json")
AFTER = frame().with_columns(pl.col("age").fill_null(30.0).alias("age_out"))
PASS, FAIL = DiagnosticStatus.PASS, DiagnosticStatus.FAIL


def request(**overrides: Any) -> dx.DiagnosticRequest:
    base: dict[str, Any] = {
        "frame": AFTER, "inputs": (PLAN_REF,), "frame_stage": FrameStage.PREPARED,
        "baseline": frame(), "row_set_hash": ROW_SET_HASH, "key_columns": ("source_row_id",),
        "expected_dtypes": {name: str(dtype) for name, dtype in AFTER.schema.items()},
        "required_columns": ("age_out",), "source_row_count": 6,
        "disposition_counts": {"retained": 3, "retained_with_missingness": 1,
                               "not_eligible_population": 2},
        "changed_by_column": {"age_out": 1},
        "changed_by_operation": {"numeric_median_imputation": 1}}
    return dx.DiagnosticRequest(**{**base, **overrides})


CASES = (
    ("row_disposition_reconciliation", "retained_rows", 4),
    ("key_uniqueness_grain", "duplicate_rows", 0),
    ("schema_type_validation", "missing_columns", 0),
    ("missingness_before_after", "age.before", 1),
    ("row_set_invariance", "frame_rows", 4),
    ("changed_cells_by_operation_column", "operation.numeric_median_imputation", 1),
    ("contract_completeness", "incomplete_columns", 0),
)


@pytest.mark.parametrize(("diagnostic_id", "key", "value"), CASES, ids=[case[0] for case in CASES])
def test_each_common_diagnostic_reports_its_denominator(diagnostic_id: str, key: str,
                                                        value: int) -> None:
    report = dx.run_diagnostic(diagnostic_id, request(), DIAGNOSTICS)
    assert report.status is PASS
    assert report.values[key] == value
    assert (report.total_rows, report.row_set_hash) == (4, ROW_SET_HASH)


FAILURES = (
    ("row_disposition_reconciliation", {"source_row_count": 99}),
    ("key_uniqueness_grain", {"frame": AFTER.with_columns(pl.lit("r1").alias("source_row_id"))}),
    ("schema_type_validation", {"expected_dtypes": {"age": "String"}}),
    ("row_set_invariance", {"baseline": frame().head(3)}),
    ("contract_completeness", {"required_columns": ("absent",)}),
)


@pytest.mark.parametrize(("diagnostic_id", "overrides"), FAILURES,
                         ids=[case[0] for case in FAILURES])
def test_a_broken_denominator_fails_closed(diagnostic_id: str, overrides: dict[str, Any]) -> None:
    assert dx.run_diagnostic(diagnostic_id, request(**overrides), DIAGNOSTICS).status is FAIL


def test_missingness_warns_when_a_step_increased_it() -> None:
    emptied = frame().with_columns(pl.lit(None, pl.Float64).alias("age_out"))
    report = dx.run_diagnostic("missingness_before_after",
                               request(frame=emptied, baseline=AFTER), DIAGNOSTICS)
    assert report.warnings == ("missingness_increased",)


REFUSALS = (("arm_counts", "diagnostic_not_implemented"), ("winsorize", "unknown_diagnostic"))


@pytest.mark.parametrize(("diagnostic_id", "code"), REFUSALS, ids=[case[1] for case in REFUSALS])
def test_an_unregistered_or_unimplemented_diagnostic_fails_closed(diagnostic_id: str,
                                                                  code: str) -> None:
    with pytest.raises(RegistryError) as refused:
        dx.run_diagnostic(diagnostic_id, request(), DIAGNOSTICS)
    assert refused.value.code == code


def test_a_diagnostic_refuses_a_frame_stage_it_does_not_read() -> None:
    with pytest.raises(RegistryError) as refused:
        dx.run_diagnostic("changed_cells_by_operation_column",
                          request(frame_stage=FrameStage.STABILIZED), DIAGNOSTICS)
    assert refused.value.code == "stage_not_allowed"


def test_every_operation_postcondition_is_a_registered_diagnostic() -> None:
    named = {name for row in REGISTRY.rows.values()
             for name in row.precondition_diagnostic_ids + row.postcondition_diagnostic_ids}
    assert len(DIAGNOSTICS) == 24 and named <= set(DIAGNOSTICS)


def test_preview_predicts_every_item_without_writing() -> None:
    item = sentinel_item().model_copy(update={"predicted_missingness_change": {"out": 1}})
    source = frame()
    result = dx.preview(plan(item), PLAN_REF, source)
    assert (result.plan_revision, len(result.items)) == (1, 1)
    assert result.items[0].expected_row_count == source.height
    assert result.items[0].expected_columns[-1] == "out"
    assert result.items[0].affected_cell_count == 1
    assert result.items[0].warnings == ()
    assert source.equals(frame())
    absent = sentinel_item().model_copy(update={"target_columns": ("absent",)})
    assert dx.preview(plan(absent), PLAN_REF, source).items[0].warnings == (
        "target_column_absent",)
