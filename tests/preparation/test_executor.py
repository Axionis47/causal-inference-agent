"""Sequential plan execution, replay, and the §17.6 mutation gate (T-017 §1.2; PRD-003 §24.2)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import polars as pl
import pytest

from causal.preparation.contracts import ColumnSchemaFieldV1, DiagnosticStatus, FrameStage
from causal.preparation.diagnostics import DiagnosticRequest, row_set_invariance
from causal.preparation.executor import ExecutionError, FrameArtifact, PlanExecutor, PriorRun
from causal.preparation.operations import OPERATIONS, OperationResult
from causal.preparation.plans import PlanPhase, PreparationPlanV1
from causal.shared.contracts import ArtifactRef
from tests.preparation.test_operations import CONTEXT, GOLDENS, REGISTRY, frame

ROW_SET_HASH = "c" * 64
PLAN_REF = ArtifactRef(artifact_id="plan-1", content_hash="d" * 64)
AGE = ColumnSchemaFieldV1(column_name="age", dtype="Float64", prepared_from=())
SOURCE = FrameArtifact(ArtifactRef(artifact_id="frame-0", content_hash="e" * 64),
                       "objects/e", (AGE,))
CLOCK = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
# The manifest permits imputing the repaired column the first item writes.
CHAINED = replace(CONTEXT, permitted_imputation_columns=("out",))


class FakeStore:
    """An in-memory frame writer, reader, and object store; T-016 owns the real ones."""

    def __init__(self) -> None:
        self.frames: dict[str, pl.DataFrame] = {}
        self.objects: dict[str, Mapping[str, object]] = {}
        self.writes = 0

    def write_frame(self, frame: pl.DataFrame,
                    schema: tuple[ColumnSchemaFieldV1, ...]) -> tuple[str, str]:
        digest = hashlib.sha256(frame.write_csv().encode("utf-8")).hexdigest()
        self.frames[f"objects/{digest}"] = frame
        self.writes += 1
        return f"objects/{digest}", digest

    def read_frame(self, locator: str, schema: tuple[ColumnSchemaFieldV1, ...]) -> pl.DataFrame:
        return self.frames[locator]

    def put_object(self, payload: Mapping[str, object]) -> str:
        locator = f"objects/{len(self.objects)}"
        self.objects[locator] = payload
        return locator


def postcondition(plan_item: Any, before: pl.DataFrame, after: pl.DataFrame) -> Any:
    return row_set_invariance(DiagnosticRequest(
        frame=after, inputs=(PLAN_REF,), frame_stage=FrameStage.PREPARED, baseline=before,
        row_set_hash=ROW_SET_HASH))


def plan(*items: Any) -> PreparationPlanV1:
    return PreparationPlanV1(
        plan_revision=1, phase=PlanPhase.PREPARATION, items=items, eligibility_rule_ids=(),
        unusable_row_rule_ids=(), recipes=(), groups=(), context_manifest=PLAN_REF,
        stabilized_frame=None, versions={"operations": "repair-operations.v1"})


def executor(store: FakeStore, postcondition_fn: Any = postcondition) -> PlanExecutor:
    return PlanExecutor(registry=REGISTRY, context=CHAINED, writer=store, reader=store,
                        objects=store, postcondition=postcondition_fn, stage_run_id="run-1",
                        clock=lambda: CLOCK)


def sentinel_item(item_id: str = "pi-1") -> Any:
    return GOLDENS[0].model_copy(update={"plan_item_id": item_id})


def imputation_item(item_id: str = "pi-2", depends_on: tuple[str, ...] = ("pi-1",)) -> Any:
    return GOLDENS[4].model_copy(update={
        "plan_item_id": item_id, "target_columns": ("out",), "output_column": "out_filled",
        "parameters": {"indicator_column": "out_missing"}, "depends_on": depends_on})


def run_plan(store: FakeStore, **kwargs: Any) -> Any:
    committed = plan(imputation_item(), sentinel_item())
    return executor(store).run(committed, PLAN_REF, frame(), SOURCE,
                               row_set_hash=ROW_SET_HASH, parents=(PLAN_REF,), **kwargs)


def test_items_run_in_dependency_order_and_the_bundle_records_the_lineage() -> None:
    store = FakeStore()
    outcome = run_plan(store)
    bundle = outcome.bundle
    assert [receipt.plan_item_id for receipt in bundle.receipts] == ["pi-1", "pi-2"]
    assert bundle.changed_counts_by_column == {"out": 1, "out_filled": 2}
    assert bundle.missingness_before["age"] == 1
    assert bundle.missingness_after["out_filled"] == 0
    assert bundle.row_set_hash == ROW_SET_HASH
    assert store.objects[bundle.imputed_cell_mask.object_locator]["columns"] == {
        "out_filled": ["r2", "r4"]}


def test_a_receipt_carries_the_shapes_counts_and_idempotency_key_of_its_item() -> None:
    outcome = run_plan(FakeStore())
    first, second = outcome.bundle.receipts
    assert (first.input_ref, second.input_ref) == (SOURCE.ref, first.output_ref)
    assert (first.shape_after.column_count, second.shape_after.column_count) == (11, 13)
    assert second.imputed_count == 2 and second.derived_count == 4
    assert first.row_set_hash_before == first.row_set_hash_after == ROW_SET_HASH


def test_fitted_values_reach_the_object_store_and_nothing_else() -> None:
    outcome = run_plan(store := FakeStore())
    stored = store.objects[outcome.fitted_params["pi-2"].object_locator]
    assert stored == {"plan_item_id": "pi-2",
                      "fitted_params": {"median": 40.0, "fit_row_count": 4}}
    assert not any("40.0" in str(receipt.model_dump()) for receipt in outcome.bundle.receipts)


def test_an_unexpected_input_frame_stops_the_item_without_a_retry() -> None:
    store = FakeStore()
    with pytest.raises(ExecutionError) as refused:
        run_plan(store, expected_inputs={"pi-2": SOURCE.ref})
    assert refused.value.code == "input_hash_mismatch"
    assert store.writes == 1


def test_a_recorded_item_is_replayed_from_its_existing_receipt() -> None:
    store = FakeStore()
    first = run_plan(store)
    writes = store.writes
    recorded = first.bundle.receipts[0]
    priors = {"pi-1": PriorRun(recorded, FrameArtifact(
        recorded.output_ref, f"objects/{recorded.output_ref.content_hash}", first.output.schema))}
    second = run_plan(store, priors=priors)
    assert second.replayed_item_ids == ("pi-1",)
    assert store.writes == writes + 1
    assert second.bundle.receipts[0] == first.bundle.receipts[0]
    assert (second.bundle.imputed_cell_mask.content_hash
            == first.bundle.imputed_cell_mask.content_hash)


def test_a_failing_postcondition_stops_the_chain_at_its_item() -> None:
    store = FakeStore()

    def failing(plan_item: Any, before: pl.DataFrame, after: pl.DataFrame) -> Any:
        report = postcondition(plan_item, before, after)
        return report.model_copy(update={"status": DiagnosticStatus.FAIL})

    committed = plan(sentinel_item())
    with pytest.raises(ExecutionError) as refused:
        executor(store, failing).run(committed, PLAN_REF, frame(), SOURCE,
                                     row_set_hash=ROW_SET_HASH, parents=(PLAN_REF,))
    assert refused.value.code == "tri_agreement_failed"
    assert "postcondition_failed" in refused.value.codes


def test_an_operation_that_moved_the_row_set_is_caught_before_it_is_written(
        monkeypatch: pytest.MonkeyPatch) -> None:
    store = FakeStore()
    monkeypatch.setitem(OPERATIONS, "missing_sentinel_normalization",
                        lambda source, plan_item, context: OperationResult(
                            frame=source.head(source.height - 1), examined=source.height))
    committed = plan(sentinel_item())
    with pytest.raises(ExecutionError) as refused:
        executor(store).run(committed, PLAN_REF, frame(), SOURCE, row_set_hash=ROW_SET_HASH,
                            parents=(PLAN_REF,))
    assert refused.value.code == "row_set_not_invariant"
    assert store.writes == 0


def test_a_plan_with_no_items_never_produces_a_bundle() -> None:
    with pytest.raises(ValueError, match="at least one item"):
        executor(FakeStore()).run(plan(), PLAN_REF, frame(), SOURCE, row_set_hash=ROW_SET_HASH,
                                  parents=(PLAN_REF,))
