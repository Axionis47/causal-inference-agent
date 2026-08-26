"""Sequential execution of a committed preparation plan (PRD-003 §17.4, §17.6, §24.2; D-057)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol

import polars as pl

from causal.preparation import contracts as pc
from causal.preparation import operations as ops
from causal.preparation import plans as pp
from causal.preparation.contracts import ColumnSchemaFieldV1, ObjectRefV1, PreparationError
from causal.preparation.diagnostics import nulls
from causal.preparation.plans import PlanItemV1
from causal.shared import receipts as rc
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef

EMPTY_PLAN, INPUT_HASH_MISMATCH = "empty_plan", "input_hash_mismatch"
OUTPUT_NOT_REOPENABLE, PLAN_NOT_A_DAG = "output_not_reopenable", "plan_not_a_dag"
TRI_AGREEMENT_FAILED: Final = "tri_agreement_failed"
MASK_SCHEMA: Final = "imputed-cell-mask.v1"


class ExecutionError(PreparationError):
    """One plan item was refused; `codes` carries the §17.6 gate codes it failed."""

    @property
    def codes(self) -> tuple[str, ...]:
        return self.detail_codes or (self.code,)


def _refuse(failed: object, message: str, code: str) -> None:
    if failed:
        raise ExecutionError(message, code)


# The one immutable store an execution needs: it writes an intermediate frame and returns its
# locator and content hash, reopens a written frame under its stored dtypes, and puts one
# canonical-JSON object (a cell mask or a fitted-parameter set).
class FrameStore(Protocol):
    def write_frame(self, frame: pl.DataFrame,
                    schema: tuple[ColumnSchemaFieldV1, ...]) -> tuple[str, str]: ...

    def read_frame(self, locator: str,
                   schema: tuple[ColumnSchemaFieldV1, ...]) -> pl.DataFrame: ...

    def put_object(self, payload: Mapping[str, object]) -> str: ...


# One frame artifact: what it is called, where its object lives, and its column schema.
@dataclass(frozen=True)
class FrameArtifact:
    ref: ArtifactRef
    locator: str
    schema: tuple[ColumnSchemaFieldV1, ...]


# Everything one plan execution produced; fitted values stay in the object store (§17.7).
@dataclass(frozen=True)
class ExecutionOutcome:
    bundle: pp.ExecutionReceiptBundleV1
    frame: pl.DataFrame
    output: FrameArtifact
    fitted_params: Mapping[str, ObjectRefV1]


# One postcondition report over (item, frame before, frame after); §15 diagnostics supply it.
PostconditionFn = Callable[[PlanItemV1, pl.DataFrame, pl.DataFrame], pc.PreparationDiagnosticV1]
_NO_EXPECTATIONS: Final[Mapping[str, ArtifactRef]] = {}


# Topological order with ties broken by declaration order (§7.4); the caller proved the DAG.
def _ordered(items: Sequence[PlanItemV1]) -> list[PlanItemV1]:
    done: set[str] = set()
    pending, ordered = list(items), []
    while pending:
        ready = [item for item in pending if set(item.depends_on) <= done]
        ordered.extend(ready)
        done.update(item.plan_item_id for item in ready)
        pending = [item for item in pending if item.plan_item_id not in done]
    return ordered


# A carried column keeps its lineage; a new column names the targets it was prepared from.
def _schema(after: pl.DataFrame, before: pl.DataFrame, item: PlanItemV1,
            prior: tuple[ColumnSchemaFieldV1, ...]) -> tuple[ColumnSchemaFieldV1, ...]:
    lineage = {field.column_name: field.prepared_from for field in prior}
    return tuple(
        ColumnSchemaFieldV1(column_name=name, dtype=str(dtype), prepared_from=lineage.get(
            name, () if name in before.columns else item.target_columns))
        for name, dtype in after.schema.items())


# Runs a committed plan item by item: guard, operate, receipt, postcondition, gate.
@dataclass(frozen=True)
class PlanExecutor:
    registry: ops.OperationRegistry
    context: ops.OperationContext
    store: FrameStore
    postcondition: PostconditionFn
    stage_run_id: str
    clock: Callable[[], datetime]
    row_id_column: str = "source_row_id"

    def run(self, plan: pp.PreparationPlanV1, plan_ref: ArtifactRef, frame: pl.DataFrame,
            source: FrameArtifact, *, row_set_hash: str, parents: tuple[ArtifactRef, ...],
            expected_inputs: Mapping[str, ArtifactRef] = _NO_EXPECTATIONS) -> ExecutionOutcome:
        """Execute every item in dependency order and assemble the receipt bundle (§24.2)."""
        _refuse(not plan.items, "a plan must carry at least one item", EMPTY_PLAN)
        _refuse(pp.has_cycle(plan.items), "plan item dependencies must form a DAG", PLAN_NOT_A_DAG)
        current, before = source, frame
        receipts: list[rc.ExecutionReceiptV1] = []
        changed: dict[str, int] = {}
        mask: dict[str, tuple[int, ...]] = {}
        fitted: dict[str, ObjectRefV1] = {}
        for item in _ordered(plan.items):
            expected = expected_inputs.get(item.plan_item_id)
            _refuse(expected is not None and expected != current.ref,
                    f"{item.plan_item_id} expected another input frame", INPUT_HASH_MISMATCH)
            frame, current, receipt, result = self._execute(
                item, frame, current, plan_ref, row_set_hash)
            receipts.append(receipt)
            for column, count in result.change_counts.items():
                changed[column] = changed.get(column, 0) + count
            mask.update(result.imputed_mask_delta)
            if result.fitted_params is not None:
                fitted[item.plan_item_id] = self._store(
                    {"plan_item_id": item.plan_item_id, "fitted_params": {**result.fitted_params}})
        bundle = pp.ExecutionReceiptBundleV1(
            plan=plan_ref, receipts=tuple(receipts), changed_counts_by_column=changed,
            missingness_before=nulls(before), missingness_after=nulls(frame),
            imputed_cell_mask=self._mask(mask, frame), row_set_hash=row_set_hash, parents=parents)
        return ExecutionOutcome(bundle, frame, current, fitted)

    def _execute(self, item: PlanItemV1, frame: pl.DataFrame, current: FrameArtifact,
                 plan_ref: ArtifactRef, row_set_hash: str,
                 ) -> tuple[pl.DataFrame, FrameArtifact, rc.ExecutionReceiptV1, ops.OperationResult]:
        """One item: operate, freeze the output, receipt it, and clear the §17.6 gate."""
        started = self.clock()
        result = ops.run_operation(frame, item, self.registry.get(item.operation_id),
                                   self.context)
        self._assert_row_set(frame, result.frame)
        schema = _schema(result.frame, frame, item, current.schema)
        locator, digest = self.store.write_frame(result.frame, schema)
        artifact_id = f"{self.stage_run_id}:{item.plan_item_id}"
        output = FrameArtifact(ArtifactRef(artifact_id=artifact_id, content_hash=digest),
                               locator, schema)
        # The harness reopens the written artifact rather than trusting the receipt (§17.6).
        reopened = self.store.read_frame(locator, schema)
        _refuse(reopened.shape != result.frame.shape,
                f"{item.plan_item_id} did not reopen as written", OUTPUT_NOT_REOPENABLE)
        receipt = self._receipt(item, current, output, result, started, plan_ref, row_set_hash)
        report = self.postcondition(item, frame, result.frame)
        if codes := rc.tri_agreement(receipt, output.ref, report):
            raise ExecutionError(f"{item.plan_item_id} failed the mutation gate",
                                 TRI_AGREEMENT_FAILED, codes)
        return result.frame, output, receipt, result

    def _receipt(self, item: PlanItemV1, current: FrameArtifact, output: FrameArtifact,
                 result: ops.OperationResult, started: datetime, plan_ref: ArtifactRef,
                 row_set_hash: str) -> rc.ExecutionReceiptV1:
        parameters_hash = content_hash({**item.parameters})
        return rc.ExecutionReceiptV1(
            stage_run_id=self.stage_run_id, plan_artifact_id=plan_ref.artifact_id,
            plan_item_id=item.plan_item_id, operation_id=item.operation_id,
            operation_version=item.operation_version,
            implementation_version=ops.IMPLEMENTATION_VERSION, input_ref=current.ref,
            output_ref=output.ref, parameters_hash=parameters_hash,
            shape_before=rc.FrameShapeV1(row_count=result.examined,
                                         column_count=len(current.schema)),
            shape_after=rc.FrameShapeV1(row_count=result.examined,
                                        column_count=len(output.schema)),
            row_set_hash_before=row_set_hash, row_set_hash_after=row_set_hash,
            examined_count=result.examined,
            changed_count=sum(result.change_counts.values()),
            derived_count=result.derived,
            imputed_count=sum(len(rows) for rows in result.imputed_mask_delta.values()),
            warning_codes=(), error_codes=(),
            attempt_id=f"{self.stage_run_id}:{item.plan_item_id}",
            idempotency_key=content_hash({"plan_item_id": item.plan_item_id,
                                          "input": current.ref.content_hash,
                                          "parameters_hash": parameters_hash,
                                          "operation_version": item.operation_version}),
            status=rc.ReceiptStatus.SUCCEEDED, started_at_utc=started,
            finished_at_utc=self.clock())

    # §9.6: an operation may add columns, never rows; the frozen row order is the row identity.
    def _assert_row_set(self, before: pl.DataFrame, after: pl.DataFrame) -> None:
        column = self.row_id_column
        held = column in before.columns
        invariant = after.height == before.height and (
            not held or (column in after.columns and after[column].equals(before[column])))
        _refuse(not invariant, "the operation changed the frozen row set",
                rc.ROW_SET_NOT_INVARIANT)

    def _store(self, payload: Mapping[str, object]) -> ObjectRefV1:
        return ObjectRefV1(object_locator=self.store.put_object(payload),
                           content_hash=content_hash(payload))

    # The imputed-cell mask: row ids (or frozen row positions) by column, as canonical JSON.
    def _mask(self, mask: Mapping[str, tuple[int, ...]],
              frame: pl.DataFrame) -> ObjectRefV1 | None:
        if not mask:
            return None
        held = self.row_id_column in frame.columns
        ids = frame[self.row_id_column].to_list() if held else []
        columns = {column: [ids[row] if held else row for row in rows]
                   for column, rows in sorted(mask.items())}
        return self._store({"schema_version": MASK_SCHEMA, "columns": columns})
