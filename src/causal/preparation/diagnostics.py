"""The registered preparation diagnostics and the write-free plan preview (PRD-003 §15, §17.2)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import polars as pl
from pydantic import ValidationError

from causal.preparation import contracts as pc
from causal.preparation.contracts import DiagnosticStatus, FrameStage, PreparationDiagnosticV1, _Row
from causal.preparation.plans import PlanItemPreviewV1, PlanPreviewV1, PreparationPlanV1
from causal.shared.contracts import ArtifactRef, Identity
from causal.shared.registry import INVALID_REGISTRY_FILE, RegistryError

IMPLEMENTATION_VERSION: Final = "preparation-diagnostics.v1"
DIAGNOSTIC_VERSION: Final = "v1"
DIAGNOSTIC_NOT_IMPLEMENTED, STAGE_NOT_ALLOWED = "diagnostic_not_implemented", "stage_not_allowed"
UNKNOWN_DIAGNOSTIC: Final = "unknown_diagnostic"
_RETAINING: Final = tuple(found.value for found in pc.RETAINING_DISPOSITIONS)
_Values = Mapping[str, float | int | str | bool | None]


# One registered diagnostic row: who computes it and which frame stages it may read.
class PreparationDiagnosticRowV1(_Row):
    diagnostic_id: Identity
    diagnostic_version: Identity
    implementation: Literal["preparation_common", "method_pack"]
    stages: tuple[FrameStage, ...]


class _RegistryFileV1(_Row):
    registry_version: Literal["preparation-diagnostics.v1"]
    diagnostics: tuple[PreparationDiagnosticRowV1, ...]


# The §15 common set plus every method-pack diagnostic the T-015 overlay names, by id.
DiagnosticRegistry = Mapping[str, PreparationDiagnosticRowV1]


# Load the frozen diagnostic registry; a duplicate or unreadable row fails closed.
def load_preparation_diagnostics(path: Path) -> DiagnosticRegistry:
    try:
        parsed = _RegistryFileV1.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise RegistryError(f"invalid registry file {path}: {error}",
                            INVALID_REGISTRY_FILE) from error
    rows = {row.diagnostic_id: row for row in parsed.diagnostics}
    if len(rows) != len(parsed.diagnostics):
        raise RegistryError("a diagnostic id may appear once", INVALID_REGISTRY_FILE)
    return rows


# One diagnostic's inputs: the frames it reads and the manifest facts it reconciles.
@dataclass(frozen=True)
class DiagnosticRequest:
    frame: pl.DataFrame
    inputs: tuple[ArtifactRef, ...]
    frame_stage: FrameStage = FrameStage.PREPARED
    baseline: pl.DataFrame | None = None
    row_set_hash: str | None = None
    row_id_column: str = "source_row_id"
    key_columns: tuple[str, ...] = ()
    expected_dtypes: Mapping[str, str] = field(default_factory=dict)
    required_columns: tuple[str, ...] = ()
    disposition_counts: Mapping[str, int] = field(default_factory=dict)
    source_row_count: int = 0
    changed_by_column: Mapping[str, int] = field(default_factory=dict)
    changed_by_operation: Mapping[str, int] = field(default_factory=dict)


DiagnosticFn = Callable[[DiagnosticRequest], PreparationDiagnosticV1]
_NO_REASONS: Final[Mapping[str, int]] = {}
_NO_EXTRA: Final[Mapping[str, DiagnosticFn]] = {}


# Null count per column, in frame column order; the one missingness reading (§15, §24.2).
def nulls(frame: pl.DataFrame) -> dict[str, int]:
    counts = frame.null_count().row(0)
    return {name: int(count) for name, count in zip(frame.columns, counts, strict=True)}


def _status(passed: bool) -> DiagnosticStatus:
    return DiagnosticStatus.PASS if passed else DiagnosticStatus.FAIL


# Every diagnostic reports the same accounting: what it read, over how many rows, and why not.
def _report(request: DiagnosticRequest, diagnostic_id: str, values: _Values,
            columns: Sequence[str] = (), used: int | None = None,
            unused: Mapping[str, int] = _NO_REASONS, warnings: tuple[str, ...] = (),
            status: DiagnosticStatus = DiagnosticStatus.PASS) -> PreparationDiagnosticV1:
    return PreparationDiagnosticV1(
        diagnostic_id=diagnostic_id, diagnostic_version=DIAGNOSTIC_VERSION,
        frame_stage=request.frame_stage, inputs=request.inputs, columns_read=tuple(columns),
        total_rows=request.frame.height, used_rows=request.frame.height if used is None else used,
        unused_reason_counts={name: count for name, count in unused.items() if count},
        row_set_hash=request.row_set_hash, values=dict(values), warnings=warnings, status=status,
        implementation_version=IMPLEMENTATION_VERSION)


# Every source row carries one disposition and the retained rows are the frame (§6.2).
def row_disposition_reconciliation(request: DiagnosticRequest) -> PreparationDiagnosticV1:
    counts = request.disposition_counts
    retained, total = sum(counts.get(name, 0) for name in _RETAINING), sum(counts.values())
    values = {"source_rows": request.source_row_count, "dispositioned_rows": total,
              "retained_rows": retained, "frame_rows": request.frame.height}
    return _report(request, "row_disposition_reconciliation", values, status=_status(
        total == request.source_row_count and retained == request.frame.height))


# The approved key selects one row; a repeated key is a grain violation (§9.4).
def key_uniqueness_grain(request: DiagnosticRequest) -> PreparationDiagnosticV1:
    keys = [column for column in request.key_columns if column in request.frame.columns]
    used = request.frame.drop_nulls(subset=keys) if keys else request.frame
    duplicated = used.height - used.select(keys).unique().height if keys else 0
    return _report(request, "key_uniqueness_grain",
                   {"key_columns": len(keys), "duplicate_rows": duplicated}, keys, used.height,
                   {"missing_key": request.frame.height - used.height},
                   status=_status(len(keys) == len(request.key_columns) and not duplicated))


# The frame's columns and dtypes are exactly the approved contract's (§14 wall 13).
def schema_type_validation(request: DiagnosticRequest) -> PreparationDiagnosticV1:
    observed = {name: str(dtype) for name, dtype in request.frame.schema.items()}
    expected = request.expected_dtypes
    missing = sorted(set(expected) - set(observed))
    wrong = sorted(name for name, dtype in expected.items() if observed.get(name, dtype) != dtype)
    values = {"expected_columns": len(expected), "missing_columns": len(missing),
              "mismatched_columns": len(wrong),
              "unexpected_columns": len(set(observed) - set(expected))}
    return _report(request, "schema_type_validation", values, tuple(observed),
                   status=_status(not missing and not wrong))


# Null counts per column before and after repair, over one frozen denominator (§15).
def missingness_before_after(request: DiagnosticRequest) -> PreparationDiagnosticV1:
    after, baseline = nulls(request.frame), request.baseline
    before = nulls(baseline) if baseline is not None else {}
    grew = any(count > before.get(name, count) for name, count in after.items())
    values: dict[str, float | int | str | bool | None] = {
        f"{name}.before": count for name, count in before.items()}
    values.update({f"{name}.after": count for name, count in after.items()})
    return _report(request, "missingness_before_after", values, tuple(after),
                   warnings=("missingness_increased",) if grew else ())


# The frozen row set survived the operation: same rows, same order (§14 wall 11).
def row_set_invariance(request: DiagnosticRequest) -> PreparationDiagnosticV1:
    baseline, column = request.baseline, request.row_id_column
    identical = baseline is None or column not in baseline.columns or (
        column in request.frame.columns and request.frame[column].equals(baseline[column]))
    rows = baseline.height if baseline is not None else request.frame.height
    values = {"baseline_rows": rows, "frame_rows": request.frame.height,
              "row_ids_identical": identical}
    return _report(request, "row_set_invariance", values,
                   status=_status(identical and rows == request.frame.height))


# Changed cells by operation and by column, from the receipt bundle's lineage (§24.2).
def changed_cells_by_operation_column(request: DiagnosticRequest) -> PreparationDiagnosticV1:
    values: dict[str, float | int | str | bool | None] = {
        **{f"column.{name}": count for name, count in request.changed_by_column.items()},
        **{f"operation.{name}": count for name, count in request.changed_by_operation.items()},
        "changed_cells": sum(request.changed_by_column.values())}
    return _report(request, "changed_cells_by_operation_column", values,
                   tuple(request.changed_by_column))


# Every required column exists and holds no null the contract forbids (§14 wall 13).
def contract_completeness(request: DiagnosticRequest) -> PreparationDiagnosticV1:
    missingness = nulls(request.frame)
    missing = sorted(set(request.required_columns) - set(request.frame.columns))
    incomplete = sorted(name for name in request.required_columns if missingness.get(name, 0))
    values = {"required_columns": len(request.required_columns), "missing_columns": len(missing),
              "incomplete_columns": len(incomplete)}
    return _report(request, "contract_completeness", values, request.required_columns,
                   status=_status(not (missing or incomplete)))


# Each common diagnostic is named for the id it computes; §15's seven are the whole set here.
COMMON_DIAGNOSTICS: Final[dict[str, DiagnosticFn]] = {
    computed.__name__: computed for computed in (
        row_disposition_reconciliation, key_uniqueness_grain, schema_type_validation,
        missingness_before_after, row_set_invariance, changed_cells_by_operation_column,
        contract_completeness)}


# Run one registered diagnostic; an unknown id or stage fails closed (§15).
def run_diagnostic(diagnostic_id: str, request: DiagnosticRequest, registry: DiagnosticRegistry,
                   extra: Mapping[str, DiagnosticFn] = _NO_EXTRA) -> PreparationDiagnosticV1:
    row = registry.get(diagnostic_id)
    computed = {**COMMON_DIAGNOSTICS, **extra}.get(diagnostic_id)
    if row is None:
        raise RegistryError(f"no registered diagnostic {diagnostic_id!r}", UNKNOWN_DIAGNOSTIC)
    if request.frame_stage not in row.stages:
        raise RegistryError(f"{diagnostic_id} reads no {request.frame_stage} frame",
                            STAGE_NOT_ALLOWED)
    if computed is None:
        raise RegistryError(f"{diagnostic_id} has no V1 implementation here",
                            DIAGNOSTIC_NOT_IMPLEMENTED)
    return computed(request)


# The ordered per-item preview a plan needs before any mutation unlocks; no write (§17.2).
def preview(plan: PreparationPlanV1, plan_ref: ArtifactRef, frame: pl.DataFrame) -> PlanPreviewV1:
    columns, items = list(frame.columns), []
    for item in plan.items:
        delta = item.predicted_missingness_change
        warnings = ("target_column_absent",) * any(c not in columns for c in item.target_columns)
        warnings += ("output_column_exists",) * (item.output_column in columns)
        if item.output_column is not None:
            columns.append(item.output_column)
        items.append(PlanItemPreviewV1(
            plan_item_id=item.plan_item_id, expected_row_count=frame.height,
            expected_columns=tuple(columns), missingness_delta=dict(delta),
            affected_cell_count=sum(abs(count) for count in delta.values()), warnings=warnings))
    return PlanPreviewV1(plan=plan_ref, plan_revision=plan.plan_revision, items=tuple(items))
