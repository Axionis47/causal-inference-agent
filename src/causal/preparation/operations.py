"""The seven registered Phase B operation families (PRD-003 §10.2, §11, §17.4; D-057)."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

import polars as pl
from pydantic import ValidationError

from causal.preparation.contracts import _Row
from causal.preparation.plans import FitScope, ItemPhase, ParameterValue, PlanItemV1
from causal.shared.contracts import Identity
from causal.shared.registry import INVALID_REGISTRY_FILE, RegistryError

__all__ = [
    "IMPLEMENTATION_VERSION", "OPERATIONS", "Operation", "OperationContext", "OperationError",
    "OperationRegistry", "OperationResult", "RepairOperationV1", "load_operation_registry",
    "run_operation", "validate_item"]

IMPLEMENTATION_VERSION: Final = "preparation-operations.v1"
COLUMN_NOT_PERMITTED, FIT_SCOPE_NOT_ALLOWED = "column_not_permitted", "fit_scope_not_allowed"
FORBIDDEN_TARGET_ROLE, LEAKAGE_GUARD = "forbidden_target_role", "leakage_guard"
LOSSY_CONVERSION, MISSING_OUTPUT_COLUMN = "lossy_conversion", "missing_output_column"
NON_NUMERIC_TARGET, NO_FIT_ROWS = "non_numeric_target", "no_fit_rows"
OUTPUT_COLUMN_EXISTS, PARAMETER_INVALID = "output_column_exists", "parameter_invalid"
PHASE_MISMATCH, RESERVED_LEVEL_COLLIDES = "phase_mismatch", "reserved_level_collides"
TARGET_ARITY_INVALID, UNKNOWN_COLUMN = "target_arity_invalid", "unknown_column"
UNKNOWN_DERIVATION, UNKNOWN_OPERATION = "unknown_derivation", "unknown_operation"
UNMAPPED_CATEGORY: Final = "unmapped_category"

# §11.1 rule 5: a fit population may never read these roles or this measurement timing.
BLINDED_ROLES: Final = ("treatment", "assignment_variable", "outcome", "mediator")
BLINDED_TIMINGS: Final = ("post_treatment",)
# The registered conversion profiles; a name outside this table is not a permitted conversion.
DTYPES: Final[dict[str, pl.DataType]] = {
    "int64": pl.Int64(), "float64": pl.Float64(), "string": pl.String(),
    "boolean": pl.Boolean(), "date": pl.Date(), "datetime": pl.Datetime("us")}


class OperationError(ValueError):
    """A registered operation refused to run; `code` is a stable contract value."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


# Every guard below reads as one line: refuse when the condition holds.
def _refuse(failed: object, message: str, code: str) -> None:
    if failed:
        raise OperationError(message, code)


# The manifest and pack facts an operation may consult; never a cell value (§7.2).
@dataclass(frozen=True)
class OperationContext:
    column_roles: Mapping[str, str] = field(default_factory=dict)
    measurement_timing: Mapping[str, str] = field(default_factory=dict)
    permitted_repair_columns: tuple[str, ...] = ()
    permitted_imputation_columns: tuple[str, ...] = ()


# One operation's new frame and the counts its receipt and cell mask are built from.
@dataclass(frozen=True)
class OperationResult:
    frame: pl.DataFrame
    change_counts: Mapping[str, int] = field(default_factory=dict)
    # Column to the row positions it filled, in the frozen row order (§11.3 as amended by §24.2).
    imputed_mask_delta: Mapping[str, tuple[int, ...]] = field(default_factory=dict)
    fitted_params: Mapping[str, ParameterValue] | None = None
    examined: int = 0
    derived: int = 0


Operation = Callable[[pl.DataFrame, PlanItemV1, OperationContext], OperationResult]


# One registered operation row: its schema, phase, targets, and guards (§10.1).
class RepairOperationV1(_Row):
    operation_id: Identity
    operation_version: Identity
    phase: ItemPhase
    permission: Literal["repair", "imputation", "none"]
    target_arity: int
    writes_output_column: bool
    required_parameters: tuple[Identity, ...]
    optional_parameters: tuple[Identity, ...]
    allowed_fit_scopes: tuple[FitScope, ...]
    forbidden_target_roles: tuple[Identity, ...]
    precondition_diagnostic_ids: tuple[Identity, ...]
    postcondition_diagnostic_ids: tuple[Identity, ...]


class _RegistryFileV1(_Row):
    registry_version: Literal["repair-operations.v1"]
    operations: tuple[RepairOperationV1, ...]


# The registered families; an unknown id fails closed rather than reaching an operation.
@dataclass(frozen=True)
class OperationRegistry:
    rows: Mapping[str, RepairOperationV1]

    def get(self, operation_id: str) -> RepairOperationV1:
        row = self.rows.get(operation_id)
        if row is None:
            raise RegistryError(f"no registered operation {operation_id!r}", UNKNOWN_OPERATION)
        return row


def load_operation_registry(path: Path) -> OperationRegistry:
    """Load the frozen registry; rows and implementations must cover each other exactly."""
    try:
        parsed = _RegistryFileV1.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise RegistryError(f"invalid registry file {path}: {error}",
                            INVALID_REGISTRY_FILE) from error
    rows = {row.operation_id: row for row in parsed.operations}
    if len(rows) != len(parsed.operations) or set(rows) != set(OPERATIONS):
        raise RegistryError(f"rows must match the {len(OPERATIONS)} implementations exactly",
                            UNKNOWN_OPERATION)
    return OperationRegistry(rows)


def validate_item(item: PlanItemV1, row: RepairOperationV1, context: OperationContext) -> None:
    """Every registry guard one plan item clears before its operation may run (§10.1)."""
    name, declared = row.operation_id, set(item.parameters)
    missing = sorted(set(row.required_parameters) - declared)
    unknown = sorted(declared - set(row.required_parameters) - set(row.optional_parameters))
    _refuse(item.phase is not row.phase, f"{item.plan_item_id} is not a {row.phase}", PHASE_MISMATCH)
    _refuse(len(item.target_columns) != row.target_arity,
            f"{name} takes {row.target_arity} targets", TARGET_ARITY_INVALID)
    _refuse(item.fit_scope not in row.allowed_fit_scopes,
            f"{name} forbids fit scope {item.fit_scope}", FIT_SCOPE_NOT_ALLOWED)
    _refuse(missing, f"{name} needs {missing}", PARAMETER_INVALID)
    _refuse(unknown, f"{name} rejects {unknown}", PARAMETER_INVALID)
    _refuse(row.writes_output_column and item.output_column is None,
            f"{name} needs an output column", MISSING_OUTPUT_COLUMN)
    permitted = {"repair": context.permitted_repair_columns,
                 "imputation": context.permitted_imputation_columns}.get(row.permission)
    for column in item.target_columns:
        _refuse(context.column_roles.get(column, "") in row.forbidden_target_roles,
                f"{column} carries a role {row.operation_id} may not target", FORBIDDEN_TARGET_ROLE)
        _refuse(permitted is not None and column not in permitted,
                f"{column} is not a permitted {row.permission} column", COLUMN_NOT_PERMITTED)


def run_operation(frame: pl.DataFrame, item: PlanItemV1, row: RepairOperationV1,
                  context: OperationContext) -> OperationResult:
    """Validate one plan item against its registry row, then run its registered family."""
    validate_item(item, row, context)
    return OPERATIONS[row.operation_id](frame, item, context)


# The one target column that must exist and the one output column that must not (§17.4).
def _columns(frame: pl.DataFrame, item: PlanItemV1) -> tuple[str, str]:
    target = item.target_columns[0]
    _refuse(target not in frame.columns, f"{target!r} is not in the frame", UNKNOWN_COLUMN)
    return target, _new(frame, item.output_column)


# Operations write new columns only; overwriting one would destroy its own lineage.
def _new(frame: pl.DataFrame, name: str | None) -> str:
    if name is None:
        raise OperationError("this operation requires an output column", MISSING_OUTPUT_COLUMN)
    _refuse(name in frame.columns, f"output column {name!r} already exists", OUTPUT_COLUMN_EXISTS)
    return name


def _text(item: PlanItemV1, key: str) -> str:
    value = item.parameters.get(key)
    if not isinstance(value, str) or not value:
        raise OperationError(f"{key!r} must be a non-empty string", PARAMETER_INVALID)
    return value


# List- and map-valued parameters travel as canonical JSON: `parameters` holds scalars.
def _json_parameter(item: PlanItemV1, key: str, kind: type[list[Any] | dict[str, Any]]) -> Any:
    try:
        parsed = json.loads(_text(item, key))
    except json.JSONDecodeError:
        raise OperationError(f"{key!r} is not canonical JSON", PARAMETER_INVALID) from None
    _refuse(not isinstance(parsed, kind), f"{key!r} must be a JSON {kind.__name__}",
            PARAMETER_INVALID)
    entries = [*parsed, *(parsed.values() if isinstance(parsed, dict) else ())]
    _refuse(not all(isinstance(entry, str) for entry in entries), f"{key!r} holds a non-string",
            PARAMETER_INVALID)
    return parsed


# Cells whose value moved, counting a null-to-value move; dtype-independent by design.
def _changed(before: pl.Series, after: pl.Series) -> int:
    same = before.cast(pl.String, strict=False).eq_missing(after.cast(pl.String, strict=False))
    return int(same.not_().sum())


def _positions(mask: pl.Series) -> tuple[int, ...]:
    return tuple(int(position) for position in mask.arg_true())


# Write one new column and count what it moved; a derived column changes no source cell.
def _apply(frame: pl.DataFrame, target: str, output: str, expr: pl.Expr,
           derived: bool = False) -> OperationResult:
    written = frame.with_columns(expr.alias(output))
    return OperationResult(
        frame=written, examined=frame.height,
        change_counts={output: 0 if derived else _changed(frame[target], written[output])},
        derived=int(written[output].is_not_null().sum()) if derived else 0)


# §11.1 rule 5: no fit population reads treatment, outcome, or post-treatment columns.
def _assert_blinded(columns: Sequence[str], context: OperationContext) -> None:
    for column in columns:
        _refuse(context.column_roles.get(column, "") in BLINDED_ROLES
                or context.measurement_timing.get(column, "") in BLINDED_TIMINGS,
                f"a fit population may not read {column}", LEAKAGE_GUARD)


def missing_sentinel_normalization(frame: pl.DataFrame, item: PlanItemV1,
                                   context: OperationContext) -> OperationResult:
    """Approved sentinel encodings become null in a new column; no other cell moves (§10.2)."""
    target, output = _columns(frame, item)
    sentinels = _json_parameter(item, "sentinels", list)
    sentinel = pl.col(target).cast(pl.String, strict=False).is_in(sentinels)
    keep = pl.when(sentinel).then(pl.lit(None, frame.schema[target])).otherwise(pl.col(target))
    return _apply(frame, target, output, keep)


def type_conversion(frame: pl.DataFrame, item: PlanItemV1,
                    context: OperationContext) -> OperationResult:
    """A strict cast into a new typed column; a lossy conversion is refused, never rounded."""
    target, output = _columns(frame, item)
    name = _text(item, "target_dtype")
    if (dtype := DTYPES.get(name)) is None:
        raise OperationError(f"{name!r} is not a registered conversion", PARAMETER_INVALID)
    source, fmt = frame[target], item.parameters.get("date_format")
    try:
        if isinstance(fmt, str):
            converted = (source.str.to_date(fmt) if dtype == pl.Date()
                         else source.str.to_datetime(fmt))
        else:
            converted = source.cast(dtype, strict=True)
    # The polars message quotes the offending cell values, which §18.4 forbids repeating.
    except (pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError):
        raise OperationError(f"{target} does not convert to {name} losslessly",
                             LOSSY_CONVERSION) from None
    _refuse(source.dtype.is_numeric() and converted.dtype.is_numeric()
            and _changed(source, converted.cast(source.dtype, strict=False)),
            f"{target} does not convert to {name} losslessly", LOSSY_CONVERSION)
    written = frame.with_columns(converted.alias(output))
    return OperationResult(frame=written, examined=frame.height,
                           change_counts={output: int(source.is_not_null().sum())})


def category_normalization(frame: pl.DataFrame, item: PlanItemV1,
                           context: OperationContext) -> OperationResult:
    """An explicit one-to-one label map; an unmapped level stops the plan (§10.3)."""
    target, output = _columns(frame, item)
    mapping = _json_parameter(item, "mapping", dict)
    source = frame[target].cast(pl.String, strict=False)
    # The unmapped levels are themselves cell values: report how many, never which (§18.4).
    unmapped = int(source.is_in(list(mapping)).not_().sum())
    _refuse(unmapped, f"{target} holds {unmapped} cells outside the approved mapping",
            UNMAPPED_CATEGORY)
    return _apply(frame, target, output, pl.col(target).cast(pl.String, strict=False)
                  .replace_strict(mapping, default=None, return_dtype=pl.String))


def registered_derivation(frame: pl.DataFrame, item: PlanItemV1,
                          context: OperationContext) -> OperationResult:
    """The closed derivation set: observed flags, period and cutoff sides, and date parts."""
    target, output = _columns(frame, item)
    name, column = _text(item, "derivation_id"), pl.col(target)
    parts: dict[str, Callable[[], pl.Expr]] = {
        "year": column.dt.year, "month": column.dt.month, "day": column.dt.day}
    if name in ("outcome_observed", "nonnull_indicator"):
        expr = column.is_not_null()
    elif name in ("post_period", "cutoff_side"):
        expr = column >= _boundary(frame.schema[target], item.parameters.get("boundary"))
    elif name == "date_component":
        part = _text(item, "component")
        _refuse(part not in parts, f"{part!r} is not a registered date component",
                PARAMETER_INVALID)
        expr = parts[part]()
    else:
        raise OperationError(f"{name!r} is not a registered derivation", UNKNOWN_DERIVATION)
    return _apply(frame, target, output, expr, derived=True)


# One period or cutoff boundary, typed to the column the derivation compares against.
def _boundary(dtype: pl.DataType, value: ParameterValue) -> pl.Expr:
    if isinstance(value, str) and dtype in (pl.Date(), pl.Datetime("us")):
        literal = pl.lit(value)
        return literal.str.to_date() if dtype == pl.Date() else literal.str.to_datetime()
    _refuse(value is None or isinstance(value, bool),
            "'boundary' must be a number, a date, or an ordered label", PARAMETER_INVALID)
    return pl.lit(value)


def numeric_median_imputation(frame: pl.DataFrame, item: PlanItemV1,
                              context: OperationContext) -> OperationResult:
    """Median plus a missingness indicator; the fit scope decides which rows fit (§11.2)."""
    target, output = _columns(frame, item)
    indicator = _new(frame, _text(item, "indicator_column"))
    source = frame[target]
    _refuse(not source.dtype.is_numeric(), f"{target} is not a numeric target", NON_NUMERIC_TARGET)
    scoped = item.fit_scope is FitScope.PRE_TREATMENT_ONLY
    read = [target, *([_text(item, "pre_period_column")] if scoped else [])]
    # The guard runs before the fit population is built: a blinded column is never even read.
    _assert_blinded(read, context)
    _refuse(any(column not in frame.columns for column in read),
            f"{read[-1]!r} is not in the frame", UNKNOWN_COLUMN)
    _refuse(scoped and frame.schema[read[-1]] != pl.Boolean(),
            f"{read[-1]} is not a boolean pre-period mask", PARAMETER_INVALID)
    fit = frame.filter(pl.col(read[-1])) if scoped else frame
    median = fit[target].median()
    if not isinstance(median, int | float):
        raise OperationError(f"the fit population for {target} has no observed value", NO_FIT_ROWS)
    missing = source.is_null()
    written = frame.with_columns(source.fill_null(median).alias(output), missing.alias(indicator))
    # The fitted median lives here and in the object store only: never in a log, event, or trace.
    return OperationResult(
        frame=written, change_counts={output: int(missing.sum())}, examined=frame.height,
        imputed_mask_delta={output: _positions(missing)}, derived=frame.height,
        fitted_params={"median": float(median), "fit_row_count": fit.height})


def categorical_missing_encoding(frame: pl.DataFrame, item: PlanItemV1,
                                 context: OperationContext) -> OperationResult:
    """The reserved explicit missing level; an existing level of that name is refused (§11.1)."""
    target, output = _columns(frame, item)
    level = _text(item, "missing_level")
    source = frame[target].cast(pl.String, strict=False)
    _refuse(source.eq(level).any(), f"{target} already uses the reserved missing level",
            RESERVED_LEVEL_COLLIDES)
    missing = source.is_null()
    written = frame.with_columns(source.fill_null(level).alias(output))
    return OperationResult(frame=written, change_counts={output: int(missing.sum())},
                           imputed_mask_delta={output: _positions(missing)}, examined=frame.height)


def estimator_scoped_recipe(frame: pl.DataFrame, item: PlanItemV1,
                            context: OperationContext) -> OperationResult:
    """Records a fold-scoped recipe for PRD-004: nothing is fitted and no column is written."""
    _refuse(item.target_columns[0] not in frame.columns,
            f"{item.target_columns[0]!r} is not in the frame", UNKNOWN_COLUMN)
    _text(item, "recipe_id")
    _text(item, "strategy_id")
    return OperationResult(frame=frame, examined=frame.height)


OPERATIONS: Final[dict[str, Operation]] = {
    "missing_sentinel_normalization": missing_sentinel_normalization,
    "type_conversion": type_conversion, "category_normalization": category_normalization,
    "registered_derivation": registered_derivation,
    "numeric_median_imputation": numeric_median_imputation,
    "categorical_missing_encoding": categorical_missing_encoding,
    "estimator_scoped_recipe": estimator_scoped_recipe}
