"""Read-only pre-repair diagnostics over the selected CSV (PRD-002 §4.2, §14; T-012 §4)."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final, NamedTuple

import polars as pl

from causal.design import compiler_v2, statistics_v2
from causal.design.contracts import DiagnosticResultV1, DiagnosticStatus
from causal.design.packs import PREREPAIR_DIAGNOSTIC_IDS
from causal.design.v2 import AgentDesignProposalV2, DiagnosticPlanV2
from causal.shared.frames import ROW_UNIT_COLUMN
from causal.shared.readers import BytesFrameSource, CsvObjectFrameSource, FrameSource, ObjectReader

__all__ = ["DIAGNOSTIC_SPECS", "DIAGNOSTIC_TOOL_CALL_LIMIT", "DIAGNOSTIC_TOOL_ID",
    "IMPLEMENTATION_VERSION", "BytesFrameSource", "CsvObjectFrameSource", "DiagnosticError",
    "DiagnosticSpec", "FrameSource", "ObjectReader", "diagnostic_result_id", "run_diagnostic",
    "run_requested_diagnostics"]

IMPLEMENTATION_VERSION, DIAGNOSTIC_VERSION = "design-diagnostics.v1", "prerepair-diagnostic.v1"
UNKNOWN_DIAGNOSTIC, DIAGNOSTIC_TOOL_ID = "unknown_diagnostic", "run_statistical_diagnostic"
DIAGNOSTIC_TOOL_CALL_LIMIT: Final = 4
DIAGNOSTIC_NOT_ALLOWED, DIAGNOSTIC_ASSESSMENT_MISMATCH = "diagnostic_not_allowed", "diagnostic_assessment_mismatch"
DIAGNOSTIC_TOOL_BUDGET_EXHAUSTED = "diagnostic_tool_budget_exhausted"
# The closed eligibility-rule grammar (PRD-002 §15); one preview's rules are a conjunction.
SIDE: Final = "side"  # the derived cutoff side, never a CSV column
_ROW, _VALUE, _BAND = "__row_index", "__numeric_value", "__band"
_MAX_GROUPS, _BANDS = 50, 5
_QUANTILES: Final = (0.25, 0.5, 0.75)
_LIST_KEYS: Final = ("columns", "key_columns", "by")
_SCALAR_KEYS: Final = ("target", "column", "running_column")

_Values = dict[str, float | int | str | bool | None]


class DiagnosticError(ValueError):
    """A diagnostic or eligibility preview was refused; `code` is a stable contract value."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class DiagnosticSpec(NamedTuple):
    diagnostic_id: str
    primitive: str
    default_params: Mapping[str, Any]
    required_params: tuple[str, ...]


# One primitive's values plus the exact row accounting it produced.
class _Outcome(NamedTuple):
    values: _Values
    warnings: tuple[str, ...]
    kept: list[int] | None
    parse_failed: int = 0


_PRIMITIVE_SPECS: Final[dict[str, tuple[dict[str, Any], tuple[str, ...]]]] = {
    "count_by": ({"columns": (), "cutoff": None, "running_column": None}, ("columns",)),
    "missing_share": ({"target": None, "by": (), "cutoff": None, "running_column": None}, ("target",)),
    "uniqueness": ({"key_columns": ()}, ("key_columns",)), "availability": ({"columns": ()}, ("columns",)),
    "level_profile": ({"column": None, "max_levels": 20, "min_level_count": 5}, ("column",)),
    "numeric_support": ({"column": None, "cutoff": None, "band": None, "top_values": 3}, ("column",)),
    "power_precision": ({"columns": (), "target": None}, ("columns", "target")),
    "rough_overlap": ({"target": None, "columns": ()}, ("target", "columns")),
    "panel_structure": ({"key_columns": ()}, ("key_columns",)),
    "did_support": ({"key_columns": (), "target": None, "adoption_time": None}, ("key_columns", "target", "adoption_time")),
    "rdd_assignment": ({"running_column": None, "target": None, "cutoff": None}, ("running_column", "target", "cutoff"))}
_OVERRIDES: Final[dict[str, dict[str, Any]]] = {
    "missingness_by_side_and_distance": {"by": (SIDE,)}}
DIAGNOSTIC_SPECS: Final[dict[str, DiagnosticSpec]] = {
    identity: DiagnosticSpec(identity, recipe.primitive, defaults | _OVERRIDES.get(identity, {}),
                             required)
    for identity, recipe in compiler_v2.DIAGNOSTIC_RECIPES.items()
    for defaults, required in (_PRIMITIVE_SPECS[recipe.primitive],)}
if set(DIAGNOSTIC_SPECS) != set(PREREPAIR_DIAGNOSTIC_IDS):
    raise RuntimeError("DIAGNOSTIC_SPECS must cover exactly the pre-repair diagnostic vocabulary")


def _key(values: Sequence[Any]) -> str:
    return "|".join("null" if value is None else str(value) for value in values)


def _share(part: int, whole: int) -> float:
    return part / whole if whole else 0.0


def _number(value: object) -> float | None:
    return None if value is None else float(value)  # type: ignore[arg-type]


# The rows a primitive may aggregate (every selection column non-null) and their row indices.
def _used(frame: pl.DataFrame, columns: Sequence[str]) -> tuple[pl.DataFrame, list[int]]:
    used = (frame.with_row_index(_ROW).drop_nulls(subset=list(columns))
            if columns else frame.with_row_index(_ROW))
    return used, [int(value) for value in used[_ROW]]


# Group sizes: arms, clusters, group-time cells, adoption cohorts, cutoff sides.
def _count_by(frame: pl.DataFrame, params: Mapping[str, Any]) -> _Outcome:
    # Several causal roles may name one measurement; it remains one grouping dimension.
    used, kept = _used(frame, columns := list(dict.fromkeys(params["columns"])))
    counts, cells = (used.group_by(columns).len().sort(columns),
                     math.prod(int(used[name].n_unique()) for name in columns))
    sizes = [int(value) for value in counts["len"]] if counts.height else []
    values: _Values = {"row_count": used.height, "group_count": counts.height,
                       "expected_cells": cells, "cell_completeness": _share(counts.height, cells),
                       "minimum_group_count": min(sizes, default=0),
                       "maximum_group_count": max(sizes, default=0)}
    values.update({f"distinct_count:{name}": int(used[name].n_unique()) for name in columns})
    for row in counts.head(_MAX_GROUPS).iter_rows():
        values[f"count:{_key(row[:-1])}"] = int(row[-1])
    return _Outcome(values, (f"only the first {_MAX_GROUPS} groups are reported",)
                    if counts.height > _MAX_GROUPS else (), kept)


# Missing share of one target column, overall and inside each requested group.
def _missing_share(frame: pl.DataFrame, params: Mapping[str, Any]) -> _Outcome:
    target, by = str(params["target"]), list(params["by"])
    used, kept = _used(frame, by)
    values: _Values = {"row_count": used.height,
                       "missing_count": (missing := int(used[target].null_count())),
                       "non_null_count": used.height - missing,
                       "missing_share": _share(missing, used.height)}
    if by:
        grouped = used.group_by(by).agg(
            pl.col(target).null_count().alias("m"), pl.len().alias("n")).sort(by)
        for row in grouped.head(_MAX_GROUPS).iter_rows():
            key = _key(row[:-2])
            values[f"missing_count:{key}"] = int(row[-2])
            values[f"missing_share:{key}"] = _share(int(row[-2]), int(row[-1]))
    return _Outcome(values, (), kept)


# Distinct and repeated key counts over the rows whose key columns are complete.
def _uniqueness(frame: pl.DataFrame, params: Mapping[str, Any]) -> _Outcome:
    used, kept = _used(frame, keys := list(params["key_columns"]))
    distinct = int(used.select(keys).n_unique())
    repeated = used.height - distinct
    values: _Values = {"row_count": used.height, "distinct_key_count": distinct,
                       "duplicate_row_count": repeated, "is_unique": not repeated}
    return _Outcome(values, () if not repeated else (f"{repeated} rows repeat a key",), kept)


# Level counts and sparsity for one categorical column.
def _level_profile(frame: pl.DataFrame, params: Mapping[str, Any]) -> _Outcome:
    column, cap, floor_count = (str(params["column"]), int(params["max_levels"]),
                                int(params["min_level_count"]))
    used, kept = _used(frame, [column])
    counts = used[column].value_counts().sort([column])
    sizes = [int(row[1]) for row in counts.iter_rows()]
    values: _Values = {"row_count": used.height, "distinct_count": counts.height,
                       "sparse_level_count": sum(1 for size in sizes if size < floor_count),
                       "min_level_count": min(sizes, default=0)}
    for row in counts.head(cap).iter_rows():
        values[f"level:{_key(row[:1])}"] = int(row[1])
    return _Outcome(values, (f"only the first {cap} levels are reported",)
                    if counts.height > cap else (), kept)


# Range, quantiles, repeated mass points, cutoff support, and a bounded distance histogram.
def _numeric_support(frame: pl.DataFrame, params: Mapping[str, Any]) -> _Outcome:
    column = str(params["column"])
    indexed = frame.with_row_index(_ROW).with_columns(
        pl.col(column).cast(pl.Float64, strict=False).alias(_VALUE))
    parse_failed = int(indexed[_VALUE].null_count()) - int(frame[column].null_count())
    series = (used := indexed.drop_nulls(subset=[_VALUE]))[_VALUE]
    low, high = _number(series.min()), _number(series.max())
    values: _Values = {"row_count": used.height, "min": low, "max": high}
    for quantile in _QUANTILES:
        values[f"q{int(quantile * 100)}"] = _number(
            series.quantile(quantile, interpolation="linear"))
    top = series.value_counts().sort(["count", _VALUE], descending=[True, False])
    for row in top.head(int(params["top_values"])).iter_rows():
        values[f"mass_point:{row[0]}"] = _share(int(row[1]), used.height)
    warnings: list[str] = []
    if (cutoff := params.get("cutoff")) is not None and used.height:
        edge = float(cutoff)
        span = high - low if low is not None and high is not None else 0.0
        width = float(params.get("band") or 0.0) or (span / _BANDS if span else 1.0)
        below = int(series.filter(series < edge).len())
        bins = used.with_columns(
            ((pl.col(_VALUE) - edge) / width).floor().clip(-_BANDS, _BANDS - 1).alias(_BAND)
        ).group_by(_BAND).len().sort(_BAND)
        values |= {"cutoff": edge, "band_width": width, "below_count": below,
                   "at_or_above_count": used.height - below}
        values |= {f"band:{int(row[0])}": int(row[1]) for row in bins.iter_rows()}
        if below in (0, used.height):
            warnings.append("the running variable has support on only one side of the cutoff")
    return _Outcome(values, tuple(warnings), [int(value) for value in used[_ROW]], parse_failed)


# Non-null share per requested column; every physical row is inspected.
def _availability(frame: pl.DataFrame, params: Mapping[str, Any]) -> _Outcome:
    values: _Values = {"row_count": frame.height}
    for name in params["columns"]:
        values.update({f"non_null_count:{name}": (count := int(frame[str(name)].count())),
                       f"non_null_share:{name}": _share(count, frame.height)})
    return _Outcome(values, (), None)


def _statistic(function: Callable[[pl.DataFrame, Mapping[str, Any]], tuple[Any, ...]],
               ) -> Callable[[pl.DataFrame, Mapping[str, Any]], _Outcome]:
    def run(frame: pl.DataFrame, params: Mapping[str, Any]) -> _Outcome:
        return _Outcome(*function(frame, params))
    return run


_PRIMITIVES: Final[dict[str, Callable[[pl.DataFrame, Mapping[str, Any]], _Outcome]]] = {
    "count_by": _count_by, "missing_share": _missing_share, "uniqueness": _uniqueness,
    "level_profile": _level_profile, "numeric_support": _numeric_support,
    "availability": _availability,
    **{name: _statistic(fn) for name, fn in {
        "power_precision": statistics_v2.power_precision, "rough_overlap": statistics_v2.rough_overlap,
        "panel_structure": statistics_v2.panel_structure, "did_support": statistics_v2.did_support,
        "rdd_assignment": statistics_v2.rdd_assignment}.items()}}


# Add the derived `side` column when a running column and a cutoff were both resolved.
def _with_side(frame: pl.DataFrame, params: Mapping[str, Any]) -> pl.DataFrame:
    running, cutoff = params.get("running_column"), params.get("cutoff")
    if running is None or cutoff is None:
        return frame
    return frame.with_columns(
        pl.when((value := pl.col(str(running))).is_null()).then(pl.lit(None, dtype=pl.String))
        .when(value >= float(cutoff)).then(pl.lit("at_or_above"))
        .otherwise(pl.lit("below")).alias(SIDE))


# Drop column names the CSV does not carry; report what is present and what is missing.
def _resolve(
    frame: pl.DataFrame, params: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str], list[str]]:
    known = set(frame.columns)
    running = params.get("running_column")
    side_ok = running is not None and str(running) in known and params.get("cutoff") is not None
    resolved: dict[str, Any] = dict(params)
    present: list[str] = []
    missing: list[str] = []
    for key in (*_LIST_KEYS, *_SCALAR_KEYS):
        raw = params.get(key)
        listed = key in _LIST_KEYS
        names = [str(name) for name in raw or ()] if listed else [str(raw)] if raw else []
        keep = [name for name in names if name in known or (name == SIDE and side_ok)]
        present.extend(name for name in keep if name != SIDE)
        missing.extend(name for name in names if name not in keep)
        resolved[key] = tuple(keep) if listed else (keep[0] if keep else None)
    return resolved, present, missing


def run_diagnostic(
    spec_id: str, source: FrameSource, params: Mapping[str, Any] | None = None
) -> DiagnosticResultV1:
    """One read-only diagnostic with exact denominators and no pass/fail judgment (§14)."""
    spec = DIAGNOSTIC_SPECS.get(spec_id)
    if spec is None:
        raise DiagnosticError(f"unknown diagnostic {spec_id!r}", UNKNOWN_DIAGNOSTIC)
    frame = source.frame()
    call_params = {**spec.default_params, **(params or {})}
    requested = {str(name) for key in _LIST_KEYS for name in (call_params.get(key) or ())}
    if ROW_UNIT_COLUMN in requested and ROW_UNIT_COLUMN not in frame.columns:
        frame = frame.with_row_index(ROW_UNIT_COLUMN)
    total = frame.height
    resolved, present, missing = _resolve(frame, call_params)
    if spec.primitive == "panel_structure" and not resolved.get("key_columns"):
        resolved["key_columns"] = resolved.get("columns") or ()
    warnings = [f"requested column {name!r} is absent from the selected CSV" for name in missing]
    outcome, used = _Outcome({}, (), None), 0
    unused: dict[str, int] = {}
    status = DiagnosticStatus.NOT_COMPUTABLE
    if absent := [key for key in spec.required_params
                  if resolved.get(key) in (None, "", (), [])]:
        warnings.append(f"{spec_id} cannot run without {absent}")
    else:
        outcome = _PRIMITIVES[spec.primitive](_with_side(frame, resolved), resolved)
        status = DiagnosticStatus.PARTIAL if missing else DiagnosticStatus.COMPUTED
        used = total if outcome.kept is None else len(outcome.kept)
        if outcome.parse_failed:
            unused["parse_failed"] = outcome.parse_failed
        if excluded := total - used - outcome.parse_failed:
            unused["null_excluded"] = excluded
        warnings.extend(outcome.warnings)
    return DiagnosticResultV1(
        diagnostic_id=spec_id, diagnostic_version=DIAGNOSTIC_VERSION, status=status,
        csv_artifact=source.csv_ref(), columns_read=tuple(dict.fromkeys(present)),
        total_rows=total, used_rows=used, unused_reason_counts=unused,
        row_set_hash=(hashlib.sha256(",".join(str(i) for i in sorted(outcome.kept)).encode()).hexdigest()
                      if outcome.kept and used < total else None),
        values=outcome.values, warnings=tuple(warnings),
        implementation_version=IMPLEMENTATION_VERSION)


def diagnostic_result_id(result: DiagnosticResultV1) -> str:
    """Stable handle for one observed result; distinct from its registered diagnostic name."""
    encoded = json.dumps(result.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return f"dr:{result.diagnostic_id}:{hashlib.sha256(encoded.encode()).hexdigest()[:24]}"


def run_requested_diagnostics(
    proposal: AgentDesignProposalV2, plan: DiagnosticPlanV2, source: FrameSource,
    observed: Sequence[DiagnosticResultV1] = (),
) -> tuple[DiagnosticResultV1, ...]:
    prior = tuple(row.diagnostic_id for row in observed)
    assessed = tuple(row.diagnostic_result_id for row in proposal.diagnostic_assessments)
    result_ids = tuple(diagnostic_result_id(row) for row in observed)
    requested = proposal.requested_diagnostic_ids
    if any(len(ids) != len(set(ids)) for ids in (prior, assessed, requested)) or set(assessed) != set(result_ids):
        raise DiagnosticError("assess exactly the observed diagnostics", DIAGNOSTIC_ASSESSMENT_MISMATCH)
    if len(prior) + len(requested) > DIAGNOSTIC_TOOL_CALL_LIMIT:
        raise DiagnosticError("diagnostic tool-call budget exhausted", DIAGNOSTIC_TOOL_BUDGET_EXHAUSTED)
    planned = {row.diagnostic_id: row for row in plan.items}
    if refused := set(requested) - set(planned) | set(requested) & set(prior):
        raise DiagnosticError(f"diagnostics not allowed: {sorted(refused)}", DIAGNOSTIC_NOT_ALLOWED)
    return tuple(run_diagnostic(item.diagnostic_id, source,
                                compiler_v2.diagnostic_parameters(item))
                 for item in (planned[diagnostic_id] for diagnostic_id in requested))
