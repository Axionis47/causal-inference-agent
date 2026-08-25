"""Read-only pre-repair diagnostics over the selected CSV (PRD-002 §4.2, §14; T-012 §4)."""

from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Final, NamedTuple, Protocol

import polars as pl

from causal.design.frame import DiagnosticResultV1, DiagnosticStatus
from causal.design.packs import PREREPAIR_DIAGNOSTIC_IDS, MethodPackRegistry
from causal.design.tools import ToolHandler
from causal.shared.contracts import ArtifactRef
from causal.shared.envelope import AgentTaskEnvelopeV1

__all__ = [
    "DIAGNOSTIC_SPECS", "IMPLEMENTATION_VERSION", "RULE_OPS", "BytesFrameSource",
    "CsvObjectFrameSource", "DiagnosticError", "DiagnosticSpec", "FrameSource",
    "ObjectReader", "make_diagnostic_handlers", "run_diagnostic",
]

IMPLEMENTATION_VERSION: Final = "design-diagnostics.v1"
DIAGNOSTIC_VERSION: Final = "prerepair-diagnostic.v1"
UNKNOWN_DIAGNOSTIC, DIAGNOSTIC_NOT_ALLOWED = "unknown_diagnostic", "diagnostic_not_allowed"
UNKNOWN_COLUMN, UNSUPPORTED_RULE = "unknown_column", "unsupported_rule"
# The closed eligibility-rule grammar (PRD-002 §15); one preview's rules are a conjunction.
RULE_OPS: Final = ("eq", "ne", "ge", "le", "not_null", "in")
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


class FrameSource(Protocol):
    """The committed CSV a diagnostic reads; nothing here may write a table (PRD-002 §4.2)."""

    def csv_ref(self) -> ArtifactRef: ...

    def frame(self) -> pl.DataFrame: ...


class ObjectReader(Protocol):
    """The one `ObjectStore` method a frame source needs."""

    def get(self, locator: str) -> bytes: ...


@dataclass(frozen=True)
class BytesFrameSource:
    """A frame source over already-held CSV bytes."""

    name: str
    data: bytes
    ref: ArtifactRef

    def csv_ref(self) -> ArtifactRef:
        return self.ref

    def frame(self) -> pl.DataFrame:
        return pl.read_csv(io.BytesIO(self.data))


@dataclass(frozen=True)
class CsvObjectFrameSource:
    """A frame source over the committed CSV object (D-019 locator)."""

    objects: ObjectReader
    locator: str
    ref: ArtifactRef

    def csv_ref(self) -> ArtifactRef:
        return self.ref

    def frame(self) -> pl.DataFrame:
        return pl.read_csv(io.BytesIO(self.objects.get(self.locator)))


class DiagnosticSpec(NamedTuple):
    """One code-level diagnostic row: the primitive that runs and its default parameters."""

    diagnostic_id: str
    primitive: str
    default_params: Mapping[str, Any]


# One primitive's values plus the exact row accounting it produced.
class _Outcome(NamedTuple):
    values: _Values
    warnings: tuple[str, ...]
    kept: list[int] | None
    parse_failed: int = 0


_DEFAULTS: Final[dict[str, dict[str, Any]]] = {
    "count_by": {"columns": (), "cutoff": None, "running_column": None},
    "missing_share": {"target": None, "by": (), "cutoff": None, "running_column": None},
    "uniqueness": {"key_columns": ()}, "availability": {"columns": ()},
    "level_profile": {"column": None, "max_levels": 20, "min_level_count": 5},
    "numeric_support": {"column": None, "cutoff": None, "band": None, "top_values": 3},
}
_REQUIRED: Final[dict[str, tuple[str, ...]]] = {
    "count_by": ("columns",), "missing_share": ("target",), "uniqueness": ("key_columns",),
    "level_profile": ("column",), "numeric_support": ("column",), "availability": ("columns",)}
# Every PRD-002 §13.1–§13.4 pre-repair id, mapped to the primitive reporting its raw inputs.
_IDS_BY_PRIMITIVE: Final[dict[str, tuple[str, ...]]] = {
    "count_by": ("arm_counts", "cluster_sizes", "power_precision_feasibility",
                 "treatment_prevalence", "rough_overlap", "effective_sample_feasibility",
                 "cross_fitting_feasibility", "group_time_counts", "panel_completeness",
                 "adoption_cohorts", "composition", "clustering_feasibility",
                 "cutoff_side_counts"),
    "missing_share": ("outcome_missingness", "missingness", "missingness_by_group_time",
                      "missingness_by_side_and_distance"),
    "uniqueness": ("assignment_unit_uniqueness", "unit_period_uniqueness", "duplicates"),
    "level_profile": ("level_sparsity",),
    "numeric_support": ("distance_to_cutoff_support", "mass_points",
                        "density_manipulation_warnings", "bandwidth_feasibility"),
    "availability": ("baseline_availability", "compliance_availability",
                     "covariate_availability", "pre_period_availability")}
_OVERRIDES: Final[dict[str, dict[str, Any]]] = {
    "cutoff_side_counts": {"columns": (SIDE,)},
    "missingness_by_side_and_distance": {"by": (SIDE,)}}
DIAGNOSTIC_SPECS: Final[dict[str, DiagnosticSpec]] = {
    identity: DiagnosticSpec(identity, primitive,
                             _DEFAULTS[primitive] | _OVERRIDES.get(identity, {}))
    for primitive, ids in _IDS_BY_PRIMITIVE.items() for identity in ids}
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
    indexed = frame.with_row_index(_ROW)
    used = indexed.drop_nulls(subset=list(columns)) if columns else indexed
    return used, [int(value) for value in used[_ROW]]


# Group sizes: arms, clusters, group-time cells, adoption cohorts, cutoff sides.
def _count_by(frame: pl.DataFrame, params: Mapping[str, Any]) -> _Outcome:
    columns = list(params["columns"])
    used, kept = _used(frame, columns)
    counts = used.group_by(columns).len().sort(columns)
    cells = math.prod(int(used[name].n_unique()) for name in columns)
    values: _Values = {"row_count": used.height, "group_count": counts.height,
                       "expected_cells": cells, "cell_completeness": _share(counts.height, cells)}
    for row in counts.head(_MAX_GROUPS).iter_rows():
        values[f"count:{_key(row[:-1])}"] = int(row[-1])
    warnings = (f"only the first {_MAX_GROUPS} groups are reported",)
    return _Outcome(values, warnings if counts.height > _MAX_GROUPS else (), kept)


# Missing share of one target column, overall and inside each requested group.
def _missing_share(frame: pl.DataFrame, params: Mapping[str, Any]) -> _Outcome:
    target, by = str(params["target"]), list(params["by"])
    used, kept = _used(frame, by)
    missing = int(used[target].null_count())
    values: _Values = {"row_count": used.height, "missing_count": missing,
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
    keys = list(params["key_columns"])
    used, kept = _used(frame, keys)
    distinct = int(used.select(keys).n_unique())
    repeated = used.height - distinct
    values: _Values = {"row_count": used.height, "distinct_key_count": distinct,
                       "duplicate_row_count": repeated, "is_unique": not repeated}
    return _Outcome(values, () if not repeated else (f"{repeated} rows repeat a key",), kept)


# Level counts and sparsity for one categorical column.
def _level_profile(frame: pl.DataFrame, params: Mapping[str, Any]) -> _Outcome:
    column, cap = str(params["column"]), int(params["max_levels"])
    floor_count = int(params["min_level_count"])
    used, kept = _used(frame, [column])
    counts = used[column].value_counts().sort([column])
    sizes = [int(row[1]) for row in counts.iter_rows()]
    values: _Values = {"row_count": used.height, "distinct_count": counts.height,
                       "sparse_level_count": sum(1 for size in sizes if size < floor_count),
                       "min_level_count": min(sizes, default=0)}
    for row in counts.head(cap).iter_rows():
        values[f"level:{_key(row[:1])}"] = int(row[1])
    warnings = (f"only the first {cap} levels are reported",)
    return _Outcome(values, warnings if counts.height > cap else (), kept)


# Range, quantiles, repeated mass points, cutoff support, and a bounded distance histogram.
def _numeric_support(frame: pl.DataFrame, params: Mapping[str, Any]) -> _Outcome:
    column = str(params["column"])
    indexed = frame.with_row_index(_ROW).with_columns(
        pl.col(column).cast(pl.Float64, strict=False).alias(_VALUE))
    parse_failed = int(indexed[_VALUE].null_count()) - int(frame[column].null_count())
    used = indexed.drop_nulls(subset=[_VALUE])
    series = used[_VALUE]
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
        count = int(frame[str(name)].count())
        values[f"non_null_count:{name}"] = count
        values[f"non_null_share:{name}"] = _share(count, frame.height)
    return _Outcome(values, (), None)


_PRIMITIVES: Final[dict[str, Callable[[pl.DataFrame, Mapping[str, Any]], _Outcome]]] = {
    "count_by": _count_by, "missing_share": _missing_share, "uniqueness": _uniqueness,
    "level_profile": _level_profile, "numeric_support": _numeric_support,
    "availability": _availability}


# Add the derived `side` column when a running column and a cutoff were both resolved.
def _with_side(frame: pl.DataFrame, params: Mapping[str, Any]) -> pl.DataFrame:
    running, cutoff = params.get("running_column"), params.get("cutoff")
    if running is None or cutoff is None:
        return frame
    value = pl.col(str(running))
    return frame.with_columns(
        pl.when(value.is_null()).then(pl.lit(None, dtype=pl.String))
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


def _row_hash(kept: list[int]) -> str:
    return hashlib.sha256(",".join(str(index) for index in sorted(kept)).encode()).hexdigest()


def run_diagnostic(
    spec_id: str, source: FrameSource, params: Mapping[str, Any] | None = None
) -> DiagnosticResultV1:
    """One read-only diagnostic with exact denominators and no pass/fail judgment (§14)."""
    spec = DIAGNOSTIC_SPECS.get(spec_id)
    if spec is None:
        raise DiagnosticError(f"unknown diagnostic {spec_id!r}", UNKNOWN_DIAGNOSTIC)
    frame = source.frame()
    total = frame.height
    resolved, present, missing = _resolve(frame, {**spec.default_params, **(params or {})})
    warnings = [f"requested column {name!r} is absent from the selected CSV" for name in missing]
    outcome, used = _Outcome({}, (), None), 0
    unused: dict[str, int] = {}
    status = DiagnosticStatus.NOT_COMPUTABLE
    if absent := [key for key in _REQUIRED[spec.primitive] if not resolved.get(key)]:
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
    selected = outcome.kept is not None and used < total
    return DiagnosticResultV1(
        diagnostic_id=spec_id, diagnostic_version=DIAGNOSTIC_VERSION, status=status,
        csv_artifact=source.csv_ref(), columns_read=tuple(dict.fromkeys(present)),
        total_rows=total, used_rows=used, unused_reason_counts=unused,
        row_set_hash=_row_hash(outcome.kept) if selected and outcome.kept else None,
        values=outcome.values, warnings=tuple(warnings),
        implementation_version=IMPLEMENTATION_VERSION)


# One closed-grammar eligibility conjunct as a polars predicate.
def _rule_expr(rule: Mapping[str, Any], known: set[str]) -> pl.Expr:
    column, op = str(rule["column"]), str(rule["op"])
    if column not in known:
        raise DiagnosticError(f"unknown eligibility column {column!r}", UNKNOWN_COLUMN)
    if op not in RULE_OPS:
        raise DiagnosticError(f"unsupported eligibility op {op!r}", UNSUPPORTED_RULE)
    target = pl.col(column)
    if op == "not_null":
        return target.is_not_null()
    if op == "in":
        return target.is_in(list(rule["value"]))
    return {"eq": target.eq, "ne": target.ne, "ge": target.ge, "le": target.le}[op](rule["value"])


def _group_counts(frame: pl.DataFrame, column: str) -> dict[str, int]:
    counts = frame.group_by(column).len().sort(column)
    return {_key(row[:1]): int(row[1]) for row in counts.head(_MAX_GROUPS).iter_rows()}


# `run_preflight_diagnostic`: one spec row, optionally bounded by the pack's allowed list.
def _run_preflight(
    source_for: Callable[[], FrameSource], packs: MethodPackRegistry,
    envelope: AgentTaskEnvelopeV1, arguments: Mapping[str, Any],
) -> dict[str, Any]:
    diagnostic_id = str(arguments["diagnostic_id"])
    if diagnostic_id not in DIAGNOSTIC_SPECS:
        raise DiagnosticError(f"unknown diagnostic {diagnostic_id!r}", UNKNOWN_DIAGNOSTIC)
    method_id = arguments.get("method_id")
    if method_id is not None and diagnostic_id not in packs.get(
            str(method_id)).allowed_prerepair_diagnostic_ids:
        raise DiagnosticError(f"{diagnostic_id!r} is not allowed for method {method_id!r}",
                              DIAGNOSTIC_NOT_ALLOWED)
    result = run_diagnostic(diagnostic_id, source_for(), arguments.get("params") or {})
    return dict(result.model_dump(mode="json"))


# `preview_eligibility_impact`: kept and excluded counts under proposed rules; no writes.
def _preview_eligibility(
    source_for: Callable[[], FrameSource], envelope: AgentTaskEnvelopeV1,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    frame = source_for().frame()
    known = set(frame.columns)
    rules = [dict(rule) for rule in arguments["rules"]]
    kept = frame
    for rule in rules:
        kept = kept.filter(_rule_expr(rule, known))
    result: dict[str, Any] = {"total_rows": frame.height, "kept_rows": kept.height,
                              "excluded_rows": frame.height - kept.height,
                              "rule_count": len(rules)}
    if (by := arguments.get("by")) is not None:
        column = str(by)
        if column not in known:
            raise DiagnosticError(f"unknown grouping column {column!r}", UNKNOWN_COLUMN)
        totals, keeps = _group_counts(frame, column), _group_counts(kept, column)
        result |= {"by": column, "kept_by": keeps, "excluded_by": {
            key: total - keeps.get(key, 0) for key, total in totals.items()}}
    return result


def make_diagnostic_handlers(
    source_for: Callable[[], FrameSource], packs: MethodPackRegistry
) -> dict[str, ToolHandler]:
    """The two read-only inspection handlers the method-design task may call (PRD-002 §15)."""
    return {"run_preflight_diagnostic": partial(_run_preflight, source_for, packs),
            "preview_eligibility_impact": partial(_preview_eligibility, source_for)}
