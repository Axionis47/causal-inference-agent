"""Dimension impact and per-method structure validation (PRD-003 §9.5, §12; T-016)."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Final

import polars as pl

from causal.preparation.contracts import DiagnosticStatus, DimensionImpactV1, _Row
from causal.preparation.plans import UNKNOWN_METHOD_PACK, PreparationPackV1
from causal.preparation.stabilize import UNKNOWN_RULE_TARGET, StabilizationError
from causal.shared.contracts import Identity

NULL_LEVEL, LEVEL_EMPTIED = "__null__", "level_emptied"
BELOW_MINIMUM_ROWS, BELOW_MINIMUM_UNITS = "below_minimum_rows", "below_minimum_unique_units"
ARM_DESTROYED, ARM_SUPPORT = "arm_destroyed_by_exclusion", "arm_support_after_stabilization"
UNIT_GRAIN_VIOLATED, CUTOFF_SIDE_EMPTIED = "unit_grain_violated", "cutoff_side_emptied"
OBSERVED_ROLE_VIOLATED: Final = "treatment_and_outcome_observed"
GROUP_TIME_CELL_EMPTIED, PERIOD_REMOVED = "group_time_cell_emptied", "period_removed"
GROUP_TIME_CELL_SUPPORT, CUTOFF_SIDE_SUPPORT = "group_time_cell_min_rows", "cutoff_side_min_rows"


class DimensionKind(StrEnum):
    """How one deletion-impact dimension resolves to levels (§9.5)."""

    OVERALL = "overall"
    COLUMN_LEVELS = "column_levels"
    PRE_POST = "pre_post"
    CUTOFF_SIDE = "cutoff_side"


# One dimension the contract names, bound to the manifest role column it reads.
class DimensionSpecV1(_Row):
    dimension_id: Identity
    kind: DimensionKind
    column: Identity | None = None
    threshold: float | str | None = None


class StructureVerdict(StrEnum):
    """What the retained data leave of the method's required structure (§12)."""

    RUNNABLE = "runnable"
    NOT_RUNNABLE = "not_runnable"
    CONFLICT = "conflict"


# The manifest role columns and thresholds one method's structure gates read.
class MethodStructureSpecV1(_Row):
    unit_columns: tuple[Identity, ...] = ()
    treatment_column: Identity | None = None
    outcome_column: Identity | None = None
    group_column: Identity | None = None
    time_column: Identity | None = None
    running_variable_column: Identity | None = None
    threshold: float | str | None = None
    minimum_cell_rows: int = 1


# The §12 structure verdict and its stable codes, ready for the stabilization record.
class MethodStructureResultV1(_Row):
    verdict: StructureVerdict
    codes: tuple[Identity, ...]

    @property
    def status(self) -> DiagnosticStatus:
        """`StabilizationRecordV1.method_structure_status`: only a runnable structure passes."""
        return (DiagnosticStatus.PASS if self.verdict is StructureVerdict.RUNNABLE
                else DiagnosticStatus.FAIL)


def _column(frame: pl.DataFrame, name: str | None, role: str) -> list[object]:
    if name is None or name not in frame.columns:
        raise StabilizationError(f"{role} column {name} absent from the frame", UNKNOWN_RULE_TARGET)
    values: list[object] = frame.get_column(name).to_list()
    return values


# Numeric comparison when both sides are numbers; ISO-lexicographic otherwise.
def _at_or_after(value: object, threshold: float | str) -> bool:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value) >= float(threshold)
    return str(value) >= str(threshold)


# Every row's level under one dimension, in frame row order.
def _levels(frame: pl.DataFrame, spec: DimensionSpecV1) -> list[str]:
    if spec.kind is DimensionKind.OVERALL:
        return ["all"] * frame.height
    values = _column(frame, spec.column, spec.dimension_id)
    if spec.kind is DimensionKind.COLUMN_LEVELS:
        return [NULL_LEVEL if value is None else str(value) for value in values]
    if spec.threshold is None:
        raise StabilizationError(f"{spec.dimension_id} needs a threshold", UNKNOWN_RULE_TARGET)
    labels = (("pre", "post") if spec.kind is DimensionKind.PRE_POST
              else ("below_cutoff", "at_or_above_cutoff"))
    return [NULL_LEVEL if value is None else labels[_at_or_after(value, spec.threshold)]
            for value in values]


# Retained and excluded counts per level of every dimension the contract names (§9.5).
def dimension_impact(frame: pl.DataFrame, retained: Sequence[bool],
                     specs: Sequence[DimensionSpecV1]) -> tuple[DimensionImpactV1, ...]:
    reports: list[DimensionImpactV1] = []
    for spec in specs:
        levels = _levels(frame, spec)
        kept = Counter(level for level, keep in zip(levels, retained, strict=True) if keep)
        lost = Counter(level for level, keep in zip(levels, retained, strict=True) if not keep)
        emptied = sorted(level for level in lost if not kept[level])
        reports.append(DimensionImpactV1(
            dimension_id=spec.dimension_id,
            retained_by_level={level: kept[level] for level in sorted(set(levels))},
            excluded_by_level={level: lost[level] for level in sorted(set(levels))},
            warning_codes=tuple(f"{LEVEL_EMPTIED}:{level}" for level in emptied)))
    return tuple(reports)


# Both arms survive stabilization, each with its minimum support (§12.1).
def _randomized_experiment(frame: pl.DataFrame,
                           spec: MethodStructureSpecV1) -> tuple[list[str], list[str]]:
    arms = Counter(value for value in _column(frame, spec.treatment_column, "treatment")
                   if value is not None)
    if len(arms) < 2:
        return [ARM_DESTROYED], []
    return [], [ARM_SUPPORT] if min(arms.values()) < spec.minimum_cell_rows else []


# One row per approved unit, with treatment and outcome observed (§12.2).
def _aipw(frame: pl.DataFrame, spec: MethodStructureSpecV1) -> tuple[list[str], list[str]]:
    units = frame.select(spec.unit_columns) if spec.unit_columns else frame
    observed = _column(frame, spec.treatment_column, "treatment") + _column(
        frame, spec.outcome_column, "outcome")
    return ([UNIT_GRAIN_VIOLATED] * (units.height != units.unique().height)
            + [OBSERVED_ROLE_VIOLATED] * any(value is None for value in observed), [])


# Every group-time cell keeps its support, on both sides of adoption (§12.3).
def _did(frame: pl.DataFrame, spec: MethodStructureSpecV1) -> tuple[list[str], list[str]]:
    groups = _column(frame, spec.group_column, "group")
    periods = _column(frame, spec.time_column, "time")
    cells = Counter(zip(groups, periods, strict=True))
    expected = {(group, period) for group in set(groups) for period in set(periods)}
    if expected - set(cells):
        return [GROUP_TIME_CELL_EMPTIED], []
    sides = set() if spec.threshold is None else {
        _at_or_after(period, spec.threshold) for period in periods if period is not None}
    if spec.threshold is not None and len(sides) < 2:
        return [PERIOD_REMOVED], []
    return [], [GROUP_TIME_CELL_SUPPORT] if min(cells.values()) < spec.minimum_cell_rows else []


# Both cutoff sides keep observations, each with its minimum support (§12.4).
def _sharp_rdd(frame: pl.DataFrame, spec: MethodStructureSpecV1) -> tuple[list[str], list[str]]:
    if spec.threshold is None:
        raise StabilizationError("sharp RDD structure needs a cutoff", UNKNOWN_RULE_TARGET)
    running = _column(frame, spec.running_variable_column, "running_variable")
    sides = Counter(_at_or_after(value, spec.threshold) for value in running if value is not None)
    if len(sides) < 2:
        return [CUTOFF_SIDE_EMPTIED], []
    return [], [CUTOFF_SIDE_SUPPORT] if min(sides.values()) < spec.minimum_cell_rows else []


_METHODS: Final = {"randomized_experiment": _randomized_experiment, "aipw": _aipw,
                   "did": _did, "sharp_rdd": _sharp_rdd}


# Validate the retained frame against one method pack's required structure (§9.1 step 7).
def validate_structure(frame: pl.DataFrame, spec: MethodStructureSpecV1,
                       pack: PreparationPackV1) -> MethodStructureResultV1:
    check = _METHODS.get(pack.method_id)
    if check is None:
        raise StabilizationError(f"no structure gates for {pack.method_id}", UNKNOWN_METHOD_PACK)
    units = frame.select(spec.unit_columns).unique().height if spec.unit_columns else frame.height
    failures = ([BELOW_MINIMUM_ROWS] * (frame.height < pack.minimum_rows)
                + [BELOW_MINIMUM_UNITS] * (units < pack.minimum_unique_units))
    conflicts, method_failures = check(frame, spec)
    codes = tuple(sorted(conflicts + failures + method_failures))
    if conflicts:
        return MethodStructureResultV1(verdict=StructureVerdict.CONFLICT, codes=codes)
    if codes:
        return MethodStructureResultV1(verdict=StructureVerdict.NOT_RUNNABLE, codes=codes)
    return MethodStructureResultV1(verdict=StructureVerdict.RUNNABLE, codes=())


# One §9.5 dimension id to the approved role whose column supplies its levels; a dimension
# with no bound column reports one `all` level rather than inventing one.
DIMENSION_ROLES: Final[dict[str, str]] = {
    "arm": "treatment", "treatment_group": "treatment", "group": "group", "time": "time",
    "cohort": "adoption_time", "cluster": "cluster", "stratum": "stratum",
    "subgroup": "stratum", "cutoff_side": "running_variable"}


# One §9.5 spec per approved deletion-impact dimension, bound to its role column.
def dimension_specs(roles: Mapping[str, str], dimensions: Sequence[str],
                    threshold: str | None) -> tuple[DimensionSpecV1, ...]:
    specs = []
    for dimension in dimensions:
        column = roles.get(DIMENSION_ROLES.get(dimension, ""))
        cutoff = dimension == "cutoff_side" and threshold is not None
        specs.append(DimensionSpecV1(
            dimension_id=dimension, column=column, threshold=threshold if cutoff else None,
            kind=DimensionKind.OVERALL if column is None
            else DimensionKind.CUTOFF_SIDE if cutoff else DimensionKind.COLUMN_LEVELS))
    return tuple(specs)
