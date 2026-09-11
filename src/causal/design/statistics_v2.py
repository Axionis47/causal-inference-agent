"""Method-neutral statistical primitives used by bound design diagnostics."""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any, cast

import polars as pl

_ROW = "__row_index"
Result = tuple[dict[str, float | int | str | bool | None], tuple[str, ...], list[int], int]


def _indexed(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_row_index(_ROW)


def power_precision(frame: pl.DataFrame, params: Mapping[str, Any]) -> Result:
    treatment, outcome = str(next(iter(params["columns"]))), str(params["target"])
    indexed = _indexed(frame).with_columns(
        pl.col(outcome).cast(pl.Float64, strict=False).alias("__outcome"))
    parse_failed = int(indexed["__outcome"].null_count()) - int(frame[outcome].null_count())
    used = indexed.drop_nulls(subset=[treatment, "__outcome"])
    groups = used.group_by(treatment).agg(
        pl.len().alias("n"), pl.col("__outcome").var().alias("variance")).sort(treatment)
    rows = [(int(row[1]), float(row[2])) for row in groups.iter_rows()
            if int(row[1]) >= 2 and row[2] is not None]
    pair_se = [math.sqrt(va / na + vb / nb)
               for (na, va), (nb, vb) in itertools.combinations(rows, 2) if na and nb]
    maximum = max(pair_se, default=0.0)
    values: dict[str, float | int | str | bool | None] = {
        "row_count": used.height, "group_count": groups.height,
        "outcome_sd": float(cast(float | int | None, used["__outcome"].std()) or 0.0),
        "standard_error_max_pair": maximum,
        "mde_80pct_95ci": 2.8016 * maximum if pair_se else None,
    }
    warnings = () if groups.height >= 2 and pair_se else (
        "power requires at least two arms with estimable outcome variance",)
    return values, warnings, [int(v) for v in used[_ROW]], parse_failed


def _numeric_support(frame: pl.DataFrame, treatment: str, column: str,
                     groups: tuple[Any, Any]) -> float:
    values = [frame.filter(pl.col(treatment) == group)[column].cast(
        pl.Float64, strict=False).drop_nulls() for group in groups]
    if any(series.is_empty() for series in values):
        return 0.0
    low = max(float(cast(float | int, series.min())) for series in values)
    high = min(float(cast(float | int, series.max())) for series in values)
    total = sum(series.len() for series in values)
    inside = sum(series.filter((series >= low) & (series <= high)).len() for series in values)
    return inside / total if low <= high and total else 0.0


def _categorical_support(frame: pl.DataFrame, treatment: str, column: str,
                         groups: tuple[Any, Any]) -> float:
    levels = [set(frame.filter(pl.col(treatment) == group)[column].drop_nulls().to_list())
              for group in groups]
    common = levels[0] & levels[1]
    denominator = len(levels[0] | levels[1])
    return len(common) / denominator if denominator else 0.0


def rough_overlap(frame: pl.DataFrame, params: Mapping[str, Any]) -> Result:
    treatment, columns = str(params["target"]), tuple(str(v) for v in params["columns"])
    indexed = _indexed(frame).drop_nulls(subset=[treatment])
    groups = tuple(indexed[treatment].unique().sort().to_list())
    if len(groups) != 2 or not columns:
        unavailable: dict[str, float | int | str | bool | None] = {
            "row_count": indexed.height, "treatment_group_count": len(groups),
            "covariate_count": len(columns), "minimum_common_support_share": 0.0}
        return unavailable, ("rough overlap requires binary treatment and covariates",), [
            int(v) for v in indexed[_ROW]], 0
    pair = (groups[0], groups[1])
    shares: list[float] = []
    values: dict[str, float | int | str | bool | None] = {
        "row_count": indexed.height, "treatment_group_count": 2,
        "covariate_count": len(columns)}
    for column in columns:
        series = indexed[column]
        share = (_numeric_support(indexed, treatment, column, pair)
                 if series.dtype.is_numeric() else _categorical_support(
                     indexed, treatment, column, pair))
        shares.append(share)
        values[f"common_support_share:{column}"] = share
    values["minimum_common_support_share"] = min(shares)
    warning = ("at least one covariate has no marginal common support",) if not min(shares) else ()
    return values, warning, [int(v) for v in indexed[_ROW]], 0


def panel_structure(frame: pl.DataFrame, params: Mapping[str, Any]) -> Result:
    keys = tuple(str(v) for v in params["key_columns"])
    indexed = _indexed(frame).drop_nulls(subset=list(keys))
    entities, periods = indexed[keys[0]].n_unique(), indexed[keys[1]].n_unique()
    cells = indexed.select(keys).n_unique()
    expected = int(entities) * int(periods)
    values: dict[str, float | int | str | bool | None] = {
        "row_count": indexed.height, "group_count": int(cells), "entity_count": int(entities),
        "period_count": int(periods), "observed_cells": int(cells),
        "expected_cells": expected, "cell_completeness": cells / expected if expected else 0.0,
        "duplicate_key_rows": indexed.height - int(cells),
    }
    counts = indexed.group_by(list(keys)).len().sort(list(keys))
    for row in counts.head(50).iter_rows():
        label = "|".join("null" if value is None else str(value) for value in row[:-1])
        values[f"count:{label}"] = int(row[-1])
    warnings = ("unit/group-period keys repeat",) if indexed.height != cells else ()
    return values, warnings, [int(v) for v in indexed[_ROW]], 0


def _ordered(value: Any) -> float | None:
    if isinstance(value, datetime):
        normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value
        return normalized.timestamp()
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time(), tzinfo=UTC).timestamp()
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            try:
                return _ordered(datetime.fromisoformat(value))
            except ValueError:
                return None
    return None


def _transition_support(rows: list[tuple[Any, Any, Any]]) -> tuple[int, int, int, int]:
    grouped: dict[Any, list[tuple[float, Any]]] = {}
    for group, period, treatment in rows:
        if (timestamp := _ordered(period)) is not None:
            grouped.setdefault(group, []).append((timestamp, treatment))
    transitions = before = after = 0
    adoption_times: set[float] = set()
    for history in grouped.values():
        sequence = sorted(history)
        if len({value for _, value in sequence}) < 2:
            continue
        first_change = next((index for index in range(1, len(sequence))
                             if sequence[index][1] != sequence[index - 1][1]), None)
        if first_change is not None:
            transitions += 1
            before += first_change
            after += len(sequence) - first_change
            adoption_times.add(sequence[first_change][0])
    return transitions, before, after, len(adoption_times)


def _history_count(rows: list[tuple[Any, Any, Any]]) -> int:
    grouped: dict[Any, list[tuple[float, str]]] = {}
    for group, period, treatment in rows:
        if (timestamp := _ordered(period)) is not None:
            grouped.setdefault(group, []).append((timestamp, repr(treatment)))
    return len({tuple(value for _, value in sorted(history))
                for history in grouped.values()})


def did_support(frame: pl.DataFrame, params: Mapping[str, Any]) -> Result:
    group, period = (str(value) for value in params["key_columns"])
    treatment, adoption = str(params["target"]), params["adoption_time"]
    adoption_column = str(adoption) if str(adoption) in frame.columns else None
    columns = list(dict.fromkeys((group, period, treatment, *((adoption_column,) if adoption_column else ()))))
    used = _indexed(frame).drop_nulls(subset=columns)
    rows = list(used.select(pl.col(group).alias("__group"), pl.col(period).alias("__period"),
                            pl.col(treatment).alias("__treatment")).iter_rows())
    transitions, transition_pre, transition_post, transition_cohorts = _transition_support(rows)
    pre = post = 0
    for threshold_row in used.select(
            period, *((adoption_column,) if adoption_column else ())).iter_rows():
        observed = _ordered(threshold_row[0])
        threshold = _ordered(threshold_row[1] if adoption_column else adoption)
        if observed is None or threshold is None:
            continue
        pre += observed < threshold
        post += observed >= threshold
    mode = "adoption_column" if adoption_column else "fixed_adoption_time"
    if not pre or not post:
        pre, post, mode = transition_pre, transition_post, "observed_treatment_transition"
    metrics: dict[str, float | int | str | bool | None] = {
        "row_count": used.height, "group_count": used[group].n_unique(),
        "period_count": used[period].n_unique(),
        "treatment_level_count": used[treatment].n_unique(),
        "treatment_history_count": _history_count(rows),
        "transition_group_count": transitions, "pre_period_rows": pre,
        "post_period_rows": post, "support_mode": mode,
        "adoption_cohort_count": max(transition_cohorts, 1),
        "adoption_profile_id": "simultaneous" if transition_cohorts <= 1 else "staggered",
    }
    warnings = () if pre and post else ("pre/post support could not be established",)
    return metrics, warnings, [int(value) for value in used[_ROW]], 0


def rdd_assignment(frame: pl.DataFrame, params: Mapping[str, Any]) -> Result:
    running, treatment = str(params["running_column"]), str(params["target"])
    cutoff = float(params["cutoff"])
    indexed = _indexed(frame).with_columns(
        pl.col(running).cast(pl.Float64, strict=False).alias("__running"),
        pl.col(treatment).cast(pl.String).alias("__treatment"))
    parse_failed = int(indexed["__running"].null_count()) - int(frame[running].null_count())
    used = indexed.drop_nulls(subset=["__running", "__treatment"])
    below = used["__running"] < cutoff
    levels = tuple(sorted(str(value) for value in used["__treatment"].unique().to_list()))
    values: dict[str, float | int | str | bool | None] = {
        "row_count": used.height, "group_count": len(levels),
        "below_count": int(below.sum()), "at_or_above_count": int((~below).sum()),
        "assignment_direction": "inconsistent", "contradiction_count": used.height}
    if len(levels) == 2:
        candidates: list[tuple[int, str, str]] = []
        for treated in levels:
            actual = used["__treatment"] == treated
            above_errors = int((actual != ~below).sum())
            below_errors = int((actual != below).sum())
            candidates.extend(((above_errors, "above", treated),
                               (below_errors, "below", treated)))
        contradictions, direction, treated = min(candidates)
        comparator = next(level for level in levels if level != treated)
        values.update({"assignment_direction": direction if contradictions == 0 else "inconsistent",
                       "contradiction_count": contradictions, "treated_value": treated,
                       "comparator_value": comparator})
    warnings = () if values["assignment_direction"] != "inconsistent" else (
        "observed treatment does not follow one sharp cutoff direction",)
    return values, warnings, [int(value) for value in used[_ROW]], parse_failed
