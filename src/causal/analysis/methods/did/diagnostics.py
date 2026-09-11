"""Numerical diagnostic measurements for did; no result-driven selection."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import polars as pl

from causal.analysis.integration import contracts as ec

if TYPE_CHECKING:
    from pyfixest.estimation.feols_ import Feols


def _leads(fitted: Feols, params: ec.ValueMap) -> ec.ValueMap:
    # §11.3: every approved lead estimate and the joint prespecified pre-period Wald test, both
    # read off the same fit. A non-significant test does not prove parallel trends.
    from causal.analysis.methods.did.estimation import REFERENCE, _coefficients, _number

    horizon = -_number(params, "event_window_leads", 3.0) + REFERENCE
    cells = [(index, rel, cohort) for index, rel, cohort in _coefficients(fitted)
             if horizon <= rel < REFERENCE]
    found: ec.ValueMap = {"leads": len(cells)} | {
        f"lead_{rel:g}_cohort_{cohort:g}": float(fitted._beta_hat[index])
        for index, rel, cohort in cells}
    if not cells:
        return found
    selector = np.eye(len(fitted._coefnames))[[index for index, _, _ in cells]]
    found["largest_lead_estimate"] = max(abs(float(fitted._beta_hat[i])) for i, _, _ in cells)
    tested = cast(Any, fitted.wald_test)(R=selector, q=np.zeros(len(cells)))
    found["joint_pre_period_p_value"] = float(tested["pvalue"])
    return found


def _support(cells: pl.DataFrame, frame: pl.DataFrame, roles: Mapping[str, str],
             params: ec.ValueMap) -> dict[str, ec.ValueMap]:
    # §11.2 and §11.3: cohort-relative support, pre and post placement, composition change across
    # time, the cluster count the covariance rests on, and the aggregate's weight spread.
    from causal.analysis.methods.did.estimation import (
        COHORT,
        POOL,
        REFERENCE,
        REL,
        SENTINEL,
        _shares,
    )

    unit, time = roles["unit_identifier"], roles["time"]
    treated, periods = cells.filter(pl.col("treated") > 0), frame[time].n_unique()
    adopting, weights = cells.filter(pl.col(COHORT) > POOL), _shares(cells, params)
    balanced = int(frame.group_by(unit).agg(
        pl.col(time).n_unique().alias("seen")).filter(pl.col("seen") == periods).height)
    units, cluster = frame[unit].n_unique(), roles.get("cluster") or unit
    return {
        "group_time_cohort_support": {
            "cell_units": int(cast(int, treated["units"].min() or 0)), "cells": cells.height,
            "treated_cells": treated.height, "cohorts": int(treated[COHORT].n_unique())},
        "pre_and_post_period_placement": {
            "pre_periods": int(adopting.filter(pl.col(REL) < 0)[REL].n_unique()),
            "post_periods": int(adopting.filter(
                (pl.col(REL) >= 0) & (pl.col(REL) < SENTINEL))[REL].n_unique()),
            "periods": periods, "reference_period": REFERENCE},
        "attrition_composition_change": {"balanced_units": balanced, "units": units,
                                         "composition_change_share": 1.0 - (
                                             balanced / float(units or 1))},
        "cluster_covariance_adequacy": {
            "clusters": frame[cluster].n_unique(), "contributing_rows": frame.height},
        "aggregate_weight_sensitivity": {
            "weight_cells": len(weights),
            "largest_cell_weight": max(weights.values(), default=0.0)}}


def harvest(frame: pl.DataFrame, cells: pl.DataFrame, roles: Mapping[str, str],
            plan: ec.EstimationPlanV1, params: ec.ValueMap,
            fitted: Feols | None, *,
            include_figures: bool = True) -> dict[str, ec.ValueMap]:
    # The §11.3 required diagnostics, harvested from the frozen frame, the one pivot above, and
    # the fits already run — plus the frozen §17 series the figure builders wrap.
    from causal.analysis.methods.did.estimation import (
        COHORT,
        POOL,
        REFERENCE,
        REL,
        _coefficients,
        _number,
        _panel,
        series,
    )

    unit, time = roles["unit_identifier"], roles["time"]
    pairs, found = frame.select(unit, time).n_unique(), _support(cells, frame, roles, params)
    comparison = str(params.get("comparison_cohort") or "never_treated")
    anticipation = _number(params, "anticipation_periods", 0.0)
    found["unit_period_schema_reconciliation"] = {
        "units": frame[unit].n_unique(), "periods": frame[time].n_unique(), "rows": frame.height,
        "unit_periods": pairs, "duplicate_unit_periods": frame.height - pairs,
        "structure": "panel" if _panel(frame, roles) else "repeated_cross_section"}
    found["treatment_timing_anticipation"] = {
        "anticipation_periods": anticipation, "comparison_cohort": comparison, "cohorts": int(
            cells.filter(pl.col(COHORT) > POOL)[COHORT].n_unique())}
    found["event_study_pre_period_test"] = _leads(fitted, params) if fitted is not None else {}
    found["concurrent_event_qualification"] = {
        "carried_rule_ids": len(plan.numerical_failure_rule_ids), "comparison_cohort": comparison}
    found["reference_period_numerical_integrity"] = {
        "reference_rows": int(frame.filter(pl.col(REL) == REFERENCE).height),
        "fitted_cells": len(_coefficients(fitted)) if fitted is not None else 0,
        "converged": fitted is not None, "aggregation": str(params.get("aggregation") or "")}
    if include_figures:
        found.update(series(frame, cells, roles, plan, fitted))
    return found

