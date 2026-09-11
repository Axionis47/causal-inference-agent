"""Numerical diagnostic measurements for aipw; no result-driven selection."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from sklearn import metrics  # type: ignore[import-untyped]

from causal.analysis.common import legacy_engine as engine
from causal.analysis.integration import contracts as ec

if TYPE_CHECKING:
    from causal.analysis.methods.aipw.estimation import CrossFitRun, Data, Score, _Bits, _Floats


def _nuisance_values(data: Data, propensity: _Floats, fitted: _Floats) -> ec.ValueMap:
    # §10.5: calibration of the held-out propensity and the registered performance measures, read
    # off out-of-fold predictions only. Good prediction is evidence, never identification.
    treated, spread = data.treated.astype(np.float64), float(np.var(propensity))
    slope = float(np.mean((propensity - propensity.mean()) * (treated - treated.mean()))
                  / spread) if spread > 0.0 else 1.0
    split = 0 < int(np.count_nonzero(data.treated)) < data.treated.size
    return {"calibration_slope": slope, "calibration_slope_deviation": abs(slope - 1.0),
            "propensity_auc": float(metrics.roc_auc_score(treated, propensity)) if split else 0.5,
            "outcome_rmse": math.sqrt(float(np.mean((data.outcome - fitted) ** 2)))}


def _positivity_values(propensity: _Floats, found: Score,
                       thresholds: Mapping[str, ec.ParameterValue]
                       ) -> tuple[ec.ValueMap, ec.ValueMap, ec.ValueMap]:
    # §10.4 and §10.5 off the same three arrays: how much propensity mass sits outside the
    # registered support (the bound is NOT applied here, since a guarded propensity would hide the
    # very mass this reports), the inverse-weight tails and effective sample the estimator leans
    # on, and the influence distribution with the single-unit share that would move the estimate.
    from causal.analysis.methods.aipw.estimation import _number

    weights, influence = found.weights, found.influence
    outside = int(np.count_nonzero((propensity < _number(thresholds, "min_propensity", 0.01))
                                   | (propensity > _number(thresholds, "max_propensity", 0.99))))
    total, top = float(weights.sum()), float(weights.max())
    effective = total * total / float((weights**2).sum()) if total > 0.0 else 0.0
    moved, largest = float(np.abs(influence).sum()), float(np.abs(influence).max())
    return ({"propensity": min(float(propensity.min()), 1.0 - float(propensity.max())),
             "propensity_max": float(propensity.max()),
             "out_of_support_share": outside / max(propensity.size, 1)},
            {"effective_sample_size": effective, "weight_max": top,
             "effective_sample_fraction": effective / max(weights.size, 1),
             "single_weight_share": top / total if total > 0.0 else 0.0},
            {"single_unit_influence_share": largest / moved if moved > 0.0 else 0.0,
             "influence_standard_deviation": float(influence.std())})


def _balance_values(view: pl.DataFrame, roles: Mapping[str, str], weights: _Floats) -> ec.ValueMap:
    # §10.5: covariate balance before and under the estimator's implied weighting, through the
    # shared engine helper so every pack reports comparability the same way.
    from causal.analysis.methods.aipw.estimation import WEIGHT_COLUMN, _number, covariates

    columns = [name for name in covariates(roles) if view[name].dtype.is_numeric()]
    arm, weighted = roles["treatment"], view.with_columns(pl.Series(WEIGHT_COLUMN, weights))
    rows = (("before", engine.balance(view, arm, columns)),
            ("weighted", engine.balance(weighted, arm, columns, weight_column=WEIGHT_COLUMN)))
    found: ec.ValueMap = {f"{prefix}:{name}": _number(row[name], "standardized_difference", 0.0)
                          for prefix, row in rows for name in columns}
    found["standardized_difference"] = max(
        (abs(_number(found, f"weighted:{name}", 0.0)) for name in columns), default=0.0)
    return found


def _overlap_values(propensity: _Floats, treated: _Bits) -> ec.ValueMap:
    # The fixed §17 overlap binning: ten equal bins over the unit interval, chosen here and never
    # after a result was seen. The counts are aggregates; no raw observation is exposed.
    from causal.analysis.methods.aipw.estimation import OVERLAP_BINS

    placed = np.clip(np.digitize(propensity, np.linspace(0.0, 1.0, OVERLAP_BINS + 1)[1:-1]),
                     0, OVERLAP_BINS - 1)
    lower = max(float(propensity[rows].min()) for rows in (treated, ~treated))
    upper = min(float(propensity[rows].max()) for rows in (treated, ~treated))
    limits: ec.ValueMap = {"support_limit_lower": lower, "support_limit_upper": upper,
                          "common_support_exists": lower <= upper}
    return limits | {
            f"{name}_bin_{index}": int(np.count_nonzero((placed == index) & rows))
            for name, rows in (("control", ~treated), ("treated", treated))
            for index in range(OVERLAP_BINS)}


def harvest(view: pl.DataFrame, roles: Mapping[str, str], plan: ec.EstimationPlanV1,
            run: CrossFitRun,
            thresholds: Mapping[str, ec.ParameterValue]) -> dict[str, ec.ValueMap]:
    # The §10.5 required diagnostics, all read off the arrays this run already holds. Nothing here
    # refits, re-splits, or re-bins anything, and no per-row value reaches a diagnostic payload.
    from causal.analysis.methods.aipw.estimation import BOUND_RULE, covariates

    data, found = run.data, run.score
    propensity, under_control, under_treated, _ = run.predictions
    support = [min(row["treated"], row["control"]) for row in run.counts_by_fold.values()]
    fitted = np.where(data.treated, under_treated, under_control)
    mass, weights, influence = _positivity_values(propensity, found, thresholds)
    overlap = _overlap_values(propensity, data.treated)
    mass.update({key: value for key, value in overlap.items() if "_bin_" not in key})
    return {
        "fold_assignment_treatment_support": {
            "folds": run.fold_count, "min_fold_treated": min(support, default=0),
            "converged_folds": run.converged_folds,
            "folds_without_both_states": sum(1 for row in support if not row)},
        "nuisance_calibration_performance": _nuisance_values(data, propensity, fitted),
        "propensity_common_support": mass, "influence_score_distribution": influence,
        "weight_tail_effective_sample": weights, "weight_summary": weights,
        "weighted_covariate_balance": _balance_values(view, roles, found.weights),
        "cross_fit_score_integrity": {
            "folds": run.fold_count, "predicted_rows": int(propensity.size),
            "standard_error": found.standard_error, "bound_rule_id": BOUND_RULE,
            "validation_rows_fitted": run.leaked_rows,
            "propensity_bound_rows": found.bounded_rows},
        "preprocessing_missingness_usage": {
            "adjustment_columns": len(covariates(roles)),
            "missingness_indicators": sum(1 for role in roles if "missing" in role),
            "recipe_ids": ",".join(plan.preprocessing_recipe_ids)},
        "unmeasured_confounding_qualification": {
            "qualification": "residual unmeasured confounding remains possible by design",
            "influence_standard_deviation": influence["influence_standard_deviation"]},
        "overlap_bins": overlap}

