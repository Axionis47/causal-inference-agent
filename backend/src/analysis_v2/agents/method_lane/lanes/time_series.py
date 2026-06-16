"""Time-series lane: interrupted time series via segmented regression.

y ~ trend + post (+ post-trend change), HAC standard errors. The level
shift at the intervention is the primary estimand; a missing control
series means history/seasonality threats ride as warnings, not claims.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analysis_v2.spec import CausalSpec, EffectEstimate, EstimateResult, MethodLane, MethodPlan

from .common import (
    LaneArtifact,
    LaneInputError,
    LaneOutcome,
    ci_from,
    effects_table,
    safe_float,
    summary_markdown,
    to_numeric,
)

MIN_TIME_POINTS = 30
MIN_SIDE_POINTS = 10


def check_ready(frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec) -> list[str]:
    """Preconditions for the segmented regression, shared with the readiness
    checker. The point counts need the time-grouping, so it is rebuilt here
    exactly as run() does."""
    lane = "time_series"
    time_col = plan.settings.get("time_column")
    date = plan.settings.get("intervention_date")
    if not time_col or not date:
        return [f"{lane}: needs time_column and intervention_date"]
    missing = [c for c in (time_col, plan.outcome) if c not in frame.columns]
    if missing:
        return [f"{lane}: columns missing from the dataset: {missing}"]
    data = frame[[time_col, plan.outcome]].copy()
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    data[plan.outcome] = to_numeric(data[plan.outcome])
    data = data.dropna().sort_values(time_col)
    series = data.groupby(time_col)[plan.outcome].mean().reset_index()
    if len(series) < MIN_TIME_POINTS:
        return [f"{lane}: only {len(series)} time points"]
    post = (series[time_col] >= pd.to_datetime(date)).astype(int)
    n_pre = int((post == 0).sum())
    n_post = int((post == 1).sum())
    if n_pre < MIN_SIDE_POINTS or n_post < MIN_SIDE_POINTS:
        return [f"{lane}: needs at least 10 points on each side, has {n_pre}/{n_post}"]
    return []


def run(frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec) -> LaneOutcome:
    lane = "time_series"
    reasons = check_ready(frame, plan, spec)
    if reasons:
        raise LaneInputError(reasons[0])
    time_col = plan.settings.get("time_column")
    date = plan.settings.get("intervention_date")

    data = frame[[time_col, plan.outcome]].copy()
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    data[plan.outcome] = to_numeric(data[plan.outcome])
    data = data.dropna().sort_values(time_col)
    series = data.groupby(time_col)[plan.outcome].mean().reset_index()

    cut = pd.to_datetime(date)
    series["post"] = (series[time_col] >= cut).astype(int)
    n_pre = int((series["post"] == 0).sum())
    n_post = int((series["post"] == 1).sum())
    series["trend"] = np.arange(len(series))
    # Re-center the post-trend at the intervention so the 'post' coefficient
    # is the level shift AT the cut, not extrapolated back to t=0.
    cut_index = int(series["post"].idxmax())
    series["post_trend"] = (series["trend"] - series.loc[cut_index, "trend"]) * series["post"]

    design_cols = series[["trend", "post", "post_trend"]]
    spacing = series[time_col].diff().dt.days.median()
    if spacing == 1:  # daily data: absorb the weekday cycle
        weekday = pd.get_dummies(
            series[time_col].dt.dayofweek, prefix="wd", drop_first=True, dtype=float
        )
        design_cols = pd.concat([design_cols, weekday], axis=1)
    design = sm.add_constant(design_cols)
    maxlags = max(1, int(round(len(series) ** (1 / 3))))
    fit = sm.OLS(series[plan.outcome], design).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )
    shift = safe_float(fit.params["post"])
    se = safe_float(fit.bse["post"])
    lo, hi = ci_from(shift, se)
    slope_change = safe_float(fit.params["post_trend"])

    warnings = [
        "single-series before/after comparison: anything else that changed at "
        "the same time is an alternative explanation"
    ]
    pre_mean = float(series.loc[series.post == 0, plan.outcome].mean())

    result = EstimateResult(
        lane=MethodLane.TIME_SERIES,
        estimator=plan.estimator,
        effects=[
            EffectEstimate(
                estimand="level_shift",
                estimate=shift,
                std_error=se,
                ci_lower=lo,
                ci_upper=hi,
                p_value=float(fit.pvalues["post"]),
                unit=f"per-period {plan.outcome} (pre mean {pre_mean:.4g})",
                interpretation=(
                    f"level change at {date} after removing the pre-trend "
                    f"({n_pre} pre / {n_post} post points, HAC lags {maxlags})"
                ),
            ),
            EffectEstimate(
                estimand="slope_change",
                estimate=slope_change,
                std_error=safe_float(fit.bse["post_trend"]),
                ci_lower=ci_from(slope_change, float(fit.bse["post_trend"]))[0],
                ci_upper=ci_from(slope_change, float(fit.bse["post_trend"]))[1],
                p_value=float(fit.pvalues["post_trend"]),
                interpretation="per-period trend change after the intervention",
            ),
        ],
        n_rows_used=len(series),
        outcome=plan.outcome,
        warnings=warnings,
    )
    summary = summary_markdown(
        "Interrupted time series",
        [
            f"{len(series):,} time points around {date}",
            f"level shift {shift:.4g} (se {se:.3g}); slope change {slope_change:.4g}",
            "no control series: this is a within-series comparison",
        ],
        fit.summary().as_text(),
    )
    return LaneOutcome(
        result=result,
        artifacts=[
            LaneArtifact("model_summary", "markdown", "ITS model summary", summary),
            LaneArtifact("effects", "table", "Estimated effects", effects_table(result)),
        ],
        warnings=warnings,
    )
