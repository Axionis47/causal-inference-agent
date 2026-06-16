"""RDD lane: local linear regression at the cutoff, sharp or fuzzy.

Within the bandwidth (default: a sample-size-aware rule of thumb), the
sharp jump comes from OLS with separate slopes and HC1 errors. When
take-up is fuzzy, crossing the cutoff instruments the treatment (2SLS),
mirroring the Wald estimate.
"""
from __future__ import annotations

import pandas as pd
import statsmodels.api as sm

from src.analysis_v2.spec import CausalSpec, EffectEstimate, EstimateResult, MethodLane, MethodPlan

from .common import (
    LaneArtifact,
    LaneInputError,
    LaneOutcome,
    ci_from,
    effects_table,
    numeric_frame,
    numeric_frame_issues,
    safe_float,
    summary_markdown,
)

MIN_LOCAL_ROWS = 50


def _takeup(frame: pd.DataFrame, index, treatment: str | None) -> pd.Series | None:
    """Binary take-up indicator (higher level = 1) within the local window, or
    None when the treatment is not a clean two-level column there. A sharp
    design needs no numeric treatment column: assignment is the cutoff itself,
    so the raw label (e.g. 'Level 1 Recovery') need never be numeric."""
    if treatment is None or treatment not in frame.columns:
        return None
    raw = frame.loc[index, treatment]
    levels = sorted(raw.dropna().unique(), key=str)
    if len(levels) != 2:
        return None
    return (raw.astype(str) == str(levels[1])).astype(float)


def _bandwidth(data: pd.DataFrame, run_col: str, plan: MethodPlan) -> float:
    """Local-linear bandwidth. An explicit plan setting wins; otherwise a
    rule-of-thumb that shrinks with sample size at the n**(-1/5) nonparametric
    rate on a tail-robust scale (the smaller of the standard deviation and the
    IQR-based estimate). The old default was the raw standard deviation, which
    ignores n and, on a skewed running variable, widens the 'local' window until
    the fit is a global-trend extrapolation rather than a discontinuity at the
    cutoff. Centering is irrelevant: both scales are shift-invariant."""
    explicit = plan.settings.get("bandwidth")
    if explicit:
        return float(explicit)
    col = data[run_col]
    std = float(col.std())
    iqr_scale = float(col.quantile(0.75) - col.quantile(0.25)) / 1.349
    scale = min(std, iqr_scale) if iqr_scale > 0 else std
    h = 1.06 * scale * len(col) ** (-0.2)
    return h if h > 0 else std


def check_ready(frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec) -> list[str]:
    """Preconditions for the local-linear fit, shared with the readiness checker.
    The bandwidth window is the precondition that needs the data, so it is
    computed here exactly as run() does."""
    lane = "rdd"
    run_col = plan.settings.get("running_variable")
    cutoff = plan.settings.get("cutoff")
    if not run_col or cutoff is None:
        return [f"{lane}: needs running_variable and cutoff in the plan"]
    columns = [plan.outcome, run_col]
    issues, data = numeric_frame_issues(frame, columns, lane)
    if issues or data is None:
        return issues
    x = data[run_col] - float(cutoff)
    bandwidth = _bandwidth(data, run_col, plan)
    local = data[x.abs() <= bandwidth]
    if len(local) < MIN_LOCAL_ROWS:
        return [f"{lane}: only {len(local)} rows within bandwidth {bandwidth:.4g}"]
    return []


def run(frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec) -> LaneOutcome:
    lane = "rdd"
    reasons = check_ready(frame, plan, spec)
    if reasons:
        raise LaneInputError(reasons[0])
    run_col = plan.settings.get("running_variable")
    cutoff = float(plan.settings.get("cutoff"))

    columns = [plan.outcome, run_col]
    data = numeric_frame(frame, columns, lane)
    x = data[run_col] - cutoff
    bandwidth = _bandwidth(data, run_col, plan)
    near = x.abs() <= bandwidth
    local = data[near].copy()
    xc = x[near]
    above = (xc >= 0).astype(int)
    design = pd.DataFrame(
        {"above": above, "xc": xc, "above_xc": above * xc}, index=local.index
    )
    y = local[plan.outcome]
    sharp_fit = sm.OLS(y, sm.add_constant(design)).fit(cov_type="HC1")
    jump = safe_float(sharp_fit.params["above"])
    se = safe_float(sharp_fit.bse["above"])
    lo, hi = ci_from(jump, se)

    warnings: list[str] = []
    effects = [
        EffectEstimate(
            estimand="itt_jump",
            estimate=jump,
            std_error=se,
            ci_lower=lo,
            ci_upper=hi,
            p_value=float(sharp_fit.pvalues["above"]),
            interpretation=(
                f"outcome jump at the cutoff {cutoff:g} "
                f"(local linear, bandwidth {bandwidth:.4g}, {len(local)} rows)"
            ),
        )
    ]
    estimator = "local_linear_sharp"

    d = _takeup(frame, local.index, plan.treatment)
    if d is not None:
        take_below = float(d[above == 0].mean())
        take_above = float(d[above == 1].mean())
        first_stage = take_above - take_below
        fuzzy = not (take_below <= 0.02 and take_above >= 0.98)
        if fuzzy:
            estimator = "local_linear_fuzzy_2sls"
            if first_stage < 0.1:
                warnings.append(
                    f"weak first stage at the cutoff (jump {first_stage:.2f})"
                )
            from statsmodels.sandbox.regression.gmm import IV2SLS

            exog = sm.add_constant(
                pd.DataFrame({"d": d, "xc": xc, "above_xc": above * xc})
            )
            instrument = sm.add_constant(
                pd.DataFrame({"above": above, "xc": xc, "above_xc": above * xc})
            )
            iv_fit = IV2SLS(y, exog, instrument).fit()
            late = safe_float(iv_fit.params["d"])
            late_se = safe_float(iv_fit.bse["d"])
            effects.insert(
                0,
                EffectEstimate(
                    estimand="late",
                    estimate=late,
                    std_error=late_se,
                    ci_lower=ci_from(late, late_se)[0],
                    ci_upper=ci_from(late, late_se)[1],
                    p_value=float(iv_fit.pvalues["d"]),
                    interpretation=(
                        "fuzzy RDD: cutoff-crossing instruments take-up "
                        f"(first stage {first_stage:.2f}); applies to cutoff compliers"
                    ),
                ),
            )

    result = EstimateResult(
        lane=MethodLane.RDD,
        estimator=estimator,
        effects=effects,
        n_rows_used=len(local),
        outcome=plan.outcome,
        treatment=plan.treatment,
        warnings=warnings,
    )
    summary = summary_markdown(
        "Regression discontinuity",
        [
            f"cutoff {cutoff:g}, bandwidth {bandwidth:.4g}, {len(local):,} local rows",
            f"primary estimate {effects[0].estimate:.4g} (se {effects[0].std_error:.3g})",
            "identification needs no manipulation of the running variable at the cutoff",
        ],
        sharp_fit.summary().as_text(),
    )
    return LaneOutcome(
        result=result,
        artifacts=[
            LaneArtifact("model_summary", "markdown", "RDD model summary", summary),
            LaneArtifact("effects", "table", "Estimated effects", effects_table(result)),
        ],
        warnings=warnings,
    )
