"""IV lane: two-stage least squares with a first-stage strength report.

The exclusion restriction is an assumption, never a test result; the
summary says so explicitly. A first-stage F below 10 raises the weak-
instrument warning.
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
    require_variation,
    safe_float,
    summary_markdown,
)


def run(frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec) -> LaneOutcome:
    lane = "iv"
    instrument = plan.settings.get("instrument")
    if not instrument or plan.treatment is None:
        raise LaneInputError(f"{lane}: needs an instrument and a treatment in the plan")

    columns = [plan.outcome, plan.treatment, instrument, *plan.covariates]
    data = numeric_frame(frame, columns, lane)
    require_variation(data[instrument], f"instrument '{instrument}'", lane)
    require_variation(data[plan.treatment], f"treatment '{plan.treatment}'", lane)

    covs = data[plan.covariates] if plan.covariates else None
    first_rhs = pd.concat([data[[instrument]], covs], axis=1) if covs is not None else data[[instrument]]
    first = sm.OLS(data[plan.treatment], sm.add_constant(first_rhs)).fit(cov_type="HC1")
    f_stat = float((first.params[instrument] / first.bse[instrument]) ** 2)
    warnings: list[str] = []
    if f_stat < 10:
        warnings.append(f"weak instrument: first-stage F {f_stat:.1f} (below 10)")

    from statsmodels.sandbox.regression.gmm import IV2SLS

    exog_cols = pd.concat([data[[plan.treatment]], covs], axis=1) if covs is not None else data[[plan.treatment]]
    inst_cols = pd.concat([data[[instrument]], covs], axis=1) if covs is not None else data[[instrument]]
    fit = IV2SLS(
        data[plan.outcome], sm.add_constant(exog_cols), sm.add_constant(inst_cols)
    ).fit()
    late = safe_float(fit.params[plan.treatment])
    se = safe_float(fit.bse[plan.treatment])
    lo, hi = ci_from(late, se)

    result = EstimateResult(
        lane=MethodLane.IV,
        estimator=plan.estimator,
        effects=[
            EffectEstimate(
                estimand="late",
                estimate=late,
                std_error=se,
                ci_lower=lo,
                ci_upper=hi,
                p_value=float(fit.pvalues[plan.treatment]),
                interpretation=(
                    f"2sls effect of {plan.treatment} instrumented by {instrument}; "
                    "applies to instrument compliers"
                ),
            )
        ],
        n_rows_used=len(data),
        outcome=plan.outcome,
        treatment=plan.treatment,
        covariates_used=list(plan.covariates),
        warnings=warnings,
    )
    summary = summary_markdown(
        "Two-stage least squares",
        [
            f"{len(data):,} rows; first-stage F {f_stat:.1f}",
            f"LATE {late:.4g} (se {se:.3g})",
            "the exclusion restriction (instrument affects the outcome only "
            "through the treatment) is assumed and cannot be tested from data",
        ],
        fit.summary().as_text() if hasattr(fit, "summary") else None,
    )
    return LaneOutcome(
        result=result,
        artifacts=[
            LaneArtifact("model_summary", "markdown", "IV model summary", summary),
            LaneArtifact("effects", "table", "Estimated effects", effects_table(result)),
        ],
        warnings=warnings,
    )
