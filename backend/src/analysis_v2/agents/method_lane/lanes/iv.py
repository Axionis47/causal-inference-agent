"""IV lane: two-stage least squares with a first-stage strength report.

The exclusion restriction is an assumption, never a test result; the
summary says so explicitly. The first-stage strength check is split: a
first stage that is not even significant means the instrument has no
residual relationship with the treatment once the controls are in (a
control collinear with the treatment can absorb it), so 2SLS is not
identified and the lane fails honestly rather than report a noise-driven,
arbitrary-sign LATE; a significant but F-below-10 first stage is the
ordinary weak-instrument warning.
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
    drop_collinear,
    effects_table,
    numeric_frame,
    numeric_frame_issues,
    safe_float,
    summary_markdown,
    usable_covariates,
    variation_issue,
)


def check_ready(frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec) -> list[str]:
    """Preconditions for 2SLS, shared with the readiness checker."""
    lane = "iv"
    instrument = plan.settings.get("instrument")
    if not instrument or plan.treatment is None:
        return [f"{lane}: needs an instrument and a treatment in the plan"]
    covariates, _ = usable_covariates(frame, plan.covariates)
    issues, data = numeric_frame_issues(
        frame, [plan.outcome, plan.treatment, instrument, *covariates], lane
    )
    if issues or data is None:
        return issues
    return variation_issue(
        data[instrument], f"instrument '{instrument}'", lane
    ) + variation_issue(data[plan.treatment], f"treatment '{plan.treatment}'", lane)


def run(frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec) -> LaneOutcome:
    lane = "iv"
    reasons = check_ready(frame, plan, spec)
    if reasons:
        raise LaneInputError(reasons[0])
    instrument = plan.settings.get("instrument")
    warnings: list[str] = []
    covariates, dropped_missing = usable_covariates(frame, plan.covariates)
    if dropped_missing:
        warnings.append(
            f"dropped {len(dropped_missing)} high-missingness covariate(s) "
            f"that would shrink the sample: {dropped_missing[:6]}"
        )
    columns = [plan.outcome, plan.treatment, instrument, *covariates]
    data = numeric_frame(frame, columns, lane)

    covs = data[covariates] if covariates else None
    if covs is not None and covs.shape[1] > 0:
        covs, dropped = drop_collinear(covs, protected=data[[plan.treatment, instrument]])
        if dropped:
            warnings.append(
                f"dropped {len(dropped)} collinear covariate(s) to keep 2SLS "
                f"identified: {dropped[:6]}"
            )
        if covs.shape[1] == 0:
            covs = None
    first_rhs = pd.concat([data[[instrument]], covs], axis=1) if covs is not None else data[[instrument]]
    first = sm.OLS(data[plan.treatment], sm.add_constant(first_rhs)).fit(cov_type="HC1")
    f_stat = float((first.params[instrument] / first.bse[instrument]) ** 2)
    inst_p = float(first.pvalues[instrument])
    if inst_p >= 0.05:
        raise LaneInputError(
            f"iv: instrument '{instrument}' has no significant first-stage "
            f"relationship with '{plan.treatment}' after conditioning on the "
            f"controls (first-stage F {f_stat:.2g}, p {inst_p:.2g}); the effect is "
            "not identified in this specification, so no estimate is reported. A "
            "control collinear with the treatment may be absorbing the instrument."
        )
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
