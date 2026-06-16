"""Mediation lane: product of coefficients with Sobel standard errors.

a: mediator ~ treatment (+ covariates); b and direct: outcome ~ mediator
+ treatment (+ covariates). indirect = a*b, total = direct + indirect.
The timing assumption (treatment before mediator before outcome) is
stated, not verified.
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
    encode_design,
    encode_design_issues,
    safe_float,
    summary_markdown,
    variation_issue,
)


def check_ready(frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec) -> list[str]:
    """Preconditions for the mediation decomposition, shared with readiness."""
    lane = "mediation"
    mediator = plan.settings.get("mediator")
    if not mediator or plan.treatment is None:
        return [f"{lane}: needs a mediator and a treatment in the plan"]
    issues, data, _ = encode_design_issues(
        frame, [plan.outcome, plan.treatment, mediator], plan.covariates, lane
    )
    if issues or data is None:
        return issues
    return variation_issue(data[plan.treatment], "treatment", lane)


def run(frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec) -> LaneOutcome:
    lane = "mediation"
    reasons = check_ready(frame, plan, spec)
    if reasons:
        raise LaneInputError(reasons[0])
    mediator = plan.settings.get("mediator")
    data, cov_names = encode_design(
        frame, [plan.outcome, plan.treatment, mediator], plan.covariates, lane
    )
    covs = data[cov_names] if cov_names else None

    m_rhs = pd.concat([data[[plan.treatment]], covs], axis=1) if covs is not None else data[[plan.treatment]]
    m_fit = sm.OLS(data[mediator], sm.add_constant(m_rhs)).fit(cov_type="HC1")
    a, sa = safe_float(m_fit.params[plan.treatment]), safe_float(m_fit.bse[plan.treatment])

    y_rhs = pd.concat([data[[mediator, plan.treatment]], covs], axis=1) if covs is not None else data[[mediator, plan.treatment]]
    y_fit = sm.OLS(data[plan.outcome], sm.add_constant(y_rhs)).fit(cov_type="HC1")
    b, sb = safe_float(y_fit.params[mediator]), safe_float(y_fit.bse[mediator])
    direct = safe_float(y_fit.params[plan.treatment])
    direct_se = safe_float(y_fit.bse[plan.treatment])

    indirect = a * b
    sobel_se = float(np.sqrt(a**2 * sb**2 + b**2 * sa**2))
    total = direct + indirect
    total_se = float(np.sqrt(direct_se**2 + sobel_se**2))  # ignores covariance; stated
    proportion = indirect / total if abs(total) > 1e-12 else float("nan")

    warnings = [
        "mediator timing (treatment -> mediator -> outcome) is assumed, not observed",
        "total-effect se ignores the direct/indirect covariance (conservative shortcut)",
    ]
    result = EstimateResult(
        lane=MethodLane.MEDIATION,
        estimator=plan.estimator,
        effects=[
            EffectEstimate(
                estimand="total",
                estimate=total,
                std_error=total_se,
                ci_lower=ci_from(total, total_se)[0],
                ci_upper=ci_from(total, total_se)[1],
                interpretation="direct + indirect effect of the treatment",
            ),
            EffectEstimate(
                estimand="indirect",
                estimate=indirect,
                std_error=sobel_se,
                ci_lower=ci_from(indirect, sobel_se)[0],
                ci_upper=ci_from(indirect, sobel_se)[1],
                interpretation=(
                    f"path through {mediator} (a={a:.4g}, b={b:.4g}, Sobel se); "
                    f"proportion mediated {proportion:.2f}"
                ),
            ),
            EffectEstimate(
                estimand="direct",
                estimate=direct,
                std_error=direct_se,
                ci_lower=ci_from(direct, direct_se)[0],
                ci_upper=ci_from(direct, direct_se)[1],
                p_value=float(y_fit.pvalues[plan.treatment]),
                interpretation="treatment effect holding the mediator fixed",
            ),
        ],
        n_rows_used=len(data),
        outcome=plan.outcome,
        treatment=plan.treatment,
        covariates_used=list(plan.covariates),
        warnings=warnings,
    )
    summary = summary_markdown(
        "Mediation decomposition",
        [
            f"{len(data):,} rows",
            f"total {total:.4g}; indirect {indirect:.4g} (Sobel se {sobel_se:.3g}); "
            f"direct {direct:.4g}",
            f"proportion mediated {proportion:.2f}",
        ],
        y_fit.summary().as_text(),
    )
    return LaneOutcome(
        result=result,
        artifacts=[
            LaneArtifact("model_summary", "markdown", "Mediation summary", summary),
            LaneArtifact("effects", "table", "Estimated effects", effects_table(result)),
        ],
        warnings=warnings,
    )
