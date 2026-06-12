"""Effect modification within the observational lane: treatment x moderator.

One OLS with an interaction term, HC1 errors. Emits the average effect
(at the moderator mean), the interaction coefficient, and conditional
effects at low/high moderator points (the two levels when binary, the
quartiles otherwise), each with a delta-method standard error. Linear
effect modification only; the model form is stated in the warnings.
"""
from __future__ import annotations

import math

import pandas as pd
import statsmodels.api as sm
from scipy import stats

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


def _conditional_effect(
    beta_t: float, beta_i: float, at: float, cov: pd.DataFrame,
    t_name: str, i_name: str,
) -> tuple[float, float]:
    """Effect of treatment at moderator==at, with a delta-method se."""
    estimate = beta_t + beta_i * at
    variance = (
        cov.loc[t_name, t_name]
        + at * at * cov.loc[i_name, i_name]
        + 2 * at * cov.loc[t_name, i_name]
    )
    return estimate, math.sqrt(max(variance, 0.0))


def _effect(estimand: str, est: float, se: float, interpretation: str) -> EffectEstimate:
    lo, hi = ci_from(est, se)
    p = float(2 * stats.norm.sf(abs(est / se))) if se > 0 else None
    return EffectEstimate(
        estimand=estimand, estimate=safe_float(est), std_error=se,
        ci_lower=lo, ci_upper=hi, p_value=p, interpretation=interpretation,
    )


def run_interaction(frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec) -> LaneOutcome:
    lane = "observational"
    moderator = plan.settings.get("moderator")
    if not moderator or plan.treatment is None:
        raise LaneInputError(f"{lane}: interaction path needs treatment and moderator")

    covariates = [c for c in plan.covariates if c != moderator]
    columns = [plan.outcome, plan.treatment, moderator, *covariates]
    data = numeric_frame(frame, columns, lane)
    require_variation(data[plan.treatment], f"treatment '{plan.treatment}'", lane)
    require_variation(data[moderator], f"moderator '{moderator}'", lane)

    t, m = data[plan.treatment], data[moderator]
    i_name = f"{plan.treatment}_x_{moderator}"
    rhs = pd.DataFrame({plan.treatment: t, moderator: m, i_name: t * m}, index=data.index)
    for c in covariates:
        rhs[c] = data[c]
    model = sm.OLS(data[plan.outcome], sm.add_constant(rhs)).fit(cov_type="HC1")
    beta_t = safe_float(model.params[plan.treatment])
    beta_i = safe_float(model.params[i_name])
    cov = model.cov_params()

    # evaluation points: the two levels when binary, the quartiles otherwise
    if m.nunique() == 2:
        low_at, high_at = sorted(float(v) for v in m.unique())
        point_label = "moderator level"
    else:
        low_at, high_at = float(m.quantile(0.25)), float(m.quantile(0.75))
        point_label = "moderator quartile"

    ate, ate_se = _conditional_effect(beta_t, beta_i, float(m.mean()), cov, plan.treatment, i_name)
    low, low_se = _conditional_effect(beta_t, beta_i, low_at, cov, plan.treatment, i_name)
    high, high_se = _conditional_effect(beta_t, beta_i, high_at, cov, plan.treatment, i_name)

    effects = [
        _effect(
            "ate", ate, ate_se,
            f"effect of {plan.treatment} at the mean of {moderator}, "
            f"adjusted for {len(covariates)} covariates",
        ),
        _effect(
            "interaction", beta_i, safe_float(model.bse[i_name]),
            f"change in the {plan.treatment} effect per unit of {moderator}",
        ),
        _effect(
            "cate_low", low, low_se,
            f"effect of {plan.treatment} at {point_label} {low_at:g}",
        ),
        _effect(
            "cate_high", high, high_se,
            f"effect of {plan.treatment} at {point_label} {high_at:g}",
        ),
    ]
    warnings = [
        "linear effect-modification model: the interaction is a single slope, "
        "not a flexible surface",
        f"conditional effects evaluated at {moderator} = {low_at:g} and {high_at:g}",
    ]
    result = EstimateResult(
        lane=MethodLane.OBSERVATIONAL,
        estimator="interaction_regression",
        effects=effects,
        n_rows_used=len(data),
        outcome=plan.outcome,
        treatment=plan.treatment,
        covariates_used=[moderator, *covariates],
        warnings=warnings,
    )
    summary = summary_markdown(
        "Effect modification (interaction regression)",
        [
            f"{len(data):,} rows used",
            f"average effect {ate:.4g} (se {ate_se:.3g})",
            f"interaction {beta_i:.4g} per unit of {moderator} "
            f"(p {model.pvalues[i_name]:.3g})",
            f"conditional effects: {low:.4g} at {moderator}={low_at:g}, "
            f"{high:.4g} at {moderator}={high_at:g}",
        ],
        model.summary().as_text(),
    )
    return LaneOutcome(
        result=result,
        artifacts=[
            LaneArtifact("model_summary", "markdown", "Model summary", summary),
            LaneArtifact("effects", "table", "Estimated effects", effects_table(result)),
        ],
        warnings=warnings,
    )
