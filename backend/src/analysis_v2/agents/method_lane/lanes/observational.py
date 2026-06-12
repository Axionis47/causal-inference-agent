"""Observational lane: covariate-adjusted regression, optional IPW check.

Primary effect: the treatment coefficient from OLS with HC1 errors.
When the plan asks (binary treatment), an IPW estimate with a re-fit
bootstrap rides along as a second effect; propensities are re-estimated
inside every bootstrap draw, never reused.
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
    binary_01,
    ci_from,
    effects_table,
    numeric_frame,
    require_variation,
    safe_float,
    summary_markdown,
)

_BOOTSTRAP_DRAWS = 200


def _ipw_ate(y: pd.Series, t: pd.Series, x: pd.DataFrame) -> float:
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(max_iter=300)
    p = model.fit(x, t).predict_proba(x)[:, 1].clip(0.02, 0.98)
    w1 = t / p
    w0 = (1 - t) / (1 - p)
    return float((w1 * y).sum() / w1.sum() - (w0 * y).sum() / w0.sum())


def run(frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec) -> LaneOutcome:
    lane = "observational"
    if plan.treatment is None and not plan.settings.get("factors"):
        raise LaneInputError(f"{lane}: no treatment and no factors to regress on")

    # effect-modification questions carry a moderator; the interaction
    # model owns that path end to end
    if plan.treatment is not None and plan.settings.get("moderator"):
        from .effect_modification import run_interaction

        return run_interaction(frame, plan, spec)

    warnings: list[str] = []
    if plan.treatment is not None:
        columns = [plan.outcome, plan.treatment, *plan.covariates]
        data = numeric_frame(frame, columns, lane)
        require_variation(data[plan.treatment], f"treatment '{plan.treatment}'", lane)
        y = data[plan.outcome]
        rhs = data[[plan.treatment, *plan.covariates]]
        model = sm.OLS(y, sm.add_constant(rhs)).fit(cov_type="HC1")
        coef = safe_float(model.params[plan.treatment])
        se = safe_float(model.bse[plan.treatment])
        lo, hi = ci_from(coef, se)
        effects = [
            EffectEstimate(
                estimand=plan.estimand,
                estimate=coef,
                std_error=se,
                ci_lower=lo,
                ci_upper=hi,
                p_value=float(model.pvalues[plan.treatment]),
                interpretation=(
                    f"adjusted association of {plan.treatment} with {plan.outcome}, "
                    f"holding {len(plan.covariates)} covariates fixed"
                ),
            )
        ]
        if not plan.covariates:
            warnings.append("no covariates adjusted; this is a raw comparison")

        if plan.settings.get("include_ipw") and plan.covariates:
            t01 = binary_01(data[plan.treatment], "treatment", lane)
            x = data[plan.covariates]
            point = _ipw_ate(y, t01, x)
            rng = np.random.default_rng(7)
            draws = []
            index = np.arange(len(data))
            for _ in range(_BOOTSTRAP_DRAWS):
                pick = rng.choice(index, size=len(index), replace=True)
                try:
                    draws.append(
                        _ipw_ate(y.iloc[pick], t01.iloc[pick], x.iloc[pick])
                    )
                except Exception:
                    continue
            boot_se = float(np.std(draws)) if len(draws) > 50 else None
            effects.append(
                EffectEstimate(
                    estimand="ate_ipw",
                    estimate=safe_float(point),
                    std_error=boot_se,
                    ci_lower=ci_from(point, boot_se)[0] if boot_se else None,
                    ci_upper=ci_from(point, boot_se)[1] if boot_se else None,
                    interpretation="inverse-propensity-weighted cross-check "
                    f"({len(draws)} bootstrap refits)",
                )
            )
            gap = abs(point - coef)
            scale = abs(coef) if coef else 1.0
            if gap > 0.5 * max(scale, 1e-9):
                warnings.append(
                    f"ipw and regression estimates disagree ({point:.4g} vs {coef:.4g})"
                )
        result = EstimateResult(
            lane=MethodLane.OBSERVATIONAL,
            estimator=plan.estimator,
            effects=effects,
            n_rows_used=len(data),
            outcome=plan.outcome,
            treatment=plan.treatment,
            covariates_used=list(plan.covariates),
            warnings=warnings,
        )
        summary = summary_markdown(
            "Regression adjustment",
            [
                f"{len(data):,} rows used",
                f"primary estimate {coef:.4g} (se {se:.3g})",
                "estimates are adjusted associations under the no-unmeasured-"
                "confounding assumption",
            ],
            model.summary().as_text(),
        )
    else:
        factors = [c for c in plan.settings.get("factors", []) if c in frame.columns]
        data = numeric_frame(frame, [plan.outcome, *factors], lane)
        y = data[plan.outcome]
        model = sm.OLS(y, sm.add_constant(data[factors])).fit(cov_type="HC1")
        effects = [
            EffectEstimate(
                estimand=f"assoc_{name}",
                estimate=safe_float(model.params[name]),
                std_error=safe_float(model.bse[name]),
                ci_lower=ci_from(model.params[name], model.bse[name])[0],
                ci_upper=ci_from(model.params[name], model.bse[name])[1],
                p_value=float(model.pvalues[name]),
                interpretation=f"joint-model association of {name} with {plan.outcome}",
            )
            for name in factors
        ]
        warnings.append("multi-factor model: associations, not causal effects per factor")
        result = EstimateResult(
            lane=MethodLane.OBSERVATIONAL,
            estimator="joint_regression",
            effects=effects,
            n_rows_used=len(data),
            outcome=plan.outcome,
            covariates_used=factors,
            warnings=warnings,
        )
        summary = summary_markdown(
            "Joint factor regression",
            [f"{len(data):,} rows; {len(factors)} factors"],
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
