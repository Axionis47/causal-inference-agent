"""Survival lane: Cox proportional hazards with a Kaplan-Meier overview.

Hazard ratio for the treatment with censoring handled through the event
indicator. The proportional-hazards assumption is flagged for the
diagnostics stage rather than assumed silently.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.analysis_v2.spec import (  # noqa: E402
    CausalSpec,
    EffectEstimate,
    EstimateResult,
    MethodLane,
    MethodPlan,
)

from .common import (  # noqa: E402
    LaneArtifact,
    LaneInputError,
    LaneOutcome,
    effects_table,
    numeric_frame,
    numeric_frame_issues,
    safe_float,
    summary_markdown,
    variation_issue,
)


def _km_png(data: pd.DataFrame, duration: str, event: str, treat: str | None) -> bytes:
    from statsmodels.duration.survfunc import SurvfuncRight

    fig, ax = plt.subplots(figsize=(6, 3.5))
    if treat is not None and data[treat].nunique() == 2:
        for value, chunk in data.groupby(treat):
            sf = SurvfuncRight(chunk[duration], chunk[event])
            ax.step(sf.surv_times, sf.surv_prob, where="post", label=f"{treat}={value}")
        ax.legend()
    else:
        sf = SurvfuncRight(data[duration], data[event])
        ax.step(sf.surv_times, sf.surv_prob, where="post")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(duration)
    ax.set_ylabel("survival probability")
    ax.set_title("Kaplan-Meier survival")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def check_ready(frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec) -> list[str]:
    """Preconditions for the Cox fit, shared with the readiness checker."""
    lane = "survival"
    duration = plan.settings.get("duration_column")
    event = plan.settings.get("event_column")
    if not duration or not event:
        return [f"{lane}: needs duration and event columns in the plan"]
    if plan.treatment is None:
        return [f"{lane}: needs a treatment column for the hazard ratio"]
    issues, data = numeric_frame_issues(
        frame, [duration, event, plan.treatment, *plan.covariates], lane
    )
    if issues or data is None:
        return issues
    varies = variation_issue(data[plan.treatment], "treatment", lane)
    if varies:
        return varies
    if not set(data[event].unique()).issubset({0, 1}):
        return [f"{lane}: event column '{event}' must be 0/1"]
    if data[event].sum() < 10:
        return [f"{lane}: fewer than 10 events observed"]
    return []


def run(frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec) -> LaneOutcome:
    lane = "survival"
    reasons = check_ready(frame, plan, spec)
    if reasons:
        raise LaneInputError(reasons[0])
    duration = plan.settings.get("duration_column")
    event = plan.settings.get("event_column")
    columns = [duration, event, plan.treatment, *plan.covariates]
    data = numeric_frame(frame, columns, lane)

    from statsmodels.duration.hazard_regression import PHReg

    exog_cols = [plan.treatment, *plan.covariates]
    model = PHReg(data[duration], data[exog_cols], status=data[event])
    fit = model.fit()
    names = list(exog_cols)
    coef = safe_float(fit.params[names.index(plan.treatment)])
    se = safe_float(fit.bse[names.index(plan.treatment)])
    from scipy.stats import norm

    hr = float(np.exp(coef))
    hr_lo, hr_hi = float(np.exp(coef - 1.96 * se)), float(np.exp(coef + 1.96 * se))
    p_value = float(2 * (1 - norm.cdf(abs(coef / se))))

    warnings = [
        "proportional hazards assumed; the diagnostics stage checks it",
    ]
    event_rate = float(data[event].mean())
    result = EstimateResult(
        lane=MethodLane.SURVIVAL,
        estimator=plan.estimator,
        effects=[
            EffectEstimate(
                estimand="hazard_ratio",
                estimate=hr,
                std_error=None,
                ci_lower=hr_lo,
                ci_upper=hr_hi,
                p_value=p_value,
                interpretation=(
                    f"multiplicative change in the event hazard per unit of "
                    f"{plan.treatment} (log-hr {coef:.4g}, se {se:.3g}); "
                    f"event rate {event_rate:.1%}"
                ),
            )
        ],
        n_rows_used=len(data),
        outcome=duration,
        treatment=plan.treatment,
        covariates_used=list(plan.covariates),
        warnings=warnings,
    )
    summary = summary_markdown(
        "Cox proportional hazards",
        [
            f"{len(data):,} rows, {int(data[event].sum())} events",
            f"hazard ratio {hr:.4g} (95% ci {hr_lo:.4g}..{hr_hi:.4g})",
            "censored observations contribute follow-up time, not events",
        ],
        str(fit.summary()),
    )
    return LaneOutcome(
        result=result,
        artifacts=[
            LaneArtifact("model_summary", "markdown", "Cox model summary", summary),
            LaneArtifact("effects", "table", "Estimated effects", effects_table(result)),
            LaneArtifact(
                "kaplan_meier", "plot", "Kaplan-Meier survival",
                _km_png(data, duration, event, plan.treatment),
            ),
        ],
        warnings=warnings,
    )
