"""DID lane: group x post interaction, long format (with wide reshape).

Two-by-two when there are two periods; with more periods the model adds
period fixed effects. Standard errors cluster on the unit when a unit
column exists, else HC1.
"""
from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf

from src.analysis_v2.spec import CausalSpec, EffectEstimate, EstimateResult, MethodLane, MethodPlan

from src.analysis_v2.agents.design_detection.rules import detect_wide_periods

from .common import (
    LaneArtifact,
    LaneInputError,
    LaneOutcome,
    ci_from,
    effects_table,
    safe_float,
    summary_markdown,
)


def _reshape_wide(frame: pd.DataFrame, group_col: str, spec: CausalSpec) -> pd.DataFrame:
    """total_emp_feb / total_emp_nov -> long rows with period + post."""
    from src.analysis_v2.agents.profiling.tools import build_profile_summary

    wide_cols = detect_wide_periods(build_profile_summary(frame))
    if len(wide_cols) < 2:
        raise LaneInputError("did: wide reshape requested but no pre/post columns found")
    ordered = sorted(wide_cols)  # alphabetical fallback; 2-period case is symmetric
    pre, post = ordered[0], ordered[-1]
    rows = []
    for idx, row in frame.iterrows():
        rows.append({"unit": idx, "group": row[group_col], "period": 0, "post": 0,
                     "outcome_value": row[pre]})
        rows.append({"unit": idx, "group": row[group_col], "period": 1, "post": 1,
                     "outcome_value": row[post]})
    long = pd.DataFrame(rows).dropna(subset=["outcome_value"])
    return long


def run(frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec) -> LaneOutcome:
    lane = "did"
    group_col = plan.settings.get("group_column")
    time_col = plan.settings.get("time_column")
    post_col = plan.settings.get("post_column") or "post"
    warnings: list[str] = []

    if plan.settings.get("needs_reshape"):
        if group_col is None:
            raise LaneInputError(f"{lane}: wide reshape needs a group column")
        long = _reshape_wide(frame, group_col, spec)
        outcome, group, unit_col = "outcome_value", "group", "unit"
        warnings.append("wide pre/post columns reshaped to long format")
    else:
        if not group_col or not time_col:
            raise LaneInputError(f"{lane}: needs group and time columns")
        long = frame[[plan.outcome, group_col, time_col]
                     + ([post_col] if post_col in frame.columns else [])].copy()
        long = long.dropna()
        outcome, group = plan.outcome, group_col
        unit_col = "unit_id" if "unit_id" in frame.columns else None
        if unit_col:
            long[unit_col] = frame.loc[long.index, unit_col]
        if post_col not in long.columns:
            periods = sorted(long[time_col].unique())
            if len(periods) < 2:
                raise LaneInputError(f"{lane}: fewer than two periods")
            split = periods[len(periods) // 2]
            long[post_col] = (long[time_col] >= split).astype(int)
            warnings.append(
                f"no post indicator supplied; periods >= {split} treated as post"
            )

    groups = long[group].dropna().unique()
    if len(groups) != 2:
        raise LaneInputError(f"{lane}: needs exactly two groups, found {len(groups)}")
    labels = sorted(map(str, groups))

    def _match(wanted: str) -> str | None:
        for label in labels:
            if label == wanted:
                return label
            try:
                if float(label) == float(wanted):
                    return label
            except (TypeError, ValueError):
                continue
        return None

    # Explicit setting wins; otherwise the higher-sorting label is treated
    # (covers 0/1 and control/treated conventions), with the question's clue
    # as a tiebreaker when it clearly names one group.
    if "treated_group" in plan.settings:
        treated_label = _match(str(plan.settings["treated_group"]))
        if treated_label is None:
            raise LaneInputError(
                f"{lane}: treated_group '{plan.settings['treated_group']}' "
                f"is not one of {labels}"
            )
    else:
        treated_label = labels[-1]
        clue = (spec.treatment.clue or "").lower()
        named = [g for g in labels if g.lower() in clue and len(g) > 1]
        if len(named) == 1:
            treated_label = named[0]
    long["_treated"] = (long[group].astype(str) == treated_label).astype(int)

    n_periods = long[post_col].nunique()
    time_for_fe = time_col if not plan.settings.get("needs_reshape") else "period"
    formula = f"Q('{outcome}') ~ _treated * Q('{post_col}')"
    if time_for_fe in long.columns and long[time_for_fe].nunique() > 2:
        formula += f" + C(Q('{time_for_fe}'))"

    model = smf.ols(formula, data=long)
    if unit_col and unit_col in long.columns:
        fit = model.fit(
            cov_type="cluster", cov_kwds={"groups": long[unit_col]}
        )
        warnings.append(f"standard errors clustered on {unit_col}")
    else:
        fit = model.fit(cov_type="HC1")

    term = f"_treated:Q('{post_col}')"
    if term not in fit.params:
        raise LaneInputError(f"{lane}: interaction term missing from the fit")
    att = safe_float(fit.params[term])
    se = safe_float(fit.bse[term])
    lo, hi = ci_from(att, se)

    result = EstimateResult(
        lane=MethodLane.DID,
        estimator=plan.estimator,
        effects=[
            EffectEstimate(
                estimand="att",
                estimate=att,
                std_error=se,
                ci_lower=lo,
                ci_upper=hi,
                p_value=float(fit.pvalues[term]),
                interpretation=(
                    f"difference-in-differences for group '{treated_label}' "
                    f"({n_periods} period blocks); requires parallel trends"
                ),
            )
        ],
        n_rows_used=len(long),
        outcome=outcome if outcome != "outcome_value" else plan.outcome,
        treatment=f"group=='{treated_label}' x post",
        warnings=warnings,
    )
    summary = summary_markdown(
        "Difference-in-differences",
        [
            f"{len(long):,} unit-period rows, treated group '{treated_label}'",
            f"DID estimate {att:.4g} (se {se:.3g})",
            "identification rests on parallel pre-trends, not testable in a 2x2",
        ],
        fit.summary().as_text(),
    )
    return LaneOutcome(
        result=result,
        artifacts=[
            LaneArtifact("model_summary", "markdown", "DID model summary", summary),
            LaneArtifact("effects", "table", "Estimated effects", effects_table(result)),
        ],
        warnings=warnings,
    )
