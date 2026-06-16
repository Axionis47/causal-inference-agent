"""RDD lane: regression discontinuity via rdrobust (Calonico-Cattaneo-Titiunik).

The data-driven bandwidth, the triangular-kernel local-linear fit, and the
robust bias-corrected inference all come from rdrobust; nothing here is a
hand-tuned bandwidth. A sharp design reports the jump in the outcome at the
cutoff. A fuzzy design, where crossing the cutoff only shifts take-up, reports
the LATE (the cutoff instruments take-up) alongside the reduced-form jump.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
import rdrobust

from src.analysis_v2.spec import CausalSpec, EffectEstimate, EstimateResult, MethodLane, MethodPlan

from .common import (
    LaneArtifact,
    LaneInputError,
    LaneOutcome,
    drop_collinear,
    effects_table,
    encode_design_issues,
    numeric_frame_issues,
    safe_float,
    summary_markdown,
    usable_covariates,
)

MIN_SIDE_ROWS = 10  # pre-flight: enough mass each side of the cutoff to fit a line
SHARP_TOL = 0.02  # take-up within this of {0 below, 1 above} is a sharp design
MIN_FIRST_STAGE_JUMP = 0.1  # smaller take-up jump means a weak fuzzy first stage


def _settings(plan: MethodPlan) -> tuple[Any, Any]:
    return plan.settings.get("running_variable"), plan.settings.get("cutoff")


def _takeup(frame: pd.DataFrame, index: Any, treatment: str | None) -> pd.Series | None:
    """Binary take-up (higher level = 1) on the given rows, or None when the
    treatment is not a clean two-level column. A sharp design needs no numeric
    treatment column: assignment is the cutoff itself, so the raw label (e.g.
    'Level 1 Recovery') need never be numeric."""
    if treatment is None or treatment not in frame.columns:
        return None
    raw = frame.loc[index, treatment]
    levels = sorted(raw.dropna().unique(), key=str)
    if len(levels) != 2:
        return None
    return (raw.astype(str) == str(levels[1])).astype(float)


def _robust(out: Any) -> tuple[float, float, float, float, float]:
    """The robust bias-corrected row of an rdrobust result: estimate (the
    bias-corrected point estimate), se, ci low/high, p. Reporting this row
    keeps the interval bracketing the estimate and the inference valid."""
    est = safe_float(out.coef.loc["Robust"].iloc[0])
    se = safe_float(out.se.loc["Robust"].iloc[0])
    lo = safe_float(out.ci.loc["Robust"].iloc[0])
    hi = safe_float(out.ci.loc["Robust"].iloc[1])
    p = float(out.pv.loc["Robust"].iloc[0])
    return est, se, lo, hi, p


def _fit(y: Any, x: Any, cutoff: float, covs: Any, h: float | None, fuzzy: Any = None) -> Any:
    try:
        return rdrobust.rdrobust(y=y, x=x, c=cutoff, covs=covs, fuzzy=fuzzy, h=h)
    except Exception as exc:  # rdrobust raises on degenerate windows / collinearity
        raise LaneInputError(f"rdd: rdrobust failed ({type(exc).__name__}: {str(exc)[:160]})")


def check_ready(frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec) -> list[str]:
    """Pre-flight for rdrobust: the running variable and cutoff are present and
    numeric, the cutoff sits strictly inside the observed range, and each side
    has enough rows to fit a local line. rdrobust selects its own window, so
    this no longer counts rows inside a hand-chosen bandwidth."""
    lane = "rdd"
    run_col, cutoff = _settings(plan)
    if not run_col or cutoff is None:
        return [f"{lane}: needs running_variable and cutoff in the plan"]
    issues, data = numeric_frame_issues(frame, [plan.outcome, run_col], lane)
    if issues or data is None:
        return issues
    cutoff = float(cutoff)
    col = data[run_col]
    lo, hi = float(col.min()), float(col.max())
    if not lo < cutoff < hi:
        return [f"{lane}: cutoff {cutoff:g} is outside the running variable range [{lo:g}, {hi:g}]"]
    below = int((col < cutoff).sum())
    above = int((col >= cutoff).sum())
    if below < MIN_SIDE_ROWS:
        return [f"{lane}: only {below} rows below the cutoff (need {MIN_SIDE_ROWS})"]
    if above < MIN_SIDE_ROWS:
        return [f"{lane}: only {above} rows above the cutoff (need {MIN_SIDE_ROWS})"]
    return []


def selected_bandwidth(frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec) -> float:
    """The bandwidth the lane will use: an explicit plan setting wins, otherwise
    the one rdrobust selects from the data. Diagnostics perturbs around this
    value, so the sensitivity sweep centres on the real window, not a guess."""
    explicit = plan.settings.get("bandwidth")
    if explicit:
        return float(explicit)
    run_col, cutoff = _settings(plan)
    issues, data = numeric_frame_issues(frame, [plan.outcome, run_col], "rdd")
    if issues or data is None:
        raise LaneInputError((issues or ["rdd: cannot compute bandwidth"])[0])
    y = data[plan.outcome].to_numpy(dtype=float)
    x = data[run_col].to_numpy(dtype=float)
    out = _fit(y, x, float(cutoff), None, None)
    return float(out.bws.loc["h"].iloc[0])


def _prepare(
    frame: pd.DataFrame, plan: MethodPlan, run_col: Any
) -> tuple[Any, Any, Any, Any]:
    """Complete-case design: outcome, running variable, optional take-up, and
    covariates (one-hot encoded, then dropped where collinear with the running
    variable). Returns (y, x, covs_array_or_None, takeup_array_or_None)."""
    usable, _ = usable_covariates(frame, plan.covariates)
    core = [plan.outcome, run_col]
    work = frame
    d_full = _takeup(frame, frame.index, plan.treatment)
    if d_full is not None:
        work = frame.assign(_takeup=d_full)
        core = [plan.outcome, run_col, "_takeup"]
    issues, design, cov_names = encode_design_issues(work, core, usable, "rdd")
    if issues or design is None:
        raise LaneInputError((issues or ["rdd: empty design"])[0])
    covs = None
    if cov_names:
        kept, _ = drop_collinear(design[cov_names], protected=design[[run_col]])
        covs = kept.to_numpy(dtype=float) if kept.shape[1] else None
    y = design[plan.outcome].to_numpy(dtype=float)
    x = design[run_col].to_numpy(dtype=float)
    d = design["_takeup"].to_numpy(dtype=float) if d_full is not None else None
    return y, x, covs, d


def run(frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec) -> LaneOutcome:
    lane = "rdd"
    reasons = check_ready(frame, plan, spec)
    if reasons:
        raise LaneInputError(reasons[0])
    run_col, _cut = _settings(plan)
    cutoff = float(_cut)
    h = plan.settings.get("bandwidth")
    h = float(h) if h else None

    y, x, covs, d = _prepare(frame, plan, run_col)

    fuzzy = None
    estimator = "rdrobust_sharp"
    if d is not None:
        above = x >= cutoff
        take_below = float(d[~above].mean()) if (~above).any() else 0.0
        take_above = float(d[above].mean()) if above.any() else 1.0
        if not (take_below <= SHARP_TOL and take_above >= 1 - SHARP_TOL):
            fuzzy = d
            estimator = "rdrobust_fuzzy"

    warnings: list[str] = []
    main = _fit(y, x, cutoff, covs, h, fuzzy=fuzzy)
    est, se, lo, hi, p = _robust(main)
    bw = float(main.bws.loc["h"].iloc[0])
    n_below, n_above = int(main.N_h[0]), int(main.N_h[1])
    n_used = n_below + n_above

    if fuzzy is None:
        effects = [
            EffectEstimate(
                estimand="itt_jump", estimate=est, std_error=se,
                ci_lower=lo, ci_upper=hi, p_value=p,
                interpretation=(
                    f"outcome jump at the cutoff {cutoff:g} (rdrobust, "
                    f"bandwidth {bw:.4g}, {n_used} effective rows)"
                ),
            )
        ]
    else:
        effects = [
            EffectEstimate(
                estimand="late", estimate=est, std_error=se,
                ci_lower=lo, ci_upper=hi, p_value=p,
                interpretation=(
                    "fuzzy RDD: cutoff-crossing instruments take-up; "
                    f"applies to cutoff compliers (bandwidth {bw:.4g})"
                ),
            )
        ]
        effects.append(_fuzzy_extras(y, x, d, cutoff, covs, h, warnings))

    result = EstimateResult(
        lane=MethodLane.RDD, estimator=estimator, effects=effects,
        n_rows_used=n_used, outcome=plan.outcome, treatment=plan.treatment,
        warnings=warnings,
    )
    summary = summary_markdown(
        "Regression discontinuity (rdrobust)",
        [
            f"cutoff {cutoff:g}, bandwidth {bw:.4g} (CCT mserd, triangular kernel)",
            f"effective sample {n_used:,} ({n_below:,} below, {n_above:,} above)",
            f"{estimator}: {effects[0].estimand} {effects[0].estimate:.4g} "
            f"(se {effects[0].std_error:.3g})",
            "identification needs no manipulation of the running variable at the cutoff",
        ],
    )
    return LaneOutcome(
        result=result,
        artifacts=[
            LaneArtifact("model_summary", "markdown", "RDD model summary", summary),
            LaneArtifact("effects", "table", "Estimated effects", effects_table(result)),
        ],
        warnings=warnings,
    )


def _fuzzy_extras(
    y: Any, x: Any, d: Any, cutoff: float, covs: Any, h: float | None, warnings: list[str]
) -> EffectEstimate:
    """Reduced-form jump in the outcome (the ITT effect), plus a weak-first-stage
    warning from the jump in take-up at the cutoff."""
    fs = _fit(d, x, cutoff, covs, h)
    fs_jump = safe_float(fs.coef.loc["Robust"].iloc[0])
    fs_p = float(fs.pv.loc["Robust"].iloc[0])
    if abs(fs_jump) < MIN_FIRST_STAGE_JUMP or fs_p > 0.05:
        warnings.append(f"weak first stage at the cutoff (take-up jump {fs_jump:.2f}, p {fs_p:.3g})")
    itt = _fit(y, x, cutoff, covs, h)
    est, se, lo, hi, p = _robust(itt)
    return EffectEstimate(
        estimand="itt_jump", estimate=est, std_error=se, ci_lower=lo, ci_upper=hi,
        p_value=p, interpretation=f"reduced-form outcome jump at the cutoff {cutoff:g}",
    )
