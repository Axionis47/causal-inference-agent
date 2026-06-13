"""Per-lane diagnostics (assumption checks) and sensitivity (perturbations).

Everything returns DiagnosticCheck records; nothing here changes the
workflow. A perturbation that moves the estimate is reported, not fixed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis_v2.spec import (
    CausalSpec,
    CheckStatus,
    DiagnosticCheck,
    EstimateResult,
    MethodPlan,
)


def _check(name: str, status: CheckStatus, detail: str, **metrics) -> DiagnosticCheck:
    return DiagnosticCheck(
        name=name, status=status, detail=detail,
        metrics={k: round(float(v), 6) for k, v in metrics.items() if v is not None},
    )


def _shift_status(base: float, alt: float) -> tuple[CheckStatus, float]:
    scale = max(abs(base), 1e-9)
    shift = abs(alt - base) / scale
    if np.sign(alt) != np.sign(base) and abs(alt) > 0.1 * scale:
        return CheckStatus.FAIL, shift
    return (CheckStatus.WARNING if shift > 0.5 else CheckStatus.PASS), shift


def detect_leakage(frame: pd.DataFrame, plan: MethodPlan) -> DiagnosticCheck:
    """A covariate that is (nearly) the outcome or treatment is a bad control."""
    suspects = []
    cols = [c for c in plan.covariates if c in frame.columns]
    for target, label in ((plan.outcome, "outcome"), (plan.treatment, "treatment")):
        if target is None or target not in frame.columns:
            continue
        t = pd.to_numeric(frame[target], errors="coerce")
        for cov in cols:
            x = pd.to_numeric(frame[cov], errors="coerce")
            ok = t.notna() & x.notna()
            if ok.sum() < 20 or x[ok].std() == 0 or t[ok].std() == 0:
                continue
            corr = abs(float(np.corrcoef(x[ok], t[ok])[0, 1]))
            if corr > 0.98:
                suspects.append(f"{cov} ~ {label} (|corr| {corr:.3f})")
    if suspects:
        return _check(
            "leakage_bad_control", CheckStatus.FAIL,
            "covariates nearly identical to the " + "; ".join(suspects),
        )
    return _check("leakage_bad_control", CheckStatus.PASS, "no near-duplicate covariates")


def evalue(result: EstimateResult, frame: pd.DataFrame) -> DiagnosticCheck:
    """VanderWeele E-value via the standardized-effect approximation."""
    primary = result.primary
    outcome = result.outcome
    if (
        primary.ci_lower is not None
        and primary.ci_upper is not None
        and primary.ci_lower <= 0 <= primary.ci_upper
    ):
        # a null finding has no nonzero estimate to explain away; failing it
        # here would misreport every true null as a broken design
        return _check(
            "e_value", CheckStatus.NOT_APPLICABLE,
            "the interval spans zero, so there is no effect to explain away; "
            "note confounding could mask a true effect rather than create one",
        )
    if outcome not in frame.columns:
        return _check("e_value", CheckStatus.NOT_APPLICABLE, "outcome column unavailable")
    # The standardized-effect approximation assumes a mean-difference
    # contrast (a binary treatment). For a continuous treatment the estimate
    # is a per-unit slope whose size depends on the treatment's units, so
    # |coef|/sd(outcome) is not a standardized effect and collapses to ~0
    # for fine-grained units; skip rather than misreport a strong effect.
    treatment = result.treatment
    if treatment is not None and treatment in frame.columns:
        levels = pd.to_numeric(frame[treatment], errors="coerce").dropna().nunique()
        if levels > 2:
            return _check(
                "e_value", CheckStatus.NOT_APPLICABLE,
                "continuous treatment: the e-value's standardized-effect "
                "approximation needs a binary contrast, not a per-unit slope",
            )
    sd = float(pd.to_numeric(frame[outcome], errors="coerce").std())
    if not sd or not np.isfinite(sd):
        return _check("e_value", CheckStatus.NOT_APPLICABLE, "outcome has no spread")
    d = abs(primary.estimate) / sd
    rr = float(np.exp(0.91 * d))
    ev = rr + float(np.sqrt(max(rr * (rr - 1), 0.0)))
    status = (
        CheckStatus.PASS if ev >= 1.5
        else CheckStatus.WARNING if ev >= 1.15
        else CheckStatus.FAIL
    )
    return _check(
        "e_value", status,
        f"unmeasured confounding would need strength {ev:.2f} (risk-ratio scale) "
        "to explain the estimate away",
        e_value=ev, standardized_effect=d,
    )


def trimming_stability(
    frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec, base: float, runner
) -> DiagnosticCheck:
    """Re-estimate without the outcome's extreme 1% tails."""
    y = pd.to_numeric(frame[plan.outcome], errors="coerce")
    lo, hi = y.quantile(0.01), y.quantile(0.99)
    trimmed = frame[(y >= lo) & (y <= hi)]
    try:
        alt = runner(trimmed, plan, spec).result.primary.estimate
    except Exception as exc:
        return _check("outlier_trimming", CheckStatus.NOT_APPLICABLE, f"re-fit failed: {exc}")
    status, shift = _shift_status(base, alt)
    return _check(
        "outlier_trimming", status,
        f"estimate {base:.4g} -> {alt:.4g} after trimming 1% outcome tails",
        trimmed_estimate=alt, relative_shift=shift,
    )


def estimator_agreement(result: EstimateResult) -> DiagnosticCheck:
    by_name = {e.estimand: e for e in result.effects}
    if "ate" in by_name and "ate_ipw" in by_name:
        a, b = by_name["ate"].estimate, by_name["ate_ipw"].estimate
        status, shift = _shift_status(a, b)
        return _check(
            "estimator_agreement", status,
            f"regression {a:.4g} vs ipw {b:.4g}", relative_gap=shift,
        )
    return _check(
        "estimator_agreement", CheckStatus.NOT_APPLICABLE, "single estimator ran"
    )


def rdd_bandwidth_sensitivity(
    frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec, base: float, runner
) -> list[DiagnosticCheck]:
    checks = []
    run_col = plan.settings.get("running_variable")
    bandwidth = plan.settings.get("bandwidth") or float(
        pd.to_numeric(frame[run_col], errors="coerce").std()
    )
    for factor in (0.5, 2.0):
        alt_plan = plan.model_copy(deep=True)
        alt_plan.settings["bandwidth"] = bandwidth * factor
        try:
            alt = runner(frame, alt_plan, spec).result.primary.estimate
            status, shift = _shift_status(base, alt)
            checks.append(
                _check(
                    f"bandwidth_x{factor:g}", status,
                    f"estimate {base:.4g} -> {alt:.4g} at bandwidth x{factor:g}",
                    alt_estimate=alt, relative_shift=shift,
                )
            )
        except Exception as exc:
            checks.append(
                _check(f"bandwidth_x{factor:g}", CheckStatus.NOT_APPLICABLE, str(exc)[:200])
            )
    return checks


def rdd_placebo_cutoff(
    frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec, runner
) -> DiagnosticCheck:
    """A jump at a fake cutoff (median of one side) undermines the design."""
    run_col = plan.settings.get("running_variable")
    cutoff = float(plan.settings.get("cutoff"))
    x = pd.to_numeric(frame[run_col], errors="coerce")
    placebo = float(x[x < cutoff].median())
    alt_plan = plan.model_copy(deep=True)
    alt_plan.settings["cutoff"] = placebo
    side = frame[x < cutoff]  # placebo estimated within one side only
    try:
        alt = runner(side, alt_plan, spec).result
        jump = next(e for e in alt.effects if e.estimand == "itt_jump")
        significant = jump.p_value is not None and jump.p_value < 0.05
        return _check(
            "placebo_cutoff",
            CheckStatus.FAIL if significant else CheckStatus.PASS,
            f"placebo cutoff {placebo:.4g}: jump {jump.estimate:.4g} "
            f"(p {jump.p_value:.3f})" if jump.p_value is not None else "no p-value",
            placebo_jump=jump.estimate, placebo_p=jump.p_value,
        )
    except Exception as exc:
        return _check("placebo_cutoff", CheckStatus.NOT_APPLICABLE, str(exc)[:200])


def did_pre_trend(frame: pd.DataFrame, plan: MethodPlan) -> DiagnosticCheck:
    time_col = plan.settings.get("time_column")
    group_col = plan.settings.get("group_column")
    post_col = plan.settings.get("post_column") or "post"
    if plan.settings.get("needs_reshape") or not all(
        c in frame.columns for c in (time_col or "", group_col or "", post_col)
    ):
        return _check(
            "pre_trend", CheckStatus.NOT_APPLICABLE,
            "two-period design: parallel trends untestable from the data",
        )
    pre = frame[frame[post_col] == 0]
    outcome = plan.outcome
    slopes = {}
    for g, chunk in pre.groupby(group_col):
        means = chunk.groupby(time_col)[outcome].mean()
        if len(means) >= 3:
            slopes[str(g)] = float(np.polyfit(range(len(means)), means.values, 1)[0])
    if len(slopes) < 2:
        return _check(
            "pre_trend", CheckStatus.NOT_APPLICABLE, "fewer than three pre-periods"
        )
    values = list(slopes.values())
    gap = abs(values[0] - values[1])
    scale = max(max(abs(v) for v in values), 1e-9)
    status = CheckStatus.WARNING if gap > 0.5 * scale else CheckStatus.PASS
    return _check(
        "pre_trend", status, f"pre-period slopes {slopes}", pre_trend_gap=gap
    )


def iv_first_stage_strength(frame: pd.DataFrame, plan: MethodPlan) -> DiagnosticCheck:
    import statsmodels.api as sm

    instrument = plan.settings.get("instrument")
    cols = [plan.treatment, instrument, *plan.covariates]
    data = frame[cols].apply(lambda s: pd.to_numeric(s, errors="coerce")).dropna()
    rhs = sm.add_constant(data[[instrument, *plan.covariates]])
    fit = sm.OLS(data[plan.treatment], rhs).fit(cov_type="HC1")
    f_stat = float((fit.params[instrument] / fit.bse[instrument]) ** 2)
    status = (
        CheckStatus.PASS if f_stat >= 10
        else CheckStatus.WARNING if f_stat >= 5
        else CheckStatus.FAIL
    )
    return _check(
        "first_stage_strength", status,
        f"first-stage F {f_stat:.1f} (10 is the usual bar)", f_stat=f_stat,
    )


def ts_window_sensitivity(
    frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec, base: float, runner
) -> DiagnosticCheck:
    date = pd.to_datetime(plan.settings.get("intervention_date"))
    shifts = []
    for days in (-14, 14):
        alt_plan = plan.model_copy(deep=True)
        alt_plan.settings["intervention_date"] = str(
            (date + pd.Timedelta(days=days)).date()
        )
        try:
            alt = runner(frame, alt_plan, spec).result.primary.estimate
            shifts.append(abs(alt - base) / max(abs(base), 1e-9))
        except Exception:
            continue
    if not shifts:
        return _check("window_sensitivity", CheckStatus.NOT_APPLICABLE, "re-fits failed")
    worst = max(shifts)
    status = CheckStatus.WARNING if worst > 0.5 else CheckStatus.PASS
    return _check(
        "window_sensitivity", status,
        f"shifting the cut by two weeks moves the estimate up to {worst:.0%}",
        max_relative_shift=worst,
    )


def survival_km_crossing(frame: pd.DataFrame, plan: MethodPlan) -> DiagnosticCheck:
    from statsmodels.duration.survfunc import SurvfuncRight

    duration = plan.settings.get("duration_column")
    event = plan.settings.get("event_column")
    treat = plan.treatment
    data = frame[[duration, event, treat]].apply(
        lambda s: pd.to_numeric(s, errors="coerce")
    ).dropna()
    if data[treat].nunique() != 2:
        return _check(
            "km_crossing", CheckStatus.NOT_APPLICABLE,
            "treatment is not binary; curve-crossing check skipped",
        )
    curves = {}
    for value, chunk in data.groupby(treat):
        sf = SurvfuncRight(chunk[duration], chunk[event])
        curves[value] = pd.Series(sf.surv_prob, index=sf.surv_times)
    a, b = curves.values()
    grid = np.linspace(data[duration].quantile(0.1), data[duration].quantile(0.9), 25)
    diff = np.array(
        [a.reindex(a.index.union(grid)).ffill().get(t, 1.0)
         - b.reindex(b.index.union(grid)).ffill().get(t, 1.0) for t in grid]
    )
    crosses = bool((diff.max() > 0.02) and (diff.min() < -0.02))
    return _check(
        "km_crossing",
        CheckStatus.WARNING if crosses else CheckStatus.PASS,
        "survival curves cross: proportional hazards is suspect"
        if crosses else "no curve crossing in the central follow-up window",
    )
