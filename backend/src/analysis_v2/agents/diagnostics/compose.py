"""Compose the per-lane diagnostic suite and the robustness rubric.

The design-specific check dispatch and the verdict rubric are the single
source of truth shared by the diagnostics agent (S8) and the generated
notebook, so the notebook's inline re-run can never drift from the pipeline.
Leakage is intentionally not here: it gates the run, so the agent runs it
on its own and this composes everything after it.
"""
from __future__ import annotations

import pandas as pd

from src.analysis_v2.spec import (
    CausalSpec,
    CheckStatus,
    DiagnosticCheck,
    EstimateResult,
    MethodLane,
    MethodPlan,
    RobustnessStatus,
)

from . import checks as C


def run_lane_checks(
    frame: pd.DataFrame, plan: MethodPlan, spec: CausalSpec,
    result: EstimateResult, runner
) -> tuple[list[DiagnosticCheck], list[DiagnosticCheck]]:
    """The design-specific diagnostic and sensitivity checks for one lane,
    returned as (diagnostics, sensitivity). Mirrors the S8 dispatch exactly."""
    diag: list[DiagnosticCheck] = []
    sens: list[DiagnosticCheck] = []
    base = result.primary.estimate
    if plan.lane in (MethodLane.OBSERVATIONAL, MethodLane.MATCHING):
        sens.append(C.evalue(result, frame))
        sens.append(C.trimming_stability(frame, plan, spec, base, runner))
        diag.append(C.estimator_agreement(result))
    elif plan.lane == MethodLane.DID:
        diag.append(C.did_pre_trend(frame, plan))
        sens.append(C.trimming_stability(frame, plan, spec, base, runner))
    elif plan.lane == MethodLane.RDD:
        sens.extend(C.rdd_bandwidth_sensitivity(frame, plan, spec, base, runner))
        sens.append(C.rdd_placebo_cutoff(frame, plan, spec, runner))
    elif plan.lane == MethodLane.IV:
        diag.append(C.iv_first_stage_strength(frame, plan))
        diag.append(
            DiagnosticCheck(
                name="exclusion_restriction",
                status=CheckStatus.NOT_APPLICABLE,
                detail="the exclusion restriction is an assumption; no data "
                "test exists for it",
            )
        )
    elif plan.lane == MethodLane.TIME_SERIES:
        sens.append(C.ts_window_sensitivity(frame, plan, spec, base, runner))
    elif plan.lane == MethodLane.MEDIATION:
        diag.append(
            DiagnosticCheck(
                name="mediator_timing",
                status=CheckStatus.WARNING,
                detail="treatment -> mediator -> outcome ordering is assumed, "
                "not observed",
            )
        )
        sens.append(C.trimming_stability(frame, plan, spec, base, runner))
    elif plan.lane == MethodLane.SURVIVAL:
        diag.append(C.survival_km_crossing(frame, plan))
    return diag, sens


def overall(checks: list[DiagnosticCheck]) -> CheckStatus:
    statuses = {c.status for c in checks}
    if CheckStatus.FAIL in statuses:
        return CheckStatus.FAIL
    if CheckStatus.WARNING in statuses:
        return CheckStatus.WARNING
    return CheckStatus.PASS


def summary_line(checks: list[DiagnosticCheck], kind: str) -> str:
    worst = [c for c in checks if c.status in (CheckStatus.WARNING, CheckStatus.FAIL)]
    if not worst:
        return f"all {kind} checks passed"
    return "; ".join(f"{c.name}: {c.detail}" for c in worst[:4])


def rubric(
    diag: list[DiagnosticCheck], sens: list[DiagnosticCheck]
) -> tuple[RobustnessStatus, str]:
    """Map the worst check status to a robustness verdict and reason."""
    hard_fails = [c for c in diag + sens if c.status == CheckStatus.FAIL]
    warns = [c for c in diag + sens if c.status == CheckStatus.WARNING]
    if hard_fails:
        return (
            RobustnessStatus.NOT_SUPPORTED,
            "a design assumption failed its check: "
            + "; ".join(c.name for c in hard_fails),
        )
    if warns:
        return (
            RobustnessStatus.FRAGILE,
            "the estimate survives but with caveats: "
            + "; ".join(c.name for c in warns[:4]),
        )
    return (
        RobustnessStatus.ROBUST,
        "the estimate is stable across the perturbations tried",
    )
