"""check_overlap - common support + KS distance + positivity check on the estimated PS.

Caches `agent._metrics.overlap_pct` so the brief's POOR_OVERLAP flag
derivation in `build_brief(state, metrics)` works.
"""

import numpy as np
from scipy import stats

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "check_overlap",
    "description": "Check overlap/common support between treatment groups based on estimated propensity scores.",
    "parameters": {},
}


async def handle(agent, state: AnalysisState, **kwargs) -> ToolResult:
    if kwargs:
        logger.debug(
            "tool_ignored_kwargs",
            tool="check_overlap",
            extra_keys=list(kwargs.keys()),
        )
    if agent._propensity_scores is None:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            error="No propensity scores estimated yet. Call estimate_propensity_scores first.",
        )

    ps = agent._propensity_scores
    T = agent._treatment

    treated_idx = T == 1
    control_idx = T == 0

    ps_treated = ps[treated_idx]
    ps_control = ps[control_idx]

    overlap_min = max(ps_control.min(), ps_treated.min())
    overlap_max = min(ps_control.max(), ps_treated.max())

    in_overlap = (ps >= overlap_min) & (ps <= overlap_max)
    pct_overlap = np.mean(in_overlap) * 100
    agent._metrics.overlap_pct = float(pct_overlap)

    ks_stat, ks_pval = stats.ks_2samp(ps_treated, ps_control)

    near_zero = int(np.sum(ps < 0.01))
    near_one = int(np.sum(ps > 0.99))
    positivity_violations = near_zero + near_one

    if pct_overlap > 95 and positivity_violations < 5:
        quality = "EXCELLENT"
    elif pct_overlap > 85 and positivity_violations < 20:
        quality = "GOOD"
    elif pct_overlap > 70:
        quality = "MODERATE"
    elif pct_overlap > 50:
        quality = "POOR"
    else:
        quality = "CRITICAL"

    recommendation = (
        "Good overlap, proceed with PS methods"
        if quality in ["EXCELLENT", "GOOD"]
        else "Consider trimming extreme PS values or using doubly robust methods"
    )

    state.push_decision(
        agent="ps_diagnostics",
        decision_type="quality_gate",
        choice=f"overlap_{quality.lower()}",
        reason=(
            f"Propensity score overlap is {quality} ({pct_overlap:.1f}% common support, "
            f"{positivity_violations} positivity violations). "
            f"{'PS methods recommended.' if quality in ['EXCELLENT', 'GOOD'] else 'Consider alternative methods.'}"
        ),
    )

    overlap_result = {
        "treated_stats": {
            "mean": float(ps_treated.mean()),
            "std": float(ps_treated.std()),
            "median": float(np.median(ps_treated)),
        },
        "control_stats": {
            "mean": float(ps_control.mean()),
            "std": float(ps_control.std()),
            "median": float(np.median(ps_control)),
        },
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pval),
        "common_support": [float(overlap_min), float(overlap_max)],
        "pct_overlap": float(pct_overlap),
        "quality": quality,
        "ps_near_zero": near_zero,
        "ps_near_one": near_one,
        "positivity_violations": positivity_violations,
        "recommendation": recommendation,
    }

    if quality == "CRITICAL":
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=overlap_result,
            error=(
                f"CRITICAL: Propensity score overlap is very poor ({pct_overlap:.1f}%). "
                "PS-based methods (IPW, AIPW, PSM) may produce unreliable estimates."
            ),
        )

    if quality == "MODERATE":
        overlap_result["warning"] = (
            f"Moderate overlap ({pct_overlap:.1f}%). Consider trimming or doubly robust methods."
        )

    return ToolResult(status=ToolResultStatus.SUCCESS, output=overlap_result)
