"""compare_results - cross-method comparison and preferred-estimate selection."""

import numpy as np

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "compare_results",
    "description": "Compare treatment effect estimates across methods.",
    "parameters": {
        "type": "object",
        "properties": {
            "interpretation_focus": {
                "type": "string",
                "enum": ["magnitude", "significance", "robustness", "all"],
                "description": "What to focus the comparison on",
            },
        },
        "required": ["interpretation_focus"],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    interpretation_focus: str,
    **kwargs,
) -> ToolResult:
    if not agent._results:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="No results to compare",
        )

    comparison: dict = {
        "n_methods": len(agent._results),
        "estimates": [],
    }

    for r in agent._results:
        comparison["estimates"].append({
            "method": r.method,
            "estimate": r.estimate,
            "std_error": r.std_error,
            "ci": [r.ci_lower, r.ci_upper],
        })

    estimates = [r.estimate for r in agent._results]
    comparison["mean_estimate"] = float(np.mean(estimates))
    comparison["std_across_methods"] = float(np.std(estimates))

    ci_overlaps = all(
        r1.ci_lower <= r2.ci_upper and r2.ci_lower <= r1.ci_upper
        for i, r1 in enumerate(agent._results)
        for r2 in agent._results[i + 1:]
    )
    comparison["ci_overlap"] = ci_overlaps

    if interpretation_focus in ["robustness", "all"]:
        comparison["robustness_assessment"] = (
            "Estimates are consistent across methods (CIs overlap)"
            if ci_overlaps
            else "CAUTION: Estimates vary substantially across methods"
        )

    if interpretation_focus in ["significance", "all"]:
        significant_count = sum(
            1 for r in agent._results if r.p_value and r.p_value < 0.05
        )
        comparison["significance"] = (
            f"{significant_count}/{len(agent._results)} methods show significant effect"
        )

    preferred = None
    for r in agent._results:
        if "aipw" in r.method.lower() or "doubly" in r.method.lower():
            preferred = r
            break
        if "ipw" in r.method.lower() or "psm" in r.method.lower() or "matching" in r.method.lower():
            preferred = preferred or r
    preferred = preferred or agent._results[0]

    comparison["preferred_method"] = preferred.method
    comparison["preferred_estimate"] = preferred.estimate
    comparison["preferred_reasoning"] = "Doubly robust methods are preferred when available"

    return ToolResult(status=ToolResultStatus.SUCCESS, output=comparison)
