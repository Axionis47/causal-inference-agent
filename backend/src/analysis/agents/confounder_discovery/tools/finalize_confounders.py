"""finalize_confounders - LLM submits final ranked list; statistical fallback fills empty."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from ..helpers import statistical_confounder_scan

SCHEMA = {
    "name": "finalize_confounders",
    "description": "Submit your final list of identified confounders after investigation",
    "parameters": {
        "confounders": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of confirmed confounders, ranked by importance",
        },
        "excluded": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Variables investigated but excluded (mediators, colliders, noise)",
        },
        "reasoning": {
            "type": "string",
            "description": "Explanation of your confounder identification process",
        },
    },
}


async def handle(
    agent,
    state: AnalysisState,
    confounders: list[str] | None = None,
    reasoning: str = "",
    excluded: list[str] | None = None,
    **kwargs,
) -> ToolResult:
    confounders = confounders or []
    excluded = excluded or []

    fallback_details = None
    if (
        len(confounders) == 0
        and agent._df is not None
        and agent._treatment_var
        and agent._outcome_var
    ):
        agent.logger.info("finalize_zero_confounders_triggering_statistical_fallback")
        scan_results = statistical_confounder_scan(
            agent._df, agent._treatment_var, agent._outcome_var, agent.logger,
        )
        if scan_results:
            confounders = [r["variable"] for r in scan_results]
            fallback_details = scan_results
            reasoning = (
                f"Statistical fallback: DAG-based confounder identification produced no results. "
                f"Correlation scan found {len(confounders)} variable(s) associated with both "
                f"treatment ({agent._treatment_var}) and outcome ({agent._outcome_var})."
            )
            state.push_decision(
                agent="confounder_discovery",
                decision_type="statistical_fallback",
                choice=f"Found {len(confounders)} confounders via correlation scan",
                reason="DAG-based confounder identification produced no results, falling back to statistical correlation with T and Y",
            )
            agent.logger.info(
                "statistical_fallback_confounders_found",
                n_confounders=len(confounders),
                top_confounders=confounders[:5],
            )

    agent.logger.info(
        "confounder_finalizing",
        n_confounders=len(confounders),
        top_confounders=confounders[:5] if confounders else [],
    )

    if state.data_profile:
        state.data_profile.potential_confounders = confounders

    state.confounder_discovery = {
        "ranked_confounders": confounders,
        "excluded_variables": excluded,
        "adjustment_strategy": reasoning,
        "investigation_log": agent._investigation_log,
    }
    if fallback_details:
        state.confounder_discovery["statistical_fallback_details"] = fallback_details

    agent._finalized = True

    state.push_decision(
        agent="confounder_discovery",
        decision_type="confounders_selected",
        choice=", ".join(confounders) if confounders else "(none)",
        reason=(
            f"Selected {len(confounders)} confounder(s) based on statistical criteria "
            f"(correlation with both {agent._treatment_var} and {agent._outcome_var}) "
            f"and domain knowledge; excluded {len(excluded)} variable(s) as mediators/colliders/noise"
        ),
    )

    agent.logger.info(
        "confounder_discovery_complete",
        n_confounders=len(confounders),
        n_excluded=len(excluded),
    )

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "confounders": confounders,
            "excluded": excluded,
            "reasoning": reasoning,
            **({"fallback_scan": fallback_details} if fallback_details else {}),
        },
    )
