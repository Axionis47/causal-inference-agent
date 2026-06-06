"""get_candidate_variables - numeric columns minus treatment and outcome."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from ..helpers import candidates_from_df

SCHEMA = {
    "name": "get_candidate_variables",
    "description": "Get the list of candidate variables to investigate as potential confounders",
    "parameters": {},
}


async def handle(agent, state: AnalysisState, **kwargs) -> ToolResult:
    candidates = candidates_from_df(agent._df, agent._treatment_var, agent._outcome_var)
    agent._investigation_log.append({
        "tool": "get_candidate_variables",
        "args": {},
        "result": candidates,
    })
    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "candidates": candidates,
            "treatment": agent._treatment_var,
            "outcome": agent._outcome_var,
        },
    )
