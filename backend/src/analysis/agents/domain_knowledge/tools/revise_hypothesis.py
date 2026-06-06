"""revise_hypothesis tool: mark a prior claim revised and append the new one."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "revise_hypothesis",
    "description": (
        "Revise a previous hypothesis based on new evidence. "
        "Use this when you find information that changes your understanding."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "original_claim": {
                "type": "string",
                "description": "The original hypothesis to revise",
            },
            "new_claim": {
                "type": "string",
                "description": "The revised hypothesis",
            },
            "reason": {
                "type": "string",
                "description": "Why you're revising",
            },
        },
        "required": ["original_claim", "new_claim", "reason"],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    original_claim: str = "",
    new_claim: str = "",
    reason: str = "",
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug("tool_ignored_kwargs", tool="revise_hypothesis", extra_keys=list(kwargs.keys()))

    found = False
    for h in agent._hypotheses:
        if h["claim"].lower() == original_claim.lower():
            h["revised"] = True
            h["superseded_by"] = new_claim
            found = True
            break

    agent._hypotheses.append({
        "claim": new_claim,
        "confidence": "medium",
        "evidence": f"Revised from: {original_claim}. Reason: {reason}",
        "revised": False,
    })

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "original_found": found,
            "revision_recorded": True,
            "new_claim": new_claim,
        },
    )
