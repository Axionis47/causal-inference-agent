"""hypothesize tool: record a causal-structure claim on the agent."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "hypothesize",
    "description": (
        "Record a hypothesis about the causal structure. "
        "You can have multiple hypotheses. They can be revised later."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "claim": {
                "type": "string",
                "description": "Your hypothesis about a causal relationship in this dataset",
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "How confident you are",
            },
            "evidence": {
                "type": "string",
                "description": "What evidence supports this hypothesis",
            },
        },
        "required": ["claim", "confidence", "evidence"],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    claim: str = "",
    confidence: str = "medium",
    evidence: str = "",
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug("tool_ignored_kwargs", tool="hypothesize", extra_keys=list(kwargs.keys()))

    hypothesis = {
        "claim": claim,
        "confidence": confidence,
        "evidence": evidence,
        "revised": False,
    }
    agent._hypotheses.append(hypothesis)

    agent.logger.info("hypothesis_recorded", claim=claim, confidence=confidence)

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "recorded": True,
            "hypothesis": hypothesis,
            "total_hypotheses": len(agent._hypotheses),
        },
    )
