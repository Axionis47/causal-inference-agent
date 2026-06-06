"""finalize_repairs - record the agent's final assessment and flip _finalized."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "finalize_repairs",
    "description": "Finalize repairs and save the repaired dataset. Call this when satisfied with data quality.",
    "parameters": {
        "quality_assessment": {
            "type": "string",
            "description": "Your assessment of final data quality",
        },
        "repairs_summary": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Summary of all repairs applied",
        },
        "cautions": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Any cautions for downstream analysis",
        },
    },
}


async def handle(
    agent,
    state: AnalysisState,
    quality_assessment: str = "acceptable",
    repairs_summary: list[str] | None = None,
    cautions: list[str] | None = None,
    **kwargs,
) -> ToolResult:
    repairs_summary = repairs_summary or []
    agent.logger.info("finalizing_repairs", quality=quality_assessment)
    agent._finalized = True

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "message": "Repairs finalized successfully",
            "quality_assessment": quality_assessment,
            "repairs_applied_count": len(agent._repairs_applied),
            "final_shape": list(agent._df.shape),
        },
    )
