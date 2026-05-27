"""finalize_eda - records the agent's final assessment and flips _finalized."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "finalize_eda",
    "description": "Finalize the EDA with your assessment. Call this ONLY after gathering sufficient evidence.",
    "parameters": {
        "type": "object",
        "properties": {
            "data_quality_score": {
                "type": "number",
                "description": "Overall data quality score 0-100",
            },
            "key_findings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Most important findings from EDA",
            },
            "data_quality_issues": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific data quality issues identified",
            },
            "recommendations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Recommendations for causal analysis",
            },
            "causal_readiness": {
                "type": "string",
                "enum": ["ready", "needs_attention", "significant_concerns"],
                "description": "Overall readiness for causal inference",
            },
        },
        "required": [
            "data_quality_score",
            "key_findings",
            "data_quality_issues",
            "recommendations",
            "causal_readiness",
        ],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    data_quality_score: float = 0.5,
    key_findings: list[str] | None = None,
    data_quality_issues: list[str] | None = None,
    recommendations: list[str] | None = None,
    causal_readiness: str = "moderate",
    **kwargs,
) -> ToolResult:
    key_findings = key_findings or []
    data_quality_issues = data_quality_issues or []
    recommendations = recommendations or []

    agent.logger.info(
        "eda_finalizing",
        data_quality_score=data_quality_score,
        num_findings=len(key_findings),
        num_issues=len(data_quality_issues),
        causal_readiness=causal_readiness,
    )

    agent._final_result = {
        "data_quality_score": data_quality_score,
        "key_findings": key_findings,
        "data_quality_issues": data_quality_issues,
        "recommendations": recommendations,
        "causal_readiness": causal_readiness,
    }
    agent._finalized = True

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "eda_finalized": True,
            "data_quality_score": data_quality_score,
            "causal_readiness": causal_readiness,
            "n_findings": len(key_findings),
            "n_issues": len(data_quality_issues),
        },
    )
