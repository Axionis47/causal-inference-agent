"""check_method_diagnostics - residuals / influence / specification on the last result."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

from ..diagnostics import (
    check_influence_diagnostics,
    check_residual_diagnostics,
    check_specification_diagnostics,
)

logger = get_logger(__name__)

SCHEMA = {
    "name": "check_method_diagnostics",
    "description": "Check diagnostics for the most recently run method (e.g., residual analysis, influence points).",
    "parameters": {
        "type": "object",
        "properties": {
            "diagnostic_type": {
                "type": "string",
                "enum": ["residuals", "influence", "specification", "all"],
                "description": "Type of diagnostic to run",
            },
        },
        "required": ["diagnostic_type"],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    diagnostic_type: str = "all",
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug(
            "tool_ignored_kwargs",
            tool="check_method_diagnostics",
            extra_keys=list(kwargs.keys()),
        )
    if not agent._last_method_result:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="No method has been run yet. Run a method first.",
        )

    result = agent._last_method_result
    diagnostics: dict = {}

    if diagnostic_type in ["residuals", "all"]:
        diagnostics["residuals"] = check_residual_diagnostics(
            agent._df, agent._treatment_var, agent._outcome_var, agent._covariates
        )

    if diagnostic_type in ["influence", "all"]:
        diagnostics["influence"] = check_influence_diagnostics(
            agent._df, agent._outcome_var
        )

    if diagnostic_type in ["specification", "all"]:
        diagnostics["specification"] = check_specification_diagnostics(result)

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "method": result.method,
            "diagnostics": diagnostics,
        },
    )
