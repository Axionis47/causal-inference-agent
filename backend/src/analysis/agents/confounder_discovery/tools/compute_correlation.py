"""compute_correlation - Pearson correlation between any two columns."""

from scipy import stats

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "compute_correlation",
    "description": "Compute Pearson correlation between two variables. Use this to check if a variable is associated with treatment or outcome.",
    "parameters": {
        "var1": {"type": "string", "description": "First variable name"},
        "var2": {"type": "string", "description": "Second variable name"},
    },
}


async def handle(agent, state: AnalysisState, var1: str = "", var2: str = "", **kwargs) -> ToolResult:
    if agent._df is None:
        return ToolResult(status=ToolResultStatus.ERROR, error="No data loaded")

    if var1 not in agent._df.columns or var2 not in agent._df.columns:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            error=f"Variable not found. Available: {list(agent._df.columns)}",
        )

    corr, pval = stats.pearsonr(agent._df[var1], agent._df[var2])
    significance = "significant" if pval < 0.05 else "not significant"
    strength = "strong" if abs(corr) > 0.3 else ("moderate" if abs(corr) > 0.1 else "weak")

    result = {
        "var1": var1,
        "var2": var2,
        "correlation": float(corr),
        "p_value": float(pval),
        "strength": strength,
        "significance": significance,
    }
    agent._investigation_log.append({
        "tool": "compute_correlation",
        "args": {"var1": var1, "var2": var2},
        "result": result,
    })
    return ToolResult(status=ToolResultStatus.SUCCESS, output=result)
