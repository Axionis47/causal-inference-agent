"""analyze_treatment - distribution + balance of the treatment column."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "analyze_treatment",
    "description": "Analyze the treatment variable distribution and identify potential issues.",
    "parameters": {
        "type": "object",
        "properties": {
            "treatment_col": {
                "type": "string",
                "description": "Name of the treatment column",
            },
        },
        "required": ["treatment_col"],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    treatment_col: str,
    **kwargs,
) -> ToolResult:
    if agent._df is None:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="No data loaded",
        )

    if treatment_col not in agent._df.columns:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error=f"Column '{treatment_col}' not found",
        )

    T = agent._df[treatment_col]
    n_unique = T.nunique()
    is_binary = n_unique == 2

    output: dict = {
        "column": treatment_col,
        "n_unique": int(n_unique),
        "is_binary": is_binary,
        "value_counts": T.value_counts().to_dict(),
        "missing": int(T.isna().sum()),
    }

    if is_binary:
        output["treatment_prevalence"] = f"{T.mean()*100:.1f}%"
        output["n_treated"] = int(T.sum())
        output["n_control"] = int(len(T) - T.sum())
    else:
        output["recommendation"] = "Consider binarizing treatment (e.g., above/below median)"

    return ToolResult(status=ToolResultStatus.SUCCESS, output=output)
