"""check_missing_patterns - missingness summary plus differential-by-treatment check."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "check_missing_patterns",
    "description": "Analyze missing data patterns and whether missingness relates to treatment.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


async def handle(agent, state: AnalysisState, **kwargs) -> ToolResult:
    if agent._df is None:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="Dataset not loaded",
        )

    df = agent._df

    missing_counts = df.isnull().sum()
    cols_with_missing = missing_counts[missing_counts > 0]

    if len(cols_with_missing) == 0:
        agent._missing_analysis = {"has_missing": False}
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={"has_missing": False, "message": "No missing values"},
        )

    n_rows = len(df)
    total_missing = df.isnull().sum().sum()
    total_cells = n_rows * len(df.columns)

    col_analysis = []
    for col in cols_with_missing.index:
        col_analysis.append({
            "column": col,
            "missing": int(cols_with_missing[col]),
            "pct": round(float(cols_with_missing[col] / n_rows * 100), 1),
        })
    col_analysis.sort(key=lambda x: x["missing"], reverse=True)

    treatment_col = agent._treatment_var
    differential_missing = []
    if treatment_col and treatment_col in df.columns:
        for col in cols_with_missing.index[:10]:
            treated_missing = df[df[treatment_col] == 1][col].isnull().mean()
            control_missing = df[df[treatment_col] == 0][col].isnull().mean()
            if abs(treated_missing - control_missing) > 0.05:
                differential_missing.append({
                    "column": col,
                    "treated_missing_pct": round(float(treated_missing * 100), 1),
                    "control_missing_pct": round(float(control_missing * 100), 1),
                })

    agent._missing_analysis = {
        "has_missing": True,
        "total_missing_pct": round(float(total_missing / total_cells * 100), 2),
        "n_cols_with_missing": len(cols_with_missing),
        "by_column": col_analysis,
        "differential_missing": differential_missing,
    }

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "has_missing": True,
            "total_missing_pct": agent._missing_analysis["total_missing_pct"],
            "n_cols_with_missing": len(cols_with_missing),
            "top_missing_columns": col_analysis[:10],
            "differential_missing": differential_missing,
            "warning": "Differential missing may indicate selection bias!" if differential_missing else None,
        },
    )
