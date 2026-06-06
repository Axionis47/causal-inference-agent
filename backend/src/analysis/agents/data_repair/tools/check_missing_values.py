"""check_missing_values - per-column missing counts plus MCAR-by-treatment check."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "check_missing_values",
    "description": "Analyze missing value patterns for specified columns or all columns.",
    "parameters": {
        "columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Columns to check (empty for all)",
        },
        "check_mcar": {
            "type": "boolean",
            "description": "Whether to test if missing is related to treatment (default: true)",
        },
    },
}


async def handle(
    agent,
    state: AnalysisState,
    columns: list[str] | None = None,
    check_mcar: bool = True,
    **kwargs,
) -> ToolResult:
    df = agent._df
    col_list = columns if columns else df.columns[df.isnull().any()].tolist()

    if not col_list:
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={"message": "No missing values found in the dataset.", "columns": []},
        )

    results = []
    n_rows = len(df)

    for col in col_list[:20]:
        if col not in df.columns:
            continue
        missing = int(df[col].isnull().sum())
        if missing == 0:
            continue
        missing_pct = float(missing / n_rows * 100)
        results.append({"column": col, "missing": missing, "pct": missing_pct})

    mcar_warnings = []
    if check_mcar and agent._current_state.treatment_variable in df.columns:
        treatment_col = agent._current_state.treatment_variable
        for r in results[:10]:
            col = r["column"]
            try:
                treated_missing = float(df[df[treatment_col] == 1][col].isnull().mean())
                control_missing = float(df[df[treatment_col] == 0][col].isnull().mean())
                diff = abs(treated_missing - control_missing)
                if diff > 0.05:
                    mcar_warnings.append({
                        "column": col,
                        "treated_missing_pct": treated_missing * 100,
                        "control_missing_pct": control_missing * 100,
                        "diff_pct": diff * 100,
                    })
            except Exception:
                agent.logger.debug("mcar_check_skipped", column=col, exc_info=True)

    results.sort(key=lambda x: x["pct"], reverse=True)

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "columns_analyzed": len(col_list),
            "columns_with_missing": results[:15],
            "mcar_warnings": mcar_warnings,
            "has_mnar_risk": len(mcar_warnings) > 0,
        },
    )
