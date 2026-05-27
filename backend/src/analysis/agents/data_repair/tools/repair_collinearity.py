"""repair_collinearity - drop high-VIF or highly-correlated covariates."""

import numpy as np

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "repair_collinearity",
    "description": "Handle collinearity by dropping or combining highly correlated variables.",
    "parameters": {
        "columns_to_drop": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Columns to drop (for manual selection)",
        },
        "strategy": {
            "type": "string",
            "enum": ["drop_specified", "drop_high_vif", "auto_drop_correlated"],
            "description": "Collinearity reduction strategy",
        },
        "vif_threshold": {
            "type": "number",
            "description": "VIF threshold for auto-dropping (default: 10)",
        },
    },
}


async def handle(
    agent,
    state: AnalysisState,
    strategy: str = "drop",
    columns_to_drop: list[str] | None = None,
    vif_threshold: float = 10,
    **kwargs,
) -> ToolResult:
    df = agent._df
    columns_to_drop = columns_to_drop or []
    protected = [agent._current_state.treatment_variable, agent._current_state.outcome_variable]

    try:
        if strategy == "drop_specified":
            columns_to_drop = [c for c in columns_to_drop if c in df.columns and c not in protected]
            if not columns_to_drop:
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output={"message": "No valid columns to drop (treatment/outcome are protected)."},
                )
            df = df.drop(columns=columns_to_drop)
            agent._df = df
            agent._repairs_applied.append({
                "type": "collinearity", "strategy": "drop_specified", "columns": columns_to_drop,
            })
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "message": f"Dropped columns: {columns_to_drop}",
                    "new_shape": list(df.shape),
                },
            )

        elif strategy == "drop_high_vif":
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in protected]

            dropped = []
            max_iters = 10
            for _ in range(max_iters):
                if len(numeric_cols) < 3:
                    break
                X = df[numeric_cols].dropna()
                if len(X) < 10:
                    break

                X_with_const = np.column_stack([np.ones(len(X)), X.values])
                vifs = []
                for i, col in enumerate(numeric_cols):
                    try:
                        vif = variance_inflation_factor(X_with_const, i + 1)
                        vifs.append((col, vif))
                    except Exception:
                        agent.logger.debug("vif_computation_skipped", column=col, exc_info=True)

                if not vifs:
                    break

                max_vif_col, max_vif = max(vifs, key=lambda x: x[1])
                if max_vif > vif_threshold:
                    numeric_cols.remove(max_vif_col)
                    df = df.drop(columns=[max_vif_col])
                    dropped.append(f"{max_vif_col} (VIF={max_vif:.1f})")
                else:
                    break

            agent._df = df

            if dropped:
                agent._repairs_applied.append({
                    "type": "collinearity", "strategy": "drop_high_vif",
                    "columns": [d.split(" ")[0] for d in dropped],
                })
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output={
                        "message": f"Dropped high VIF columns: {dropped}",
                        "new_shape": list(df.shape),
                    },
                )
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={"message": "No columns exceeded the VIF threshold."},
            )

        elif strategy == "auto_drop_correlated":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c not in protected]

            corr_matrix = df[numeric_cols].corr().abs()
            dropped = []

            for i, col1 in enumerate(corr_matrix.columns):
                for j, col2 in enumerate(corr_matrix.columns):
                    if i < j and col1 in df.columns and col2 in df.columns:
                        if corr_matrix.iloc[i, j] > 0.9:
                            if df[col1].isnull().sum() >= df[col2].isnull().sum():
                                to_drop = col1
                            else:
                                to_drop = col2
                            if to_drop not in dropped:
                                dropped.append(to_drop)
                                df = df.drop(columns=[to_drop])

            agent._df = df

            if dropped:
                agent._repairs_applied.append({
                    "type": "collinearity", "strategy": "auto_drop_correlated", "columns": dropped,
                })
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output={
                        "message": f"Dropped highly correlated columns: {dropped}",
                        "new_shape": list(df.shape),
                    },
                )
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={"message": "No highly correlated column pairs found to drop."},
            )

        return ToolResult(status=ToolResultStatus.ERROR, error=f"Unknown strategy: {strategy}")

    except Exception as e:
        return ToolResult(status=ToolResultStatus.ERROR, error=f"Error repairing collinearity: {str(e)}")
