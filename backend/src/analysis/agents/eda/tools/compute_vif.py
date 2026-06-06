"""compute_vif - Variance Inflation Factor across numeric covariates."""

import numpy as np
import pandas as pd

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "compute_vif",
    "description": "Compute Variance Inflation Factor to assess multicollinearity severity. Use after finding high correlations.",
    "parameters": {
        "type": "object",
        "properties": {
            "covariates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Covariates to check (empty for all numeric except treatment/outcome)",
            },
        },
        "required": [],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    covariates: list[str] | None = None,
    **kwargs,
) -> ToolResult:
    if agent._df is None:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="Dataset not loaded",
        )

    df = agent._df

    if not covariates:
        covariates = df.select_dtypes(include=[np.number]).columns.tolist()
        for var in [agent._treatment_var, agent._outcome_var]:
            if var in covariates:
                covariates.remove(var)

    covariates = [c for c in covariates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    if len(covariates) < 2:
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={"error": "Need at least 2 covariates for VIF"},
        )

    X = df[covariates].dropna()
    if len(X) < 10:
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={"error": "Not enough complete cases for VIF"},
        )

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        X_with_const = np.column_stack([np.ones(len(X)), X.values])

        vif_results = []
        warnings: list[str] = []

        for i, col in enumerate(covariates):
            try:
                vif = variance_inflation_factor(X_with_const, i + 1)
                if np.isinf(vif) or np.isnan(vif):
                    vif_results.append({"variable": col, "vif": float("inf"), "note": "perfect_collinearity"})
                    agent._vif_results[col] = float("inf")
                    warnings.append(f"PERFECT COLLINEARITY: {col} (VIF=inf)")
                else:
                    vif_results.append({"variable": col, "vif": round(float(vif), 2)})
                    agent._vif_results[col] = float(vif)
                    if vif > 10:
                        warnings.append(f"SEVERE: {col} (VIF={vif:.1f})")
                    elif vif > 5:
                        warnings.append(f"MODERATE: {col} (VIF={vif:.1f})")
            except Exception as e:
                agent.logger.debug("vif_computation_error", column=col, error=str(e))

        agent._eda_result.vif_scores = agent._vif_results
        agent._eda_result.multicollinearity_warnings = warnings

        vif_results.sort(key=lambda x: x.get("vif", 0), reverse=True)

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "n_covariates": len(covariates),
                "severe_count": sum(1 for r in vif_results if r.get("vif", 0) > 10),
                "moderate_count": sum(1 for r in vif_results if 5 < r.get("vif", 0) <= 10),
                "top_vif": vif_results[:10],
                "warnings": warnings,
            },
        )
    except ImportError:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="statsmodels not available for VIF calculation",
        )
