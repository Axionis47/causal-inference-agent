"""check_covariate_balance - SMD between treated and control groups."""

import numpy as np
import pandas as pd

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "check_covariate_balance",
    "description": "Check balance of covariates between treatment and control groups using Standardized Mean Difference (SMD). Critical for causal inference.",
    "parameters": {
        "type": "object",
        "properties": {
            "covariates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Covariates to check (empty for all potential confounders)",
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
    treatment_col = agent._treatment_var

    if not treatment_col or treatment_col not in df.columns:
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={"error": "Treatment variable not specified or not found"},
        )

    treatment_values = df[treatment_col].dropna().unique()
    if len(treatment_values) != 2:
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={"error": f"Treatment variable is not binary (found {len(treatment_values)} values)"},
        )

    treated = df[df[treatment_col] == treatment_values.max()]
    control = df[df[treatment_col] == treatment_values.min()]

    if not covariates:
        if state.data_profile and state.data_profile.potential_confounders:
            covariates = state.data_profile.potential_confounders
        else:
            covariates = df.select_dtypes(include=[np.number]).columns.tolist()
            for var in [treatment_col, agent._outcome_var]:
                if var in covariates:
                    covariates.remove(var)

    results = []
    imbalanced = []

    for cov in covariates:
        if cov not in df.columns or not pd.api.types.is_numeric_dtype(df[cov]):
            continue

        treated_vals = treated[cov].dropna()
        control_vals = control[cov].dropna()

        if len(treated_vals) < 5 or len(control_vals) < 5:
            continue

        mean_diff = treated_vals.mean() - control_vals.mean()
        pooled_std = np.sqrt((treated_vals.std() ** 2 + control_vals.std() ** 2) / 2)
        smd = abs(mean_diff / pooled_std) if pooled_std > 0 else 0.0

        balance_info = {
            "covariate": cov,
            "treated_mean": round(float(treated_vals.mean()), 3),
            "control_mean": round(float(control_vals.mean()), 3),
            "smd": round(float(smd), 3),
            "is_balanced": smd < 0.1,
        }
        results.append(balance_info)
        agent._balance_results[cov] = balance_info

        if smd >= 0.1:
            imbalanced.append({"covariate": cov, "smd": round(smd, 3)})

    agent._eda_result.covariate_balance = agent._balance_results

    if imbalanced:
        imb_str = ", ".join([f"{i['covariate']} (SMD={i['smd']})" for i in imbalanced[:5]])
        agent._eda_result.balance_summary = f"Imbalanced: {imb_str}"
    else:
        agent._eda_result.balance_summary = "All covariates well-balanced (SMD < 0.1)"

    imbalanced.sort(key=lambda x: x["smd"], reverse=True)

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "treatment_variable": treatment_col,
            "n_treated": len(treated),
            "n_control": len(control),
            "n_covariates_checked": len(results),
            "n_imbalanced": len(imbalanced),
            "imbalanced_covariates": imbalanced[:10],
            "balance_summary": agent._eda_result.balance_summary,
        },
    )
