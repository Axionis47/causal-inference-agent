"""check_covariate_balance - SMD per covariate between treatment groups."""

import numpy as np
import pandas as pd

SCHEMA = {
    "name": "check_covariate_balance",
    "description": "Check actual covariate balance between treatment groups. Critical for verifying selection bias.",
    "parameters": {
        "type": "object",
        "properties": {
            "covariates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Covariates to check (empty for all)",
            },
        },
        "required": [],
    },
}


def handle(agent, covariates: list[str] | None = None, **kwargs) -> str:
    if agent._df is None:
        return "No data available for balance check."

    state = agent._state
    df = agent._df
    treatment_col, outcome_col = agent._resolve_treatment_outcome()

    if not treatment_col:
        return "Treatment variable not identified."

    if treatment_col not in df.columns:
        return f"Treatment variable '{treatment_col}' not found."

    try:
        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]
    except Exception:
        treated = df[df[treatment_col] == df[treatment_col].max()]
        control = df[df[treatment_col] == df[treatment_col].min()]

    covariates = covariates or []
    if not covariates:
        if state.data_profile and state.data_profile.potential_confounders:
            covariates = state.data_profile.potential_confounders[:10]
        else:
            covariates = df.select_dtypes(include=[np.number]).columns.tolist()
            covariates = [c for c in covariates if c not in [treatment_col, outcome_col]][:10]

    output = "Covariate Balance Check:\n"
    output += "=" * 50 + "\n"
    output += f"Treated: n={len(treated)}, Control: n={len(control)}\n\n"

    imbalanced: list[tuple[str, float]] = []
    for cov in covariates:
        if cov not in df.columns or not pd.api.types.is_numeric_dtype(df[cov]):
            continue

        t_vals = treated[cov].dropna()
        c_vals = control[cov].dropna()

        if len(t_vals) < 5 or len(c_vals) < 5:
            continue

        mean_diff = t_vals.mean() - c_vals.mean()
        pooled_std = np.sqrt((t_vals.std() ** 2 + c_vals.std() ** 2) / 2)
        smd = abs(mean_diff / pooled_std) if pooled_std > 0 else 0

        status = ""
        if smd > 0.25:
            status = " *** SEVERE ***"
            imbalanced.append((cov, smd))
        elif smd > 0.1:
            status = " ** IMBALANCED **"
            imbalanced.append((cov, smd))

        output += f"  {cov}: SMD={smd:.3f}{status}\n"

    output += f"\nSummary: {len(imbalanced)} of {len(covariates)} covariates imbalanced (SMD > 0.1)\n"

    if imbalanced:
        output += "\nIMBALANCED COVARIATES (potential confounding):\n"
        for cov, smd in sorted(imbalanced, key=lambda x: x[1], reverse=True)[:5]:
            output += f"  - {cov}: SMD={smd:.3f}\n"
        agent._investigation_evidence.append(
            f"Found {len(imbalanced)} imbalanced covariates"
        )
    else:
        agent._investigation_evidence.append("Covariates are well-balanced")

    return output
