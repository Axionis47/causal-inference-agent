"""check_influential_observations - residual-based outlier sensitivity check."""

import numpy as np
import pandas as pd

SCHEMA = {
    "name": "check_influential_observations",
    "description": "Identify observations that strongly influence the treatment effect estimate.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


def handle(agent, **kwargs) -> str:
    if agent._df is None:
        return "No data available for influence check."

    state = agent._state
    df = agent._df
    treatment_col, outcome_col = agent._resolve_treatment_outcome()

    if not treatment_col or not outcome_col:
        return "Treatment or outcome variable not identified."

    if treatment_col not in df.columns or outcome_col not in df.columns:
        return "Treatment or outcome variable not found."

    output = "Influential Observation Check:\n"
    output += "=" * 50 + "\n"

    try:
        from sklearn.linear_model import LinearRegression

        covariates: list[str] = []
        if state.data_profile and state.data_profile.potential_confounders:
            covariates = [
                c for c in state.data_profile.potential_confounders[:5]
                if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ]

        X = df[[treatment_col] + covariates].dropna()
        y = df.loc[X.index, outcome_col]

        if len(X) < 50:
            return "Not enough complete cases for influence analysis."

        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred

        std_residuals = (residuals - residuals.mean()) / residuals.std()

        n_high_leverage = int((np.abs(std_residuals) > 3).sum())
        pct_influential = n_high_leverage / len(residuals) * 100

        output += f"Total observations: {len(residuals)}\n"
        output += f"High-leverage (|resid| > 3): {n_high_leverage} ({pct_influential:.1f}%)\n"

        if pct_influential > 5:
            output += "\nWARNING: Many influential observations - results may be unstable\n"
            agent._investigation_evidence.append(
                f"Found {n_high_leverage} influential observations"
            )
        elif pct_influential > 1:
            output += "\nSome influential observations present - consider robust methods\n"
            agent._investigation_evidence.append("Some influential observations present")
        else:
            output += "\nFew influential observations - results likely stable\n"
            agent._investigation_evidence.append("Few influential observations")

        non_outlier_mask = np.abs(std_residuals) <= 3
        X_clean = X[non_outlier_mask]
        y_clean = y[non_outlier_mask]

        treated_clean = y_clean[X_clean[treatment_col] == X_clean[treatment_col].max()]
        control_clean = y_clean[X_clean[treatment_col] == X_clean[treatment_col].min()]

        effect_all = df.groupby(treatment_col)[outcome_col].mean().diff().iloc[-1]
        effect_clean = treated_clean.mean() - control_clean.mean()

        output += f"\nEffect (all data): {effect_all:.4f}\n"
        output += f"Effect (excl. outliers): {effect_clean:.4f}\n"
        output += f"Change: {((effect_clean - effect_all) / effect_all * 100):.1f}%\n"

        if abs((effect_clean - effect_all) / effect_all) > 0.2:
            output += "\nWARNING: Effect changes >20% when excluding outliers!\n"
            agent._investigation_evidence.append("Effect sensitive to outliers")

    except Exception as e:
        output += f"\nCould not complete influence check: {str(e)}\n"

    return output
