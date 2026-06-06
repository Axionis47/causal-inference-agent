"""inspect_data - examine the dataset's overall, treatment, outcome, or covariate shape."""

import numpy as np

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "inspect_data",
    "description": "Inspect the dataset to understand its characteristics for causal inference.",
    "parameters": {
        "type": "object",
        "properties": {
            "focus": {
                "type": "string",
                "enum": ["overview", "treatment", "outcome", "covariates"],
                "description": "What aspect to focus on",
            },
        },
        "required": ["focus"],
    },
}


async def handle(agent, state: AnalysisState, focus: str, **kwargs) -> ToolResult:
    """Return a focus-specific summary of the in-memory dataframe."""
    if agent._df is None:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="No data loaded",
        )

    df = agent._df
    output: dict = {}

    if focus == "overview":
        output = {
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": list(df.columns),
            "dtypes": {col: str(df[col].dtype) for col in df.columns[:20]},
            "missing_pct": {
                col: f"{df[col].isna().mean()*100:.1f}%"
                for col in df.columns if df[col].isna().any()
            },
        }
    elif focus == "treatment" and state.data_profile:
        candidates = state.data_profile.treatment_candidates[:5]
        output = {"candidates": candidates, "distributions": {}}
        for col in candidates:
            if col in df.columns:
                output["distributions"][col] = {
                    "unique_values": int(df[col].nunique()),
                    "value_counts": df[col].value_counts().head(5).to_dict(),
                }
    elif focus == "outcome" and state.data_profile:
        candidates = state.data_profile.outcome_candidates[:5]
        output = {"candidates": candidates, "statistics": {}}
        for col in candidates:
            if col in df.columns and np.issubdtype(df[col].dtype, np.number):
                output["statistics"][col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "max": float(df[col].max()),
                }
    elif focus == "covariates" and state.data_profile:
        confounders = state.data_profile.potential_confounders[:10]
        output = {
            "potential_confounders": confounders,
            "n_confounders": len(state.data_profile.potential_confounders),
            "numeric_covariates": [
                c for c in confounders
                if c in df.columns and np.issubdtype(df[c].dtype, np.number)
            ],
        }

    return ToolResult(status=ToolResultStatus.SUCCESS, output=output)
