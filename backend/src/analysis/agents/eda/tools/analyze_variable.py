"""analyze_variable - distribution stats and normality tests for one column."""

import pandas as pd
from scipy import stats
from scipy.stats import jarque_bera, normaltest, shapiro

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "analyze_variable",
    "description": "Analyze a specific variable's distribution - summary stats, skewness, normality tests. Use for treatment, outcome, or suspicious variables.",
    "parameters": {
        "type": "object",
        "properties": {
            "variable": {
                "type": "string",
                "description": "Name of the variable to analyze",
            },
            "include_normality_tests": {
                "type": "boolean",
                "description": "Whether to run normality tests",
            },
        },
        "required": ["variable"],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    variable: str = "",
    include_normality_tests: bool = True,
    **kwargs,
) -> ToolResult:
    if agent._df is None:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="Dataset not loaded",
        )

    if variable not in agent._df.columns:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error=f"Variable '{variable}' not found",
        )

    data = agent._df[variable].dropna()
    n = len(data)

    if n == 0:
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={"variable": variable, "error": "No non-missing values"},
        )

    result_dict: dict = {
        "variable": variable,
        "n": n,
        "missing": int(len(agent._df) - n),
    }

    if pd.api.types.is_numeric_dtype(agent._df[variable]):
        result_dict["type"] = "numeric"
        result_dict["mean"] = round(float(data.mean()), 4)
        result_dict["median"] = round(float(data.median()), 4)
        result_dict["std"] = round(float(data.std()), 4)
        result_dict["min"] = round(float(data.min()), 4)
        result_dict["max"] = round(float(data.max()), 4)
        result_dict["q1"] = round(float(data.quantile(0.25)), 4)
        result_dict["q3"] = round(float(data.quantile(0.75)), 4)
        result_dict["skewness"] = round(float(stats.skew(data)), 3)
        result_dict["kurtosis"] = round(float(stats.kurtosis(data)), 3)
        result_dict["unique_values"] = int(data.nunique())

        skew = result_dict["skewness"]
        if abs(skew) > 1:
            result_dict["skewness_interpretation"] = "highly_skewed"
        elif abs(skew) > 0.5:
            result_dict["skewness_interpretation"] = "moderately_skewed"
        else:
            result_dict["skewness_interpretation"] = "approximately_symmetric"

        if include_normality_tests and n >= 8:
            normality: dict = {}
            try:
                if n <= 5000:
                    _, p = shapiro(data.sample(min(n, 5000), random_state=42))
                    normality["shapiro_p"] = round(float(p), 4)
                    normality["shapiro_normal"] = p > 0.05
                if n >= 20:
                    _, p = normaltest(data)
                    normality["dagostino_p"] = round(float(p), 4)
                    normality["dagostino_normal"] = p > 0.05
                _, p = jarque_bera(data)
                normality["jarque_bera_p"] = round(float(p), 4)
                normality["jarque_bera_normal"] = p > 0.05
            except Exception:
                agent.logger.debug("normality_test_skipped", variable=variable, exc_info=True)
            if normality:
                result_dict["normality_tests"] = normality

        agent._analyzed_distributions[variable] = result_dict
    else:
        result_dict["type"] = "categorical"
        value_counts = data.value_counts().head(10).to_dict()
        result_dict["unique_values"] = int(data.nunique())
        result_dict["top_values"] = {str(k): int(v) for k, v in value_counts.items()}

    return ToolResult(status=ToolResultStatus.SUCCESS, output=result_dict)
