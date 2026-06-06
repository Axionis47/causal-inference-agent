"""check_column_relationship tool: correlation or chi-square between two columns."""

import numpy as np
import pandas as pd
from scipy import stats

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "check_column_relationship",
    "description": (
        "Check relationship between two columns (correlation for numeric, association "
        "for categorical). Use to identify confounders."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "column1": {"type": "string", "description": "First column name"},
            "column2": {"type": "string", "description": "Second column name"},
        },
        "required": ["column1", "column2"],
    },
}


async def handle(
    agent, state: AnalysisState, column1: str = "", column2: str = "", **kwargs
) -> ToolResult:
    if kwargs:
        logger.debug("tool_ignored_kwargs", tool="check_relationship", extra_keys=list(kwargs.keys()))

    if agent._df is None:
        return ToolResult(status=ToolResultStatus.ERROR, output=None, error="Dataset not loaded")
    if column1 not in agent._df.columns:
        return ToolResult(status=ToolResultStatus.ERROR, output=None, error=f"Column '{column1}' not found")
    if column2 not in agent._df.columns:
        return ToolResult(status=ToolResultStatus.ERROR, output=None, error=f"Column '{column2}' not found")

    df = agent._df
    type1 = agent._profile.feature_types.get(column1, "unknown")
    type2 = agent._profile.feature_types.get(column2, "unknown")

    valid_data = df[[column1, column2]].dropna()
    n = len(valid_data)

    if n < 10:
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={"error": "Not enough complete cases", "n": n},
        )

    result: dict = {
        "column1": column1,
        "column2": column2,
        "type1": type1,
        "type2": type2,
        "complete_cases": n,
    }

    if type1 == "binary" and type2 in ["numeric", "ordinal"]:
        group_0 = valid_data[valid_data[column1] == valid_data[column1].min()][column2]
        group_1 = valid_data[valid_data[column1] == valid_data[column1].max()][column2]

        result["group_0_mean"] = round(float(group_0.mean()), 3)
        result["group_1_mean"] = round(float(group_1.mean()), 3)
        result["mean_difference"] = round(float(group_1.mean() - group_0.mean()), 3)

        t_stat, p_value = stats.ttest_ind(group_0, group_1)
        result["t_statistic"] = round(float(t_stat), 3)
        result["p_value"] = round(float(p_value), 4)

    elif type1 in ["numeric", "ordinal"] and type2 in ["numeric", "ordinal", "binary"]:
        corr, p_value = stats.pearsonr(valid_data[column1], valid_data[column2])
        result["pearson_correlation"] = round(float(corr), 3)
        result["p_value"] = round(float(p_value), 4)

        if abs(corr) > 0.7:
            result["strength"] = "STRONG"
        elif abs(corr) > 0.3:
            result["strength"] = "MODERATE"
        else:
            result["strength"] = "WEAK"

        spearman_corr, _ = stats.spearmanr(valid_data[column1], valid_data[column2])
        result["spearman_correlation"] = round(float(spearman_corr), 3)

    elif type1 in ["categorical", "binary"] and type2 in ["categorical", "binary"]:
        contingency = pd.crosstab(valid_data[column1], valid_data[column2])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        result["chi_squared"] = round(float(chi2), 3)
        result["p_value"] = round(float(p_value), 4)

        n_total = contingency.sum().sum()
        min_dim = min(contingency.shape) - 1
        cramers_v = np.sqrt(chi2 / (n_total * min_dim)) if min_dim > 0 else 0
        result["cramers_v"] = round(float(cramers_v), 3)

    else:
        result["note"] = "Relationship analysis not available for this combination"

    return ToolResult(status=ToolResultStatus.SUCCESS, output=result)
