"""check_treatment_balance tool: assess a column's suitability as a treatment variable."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "check_treatment_balance",
    "description": (
        "Check if a column is suitable as a treatment variable by examining its balance "
        "and distribution. Binary treatments need 10-50% in minority class."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "column": {"type": "string", "description": "Name of the potential treatment column"},
        },
        "required": ["column"],
    },
}


async def handle(agent, state: AnalysisState, column: str = "", **kwargs) -> ToolResult:
    if kwargs:
        logger.debug("tool_ignored_kwargs", tool="check_treatment_balance", extra_keys=list(kwargs.keys()))

    if agent._df is None:
        return ToolResult(status=ToolResultStatus.ERROR, output=None, error="Dataset not loaded")

    if column not in agent._df.columns:
        return ToolResult(status=ToolResultStatus.ERROR, output=None, error=f"Column '{column}' not found")

    df = agent._df
    data = df[column].dropna()
    n = len(data)
    unique_vals = data.unique()
    n_unique = len(unique_vals)

    result: dict = {
        "column": column,
        "n_unique": n_unique,
        "value_counts": data.value_counts().head(10).to_dict(),
    }

    if n_unique == 2:
        value_counts = data.value_counts()
        minority_pct = value_counts.min() / n * 100
        majority_pct = value_counts.max() / n * 100

        result["treatment_type"] = "binary"
        result["minority_pct"] = round(minority_pct, 1)
        result["majority_pct"] = round(majority_pct, 1)

        if minority_pct >= 20:
            result["assessment"] = "EXCELLENT"
            result["suitable_methods"] = ["IPW", "Matching", "AIPW", "DML"]
        elif minority_pct >= 10:
            result["assessment"] = "GOOD"
            result["suitable_methods"] = ["IPW", "Matching", "AIPW"]
        elif minority_pct >= 5:
            result["assessment"] = "MODERATE"
            result["suitable_methods"] = ["Matching", "Stabilized Weights"]
        else:
            result["assessment"] = "POOR"
            result["suitable_methods"] = ["Consider alternative treatment definition"]

    elif 2 < n_unique <= 5:
        result["treatment_type"] = "multi-level"
        result["assessment"] = "USABLE"
        result["values"] = [str(v) for v in unique_vals]
        result["note"] = (
            "Multi-level categorical treatment. Set treatment_encoding_strategy='collapse_to_binary' "
            "and treatment_control_value to the control/reference category (e.g., the 'no treatment' group). "
            f"Available values: {[str(v) for v in unique_vals]}"
        )
        if data.dtype == object:
            result["requires_encoding"] = True

    elif n_unique > 5 and n_unique <= 20:
        result["treatment_type"] = "categorical"
        result["assessment"] = "CONSIDER_COLLAPSING"
        result["values"] = [str(v) for v in unique_vals[:10]]
        result["note"] = "Consider collapsing categories for treatment analysis"

    else:
        result["treatment_type"] = "continuous"
        result["assessment"] = "DOSE_RESPONSE"
        result["note"] = "Can be used for dose-response analysis or discretized"

    return ToolResult(status=ToolResultStatus.SUCCESS, output=result)
