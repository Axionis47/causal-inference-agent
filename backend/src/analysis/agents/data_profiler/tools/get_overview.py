"""get_dataset_overview tool: dataset shape, types, missing-value summary."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "get_dataset_overview",
    "description": (
        "Get overview of the dataset including shape, columns, types, and missing values. "
        "Call this early to understand the data structure."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


async def handle(agent, state: AnalysisState, **kwargs) -> ToolResult:
    if kwargs:
        logger.debug("tool_ignored_kwargs", tool="get_overview", extra_keys=list(kwargs.keys()))

    if agent._df is None or agent._profile is None:
        return ToolResult(status=ToolResultStatus.ERROR, output=None, error="Dataset not loaded")

    df = agent._df
    profile = agent._profile

    binary_cols = [c for c, t in profile.feature_types.items() if t == "binary"]
    numeric_cols = [c for c, t in profile.feature_types.items() if t == "numeric"]
    ordinal_cols = [c for c, t in profile.feature_types.items() if t == "ordinal"]
    categorical_cols = [c for c, t in profile.feature_types.items() if t == "categorical"]
    datetime_cols = [c for c, t in profile.feature_types.items() if t == "datetime"]
    text_cols = [c for c, t in profile.feature_types.items() if t == "text"]

    cols_with_missing = {c: v for c, v in profile.missing_values.items() if v > 0}
    total_cells = df.shape[0] * df.shape[1]
    total_missing_pct = (sum(cols_with_missing.values()) / total_cells * 100) if total_cells > 0 else 0.0

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "n_samples": profile.n_samples,
            "n_features": profile.n_features,
            "total_missing_pct": round(total_missing_pct, 2),
            "binary_columns": binary_cols[:10],
            "numeric_columns": numeric_cols[:10],
            "ordinal_columns": ordinal_cols[:5],
            "categorical_columns": categorical_cols[:5],
            "datetime_columns": datetime_cols,
            "text_columns": text_cols[:3],
            "columns_with_missing": list(cols_with_missing.keys())[:10],
            "suggestions": [
                "Binary columns are good TREATMENT candidates - check balance",
                "Numeric columns are good OUTCOME candidates - check variance",
                "Query domain knowledge for hints before investigating",
            ],
        },
    )
