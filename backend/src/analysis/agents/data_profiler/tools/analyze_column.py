"""analyze_column tool: distribution, statistics, and suitability hints for one column."""

from scipy import stats

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "analyze_column",
    "description": (
        "Analyze a specific column in detail - distribution, unique values, statistics. "
        "Use to verify if a column matches domain hints."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "column": {"type": "string", "description": "Name of the column to analyze"},
        },
        "required": ["column"],
    },
}


async def handle(agent, state: AnalysisState, column: str = "", **kwargs) -> ToolResult:
    if kwargs:
        logger.debug("tool_ignored_kwargs", tool="analyze_column", extra_keys=list(kwargs.keys()))

    if agent._df is None:
        return ToolResult(status=ToolResultStatus.ERROR, output=None, error="Dataset not loaded")

    if column not in agent._df.columns:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error=f"Column '{column}' not found. Available: {list(agent._df.columns)[:10]}...",
        )

    df = agent._df
    col_type = agent._profile.feature_types.get(column, "unknown")
    data = df[column].dropna()
    n = len(data)
    missing = df[column].isnull().sum()

    analysis = {
        "column": column,
        "type": col_type,
        "non_missing": n,
        "missing": missing,
        "missing_pct": round(missing / len(df) * 100, 1),
        "unique_values": int(data.nunique()),
    }

    if col_type in ["binary", "ordinal", "categorical"]:
        value_counts = data.value_counts().head(10).to_dict()
        analysis["value_distribution"] = {str(k): int(v) for k, v in value_counts.items()}

        if col_type == "binary" and data.nunique() == 2:
            min_class = data.value_counts().min()
            balance = min_class / n
            analysis["treatment_suitability"] = {
                "minority_class_pct": round(balance * 100, 1),
                "is_balanced": 0.1 <= balance <= 0.5,
                "assessment": "GOOD" if 0.1 <= balance <= 0.5 else "IMBALANCED",
            }

    elif col_type == "numeric":
        analysis["statistics"] = {
            "mean": round(float(data.mean()), 4),
            "std": round(float(data.std()), 4),
            "min": round(float(data.min()), 4),
            "max": round(float(data.max()), 4),
            "median": round(float(data.median()), 4),
        }
        skew = stats.skew(data)
        analysis["skewness"] = round(float(skew), 3)
        analysis["has_variance"] = data.std() > 0
        analysis["outcome_suitability"] = "GOOD" if data.std() > 0 else "NO_VARIANCE"

    elif col_type == "datetime":
        analysis["time_range"] = {"min": str(data.min()), "max": str(data.max())}

    return ToolResult(status=ToolResultStatus.SUCCESS, output=analysis)
