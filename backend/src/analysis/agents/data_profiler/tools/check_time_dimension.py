"""check_time_dimension tool: find columns that signal time / panel structure."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "check_time_dimension",
    "description": "Check if the dataset has a time dimension suitable for DiD or panel methods.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

_TIME_KEYWORDS = ["time", "date", "year", "month", "period", "quarter", "week", "day"]


async def handle(agent, state: AnalysisState, **kwargs) -> ToolResult:
    if kwargs:
        logger.debug("tool_ignored_kwargs", tool="check_time_dimension", extra_keys=list(kwargs.keys()))

    if agent._df is None or agent._profile is None:
        return ToolResult(status=ToolResultStatus.ERROR, output=None, error="Dataset not loaded")

    df = agent._df
    profile = agent._profile
    time_candidates: list[dict] = []

    for col in df.columns:
        col_lower = col.lower()

        if profile.feature_types.get(col) == "datetime":
            time_candidates.append({
                "column": col,
                "type": "datetime",
                "reason": "Datetime type",
                "unique_values": int(df[col].nunique()),
            })
            continue

        for kw in _TIME_KEYWORDS:
            if kw in col_lower:
                if profile.feature_types.get(col) in ["numeric", "ordinal"]:
                    data = df[col].dropna()
                    time_candidates.append({
                        "column": col,
                        "type": profile.feature_types.get(col),
                        "reason": f"Contains '{kw}'",
                        "unique_values": int(data.nunique()),
                        "range": [float(data.min()), float(data.max())],
                    })
                break

    result: dict = {
        "has_time_dimension": len(time_candidates) > 0,
        "candidates": time_candidates,
    }

    if time_candidates:
        result["methods_enabled"] = ["DiD", "Panel methods", "Event study"]
    else:
        result["note"] = "No obvious time dimension found"

    return ToolResult(status=ToolResultStatus.SUCCESS, output=result)
