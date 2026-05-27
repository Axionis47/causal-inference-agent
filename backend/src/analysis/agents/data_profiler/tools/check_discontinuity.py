"""check_discontinuity_candidates tool: find numeric columns suitable for RDD running variables."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "check_discontinuity_candidates",
    "description": "Check for running variables that could enable regression discontinuity design.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

_RDD_KEYWORDS = ["score", "threshold", "cutoff", "grade", "test", "rank", "percentile", "rating"]


async def handle(agent, state: AnalysisState, **kwargs) -> ToolResult:
    if kwargs:
        logger.debug("tool_ignored_kwargs", tool="check_discontinuity", extra_keys=list(kwargs.keys()))

    if agent._df is None or agent._profile is None:
        return ToolResult(status=ToolResultStatus.ERROR, output=None, error="Dataset not loaded")

    df = agent._df
    profile = agent._profile
    candidates: list[dict] = []

    for col in df.columns:
        col_lower = col.lower()
        if profile.feature_types.get(col) in ["numeric", "ordinal"]:
            for kw in _RDD_KEYWORDS:
                if kw in col_lower:
                    data = df[col].dropna()
                    candidates.append({
                        "column": col,
                        "matched_keyword": kw,
                        "range": [float(data.min()), float(data.max())],
                        "mean": float(data.mean()),
                        "median": float(data.median()),
                        "unique_values": int(data.nunique()),
                    })
                    break

    result: dict = {
        "has_rdd_candidates": len(candidates) > 0,
        "candidates": candidates,
    }

    if candidates:
        result["note"] = "RDD may be applicable if there's a known cutoff/threshold"
    else:
        result["note"] = "No obvious running variables found"

    return ToolResult(status=ToolResultStatus.SUCCESS, output=result)
