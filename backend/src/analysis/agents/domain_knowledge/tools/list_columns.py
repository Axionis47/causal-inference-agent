"""list_columns tool: list dataset columns from profile or metadata."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "list_columns",
    "description": (
        "Get list of all column names in the dataset. "
        "Column names often reveal what variables represent."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


async def handle(agent, state: AnalysisState, **kwargs) -> ToolResult:
    if kwargs:
        logger.debug("tool_ignored_kwargs", tool="list_columns", extra_keys=list(kwargs.keys()))

    if state.data_profile:
        columns = state.data_profile.feature_names
    else:
        metadata = state.raw_metadata or {}
        columns = list(metadata.get("column_descriptions", {}).keys())
        if not columns:
            files = metadata.get("files", [])
            if files:
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output={
                        "columns": [],
                        "message": f"Column names not available yet. Dataset has {len(files)} files.",
                    },
                )
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "columns": [],
                    "message": "Column names not available in metadata. Will be discovered during profiling.",
                },
            )

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={"columns": columns, "count": len(columns)},
    )
