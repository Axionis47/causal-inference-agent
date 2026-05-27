"""read_description tool: surface the dataset description text."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "read_description",
    "description": (
        "Read the dataset description. This often contains crucial information "
        "about what the study is, how data was collected, and what variables mean. "
        "Call this FIRST."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


async def handle(agent, state: AnalysisState, **kwargs) -> ToolResult:
    if kwargs:
        logger.debug("tool_ignored_kwargs", tool="read_description", extra_keys=list(kwargs.keys()))
    metadata = state.raw_metadata or {}

    description = metadata.get("description", "")
    subtitle = metadata.get("subtitle", "")
    title = metadata.get("title", "")

    if not description and not subtitle:
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "title": title,
                "description": "No description available",
                "has_description": False,
            },
        )

    full_text = f"{title}\n\n{subtitle}\n\n{description}" if subtitle else f"{title}\n\n{description}"

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "title": title,
            "description": full_text.strip(),
            "has_description": True,
            "length": len(description),
        },
    )
