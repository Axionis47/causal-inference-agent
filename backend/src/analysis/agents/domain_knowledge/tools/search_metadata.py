"""search_metadata tool: keyword search across title, description, tags, column docs."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "search_metadata",
    "description": (
        "Search the metadata for specific keywords or phrases. "
        "Use this to find evidence for your hypotheses. "
        "Examples: 'random', 'treatment', 'outcome', 'baseline', 'before', 'after'"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Keyword or phrase to search for",
            }
        },
        "required": ["query"],
    },
}


async def handle(agent, state: AnalysisState, query: str = "", **kwargs) -> ToolResult:
    if kwargs:
        logger.debug("tool_ignored_kwargs", tool="search_metadata", extra_keys=list(kwargs.keys()))
    metadata = state.raw_metadata or {}

    searchable = ""
    searchable += metadata.get("title", "") + "\n"
    searchable += metadata.get("subtitle", "") + "\n"
    searchable += metadata.get("description", "") + "\n"
    searchable += " ".join(metadata.get("tags", [])) + "\n"
    searchable += " ".join(metadata.get("keywords", [])) + "\n"

    for col, desc in metadata.get("column_descriptions", {}).items():
        searchable += f"{col}: {desc}\n"

    query_lower = query.lower()
    matches = [line.strip() for line in searchable.split("\n") if query_lower in line.lower()]

    if matches:
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "query": query,
                "found": True,
                "matches": matches[:5],
                "total_matches": len(matches),
            },
        )

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "query": query,
            "found": False,
            "message": f"No matches found for '{query}' in metadata",
        },
    )
