"""get_tags tool: surface dataset tags/keywords and infer a domain hint."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "get_tags",
    "description": (
        "Get the dataset tags/categories. Tags indicate the domain "
        "(healthcare, economics, etc.) which helps interpret variables."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

_DOMAIN_RULES: list[tuple[tuple[str, ...], str]] = [
    (("healthcare", "health", "medical", "medicine", "clinical"), "Healthcare/Medical domain"),
    (("economics", "economic", "finance", "income", "employment"), "Economics/Finance domain"),
    (("education", "school", "student", "learning"), "Education domain"),
    (("social", "sociology", "demographics"), "Social science domain"),
    (("causal", "treatment", "experiment", "rct"), "Explicitly tagged as causal/experimental"),
]


async def handle(agent, state: AnalysisState, **kwargs) -> ToolResult:
    if kwargs:
        logger.debug("tool_ignored_kwargs", tool="get_tags", extra_keys=list(kwargs.keys()))
    metadata = state.raw_metadata or {}

    tags = metadata.get("tags", [])
    keywords = metadata.get("keywords", [])
    all_tags = [t.lower() for t in tags + keywords]

    domain_hints = [hint for keywords_set, hint in _DOMAIN_RULES if any(k in all_tags for k in keywords_set)]

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "tags": tags,
            "keywords": keywords,
            "domain_hints": domain_hints if domain_hints else ["Domain not clear from tags"],
        },
    )
