"""investigate_column tool: name-heuristics + metadata lookup for one column."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "investigate_column",
    "description": (
        "Investigate what a specific column might mean based on its name "
        "and any available description. Use this to understand potential "
        "treatment, outcome, or confounder variables."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "column": {
                "type": "string",
                "description": "Name of the column to investigate",
            }
        },
        "required": ["column"],
    },
}


_HINT_RULES: list[tuple[tuple[str, ...], str]] = [
    (("treat", "intervention", "program", "policy", "exposed"),
     "Name suggests this could be a TREATMENT variable"),
    (("outcome", "result", "earnings", "income", "score", "effect", "response"),
     "Name suggests this could be an OUTCOME variable"),
    (("age", "sex", "gender", "race", "ethnicity", "birth", "dob"),
     "Name suggests this is a DEMOGRAPHIC/IMMUTABLE variable"),
    (("year", "month", "date", "time", "period", "wave"),
     "Name suggests this is a TIME variable"),
    (("id", "index", "key", "code"),
     "Name suggests this is an IDENTIFIER (not for analysis)"),
    (("pre", "before", "baseline", "prior"),
     "Name suggests this is measured BEFORE treatment (potential confounder)"),
    (("post", "after", "follow"),
     "Name suggests this is measured AFTER treatment"),
]


async def handle(agent, state: AnalysisState, column: str = "", **kwargs) -> ToolResult:
    if kwargs:
        logger.debug("tool_ignored_kwargs", tool="investigate_column", extra_keys=list(kwargs.keys()))
    metadata = state.raw_metadata or {}

    col_descriptions = metadata.get("column_descriptions", {})
    description = col_descriptions.get(column, None)

    name_lower = column.lower()
    clues = [hint for keywords, hint in _HINT_RULES if any(k in name_lower for k in keywords)]

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "column": column,
            "description": description,
            "has_description": description is not None,
            "name_clues": clues if clues else ["No obvious clues from name"],
        },
    )
