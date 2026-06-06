"""analyze_domain - pull domain hints from Kaggle metadata for downstream reasoning."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from ..helpers import patterns_for_domain

SCHEMA = {
    "name": "analyze_domain",
    "description": (
        "Analyze the dataset domain from metadata to understand "
        "the causal context. Returns domain type, typical causal patterns, "
        "and variable role hints."
    ),
    "parameters": {"type": "object", "properties": {}},
}


async def handle(agent, state: AnalysisState, **kwargs) -> ToolResult:
    domain_info: dict = {
        "domain": state.dataset_info.kaggle_domain or "unknown",
        "tags": state.dataset_info.kaggle_tags or [],
        "description": None,
        "typical_patterns": [],
        "variable_hints": {},
    }

    if state.dataset_info.kaggle_description:
        domain_info["description"] = state.dataset_info.kaggle_description[:500]
    elif state.raw_metadata:
        desc = state.raw_metadata.get("description", "")
        domain_info["description"] = desc[:500] if desc else None

    domain_info["typical_patterns"] = patterns_for_domain(domain_info["domain"])

    if state.dataset_info.kaggle_column_descriptions:
        for col, desc in state.dataset_info.kaggle_column_descriptions.items():
            domain_info["variable_hints"][col] = desc

    return ToolResult(status=ToolResultStatus.SUCCESS, output=domain_info)
