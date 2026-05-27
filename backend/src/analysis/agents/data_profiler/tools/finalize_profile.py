"""finalize_profile tool: agent's commitment to a causal structure."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "finalize_profile",
    "description": (
        "Finalize the data profile with identified causal structure. "
        "Call this ONLY after you have investigated and verified the structure."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "treatment_candidates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Columns suitable as treatment variables (best first)",
            },
            "outcome_candidates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Columns suitable as outcome variables (best first)",
            },
            "potential_confounders": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Columns that could be confounders",
            },
            "potential_instruments": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Columns that could be instrumental variables",
            },
            "has_time_dimension": {
                "type": "boolean",
                "description": "Whether dataset has a time dimension",
            },
            "time_column": {
                "type": "string",
                "description": "Name of time column if exists",
            },
            "discontinuity_candidates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Columns suitable for RDD running variable",
            },
            "recommended_methods": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Causal inference methods suitable for this data",
            },
            "treatment_encoding_strategy": {
                "type": "string",
                "enum": ["none", "label_encode", "collapse_to_binary"],
                "description": (
                    "How to encode the treatment variable. 'none' for numeric/binary treatments, "
                    "'label_encode' for 2-level string treatments, 'collapse_to_binary' for 3+ level "
                    "categorical treatments"
                ),
            },
            "treatment_control_value": {
                "type": "string",
                "description": (
                    "For collapse_to_binary: the string value representing the control/reference group "
                    "(e.g., 'No E-Mail', 'Control', 'Placebo'). This value becomes 0, all others become 1."
                ),
            },
        },
        "required": ["treatment_candidates", "outcome_candidates", "potential_confounders"],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    treatment_candidates: list[str] | None = None,
    outcome_candidates: list[str] | None = None,
    potential_confounders: list[str] | None = None,
    potential_instruments: list[str] | None = None,
    has_time_dimension: bool = False,
    time_column: str | None = None,
    discontinuity_candidates: list[str] | None = None,
    recommended_methods: list[str] | None = None,
    treatment_encoding_strategy: str | None = None,
    treatment_control_value: str | None = None,
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug("tool_ignored_kwargs", tool="finalize_profile", extra_keys=list(kwargs.keys()))

    treatment_candidates = treatment_candidates or []
    outcome_candidates = outcome_candidates or []
    potential_confounders = potential_confounders or []

    agent.logger.info(
        "profiler_finalizing",
        treatment_candidates=treatment_candidates,
        outcome_candidates=outcome_candidates,
        confounders=potential_confounders[:5],
        recommended_methods=recommended_methods or [],
        treatment_encoding_strategy=treatment_encoding_strategy,
    )

    agent._final_result = {
        "treatment_candidates": treatment_candidates,
        "outcome_candidates": outcome_candidates,
        "potential_confounders": potential_confounders,
        "potential_instruments": potential_instruments or [],
        "has_time_dimension": has_time_dimension,
        "time_column": time_column,
        "discontinuity_candidates": discontinuity_candidates or [],
        "recommended_methods": recommended_methods or ["IPW", "AIPW", "Matching"],
        "treatment_encoding_strategy": treatment_encoding_strategy,
        "treatment_control_value": treatment_control_value,
    }
    agent._finalized = True

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "profile_finalized": True,
            "treatment_candidates": treatment_candidates,
            "outcome_candidates": outcome_candidates,
            "n_confounders": len(potential_confounders),
            "recommended_methods": recommended_methods or ["IPW", "AIPW", "Matching"],
            "treatment_encoding_strategy": treatment_encoding_strategy or "none",
        },
    )
