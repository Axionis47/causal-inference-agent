"""Pure helpers for the domain knowledge agent."""

from src.analysis.agents.base import AnalysisState


def initial_observation_text(state: AnalysisState) -> str:
    """Build the lean initial observation the ReAct loop sees on step zero."""
    metadata = state.raw_metadata or {}
    return f"""You are investigating a new dataset for causal analysis.

Dataset: {metadata.get('title', state.dataset_info.name or 'Unknown')}
Source: Kaggle
Metadata quality: {metadata.get('metadata_quality', 'unknown')}

You have tools to investigate the metadata. Start by reading the description.
"""


def has_treatment_and_outcome_hypotheses(hypotheses: list[dict]) -> bool:
    """True iff we have at least one treatment and one outcome hypothesis at
    medium-or-better confidence. This is the completion signal for the agent.
    """
    has_treatment = any(
        "treatment" in h["claim"].lower() and h["confidence"] in ["medium", "high"]
        for h in hypotheses
    )
    has_outcome = any(
        "outcome" in h["claim"].lower() and h["confidence"] in ["medium", "high"]
        for h in hypotheses
    )
    return has_treatment and has_outcome
