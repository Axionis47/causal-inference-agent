"""Pure helpers for the domain knowledge agent."""

from src.analysis.agents.base import AnalysisState


def initial_observation_text(state: AnalysisState) -> str:
    """Build the lean initial observation the ReAct loop sees on step zero.

    Surfaces every metadata signal the agent could use: Kaggle's authored
    description + subtitle, the analyst's optional user_provided_context,
    and the user-designated treatment / outcome pair. The agent's tools
    let it pull more detail (column descriptions, tags, etc.) on demand.
    """
    metadata = state.raw_metadata or {}
    ds = state.dataset_info

    lines = [
        "You are investigating a new dataset for causal analysis.",
        "",
        f"Dataset: {metadata.get('title', ds.name or 'Unknown')}",
        "Source: Kaggle",
        f"Metadata quality: {metadata.get('metadata_quality', 'unknown')}",
    ]

    if ds.kaggle_subtitle:
        lines.append(f"Subtitle: {ds.kaggle_subtitle}")
    if state.treatment_variable or state.outcome_variable:
        lines.append(
            "User-designated pair: "
            f"treatment={state.treatment_variable or '?'}, "
            f"outcome={state.outcome_variable or '?'}"
        )
    # Analyst-provided context is pushed into every agent's initial observation
    # by the base ReAct agent (analyst_note_block), so it is not repeated here.

    lines.append("")
    lines.append(
        "You have tools to investigate the metadata. "
        "Start by reading the description."
    )
    return "\n".join(lines)


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
