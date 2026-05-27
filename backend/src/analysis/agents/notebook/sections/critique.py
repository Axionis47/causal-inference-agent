"""Critique section renderer."""

from nbformat.v4 import new_markdown_cell

from src.analysis.agents.base import AnalysisState

from ._skip import render_skipped_cell


def render_critique_section(state: AnalysisState) -> list:
    """Report critique agent findings."""
    if not state.critique_history:
        return render_skipped_cell(
            "Analysis Quality & Critique",
            reason=(
                "The Critique agent did not run, so the analysis was not "
                "audited for quality issues. Treat the rest of the report "
                "with corresponding caution."
            ),
            upstream_agent="critique_agent",
        )

    cells = []

    md = "## Analysis Quality & Critique\n\n"
    md += "*Assessment from the automated Critique agent.*\n\n"

    for critique in state.critique_history:
        md += f"### Iteration {critique.iteration}\n\n"
        md += f"**Decision**: **{critique.decision.value}**\n\n"

        # Quality scores
        if critique.scores:
            md += "**Quality Scores:**\n\n"
            md += "| Dimension | Score |\n|-----------|-------|\n"
            for dim, score in critique.scores.items():
                bar = "\u2588" * score + "\u2591" * (5 - score)
                md += f"| {dim.replace('_', ' ').title()} | {bar} {score}/5 |\n"
            md += "\n"

        # Issues
        if critique.issues:
            md += "**Issues Identified:**\n"
            for issue in critique.issues:
                md += f"- {issue}\n"
            md += "\n"

        # Improvements
        if critique.improvements:
            md += "**Improvements:**\n"
            for imp in critique.improvements:
                md += f"- {imp}\n"
            md += "\n"

        # Reasoning
        if critique.reasoning:
            md += f"**Reasoning**: {critique.reasoning[:500]}\n\n"

        md += "---\n\n"

    cells.append(new_markdown_cell(md))
    return cells
