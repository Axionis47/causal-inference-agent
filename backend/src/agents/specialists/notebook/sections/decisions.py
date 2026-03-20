"""Decision audit trail section for the generated notebook."""

from nbformat.v4 import new_markdown_cell

from src.agents.base import AnalysisState


def render_decisions(state: AnalysisState) -> list:
    """Render methodology decisions section."""
    if not hasattr(state, "decisions") or not state.decisions:
        return []

    cells = []

    md = "## Methodology Decisions\n\n"
    md += "The following key decisions were made during the automated analysis:\n\n"
    md += "| Agent | Decision | Rationale |\n"
    md += "|-------|----------|----------|\n"

    for d in state.decisions:
        # Clean up decision type for display
        dtype = d.decision_type.replace("_", " ").title()
        choice = d.choice
        reason = d.reason.replace("|", "\\|")  # Escape pipes for markdown table
        md += f"| {d.agent} | {dtype}: {choice} | {reason} |\n"

    cells.append(new_markdown_cell(md))

    # Add alternatives if any decisions have them
    alt_decisions = [d for d in state.decisions if d.alternatives]
    if alt_decisions:
        alt_md = "### Alternatives Considered\n\n"
        for d in alt_decisions:
            alt_md += f"**{d.choice}** was selected over:\n"
            for alt in d.alternatives:
                option = alt.get("option", alt.get("method", "unknown"))
                reason = alt.get("reason", "no reason given")
                alt_md += f"- {option}: {reason}\n"
            alt_md += "\n"
        cells.append(new_markdown_cell(alt_md))

    return cells
