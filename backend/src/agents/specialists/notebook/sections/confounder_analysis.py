"""Confounder analysis section renderer."""

from nbformat.v4 import new_markdown_cell

from src.agents.base import AnalysisState


def render_confounder_analysis(state: AnalysisState) -> list:
    """Report confounder identification from the pipeline."""
    cells = []

    md = "## Confounder Analysis\n\n"
    md += "*Variables identified as potential confounders by the pipeline.*\n\n"

    # From confounder_discovery if available
    if state.confounder_discovery:
        cd = state.confounder_discovery
        ranked = cd.get("ranked_confounders", [])
        if ranked:
            md += "### Ranked Confounders\n\n"
            if isinstance(ranked[0], dict):
                md += "| Rank | Variable | Evidence |\n|------|----------|----------|\n"
                for i, c in enumerate(ranked, 1):
                    name = c.get("variable", c.get("name", str(c)))
                    evidence = c.get("evidence", c.get("reason", "-"))
                    md += f"| {i} | {name} | {evidence} |\n"
            else:
                md += "| Rank | Variable |\n|------|----------|\n"
                for i, c in enumerate(ranked, 1):
                    md += f"| {i} | {c} |\n"
            md += "\n"

        strategy = cd.get("adjustment_strategy", "")
        if strategy:
            md += f"**Adjustment Strategy**: {strategy}\n\n"

    # From data profile
    elif state.data_profile and state.data_profile.potential_confounders:
        confounders = state.data_profile.potential_confounders
        md += "### Potential Confounders (from Data Profile)\n\n"
        for c in confounders:
            ftype = state.data_profile.feature_types.get(c, "unknown") if state.data_profile.feature_types else "unknown"
            md += f"- **{c}** ({ftype})\n"
        md += "\n"

    # Analyzed pairs
    if state.analyzed_pairs:
        md += "### Analyzed Treatment-Outcome Pairs\n\n"
        md += "| Treatment | Outcome | Priority | Rationale |\n"
        md += "|-----------|---------|----------|-----------|\n"
        for pair in state.analyzed_pairs:
            md += f"| {pair.treatment} | {pair.outcome} | {pair.priority} | {pair.rationale} |\n"
        md += "\n"

    cells.append(new_markdown_cell(md))
    return cells
