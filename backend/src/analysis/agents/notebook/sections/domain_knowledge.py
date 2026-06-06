"""Domain knowledge section renderer."""

from nbformat.v4 import new_markdown_cell

from src.analysis.agents.base import AnalysisState

from ._skip import render_skipped_cell


def render_domain_knowledge(state: AnalysisState) -> list:
    """Report domain knowledge agent findings."""
    if not state.domain_knowledge:
        return render_skipped_cell(
            "Domain Knowledge & Causal Framework",
            reason=(
                "The Domain Knowledge agent did not produce a result for "
                "this run. Causal hypotheses, temporal ordering, and "
                "uncertainty notes were not extracted."
            ),
            upstream_agent="domain_knowledge",
        )

    cells = []
    dk = state.domain_knowledge

    md = "## Domain Knowledge & Causal Framework\n\n"
    md += "The domain knowledge agent extracted the following understanding from dataset metadata.\n\n"

    if dk.hypotheses:
        md += "### Causal Hypotheses\n\n"
        md += "| # | Hypothesis | Confidence | Evidence |\n"
        md += "|---|-----------|------------|----------|\n"
        for i, h in enumerate(dk.hypotheses, 1):
            md += f"| {i} | {h.claim} | {h.confidence} | {h.evidence[:80]} |\n"
        md += "\n"

    if dk.temporal_understanding:
        md += "### Temporal Ordering\n\n"
        md += f"{dk.temporal_understanding}\n\n"

    if dk.immutable_vars:
        md += "### Immutable Variables\n\n"
        md += "These variables cannot be caused by the treatment and are safe to condition on:\n\n"
        for var in dk.immutable_vars:
            md += f"- {var}\n"
        md += "\n"

    if dk.uncertainties:
        md += "### Uncertainties & Limitations\n\n"
        for u in dk.uncertainties:
            md += f"- {u.issue}\n"
        md += "\n"

    cells.append(new_markdown_cell(md))
    return cells
