"""Domain knowledge section renderer."""

from nbformat.v4 import new_markdown_cell

from src.agents.base import AnalysisState


def render_domain_knowledge(state: AnalysisState) -> list:
    """Report domain knowledge agent findings."""
    cells = []
    dk = state.domain_knowledge

    md = "## Domain Knowledge & Causal Framework\n\n"
    md += "The domain knowledge agent extracted the following understanding from dataset metadata.\n\n"

    # Hypotheses
    hypotheses = dk.get("hypotheses", [])
    if hypotheses:
        md += "### Causal Hypotheses\n\n"
        md += "| # | Hypothesis | Confidence | Evidence |\n"
        md += "|---|-----------|------------|----------|\n"
        for i, h in enumerate(hypotheses, 1):
            if isinstance(h, dict):
                claim = h.get("claim", h.get("hypothesis", str(h)))
                conf = h.get("confidence", "unknown")
                evidence = h.get("evidence", "")
                md += f"| {i} | {claim} | {conf} | {evidence[:80]} |\n"
            else:
                md += f"| {i} | {h} | - | - |\n"
        md += "\n"

    # Temporal ordering
    temporal = dk.get("temporal_understanding")
    if temporal:
        md += "### Temporal Ordering\n\n"
        if isinstance(temporal, dict):
            for key, val in temporal.items():
                md += f"- **{key}**: {val}\n"
        else:
            md += f"{temporal}\n"
        md += "\n"

    # Immutable variables
    immutable = dk.get("immutable_vars", [])
    if immutable:
        md += "### Immutable Variables\n\n"
        md += "These variables cannot be caused by the treatment and are safe to condition on:\n\n"
        for var in immutable:
            if isinstance(var, dict):
                md += f"- **{var.get('name', var.get('variable', str(var)))}**: {var.get('reason', '')}\n"
            else:
                md += f"- {var}\n"
        md += "\n"

    # Uncertainties
    uncertainties = dk.get("uncertainties", [])
    if uncertainties:
        md += "### Uncertainties & Limitations\n\n"
        for u in uncertainties:
            if isinstance(u, dict):
                md += f"- {u.get('issue', u.get('description', str(u)))}\n"
            else:
                md += f"- {u}\n"
        md += "\n"

    cells.append(new_markdown_cell(md))
    return cells
