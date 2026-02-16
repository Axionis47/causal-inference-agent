"""Data repairs section renderer."""

from nbformat.v4 import new_markdown_cell

from src.agents.base import AnalysisState


def render_data_repairs(state: AnalysisState) -> list:
    """Report data repair agent findings — what preprocessing was applied."""
    cells = []
    repairs = state.data_repairs
    if not repairs:
        return cells

    md = "## Data Preprocessing & Repairs\n\n"
    md += "*Findings from the Data Repair agent. "
    md += "These repairs were applied before causal analysis.*\n\n"

    # Repairs summary table
    md += "### Repairs Applied\n\n"
    md += "| # | Type | Strategy | Columns |\n"
    md += "|---|------|----------|---------|\n"
    for i, repair in enumerate(repairs, 1):
        rtype = repair.get("type", "unknown")
        strategy = repair.get("strategy", "unknown")
        columns = repair.get("columns", [])
        col_str = ", ".join(columns[:5])
        if len(columns) > 5:
            col_str += f" (+{len(columns) - 5} more)"
        md += f"| {i} | {rtype} | {strategy} | {col_str} |\n"
    md += "\n"

    # Detail each repair type
    missing_repairs = [r for r in repairs if r.get("type") == "missing"]
    outlier_repairs = [r for r in repairs if r.get("type") == "outliers"]
    collinearity_repairs = [r for r in repairs if r.get("type") == "collinearity"]

    if missing_repairs:
        md += "### Missing Value Handling\n\n"
        for r in missing_repairs:
            strategy = r.get("strategy", "unknown")
            columns = r.get("columns", [])
            md += f"- **Strategy**: {strategy}\n"
            md += f"- **Columns**: {', '.join(columns)}\n"
            if r.get("rows_dropped"):
                md += f"- **Rows dropped**: {r['rows_dropped']:,}\n"
            if r.get("before") is not None and r.get("after") is not None:
                md += f"- **Missing values**: {r['before']:,} → {r['after']:,}\n"
            md += "\n"

    if outlier_repairs:
        md += "### Outlier Treatment\n\n"
        for r in outlier_repairs:
            strategy = r.get("strategy", "unknown")
            columns = r.get("columns", [])
            md += f"- **Strategy**: {strategy}\n"
            md += f"- **Columns**: {', '.join(columns)}\n\n"

    if collinearity_repairs:
        md += "### Collinearity Resolution\n\n"
        for r in collinearity_repairs:
            strategy = r.get("strategy", "unknown")
            columns = r.get("columns", [])
            md += f"- **Strategy**: {strategy}\n"
            md += f"- **Columns removed/adjusted**: {', '.join(columns)}\n\n"

    # Quality assessment and cautions
    quality = None
    summary_lines = None
    cautions = None
    for r in repairs:
        if "quality_assessment" in r:
            quality = r["quality_assessment"]
        if "repairs_summary" in r:
            summary_lines = r["repairs_summary"]
        if "cautions" in r:
            cautions = r["cautions"]

    if quality:
        md += f"### Data Quality Assessment\n\n{quality}\n\n"

    if cautions:
        md += "### Cautions\n\n"
        md += "*These caveats may affect the validity of downstream results.*\n\n"
        for c in cautions:
            md += f"- {c}\n"
        md += "\n"

    cells.append(new_markdown_cell(md))
    return cells
