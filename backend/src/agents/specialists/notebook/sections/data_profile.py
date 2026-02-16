"""Data profile report section renderer."""

from nbformat.v4 import new_markdown_cell

from src.agents.base import AnalysisState


def render_data_profile_report(state: AnalysisState) -> list:
    """Report data profiler agent findings."""
    cells = []
    profile = state.data_profile
    if not profile:
        return cells

    md = "## Data Profile\n\n"
    md += "*Findings from the Data Profiler agent.*\n\n"

    # Overview
    md += "| Property | Value |\n"
    md += "|----------|-------|\n"
    md += f"| Samples | {profile.n_samples:,} |\n"
    md += f"| Features | {profile.n_features} |\n"
    md += f"| Has Time Dimension | {'Yes' if profile.has_time_dimension else 'No'} |\n"
    if profile.time_column:
        md += f"| Time Column | {profile.time_column} |\n"
    md += "\n"

    # Feature types
    if profile.feature_types:
        md += "### Feature Types\n\n"
        md += "| Feature | Type |\n|---------|------|\n"
        for feat, ftype in sorted(profile.feature_types.items()):
            md += f"| {feat} | {ftype} |\n"
        md += "\n"

    # Missing values
    if profile.missing_values:
        has_missing = {k: v for k, v in profile.missing_values.items() if v > 0}
        if has_missing:
            md += "### Missing Values\n\n"
            md += "| Feature | Missing Count | % Missing |\n"
            md += "|---------|--------------|----------|\n"
            for feat, count in sorted(has_missing.items(), key=lambda x: -x[1]):
                pct = 100 * count / profile.n_samples if profile.n_samples > 0 else 0
                md += f"| {feat} | {count} | {pct:.1f}% |\n"
            md += "\n"
        else:
            md += "**No missing values detected.**\n\n"

    # Causal candidates
    md += "### Variable Roles (Identified by Profiler)\n\n"
    md += f"- **Treatment**: {state.treatment_variable}\n"
    md += f"- **Outcome**: {state.outcome_variable}\n"
    if profile.treatment_candidates:
        md += f"- **Treatment candidates**: {', '.join(profile.treatment_candidates)}\n"
    if profile.outcome_candidates:
        md += f"- **Outcome candidates**: {', '.join(profile.outcome_candidates)}\n"
    if profile.potential_confounders:
        md += f"- **Potential confounders**: {', '.join(profile.potential_confounders)}\n"
    if profile.potential_instruments:
        md += f"- **Potential instruments**: {', '.join(profile.potential_instruments)}\n"
    if profile.discontinuity_candidates:
        md += f"- **Discontinuity candidates**: {', '.join(profile.discontinuity_candidates)}\n"
    md += "\n"

    cells.append(new_markdown_cell(md))
    return cells
