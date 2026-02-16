"""EDA report section renderer."""

import json

from nbformat.v4 import new_code_cell, new_markdown_cell

from src.agents.base import AnalysisState


def render_eda_report(state: AnalysisState) -> list:
    """Report EDA agent findings with visualizations from pipeline data."""
    cells = []

    cells.append(new_markdown_cell(
        "## Exploratory Data Analysis\n\n*Findings from the EDA agent.*"
    ))

    eda = state.eda_result

    # Quality summary
    if eda:
        quality_md = "### Data Quality\n\n"
        quality_md += f"**Overall Quality Score**: {eda.data_quality_score:.1f}/100\n\n"
        if eda.data_quality_issues:
            quality_md += "**Issues Found:**\n"
            for issue in eda.data_quality_issues:
                quality_md += f"- {issue}\n"
        else:
            quality_md += "No significant data quality issues detected.\n"
        quality_md += "\n"
        cells.append(new_markdown_cell(quality_md))

    # High correlations
    if eda and eda.high_correlations:
        corr_md = "### High Correlations (|r| > 0.7)\n\n"
        corr_md += "| Variable 1 | Variable 2 | Correlation |\n"
        corr_md += "|-----------|-----------|------------|\n"
        for hc in eda.high_correlations[:15]:
            if isinstance(hc, dict):
                v1 = hc.get("var1", hc.get("column1", "?"))
                v2 = hc.get("var2", hc.get("column2", "?"))
                r = hc.get("correlation", hc.get("r", "?"))
                r_str = f"{r:.3f}" if isinstance(r, (int, float)) else str(r)
                corr_md += f"| {v1} | {v2} | {r_str} |\n"
        corr_md += "\n"
        cells.append(new_markdown_cell(corr_md))

    # Multicollinearity
    if eda and (eda.vif_scores or eda.multicollinearity_warnings):
        vif_md = "### Multicollinearity\n\n"
        if eda.multicollinearity_warnings:
            for w in eda.multicollinearity_warnings:
                vif_md += f"- {w}\n"
            vif_md += "\n"
        if eda.vif_scores:
            vif_md += "| Variable | VIF |\n|----------|-----|\n"
            for var, vif in sorted(
                eda.vif_scores.items(), key=lambda x: -x[1]
            )[:10]:
                severity = " **SEVERE**" if vif > 10 else " (moderate)" if vif > 5 else ""
                vif_md += f"| {var} | {vif:.2f}{severity} |\n"
            vif_md += "\n"
        cells.append(new_markdown_cell(vif_md))

    # Covariate balance
    if eda and eda.covariate_balance:
        bal_md = "### Covariate Balance\n\n"
        if eda.balance_summary:
            bal_md += f"{eda.balance_summary}\n\n"
        bal_md += "| Covariate | Treated Mean | Control Mean | SMD | Balanced |\n"
        bal_md += "|-----------|-------------|-------------|-----|----------|\n"
        for cov, vals in eda.covariate_balance.items():
            if isinstance(vals, dict):
                t_mean = vals.get("treated_mean", vals.get("mean_treated", "?"))
                c_mean = vals.get("control_mean", vals.get("mean_control", "?"))
                smd = vals.get("smd", vals.get("std_mean_diff", "?"))
                balanced = vals.get("is_balanced", vals.get("balanced", "?"))
                t_str = f"{t_mean:.3f}" if isinstance(t_mean, (int, float)) else str(t_mean)
                c_str = f"{c_mean:.3f}" if isinstance(c_mean, (int, float)) else str(c_mean)
                s_str = f"{smd:.3f}" if isinstance(smd, (int, float)) else str(smd)
                b_str = "Yes" if balanced else "No"
                bal_md += f"| {cov} | {t_str} | {c_str} | {s_str} | {b_str} |\n"
        bal_md += "\n"
        cells.append(new_markdown_cell(bal_md))

    # EDA summary findings
    if eda and eda.summary_table:
        if isinstance(eda.summary_table, dict):
            findings = eda.summary_table.get("key_findings", [])
            recs = eda.summary_table.get("recommendations", [])
            readiness = eda.summary_table.get("causal_readiness", "")

            if findings or recs or readiness:
                summary_md = "### EDA Summary\n\n"
                if findings:
                    summary_md += "**Key Findings:**\n"
                    for f in findings:
                        summary_md += f"- {f}\n"
                    summary_md += "\n"
                if recs:
                    summary_md += "**Recommendations:**\n"
                    for r in recs:
                        summary_md += f"- {r}\n"
                    summary_md += "\n"
                if readiness:
                    summary_md += f"**Causal Readiness**: {readiness}\n\n"
                cells.append(new_markdown_cell(summary_md))

    # Visualization: Distribution plots for treatment & outcome
    cells.append(new_markdown_cell("### Distribution Visualizations"))
    dist_code = f'''# Distribution plots for treatment and outcome
treatment_var = "{state.treatment_variable}"
outcome_var = "{state.outcome_variable}"

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Treatment distribution
ax = axes[0]
if df[treatment_var].nunique() <= 10:
    df[treatment_var].value_counts().sort_index().plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
else:
    df[treatment_var].hist(bins=30, ax=ax, color='steelblue', alpha=0.7, edgecolor='black')
ax.set_title(f'{{treatment_var}} (Treatment)')
ax.set_xlabel(treatment_var)
ax.set_ylabel('Count')

# Outcome distribution
ax = axes[1]
df[outcome_var].hist(bins=30, ax=ax, color='coral', alpha=0.7, edgecolor='black')
ax.set_title(f'{{outcome_var}} (Outcome)')
ax.set_xlabel(outcome_var)
ax.set_ylabel('Count')

plt.tight_layout()
plt.show()'''
    cells.append(new_code_cell(dist_code))

    # Correlation heatmap from pipeline data
    if eda and eda.correlation_matrix:
        cells.append(new_markdown_cell("### Correlation Heatmap"))
        corr_data = json.dumps(eda.correlation_matrix)
        heatmap_code = f'''# Correlation matrix from pipeline
import json
corr_data = json.loads('{corr_data}')
corr_df = pd.DataFrame(corr_data)

# Limit display to 15 variables
if len(corr_df.columns) > 15:
    cols = corr_df.columns[:15]
    corr_df = corr_df.loc[cols, cols]

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_df, dtype=bool))
sns.heatmap(corr_df, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax)
ax.set_title('Correlation Matrix (from EDA agent)')
plt.tight_layout()
plt.show()'''
        cells.append(new_code_cell(heatmap_code))

    return cells
