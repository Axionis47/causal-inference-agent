"""Treatment effects section renderer."""

import json

from nbformat.v4 import new_code_cell, new_markdown_cell

from src.agents.base import AnalysisState

from ..helpers import deduplicate_effects


def render_treatment_effects(state: AnalysisState) -> list:
    """Report treatment effect estimation results."""
    cells = []

    effects = deduplicate_effects(state.treatment_effects)

    md = "## Treatment Effect Estimation\n\n"
    md += "*Results from the Effect Estimator agent.*\n\n"
    md += f"**Treatment**: {state.treatment_variable}\n"
    md += f"**Outcome**: {state.outcome_variable}\n"
    md += f"**Methods applied**: {len(effects)}\n\n"
    cells.append(new_markdown_cell(md))

    # Results table
    if effects:
        table_md = "### Results Summary\n\n"
        table_md += "| Method | Estimand | Estimate | Std Error | 95% CI | p-value |\n"
        table_md += "|--------|----------|----------|-----------|--------|--------|\n"
        for e in effects:
            pval = f"{e.p_value:.4f}" if e.p_value is not None else "N/A"
            table_md += (
                f"| {e.method} | {e.estimand} | {e.estimate:.4f} | "
                f"{e.std_error:.4f} | [{e.ci_lower:.4f}, {e.ci_upper:.4f}] | {pval} |\n"
            )
        table_md += "\n"
        cells.append(new_markdown_cell(table_md))

    # Per-method details
    for e in effects:
        if e.details or e.assumptions_tested:
            detail_md = f"#### {e.method} Details\n\n"
            if e.assumptions_tested:
                detail_md += "**Assumptions tested:**\n"
                for a in e.assumptions_tested:
                    detail_md += f"- {a}\n"
                detail_md += "\n"
            if e.details:
                detail_md += "**Diagnostics:**\n"
                for k, v in e.details.items():
                    if isinstance(v, float):
                        detail_md += f"- {k}: {v:.4f}\n"
                    elif not isinstance(v, (list, dict)):
                        detail_md += f"- {k}: {v}\n"
                detail_md += "\n"
            cells.append(new_markdown_cell(detail_md))

    # Forest plot
    if effects:
        cells.append(new_markdown_cell("### Treatment Effect Comparison"))
        results_json = json.dumps([
            {
                "method": e.method,
                "estimate": e.estimate,
                "ci_lower": e.ci_lower,
                "ci_upper": e.ci_upper,
            }
            for e in effects
        ])

        plot_code = f'''# Forest plot of treatment effect estimates
import json
results = json.loads('{results_json}')

fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 1.2)))

methods = [r['method'] for r in results]
estimates = [r['estimate'] for r in results]
ci_lower = [r['ci_lower'] for r in results]
ci_upper = [r['ci_upper'] for r in results]

y_pos = list(range(len(methods)))
xerr_lower = [e - l for e, l in zip(estimates, ci_lower)]
xerr_upper = [u - e for e, u in zip(estimates, ci_upper)]

# Traditional forest plot: point estimates with CI whiskers
ax.errorbar(estimates, y_pos, xerr=[xerr_lower, xerr_upper],
            fmt='o', color='steelblue', markersize=8, capsize=6,
            elinewidth=2, markeredgewidth=2)
ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, label='Zero effect')
ax.set_yticks(y_pos)
ax.set_yticklabels(methods)
ax.set_xlabel('Treatment Effect Estimate')
ax.set_title('Forest Plot: Treatment Effect Estimates Across Methods')
ax.legend()
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()'''
        cells.append(new_code_cell(plot_code))

    # Verification OLS code (reproducibility check)
    if effects:
        # Get numeric covariates only
        numeric_confounders: list[str] = []
        if state.data_profile and state.data_profile.potential_confounders:
            numeric_types = {"numeric", "binary", "ordinal"}
            ft = state.data_profile.feature_types or {}
            numeric_confounders = [
                c
                for c in state.data_profile.potential_confounders[:10]
                if ft.get(c) in numeric_types
                and c != state.treatment_variable
                and c != state.outcome_variable
            ]

        cells.append(new_markdown_cell(
            "### Verification: OLS Regression\n\n"
            "Run this cell to independently verify the OLS estimate."
        ))

        covariates_json = json.dumps(numeric_confounders)
        verify_code = f'''# Verification: OLS regression
import statsmodels.api as sm

treatment_var = "{state.treatment_variable}"
outcome_var = "{state.outcome_variable}"
covariates = {covariates_json}

# Filter to numeric covariates only (safety check)
covariates = [c for c in covariates if c in df.columns
              and pd.api.types.is_numeric_dtype(df[c])]

all_cols = [treatment_var, outcome_var] + covariates
df_clean = df[all_cols].dropna()

T = df_clean[treatment_var].values.astype(float)
Y = df_clean[outcome_var].values.astype(float)

# Binarize continuous treatment at median
if len(np.unique(T)) > 2:
    median_t = np.median(T)
    T_binary = (T > median_t).astype(int)
    print(f"Treatment binarized at median ({{median_t:.2f}})")
else:
    T_binary = T

if covariates:
    X = df_clean[covariates].values.astype(float)
    design = np.column_stack([np.ones(len(T_binary)), T_binary, X])
else:
    design = np.column_stack([np.ones(len(T_binary)), T_binary])

model = sm.OLS(Y, design)
results = model.fit()

print(f"\\nVerification OLS Results:")
print(f"  ATE:      {{results.params[1]:.4f}}")
print(f"  SE:       {{results.bse[1]:.4f}}")
print(f"  95% CI:   [{{results.conf_int()[1][0]:.4f}}, {{results.conf_int()[1][1]:.4f}}]")
print(f"  p-value:  {{results.pvalues[1]:.4f}}")
print(f"  R-squared: {{results.rsquared:.4f}}")
print(f"  N:        {{len(df_clean)}}")'''
        cells.append(new_code_cell(verify_code))

    return cells
