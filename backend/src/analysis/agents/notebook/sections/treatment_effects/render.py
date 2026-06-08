"""Main entry point for the treatment-effects section renderer.

Builds the results table, forest plot, and verification cells.
Per-method cells live in sibling modules so the renderer here can
stay focused on layout.
"""

import json

import numpy as np
from nbformat.v4 import new_code_cell, new_markdown_cell

from src.analysis.agents.base import AnalysisState

from ...helpers import deduplicate_effects
from .aipw import make_aipw_cell
from .dml import make_dml_cell
from .ipw import make_ipw_cell
from .ols import make_ols_cell

_METHOD_ALIASES = {
    "ipw": "ipw",
    "inverse probability weighting": "ipw",
    "inverse_probability_weighting": "ipw",
    "aipw": "aipw",
    "augmented ipw": "aipw",
    "augmented_ipw": "aipw",
    "doubly robust": "aipw",
    "doubly_robust": "aipw",
    "double_ml": "dml",
    "double ml": "dml",
    "doubleml": "dml",
    "dml": "dml",
}


def _holm_bonferroni(pvals: list[float | None]) -> list[float | None]:
    """Holm-Bonferroni adjustment preserving order and None entries."""
    valid = [(i, p) for i, p in enumerate(pvals) if p is not None]
    if len(valid) <= 1:
        return pvals

    valid_sorted = sorted(valid, key=lambda x: x[1])
    k = len(valid_sorted)

    adjusted: list[float | None] = [None] * len(pvals)
    cumulative_max = 0.0
    for rank, (orig_idx, p) in enumerate(valid_sorted):
        adj = p * (k - rank)
        cumulative_max = max(cumulative_max, adj)
        adjusted[orig_idx] = min(cumulative_max, 1.0)

    return adjusted


def _resolve_covariates(state: AnalysisState) -> tuple[list[str], str]:
    """Priority chain: DAG adjustment set > confounder discovery > profile."""
    treatment = state.treatment_variable
    outcome = state.outcome_variable

    def _exclude_t_y(cols: list[str]) -> list[str]:
        return [c for c in cols if c != treatment and c != outcome]

    if state.proposed_dag and state.proposed_dag.adjustment_set:
        covs = _exclude_t_y(state.proposed_dag.adjustment_set)
        if covs:
            return covs, "DAG adjustment set (backdoor criterion)"

    if state.confounder_discovery:
        ranked = state.confounder_discovery.get("ranked_confounders", [])
        if ranked:
            if isinstance(ranked[0], dict):
                names = [c.get("variable", c.get("name", "")) for c in ranked]
            else:
                names = list(ranked)
            covs = _exclude_t_y([n for n in names if n])
            if covs:
                return covs, "confounder discovery agent"

    if state.data_profile and state.data_profile.potential_confounders:
        covs = _exclude_t_y(state.data_profile.potential_confounders)
        if covs:
            return covs, "data profiler (potential confounders)"

    return [], "none (no confounders identified)"


def _method_family(method: str) -> str:
    """Canonical family key for a result.method string."""
    key = method.lower().replace("-", "_").replace(" ", "_")
    return _METHOD_ALIASES.get(key, "")


def render_treatment_effects(state: AnalysisState) -> list:
    """Report treatment effect estimation results with verification cells."""
    effects = deduplicate_effects(state.treatment_effects)

    # No usable estimates: skip the section rather than render an empty table.
    if not effects:
        from .._skip import render_skipped_cell

        return render_skipped_cell(
            "Treatment Effect Estimation",
            reason=(
                "No treatment effect estimates were produced for this run. "
                "All estimation methods either failed or were not attempted. "
                "Without estimates, downstream interpretation, sensitivity "
                "analysis, and the results conclusions cannot be grounded."
            ),
            upstream_agent="effect_estimator",
        )

    cells: list = []

    md = "## Treatment Effect Estimation\n\n"
    md += "*Results from the Effect Estimator agent.*\n\n"
    md += f"**Treatment**: {state.treatment_variable}\n"
    md += f"**Outcome**: {state.outcome_variable}\n"
    md += f"**Methods applied**: {len(effects)}\n\n"
    md += (
        "The results table and forest plot below display the estimates the "
        "pipeline computed. Further down, OLS is re-fit from the data as an "
        "independent check; the other methods are shown as the pipeline produced "
        "them, not recomputed in this notebook.\n\n"
    )
    cells.append(new_markdown_cell(md))

    caveat_md = (
        "> **Interpretation Note**: These are *estimated* treatment effects from "
        "observational data. They rely on the assumption that all relevant confounders "
        "have been measured and adjusted for (no unmeasured confounding). "
        "Results should be interpreted as associations adjusted for observed covariates, "
        "not as proof of causation.\n\n"
    )
    cells.append(new_markdown_cell(caveat_md))

    # Multiple testing correction
    raw_pvals = [e.p_value for e in effects]
    adjusted_pvals = _holm_bonferroni(raw_pvals)
    n_tests = sum(1 for p in raw_pvals if p is not None)

    # Results table
    table_md = "### Results Summary\n\n"
    if n_tests > 1:
        table_md += "| Method | Estimand | Estimate | Std Error | 95% CI | p-value | Adjusted p |\n"
        table_md += "|--------|----------|----------|-----------|--------|---------|------------|\n"
        for e, adj_p in zip(effects, adjusted_pvals):
            pval = f"{e.p_value:.4f}" if e.p_value is not None else "N/A"
            adj_pval = f"{adj_p:.4f}" if adj_p is not None else "N/A"
            table_md += (
                f"| {e.method} | {e.estimand} | {e.estimate:.4f} | "
                f"{e.std_error:.4f} | [{e.ci_lower:.4f}, {e.ci_upper:.4f}] | {pval} | {adj_pval} |\n"
            )
        table_md += f"\n*p-values adjusted for multiple comparisons (Holm-Bonferroni, k={n_tests}).*\n\n"
    else:
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

    # Verification cells
    covariates, cov_source = _resolve_covariates(state)

    if state.data_profile and state.data_profile.feature_types:
        numeric_types = {"numeric", "binary", "ordinal"}
        ft = state.data_profile.feature_types
        covariates = [c for c in covariates if ft.get(c) in numeric_types]

    covariates_json = json.dumps(covariates)

    cov_md = (
        "### Verification Cells\n\n"
        "The cells below independently re-estimate treatment effects using standard "
        "Python data-science packages (no backend imports). Each cell is self-contained.\n\n"
        f"**Covariates source**: {cov_source}\n"
        f"**Covariates ({len(covariates)})**: {', '.join(covariates) if covariates else 'none'}\n"
    )
    cells.append(new_markdown_cell(cov_md))

    # OLS verification always emitted.
    cells.append(new_markdown_cell(
        "#### Verification: OLS Regression\n\n"
        "Run this cell to independently verify the OLS estimate."
    ))
    cells.append(new_code_cell(make_ols_cell(state, covariates_json)))

    # Method-conditional verification cells.
    ipw_effects = [e for e in effects if _method_family(e.method) == "ipw"]
    if ipw_effects:
        e = ipw_effects[0]
        cells.append(new_markdown_cell(
            "#### Verification: Inverse Probability Weighting\n\n"
            "Hajek estimator with bootstrap SEs (200 iterations)."
        ))
        cells.append(new_code_cell(
            make_ipw_cell(state, covariates_json, e.estimate, e.std_error)
        ))

    aipw_effects = [e for e in effects if _method_family(e.method) == "aipw"]
    if aipw_effects:
        e = aipw_effects[0]
        cells.append(new_markdown_cell(
            "#### Verification: Augmented IPW (Doubly Robust)\n\n"
            "5-fold cross-fitted AIPW with influence function SEs."
        ))
        cells.append(new_code_cell(
            make_aipw_cell(state, covariates_json, e.estimate, e.std_error)
        ))

    dml_effects = [e for e in effects if _method_family(e.method) == "dml"]
    if dml_effects:
        e = dml_effects[0]
        cells.append(new_markdown_cell(
            "#### Verification: Double/Debiased ML\n\n"
            "Gradient boosting nuisance models with 5-fold cross-fitting."
        ))
        cells.append(new_code_cell(
            make_dml_cell(state, covariates_json, e.estimate, e.std_error)
        ))

    return cells
