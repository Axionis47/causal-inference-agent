"""Confounder analysis section renderer.

Reports identified confounders and includes an executable SMD love plot
for covariate balance verification.
"""

import json

import numpy as np
from nbformat.v4 import new_code_cell, new_markdown_cell

from src.agents.base import AnalysisState


def _resolve_covariates_for_balance(state: AnalysisState) -> list[str]:
    """Get covariates for balance checking (same priority chain as treatment_effects)."""
    treatment = state.treatment_variable
    outcome = state.outcome_variable

    def _exclude(cols: list[str]) -> list[str]:
        return [c for c in cols if c != treatment and c != outcome]

    if state.proposed_dag and state.proposed_dag.adjustment_set:
        covs = _exclude(state.proposed_dag.adjustment_set)
        if covs:
            return covs

    if state.confounder_discovery:
        ranked = state.confounder_discovery.get("ranked_confounders", [])
        if ranked:
            if isinstance(ranked[0], dict):
                names = [c.get("variable", c.get("name", "")) for c in ranked]
            else:
                names = list(ranked)
            covs = _exclude([n for n in names if n])
            if covs:
                return covs

    if state.data_profile and state.data_profile.potential_confounders:
        covs = _exclude(state.data_profile.potential_confounders)
        if covs:
            return covs

    return []


def render_confounder_analysis(state: AnalysisState) -> list:
    """Report confounder identification and add balance verification plot."""
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

    # ── Executable: SMD Love Plot ──────────────────────────────
    covariates = _resolve_covariates_for_balance(state)
    if covariates and state.treatment_variable:
        covariates_json = json.dumps(covariates)
        treatment_var = state.treatment_variable

        # Binarization code (consistent with treatment_effects)
        if state.treatment_encoding and state.treatment_encoding.value_mapping:
            mapping = state.treatment_encoding.value_mapping
            binarize = (
                f"_mapping = {repr(mapping)}\n"
                f"T = df['{treatment_var}'].map(_mapping).values.astype(float)\n"
                f"T_binary = T\n"
            )
        elif state.treatment_binarization_threshold is not None:
            thr = state.treatment_binarization_threshold
            binarize = (
                f"T = df['{treatment_var}'].values.astype(float)\n"
                f"T_binary = (T > {thr}).astype(int)\n"
            )
        else:
            binarize = (
                f"T = df['{treatment_var}'].values.astype(float)\n"
                f"if len(np.unique(T[~np.isnan(T)])) <= 2:\n"
                f"    T_binary = T.astype(int)\n"
                f"else:\n"
                f"    T_binary = (T > np.median(T[~np.isnan(T)])).astype(int)\n"
            )

        cells.append(new_markdown_cell(
            "### Covariate Balance: Standardized Mean Differences\n\n"
            "A love plot shows balance between treatment and control groups. "
            "Variables with |SMD| < 0.1 are considered well-balanced."
        ))

        balance_code = f'''# Covariate balance: Standardized Mean Differences (Love Plot)
COVARIATES = {covariates_json}
covariates = [c for c in COVARIATES if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

{binarize}
# Compute SMD for each covariate
smd_data = []
for c in covariates:
    vals = df[c].values.astype(float)
    mask = ~(np.isnan(vals) | np.isnan(T_binary))
    v, t = vals[mask], T_binary[mask]

    mean_t = v[t == 1].mean()
    mean_c = v[t == 0].mean()
    sd_t = v[t == 1].std()
    sd_c = v[t == 0].std()
    pooled_sd = np.sqrt((sd_t**2 + sd_c**2) / 2)

    if pooled_sd > 0:
        smd = (mean_t - mean_c) / pooled_sd
    else:
        smd = 0.0
    smd_data.append((c, smd))

# Sort by absolute SMD
smd_data.sort(key=lambda x: abs(x[1]))
names = [x[0] for x in smd_data]
smds = [x[1] for x in smd_data]

# Love plot
fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.4)))
colors = ['green' if abs(s) < 0.1 else 'red' for s in smds]
y_pos = range(len(names))
ax.barh(y_pos, smds, color=colors, alpha=0.7, height=0.6)
ax.axvline(x=-0.1, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='black', linewidth=0.5)
ax.set_yticks(list(y_pos))
ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel('Standardized Mean Difference (SMD)')
ax.set_title('Love Plot: Covariate Balance Between Treatment Groups')

n_balanced = sum(1 for s in smds if abs(s) < 0.1)
n_total = len(smds)
ax.text(0.02, 0.98, f'Balanced: {{n_balanced}}/{{n_total}} (|SMD| < 0.1)',
        transform=ax.transAxes, va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
plt.tight_layout()
plt.show()

print(f"\\nCovariate balance summary: {{n_balanced}}/{{n_total}} covariates balanced (|SMD| < 0.1)")
for name, smd in sorted(smd_data, key=lambda x: -abs(x[1])):
    status = "✓" if abs(smd) < 0.1 else "✗"
    print(f"  {{status}} {{name}}: SMD = {{smd:.4f}}")'''

        cells.append(new_code_cell(balance_code))

    return cells
