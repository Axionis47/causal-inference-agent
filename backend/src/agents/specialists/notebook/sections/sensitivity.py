"""Sensitivity analysis section renderer."""

from nbformat.v4 import new_code_cell, new_markdown_cell

from src.agents.base import AnalysisState

from ..helpers import deduplicate_sensitivity


def render_sensitivity(state: AnalysisState) -> list:
    """Report sensitivity analysis results with executable verification code."""
    cells = []
    sensitivity_results = deduplicate_sensitivity(state.sensitivity_results)

    cells.append(new_markdown_cell(
        "## Sensitivity Analysis\n\n*Findings from the Sensitivity Analyst agent.*"
    ))

    # Results summary table
    if sensitivity_results:
        table_md = "### Results Summary\n\n"
        table_md += "| Analysis | Robustness Value | Interpretation |\n"
        table_md += "|----------|------------------|----------------|\n"
        for s in sensitivity_results:
            interp = s.interpretation[:100] + ("..." if len(s.interpretation) > 100 else "")
            table_md += f"| {s.method} | {s.robustness_value:.2f} | {interp} |\n"
        table_md += "\n"
        cells.append(new_markdown_cell(table_md))

    # Per-method details
    for s in sensitivity_results:
        if s.details:
            detail_md = f"#### {s.method}\n\n"
            detail_md += f"**Interpretation**: {s.interpretation}\n\n"
            detail_md += "**Details:**\n"
            for k, v in s.details.items():
                if k == "subgroup_effects" and isinstance(v, list):
                    detail_md += "\n**Per-subgroup Effects:**\n\n"
                    detail_md += "| Subgroup | Effect |\n|----------|--------|\n"
                    for sg in v:
                        label = sg.get("label", "?")
                        effect = sg.get("effect", 0)
                        detail_md += f"| {label} | {effect:.4f} |\n"
                    detail_md += "\n"
                elif isinstance(v, float):
                    detail_md += f"- {k}: {v:.4f}\n"
                elif isinstance(v, (str, int, bool)):
                    detail_md += f"- {k}: {v}\n"
            detail_md += "\n"
            cells.append(new_markdown_cell(detail_md))

    # Executable: E-value computation
    cells.append(new_markdown_cell(
        "### Verification: E-value Computation\n\n"
        "The E-value quantifies how strong unmeasured confounding "
        "would need to be to explain away the observed effect."
    ))

    evalue_code = '''# E-value computation
def compute_e_value(ate, se, y_std):
    """Compute E-value from treatment effect estimate."""
    if abs(ate) < 1e-10 or y_std < 1e-10:
        return 1.0, 1.0

    # Approximate risk ratio
    rr = np.exp(0.91 * ate / y_std)
    if rr < 1:
        rr = 1 / rr

    e_val = rr + np.sqrt(rr * (rr - 1))

    # E-value for CI bound
    ci_bound = abs(ate) - 1.96 * se
    if ci_bound > 0:
        rr_ci = np.exp(0.91 * ci_bound / y_std)
        if rr_ci < 1:
            rr_ci = 1 / rr_ci
        e_val_ci = rr_ci + np.sqrt(rr_ci * (rr_ci - 1))
    else:
        e_val_ci = 1.0

    return e_val, e_val_ci

# Use the verification OLS results
e_val, e_val_ci = compute_e_value(results.params[1], results.bse[1], np.std(Y))
print(f"E-value (point estimate): {e_val:.2f}")
print(f"E-value (CI bound):       {e_val_ci:.2f}")
print()
if e_val > 3:
    print("STRONG robustness: A very strong unmeasured confounder would be needed.")
elif e_val > 2:
    print("MODERATE robustness: A moderately strong confounder could explain this.")
else:
    print("WEAK robustness: A relatively weak confounder could explain the effect.")'''
    cells.append(new_code_cell(evalue_code))

    # Executable: Placebo test
    cells.append(new_markdown_cell(
        "### Verification: Placebo Test\n\n"
        "Checks whether the treatment effect disappears under random treatment assignment."
    ))

    placebo_code = '''# Placebo test: permutation-based
n_permutations = 500
placebo_effects = []

for _ in range(n_permutations):
    T_placebo = np.random.permutation(T_binary)
    if covariates:
        X_p = df_clean[covariates].values.astype(float)
        design_p = np.column_stack([np.ones(len(T_placebo)), T_placebo, X_p])
    else:
        design_p = np.column_stack([np.ones(len(T_placebo)), T_placebo])

    try:
        res_p = sm.OLS(Y, design_p).fit()
        placebo_effects.append(res_p.params[1])
    except Exception:
        pass

real_ate = results.params[1]
p_value_perm = np.mean(np.abs(placebo_effects) >= np.abs(real_ate))

print(f"Real ATE:             {real_ate:.4f}")
print(f"Mean placebo effect:  {np.mean(placebo_effects):.4f}")
print(f"Permutation p-value:  {p_value_perm:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(placebo_effects, bins=50, alpha=0.7, color='gray', edgecolor='black', label='Placebo effects')
ax.axvline(x=real_ate, color='red', linewidth=2, label=f'Real ATE: {real_ate:.4f}')
ax.set_xlabel('Effect Estimate')
ax.set_ylabel('Frequency')
ax.set_title('Placebo Test: Real Effect vs Permuted Effects')
ax.legend()
plt.tight_layout()
plt.show()'''
    cells.append(new_code_cell(placebo_code))

    return cells
