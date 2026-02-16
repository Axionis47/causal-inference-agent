"""Propensity score diagnostics section renderer."""

from nbformat.v4 import new_code_cell, new_markdown_cell

from src.agents.base import AnalysisState


def render_ps_diagnostics(state: AnalysisState) -> list:
    """Report propensity score diagnostics agent findings."""
    cells = []
    ps = state.ps_diagnostics
    if not ps:
        return cells

    md = "## Propensity Score Diagnostics\n\n"
    md += "*Assessment from the PS Diagnostics agent — validates whether "
    md += "propensity-score-based estimation methods are reliable for this data.*\n\n"

    # Model quality badge
    quality = ps.get("model_quality", "unknown")
    proceed = ps.get("proceed_with_analysis", True)
    recommended = ps.get("recommended_method", "unknown")

    md += "### Summary\n\n"
    md += "| Diagnostic | Result |\n"
    md += "|-----------|--------|\n"
    md += f"| PS Model Quality | **{quality.replace('_', ' ').title()}** |\n"
    md += f"| Proceed with Analysis | **{'Yes' if proceed else 'No'}** |\n"
    md += f"| Recommended Method | **{recommended.upper().replace('_', ' ')}** |\n"

    # Trimming bounds
    trimming = ps.get("trimming_bounds")
    if trimming:
        md += f"| PS Trimming Bounds | [{trimming[0]:.3f}, {trimming[1]:.3f}] |\n"
    else:
        md += "| PS Trimming Bounds | None (no trimming needed) |\n"
    md += "\n"

    # Warnings
    warnings_list = ps.get("warnings", [])
    if warnings_list:
        md += "### Warnings\n\n"
        for w in warnings_list:
            md += f"- {w}\n"
        md += "\n"

    # Reasoning
    reasoning = ps.get("reasoning", "")
    if reasoning:
        md += "### Diagnostic Reasoning\n\n"
        md += f"{reasoning}\n\n"

    # Not-proceed warning box
    if not proceed:
        md += "> **WARNING**: The PS diagnostics agent recommends **not proceeding** "
        md += "with propensity-score-based methods for this dataset. Treatment effect "
        md += "estimates using IPW or matching should be interpreted with extreme caution.\n\n"

    cells.append(new_markdown_cell(md))

    # PS distribution visualization code (if we have the data)
    cells.append(new_markdown_cell(
        "### Propensity Score Distribution\n\n"
        "Visualize the estimated propensity scores by treatment group to assess overlap."
    ))

    trimming_code = ""
    if trimming:
        trimming_code = (
            f"\n    # Apply trimming bounds from PS diagnostics\n"
            f"    trim_lower, trim_upper = {trimming[0]}, {trimming[1]}\n"
            f"    mask = (ps_scores >= trim_lower) & (ps_scores <= trim_upper)\n"
            f"    print(f\"Trimming: {{{{(~mask).sum()}}}} observations outside "
            f"[{trimming[0]:.3f}, {trimming[1]:.3f}]\")\n"
        )

    ps_code = f'''# Propensity score estimation and overlap check
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

treatment_var = "{state.treatment_variable}"
outcome_var = "{state.outcome_variable}"

# Get covariates (all numeric columns except treatment and outcome)
covariates = [c for c in df.select_dtypes(include=[np.number]).columns
              if c != treatment_var and c != outcome_var]

if covariates:
    X = df[covariates].fillna(df[covariates].median())
    T = df[treatment_var].values

    # Binarize treatment if continuous
    if len(np.unique(T)) > 2:
        T = (T > np.median(T)).astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, T)
    ps_scores = lr.predict_proba(X_scaled)[:, 1]
{trimming_code}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution by group
    ax = axes[0]
    ax.hist(ps_scores[T == 0], bins=30, alpha=0.6, label="Control", color="steelblue", edgecolor="black")
    ax.hist(ps_scores[T == 1], bins=30, alpha=0.6, label="Treated", color="coral", edgecolor="black")
    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Count")
    ax.set_title("PS Distribution by Treatment Group")
    ax.legend()

    # Overlap assessment
    ax = axes[1]
    ax.boxplot([ps_scores[T == 0], ps_scores[T == 1]], labels=["Control", "Treated"])
    ax.set_ylabel("Propensity Score")
    ax.set_title("PS Overlap Assessment")

    plt.tight_layout()
    plt.show()

    # Print diagnostics
    print(f"PS Model Quality: {quality}")
    print(f"Recommended Method: {recommended}")
    print(f"Overlap: ({{ps_scores[T == 1].min():.3f}}-{{ps_scores[T == 1].max():.3f}}) treated, "
          f"({{ps_scores[T == 0].min():.3f}}-{{ps_scores[T == 0].max():.3f}}) control")
else:
    print("No numeric covariates available for PS estimation.")'''
    cells.append(new_code_cell(ps_code))

    return cells
