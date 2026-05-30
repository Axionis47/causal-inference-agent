"""check_subgroup_effects - effect heterogeneity across one subgroup variable."""

import numpy as np

SCHEMA = {
    "name": "check_subgroup_effects",
    "description": "Check if treatment effect varies across subgroups (heterogeneity).",
    "parameters": {
        "type": "object",
        "properties": {
            "subgroup_variable": {
                "type": "string",
                "description": "Variable to split by (empty to auto-select)",
            },
        },
        "required": [],
    },
}


def handle(agent, subgroup_variable: str | None = None, **kwargs) -> str:
    if agent._df is None:
        return "No data available for subgroup analysis."

    df = agent._df
    treatment_col, outcome_col = agent._resolve_treatment_outcome()

    if not treatment_col or not outcome_col:
        return "Treatment or outcome variable not identified."

    if treatment_col not in df.columns or outcome_col not in df.columns:
        return "Treatment or outcome variable not found."

    if not subgroup_variable:
        for col in df.columns:
            if col not in [treatment_col, outcome_col]:
                if 2 <= df[col].nunique() <= 5:
                    subgroup_variable = col
                    break

    if not subgroup_variable:
        return "No suitable subgroup variable found."

    output = f"Subgroup Analysis ({subgroup_variable}):\n"
    output += "=" * 50 + "\n"

    subgroup_effects: list[tuple[object, float, int, int]] = []
    for val in df[subgroup_variable].dropna().unique():
        subgroup = df[df[subgroup_variable] == val]
        treated = subgroup[subgroup[treatment_col] == subgroup[treatment_col].max()]
        control = subgroup[subgroup[treatment_col] == subgroup[treatment_col].min()]

        if len(treated) >= 10 and len(control) >= 10:
            effect = treated[outcome_col].mean() - control[outcome_col].mean()
            subgroup_effects.append((val, effect, len(treated), len(control)))

    if not subgroup_effects:
        return "Not enough data in subgroups for analysis."

    output += f"Variable: {subgroup_variable}\n\n"
    effects: list[float] = []
    for val, effect, n_t, n_c in subgroup_effects:
        output += f"  {val}: Effect={effect:.4f} (n_t={n_t}, n_c={n_c})\n"
        effects.append(effect)

    if len(effects) > 1:
        effect_std = float(np.std(effects))
        effect_mean = float(np.mean(effects))
        heterogeneity = effect_std / (abs(effect_mean) + 1e-10)

        output += f"\nHeterogeneity: {heterogeneity:.2%}\n"

        if heterogeneity > 0.5:
            output += "WARNING: Substantial heterogeneity - effect varies across subgroups\n"
            agent._investigation_evidence.append(
                f"Substantial heterogeneity in {subgroup_variable}"
            )
        else:
            output += "Effect is relatively consistent across subgroups\n"
            agent._investigation_evidence.append(
                f"Effects consistent across {subgroup_variable}"
            )

    return output
