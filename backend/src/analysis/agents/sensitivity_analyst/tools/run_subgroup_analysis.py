"""run_subgroup_analysis - effect heterogeneity across a subgroup variable."""

import numpy as np
import pandas as pd

from src.analysis.agents.base import (
    AnalysisState,
    SensitivityResult,
    ToolResult,
    ToolResultStatus,
)
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "run_subgroup_analysis",
    "description": "Analyze treatment effect across subgroups to check for consistency.",
    "parameters": {
        "type": "object",
        "properties": {
            "subgroup_variable": {
                "type": "string",
                "description": "Variable to use for subgroups. If not specified, uses a suitable categorical variable.",
            },
        },
        "required": [],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    subgroup_variable: str | None = None,
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug(
            "tool_ignored_kwargs",
            tool="run_subgroup_analysis",
            extra_keys=list(kwargs.keys()),
        )
    from sklearn.linear_model import LinearRegression

    df = agent._df
    T_col, Y_col = agent._resolve_treatment_outcome()

    if not T_col or not Y_col:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="Treatment or outcome not identified.",
        )

    subgroup_var = subgroup_variable

    if not subgroup_var:
        if agent._current_state.data_profile:
            for col, dtype in agent._current_state.data_profile.feature_types.items():
                if dtype in ["categorical", "binary"] and col not in [T_col, Y_col]:
                    if 2 <= df[col].nunique() <= 5:
                        subgroup_var = col
                        break

    if not subgroup_var:
        subgroup_var = "_quartile"
        df = df.copy()
        df[subgroup_var] = pd.qcut(df[Y_col], 4, labels=False, duplicates="drop")

    subgroup_effects: list[float] = []
    subgroup_labels: list[str] = []
    skipped_groups = 0

    for sg_val in df[subgroup_var].unique():
        sg_mask = df[subgroup_var] == sg_val
        T_sg = df.loc[sg_mask, T_col].values
        Y_sg = df.loc[sg_mask, Y_col].values

        valid = ~(np.isnan(T_sg) | np.isnan(Y_sg))
        T_sg = T_sg[valid]
        Y_sg = Y_sg[valid]

        if len(T_sg) < 20:
            skipped_groups += 1
            continue

        model = LinearRegression()
        model.fit(T_sg.reshape(-1, 1), Y_sg)
        subgroup_effects.append(float(model.coef_[0]))
        subgroup_labels.append(f"{subgroup_var}={sg_val}")

    if len(subgroup_effects) < 2:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="Insufficient subgroups.",
        )

    effect_mean = float(np.mean(subgroup_effects))
    effect_std = float(np.std(subgroup_effects))
    all_same_sign = (
        all(e > 0 for e in subgroup_effects)
        or all(e < 0 for e in subgroup_effects)
    )
    cv = effect_std / abs(effect_mean) if effect_mean != 0 else float("inf")

    if cv < 0.3 and all_same_sign:
        interpretation = "CONSISTENT: Effect similar across subgroups"
    elif all_same_sign:
        interpretation = "DIRECTION CONSISTENT: Magnitude varies but direction stable"
    else:
        interpretation = "HETEROGENEOUS: Effect varies including sign changes"

    sens_result = SensitivityResult(
        method="Subgroup Analysis",
        robustness_value=float(1 - min(cv, 1)),
        interpretation=interpretation,
        details={
            "subgroup_variable": subgroup_var,
            "n_subgroups": len(subgroup_effects),
            "cv": float(cv),
            "subgroup_effects": [
                {"label": label, "effect": round(float(e), 4)}
                for label, e in zip(subgroup_labels, subgroup_effects, strict=False)
            ],
            "mean_effect": round(float(effect_mean), 4),
            "std_effect": round(float(effect_std), 4),
        },
    )
    agent._results.append(sens_result)

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "subgroup_variable": subgroup_var,
            "n_subgroups": len(subgroup_effects),
            "subgroup_effects": [
                {"label": label, "effect": round(e, 4)}
                for label, e in zip(subgroup_labels, subgroup_effects, strict=False)
            ],
            "mean_effect": round(effect_mean, 4),
            "std_effect": round(effect_std, 4),
            "all_same_sign": all_same_sign,
            "cv": round(cv, 4),
            "subgroups_skipped": skipped_groups,
            "skip_reason": "fewer than 20 observations" if skipped_groups > 0 else None,
            "interpretation": interpretation,
        },
    )
