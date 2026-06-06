"""run_placebo_test - placebo-treatment or placebo-outcome falsification."""

import numpy as np

from src.analysis.agents.base import (
    AnalysisState,
    SensitivityResult,
    ToolResult,
    ToolResultStatus,
)
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "run_placebo_test",
    "description": "Run placebo tests using fake treatments or outcomes to check for spurious effects.",
    "parameters": {
        "type": "object",
        "properties": {
            "test_type": {
                "type": "string",
                "enum": ["placebo_treatment", "placebo_outcome", "both"],
                "description": "Type of placebo test to run",
            },
            "n_placebos": {
                "type": "integer",
                "description": "Number of placebo iterations (default: 100)",
            },
        },
        "required": ["test_type"],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    test_type: str = "both",
    n_placebos: int = 100,
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug(
            "tool_ignored_kwargs",
            tool="run_placebo_test",
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

    # CR2: Drop NaN from both columns simultaneously to keep rows aligned;
    # independent dropna() misaligns rows when NaN positions differ.
    mask = df[[T_col, Y_col]].notna().all(axis=1)
    T = df.loc[mask, T_col].values
    Y = df.loc[mask, Y_col].values

    actual_effect = (
        abs(agent._current_state.treatment_effects[0].estimate)
        if agent._current_state.treatment_effects else 0
    )

    np.random.seed(42)
    results_output: dict = {"test_type": test_type, "actual_effect": round(actual_effect, 4)}
    interpretation = ""
    ratio = 1.0

    if test_type in ["placebo_treatment", "both"]:
        placebo_effects = []
        for _ in range(n_placebos):
            T_placebo = np.random.binomial(1, 0.5, size=len(Y))
            model = LinearRegression()
            model.fit(T_placebo.reshape(-1, 1), Y)
            placebo_effects.append(abs(model.coef_[0]))

        placebo_mean = float(np.mean(placebo_effects))
        placebo_p95 = float(np.percentile(placebo_effects, 95))
        ratio = actual_effect / (placebo_mean + 0.001)

        if actual_effect > 2 * placebo_p95:
            interp_t = "PASSED: Real effect far exceeds placebo"
        elif actual_effect > placebo_p95:
            interp_t = "PASSED: Real effect exceeds 95th percentile"
        else:
            interp_t = "CONCERNING: Real effect within placebo distribution"

        results_output["placebo_treatment"] = {
            "placebo_mean": round(placebo_mean, 4),
            "placebo_p95": round(placebo_p95, 4),
            "ratio": round(ratio, 2),
            "interpretation": interp_t,
        }
        interpretation = interp_t

    if test_type in ["placebo_outcome", "both"]:
        placebo_effects = []
        for _ in range(n_placebos):
            Y_placebo = np.random.randn(len(T))
            model = LinearRegression()
            model.fit(T.reshape(-1, 1), Y_placebo)
            placebo_effects.append(abs(model.coef_[0]))

        placebo_mean = float(np.mean(placebo_effects))
        ratio = actual_effect / (placebo_mean + 0.001)

        if actual_effect > 3 * placebo_mean:
            interp_o = "PASSED: Much larger effect on real outcome"
        elif actual_effect > 2 * placebo_mean:
            interp_o = "PASSED: Notably larger on real outcome"
        else:
            interp_o = "CONCERNING: Similar on real and placebo outcomes"

        results_output["placebo_outcome"] = {
            "placebo_mean": round(placebo_mean, 4),
            "ratio": round(ratio, 2),
            "interpretation": interp_o,
        }
        interpretation = (
            interp_o if test_type == "placebo_outcome" else f"{interpretation}; {interp_o}"
        )

    sens_result = SensitivityResult(
        method=f"Placebo Test ({test_type})",
        robustness_value=float(ratio),
        interpretation=interpretation,
        details={"test_type": test_type, "n_placebos": n_placebos},
    )
    agent._results.append(sens_result)

    return ToolResult(status=ToolResultStatus.SUCCESS, output=results_output)
