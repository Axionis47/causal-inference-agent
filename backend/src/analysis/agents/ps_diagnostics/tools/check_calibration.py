"""check_calibration - calibration curve over n_bins on the PS model.

Caches `agent._metrics.calibration_mae` so the brief's CALIBRATION_OFF
flag derivation in `build_brief(state, metrics)` works.
"""

import numpy as np
from sklearn.calibration import calibration_curve

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

SCHEMA = {
    "name": "check_calibration",
    "description": "Check calibration of the propensity score model (how well predicted probabilities match observed rates).",
    "parameters": {
        "n_bins": {
            "type": "integer",
            "description": "Number of bins for calibration curve",
        },
    },
}


async def handle(
    agent,
    state: AnalysisState,
    n_bins: int = 10,
    **kwargs,
) -> ToolResult:
    if kwargs:
        logger.debug(
            "tool_ignored_kwargs",
            tool="check_calibration",
            extra_keys=list(kwargs.keys()),
        )
    if agent._propensity_scores is None:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            error="No propensity scores estimated yet.",
        )

    ps = agent._propensity_scores
    T = agent._treatment

    try:
        prob_true, prob_pred = calibration_curve(T, ps, n_bins=n_bins, strategy="uniform")
        calibration_error = float(np.mean(np.abs(prob_true - prob_pred)))
        agent._metrics.calibration_mae = calibration_error

        well_calibrated = calibration_error < 0.1

        bin_results = []
        for i, (true, pred) in enumerate(zip(prob_true, prob_pred, strict=False)):
            bin_results.append({
                "bin": i + 1,
                "predicted": float(pred),
                "observed": float(true),
                "diff": float(abs(true - pred)),
            })

        interpretation = (
            "Model probabilities are reliable"
            if well_calibrated
            else "Model may be misspecified - consider adding terms or using different link"
        )

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "n_bins": n_bins,
                "calibration_curve": bin_results,
                "mean_calibration_error": calibration_error,
                "well_calibrated": well_calibrated,
                "interpretation": interpretation,
            },
        )
    except Exception as e:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            error=f"Calibration check failed: {str(e)}",
        )
