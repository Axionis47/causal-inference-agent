"""get_data_summary - sample sizes, treatment split, outcome stats, method recommendations."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.logging_config.structured import get_logger

from ..method_selector import SampleSizeThresholds

logger = get_logger(__name__)

SCHEMA = {
    "name": "get_data_summary",
    "description": "Get summary statistics of the dataset including sample sizes, treatment/control split, outcome distribution, and available covariates.",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


async def handle(agent, state: AnalysisState, **kwargs) -> ToolResult:
    if kwargs:
        logger.debug(
            "tool_ignored_kwargs",
            tool="get_data_summary",
            extra_keys=list(kwargs.keys()),
        )
    df = agent._df
    T = df[agent._treatment_var]
    Y = df[agent._outcome_var]

    n_total = len(df)
    n_treated = int(T.sum())
    n_control = n_total - n_treated

    warning = SampleSizeThresholds.get_sample_size_warning(n_treated, n_control)
    recommended = SampleSizeThresholds.get_recommended_methods(n_treated, n_control)

    if n_treated < 100 or n_control < 100:
        guidance = "Small sample: PREFER OLS, IPW, AIPW. AVOID T/X-Learner, Causal Forest."
    elif n_treated < 200:
        guidance = "Moderate sample: SAFE OLS, IPW, AIPW, Matching. CAUTIOUS with T/X-Learner."
    else:
        guidance = "Adequate sample: All methods viable. Use ML methods for heterogeneity."

    return ToolResult(
        status=ToolResultStatus.SUCCESS,
        output={
            "n_total": n_total,
            "n_treated": n_treated,
            "n_control": n_control,
            "treatment_pct": round(100 * n_treated / n_total, 1),
            "outcome_stats": {
                "overall_mean": round(Y.mean(), 4),
                "overall_std": round(Y.std(), 4),
                "treated_mean": round(Y[T == 1].mean(), 4),
                "control_mean": round(Y[T == 0].mean(), 4),
                "naive_difference": round(Y[T == 1].mean() - Y[T == 0].mean(), 4),
            },
            "n_covariates": len(agent._covariates),
            "covariates_sample": agent._covariates[:10],
            "recommended_methods": recommended,
            "guidance": guidance,
            "warning": warning,
        },
    )
