"""run_method - execute one causal inference method via the unified engine."""

from src.analysis.agents.base import AnalysisState, ToolResult, ToolResultStatus

SCHEMA = {
    "name": "run_method",
    "description": "Execute a causal inference method to estimate treatment effects.",
    "parameters": {
        "type": "object",
        "properties": {
            "method": {
                "type": "string",
                "enum": [
                    "ols", "psm", "ipw", "aipw", "did", "iv", "rdd",
                    "s_learner", "t_learner", "causal_forest",
                ],
                "description": "Method to run",
            },
            "treatment_col": {
                "type": "string",
                "description": "Treatment column",
            },
            "outcome_col": {
                "type": "string",
                "description": "Outcome column",
            },
            "covariates": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Covariate columns for adjustment",
            },
        },
        "required": ["method", "treatment_col", "outcome_col"],
    },
}


async def handle(
    agent,
    state: AnalysisState,
    method: str,
    treatment_col: str,
    outcome_col: str,
    covariates: list[str] | None = None,
    **kwargs,
) -> ToolResult:
    if agent._df is None:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error="No data loaded",
        )

    try:
        from src.causal.estimators.effect_estimator import EffectEstimatorEngine

        engine = EffectEstimatorEngine(confidence_level=0.95)
        effect_result = engine.run_method_safe(
            method=method,
            df=agent._df,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            covariates=covariates or [],
            current_state=state,
        )

        if effect_result is None:
            state.push_decision(
                agent="effect_estimator_react",
                decision_type="method_failed",
                choice=method,
                reason=f"Method {method} returned no result (insufficient data or inapplicable).",
            )
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Method {method} returned no result (insufficient data or inapplicable)",
            )

        agent._results.append(effect_result)
        state.treatment_effects.append(effect_result)

        state.push_decision(
            agent="effect_estimator_react",
            decision_type="method_succeeded",
            choice=effect_result.method,
            reason=f"Estimate={effect_result.estimate:.4f}, SE={effect_result.std_error:.4f}, 95% CI=[{effect_result.ci_lower:.4f}, {effect_result.ci_upper:.4f}]"
            + (f", p={effect_result.p_value:.4f}" if effect_result.p_value else ""),
        )

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "method": effect_result.method,
                "estimand": effect_result.estimand,
                "estimate": f"{effect_result.estimate:.4f}",
                "std_error": f"{effect_result.std_error:.4f}",
                "ci": f"[{effect_result.ci_lower:.4f}, {effect_result.ci_upper:.4f}]",
                "p_value": f"{effect_result.p_value:.4f}" if effect_result.p_value else "N/A",
                "n_treated": effect_result.details.get("n_treated"),
                "n_control": effect_result.details.get("n_control"),
                "significant": effect_result.p_value < 0.05 if effect_result.p_value else None,
            },
        )

    except Exception as e:
        agent.logger.warning("method_failed", method=method, error=str(e))
        state.push_decision(
            agent="effect_estimator_react",
            decision_type="method_failed",
            choice=method,
            reason=f"Method {method} raised exception: {str(e)}",
        )
        return ToolResult(
            status=ToolResultStatus.ERROR,
            output=None,
            error=f"Method {method} failed: {str(e)}",
        )
