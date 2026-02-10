"""Sensitivity Analyst Agent - ReAct-based sensitivity analysis.

This agent uses the ReAct pattern to iteratively:
1. Review current treatment effect estimates
2. Run sensitivity analyses one at a time
3. Interpret results and decide if more investigation needed
4. Assess overall robustness of causal findings

Uses pull-based context - queries for information on demand.
"""

import time
from typing import Any

import numpy as np
import pandas as pd

from src.agents.base import (
    AnalysisState,
    JobStatus,
    SensitivityResult,
    ToolResult,
    ToolResultStatus,
)
from src.agents.base.context_tools import ContextTools
from src.agents.base.react_agent import ReActAgent
from src.agents.registry import register_agent
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


@register_agent("sensitivity_analyst")
class SensitivityAnalystAgent(ReActAgent, ContextTools):
    """ReAct-based sensitivity analyst.

    This agent iteratively:
    1. Queries previous agent findings for context
    2. Reviews treatment effect estimates
    3. Chooses and runs sensitivity analyses one at a time
    4. Interprets results before deciding next analysis
    5. Provides final robustness assessment

    Uses pull-based context - queries for information on demand.
    """

    AGENT_NAME = "sensitivity_analyst"
    MAX_STEPS = 15

    SYSTEM_PROMPT = """You are an expert in sensitivity analysis for causal inference.
Your task is to assess how robust the causal estimates are to potential assumption violations.

WORKFLOW:
1. FIRST: Query previous findings from effect_estimator agent
2. THEN: Run analyses one at a time based on what you learn
3. INTERPRET: Each result before choosing the next analysis
4. FINALLY: Provide overall robustness assessment

CONTEXT TOOLS:
- ask_domain_knowledge: Query domain knowledge for causal constraints
- get_previous_finding: Get findings from previous agents (especially effect_estimator)
- get_eda_finding: Query EDA results
- get_treatment_outcome: Get treatment/outcome variables

AVAILABLE ANALYSES AND WHEN TO USE THEM:

1. E-VALUE: Assesses sensitivity to unmeasured confounding
   - Use for ANY observational study
   - Higher E-value = more robust (E > 2 is generally good)
   - Always run this first

2. ROSENBAUM BOUNDS: Sensitivity for matched designs
   - Use when matching/propensity score methods were used
   - Gamma > 2 suggests robustness

3. SPECIFICATION CURVE: How estimates vary across modeling choices
   - Use when concerned about arbitrary analytical decisions
   - Look for sign stability and low variance

4. PLACEBO TESTS: Test effects where none should exist
   - Use to validate the analysis approach
   - Significant placebo effects are concerning

5. SUBGROUP ANALYSIS: Check effect consistency across subgroups
   - Use to detect heterogeneity or specification issues
   - Effect should be directionally consistent

6. BOOTSTRAP VARIANCE: Assess estimate precision
   - Use to verify standard errors are reasonable

Run analyses based on what you learn. If E-value is low, no need for extensive other tests.
If specification curve shows instability, investigate further."""

    def __init__(self) -> None:
        """Initialize the sensitivity analyst agent."""
        super().__init__()

        # Register context query tools from mixin
        self.register_context_tools()

        # Internal state
        self._df: pd.DataFrame | None = None
        self._current_state: AnalysisState | None = None
        self._results: list[SensitivityResult] = []
        self._treatment_var: str | None = None
        self._outcome_var: str | None = None
        self._finalized: bool = False
        self._final_result: dict[str, Any] = {}

        # Register sensitivity-specific tools
        self._register_sensitivity_tools()

    def _register_sensitivity_tools(self) -> None:
        """Register tools for sensitivity analysis."""
        self.register_tool(
            name="get_estimates_summary",
            description="Get summary of current treatment effect estimates to understand what needs sensitivity testing.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=self._tool_get_estimates_summary,
        )

        self.register_tool(
            name="compute_e_value",
            description="Compute E-value for sensitivity to unmeasured confounding. This quantifies how strong an unmeasured confounder would need to be to explain away the observed effect.",
            parameters={
                "type": "object",
                "properties": {
                    "method_index": {
                        "type": "integer",
                        "description": "Index of the method to analyze (0 for first/primary estimate)",
                    },
                },
                "required": [],
            },
            handler=self._tool_compute_e_value,
        )

        self.register_tool(
            name="compute_rosenbaum_bounds",
            description="Compute Rosenbaum bounds (Gamma) for sensitivity to unmeasured confounding in matched studies.",
            parameters={
                "type": "object",
                "properties": {
                    "method_index": {
                        "type": "integer",
                        "description": "Index of the method to analyze",
                    },
                },
                "required": [],
            },
            handler=self._tool_compute_rosenbaum_bounds,
        )

        self.register_tool(
            name="run_specification_curve",
            description="Run specification curve analysis to see how estimates vary across different model specifications.",
            parameters={
                "type": "object",
                "properties": {
                    "n_specifications": {
                        "type": "integer",
                        "description": "Number of specifications to try (default: 10)",
                    },
                },
                "required": [],
            },
            handler=self._tool_run_specification_curve,
        )

        self.register_tool(
            name="run_placebo_test",
            description="Run placebo tests using fake treatments or outcomes to check for spurious effects.",
            parameters={
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
            handler=self._tool_run_placebo_test,
        )

        self.register_tool(
            name="run_subgroup_analysis",
            description="Analyze treatment effect across subgroups to check for consistency.",
            parameters={
                "type": "object",
                "properties": {
                    "subgroup_variable": {
                        "type": "string",
                        "description": "Variable to use for subgroups. If not specified, uses a suitable categorical variable.",
                    },
                },
                "required": [],
            },
            handler=self._tool_run_subgroup_analysis,
        )

        self.register_tool(
            name="check_variance_stability",
            description="Check if standard errors are stable via bootstrap resampling.",
            parameters={
                "type": "object",
                "properties": {
                    "n_bootstrap": {
                        "type": "integer",
                        "description": "Number of bootstrap iterations (default: 200)",
                    },
                },
                "required": [],
            },
            handler=self._tool_check_variance_stability,
        )

        self.register_tool(
            name="finalize_sensitivity",
            description="Finalize the sensitivity analysis with overall robustness assessment.",
            parameters={
                "type": "object",
                "properties": {
                    "overall_robustness": {
                        "type": "string",
                        "enum": ["high", "moderate", "low", "uncertain"],
                        "description": "Overall assessment of causal estimate robustness",
                    },
                    "key_findings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Key findings from sensitivity analysis",
                    },
                    "concerns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Any concerns about robustness",
                    },
                    "recommendations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Recommendations for interpreting results",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Detailed reasoning for robustness assessment",
                    },
                },
                "required": ["overall_robustness", "key_findings", "reasoning"],
            },
            handler=self._tool_finalize_sensitivity,
        )

    # =========================================================================
    # ReAct Required Methods
    # =========================================================================

    def _get_initial_observation(self, state: AnalysisState) -> str:
        """Get the initial observation for the ReAct loop."""
        # Get primary pair
        treatment_var, outcome_var = state.get_primary_pair()

        # Get effects for the primary pair
        if treatment_var and outcome_var:
            effects = state.get_effects_for_pair(treatment_var, outcome_var)
        else:
            effects = state.treatment_effects

        effects_summary = ""
        for i, effect in enumerate(effects[:5]):  # Limit to first 5
            effects_summary += f"  {i}. {effect.method}: {effect.estimate:.4f} (SE: {effect.std_error:.4f})\n"

        obs = f"""You are assessing the robustness of causal effect estimates.

Treatment: {treatment_var or 'Unknown'}
Outcome: {outcome_var or 'Unknown'}
Sample size: {state.data_profile.n_samples if state.data_profile else 'Unknown'}

Treatment Effects to assess:
{effects_summary}

WORKFLOW:
1. First, query previous findings from effect_estimator
2. Run E-value analysis (most fundamental sensitivity test)
3. Based on results, run additional analyses as needed
4. Finalize with overall robustness assessment"""

        return obs

    async def is_task_complete(self, state: AnalysisState) -> bool:
        """Check if sensitivity analysis is complete."""
        return self._finalized

    def _resolve_treatment_outcome(self) -> tuple[str | None, str | None]:
        """Get treatment and outcome variables using state helper."""
        if self._treatment_var and self._outcome_var:
            return self._treatment_var, self._outcome_var

        if self._current_state:
            t, o = self._current_state.get_primary_pair()
            self._treatment_var = t
            self._outcome_var = o
            return t, o

        return None, None

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute sensitivity analysis using ReAct loop.

        Args:
            state: Current analysis state with treatment effects

        Returns:
            Updated state with sensitivity results
        """
        self.logger.info(
            "sensitivity_start",
            job_id=state.job_id,
            n_effects=len(state.treatment_effects),
        )

        state.status = JobStatus.SENSITIVITY_ANALYSIS
        start_time = time.time()

        try:
            if not state.treatment_effects:
                self.logger.warning("no_effects_for_sensitivity")
                return state

            # Load data
            if state.dataframe_path is None:
                return state

            self._df = pd.read_parquet(state.dataframe_path)

            self._current_state = state
            self._results = []
            self._finalized = False

            # Run ReAct loop
            await super().execute(state)

            # If not finalized via tool, auto-finalize
            if not self._finalized:
                self.logger.warning("sensitivity_auto_finalize")
                self._auto_finalize()
                self._finalized = True

            # Store results
            state.sensitivity_results = self._results

            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.info(
                "sensitivity_complete",
                n_results=len(self._results),
                robustness=self._final_result.get("overall_robustness"),
                duration_ms=duration_ms,
            )

        except Exception as e:
            self.logger.exception("sensitivity_failed", error=str(e))

        return state

    def _auto_finalize(self) -> None:
        """Auto-finalize when LLM doesn't explicitly finalize."""
        if not self._results:
            self._final_result = {
                "overall_robustness": "uncertain",
                "key_findings": ["No sensitivity analyses completed"],
                "concerns": ["Unable to assess robustness"],
                "reasoning": "Auto-finalized: no sensitivity results obtained",
            }
            return

        # Assess based on results
        e_value_result = next((r for r in self._results if "e-value" in r.method.lower()), None)
        robustness = "moderate"

        if e_value_result and e_value_result.robustness_value is not None:
            if e_value_result.robustness_value >= 2.5:
                robustness = "high"
            elif e_value_result.robustness_value >= 1.5:
                robustness = "moderate"
            else:
                robustness = "low"

        key_findings = [r.interpretation for r in self._results[:3]]

        self._final_result = {
            "overall_robustness": robustness,
            "key_findings": key_findings,
            "concerns": [],
            "reasoning": f"Auto-finalized with {len(self._results)} analyses completed",
        }

    # =========================================================================
    # Tool Handlers (async, return ToolResult)
    # =========================================================================

    async def _tool_get_estimates_summary(
        self,
        state: AnalysisState,
    ) -> ToolResult:
        """Get summary of treatment effect estimates."""
        if not self._current_state.treatment_effects:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="No treatment effects estimated yet."
            )

        estimates_data = []
        for i, effect in enumerate(self._current_state.treatment_effects):
            estimates_data.append({
                "index": i,
                "method": effect.method,
                "estimand": effect.estimand,
                "estimate": round(effect.estimate, 4),
                "std_error": round(effect.std_error, 4),
                "ci": [round(effect.ci_lower, 4), round(effect.ci_upper, 4)],
                "p_value": round(effect.p_value, 4) if effect.p_value else None,
            })

        estimates = [e.estimate for e in self._current_state.treatment_effects]
        summary_stats = None
        if len(estimates) > 1:
            summary_stats = {
                "mean": round(float(np.mean(estimates)), 4),
                "std": round(float(np.std(estimates)), 4),
                "range": [round(min(estimates), 4), round(max(estimates), 4)],
                "all_same_sign": all(e > 0 for e in estimates) or all(e < 0 for e in estimates),
            }

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "n_estimates": len(estimates_data),
                "estimates": estimates_data,
                "cross_method_summary": summary_stats,
            }
        )

    async def _tool_compute_e_value(
        self,
        state: AnalysisState,
        method_index: int = 0,
    ) -> ToolResult:
        """Compute E-value for unmeasured confounding."""
        if method_index >= len(self._current_state.treatment_effects):
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Invalid method index {method_index}"
            )

        effect = self._current_state.treatment_effects[method_index]
        estimate = effect.estimate
        se = effect.std_error

        # Convert to approximate risk ratio for E-value calculation
        if estimate >= 0:
            rr = max(1.01, 1 + estimate / (se * 2 + 0.01))
        else:
            rr = max(1.01, 1 / (1 + abs(estimate) / (se * 2 + 0.01)))

        # E-value formula: E = RR + sqrt(RR * (RR - 1))
        e_value = rr + np.sqrt(rr * (rr - 1))

        # E-value for CI bound
        ci_bound = effect.ci_lower if estimate > 0 else effect.ci_upper
        if (estimate > 0 and ci_bound > 0) or (estimate < 0 and ci_bound < 0):
            rr_ci = max(1.01, 1 + abs(ci_bound) / (se * 2 + 0.01))
            e_value_ci = rr_ci + np.sqrt(rr_ci * (rr_ci - 1))
        else:
            e_value_ci = 1.0

        # Interpretation
        if e_value >= 3 and e_value_ci >= 1.5:
            interpretation = "ROBUST: Would need very strong unmeasured confounding"
            robustness_level = "high"
        elif e_value >= 2 and e_value_ci >= 1.2:
            interpretation = "MODERATELY ROBUST: Moderate confounding strength needed"
            robustness_level = "moderate"
        elif e_value >= 1.5:
            interpretation = "SOMEWHAT SENSITIVE: Relatively weak confounding could explain"
            robustness_level = "low"
        else:
            interpretation = "SENSITIVE: Even weak confounding could explain the effect"
            robustness_level = "very low"

        # Store result
        sens_result = SensitivityResult(
            method="E-value",
            robustness_value=float(e_value),
            interpretation=f"E-value = {e_value:.2f} (CI: {e_value_ci:.2f}): {interpretation}",
            details={
                "e_value_point": float(e_value),
                "e_value_ci": float(e_value_ci),
                "approximate_rr": float(rr),
                "robustness_level": robustness_level,
            },
        )
        self._results.append(sens_result)

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "method_analyzed": effect.method,
                "e_value_point": round(float(e_value), 2),
                "e_value_ci": round(float(e_value_ci), 2),
                "approximate_rr": round(float(rr), 2),
                "robustness_level": robustness_level,
                "interpretation": interpretation,
            }
        )

    async def _tool_compute_rosenbaum_bounds(
        self,
        state: AnalysisState,
        method_index: int = 0,
    ) -> ToolResult:
        """Compute Rosenbaum bounds."""
        if method_index >= len(self._current_state.treatment_effects):
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Invalid method index {method_index}"
            )

        effect = self._current_state.treatment_effects[method_index]
        estimate = effect.estimate
        se = effect.std_error

        # Approximate Gamma calculation
        z_score = abs(estimate / se) if se > 0 else 0
        gamma = 1 + z_score / 2

        # Interpretation
        if gamma >= 3:
            interpretation = "VERY ROBUST: Would need very strong hidden bias"
        elif gamma >= 2:
            interpretation = "MODERATELY ROBUST: Hidden bias would need to double odds"
        elif gamma >= 1.5:
            interpretation = "SOMEWHAT SENSITIVE: Moderate hidden bias could explain effect"
        else:
            interpretation = "SENSITIVE: Small hidden bias could explain effect"

        sens_result = SensitivityResult(
            method="Rosenbaum Bounds (approximate)",
            robustness_value=float(gamma),
            interpretation=f"Gamma = {gamma:.2f}: {interpretation}",
            details={
                "gamma": float(gamma),
                "z_score": float(z_score),
            },
        )
        self._results.append(sens_result)

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "gamma": round(float(gamma), 2),
                "z_score": round(float(z_score), 2),
                "interpretation": interpretation,
            }
        )

    async def _tool_run_specification_curve(
        self,
        state: AnalysisState,
        n_specifications: int = 10,
    ) -> ToolResult:
        """Run specification curve analysis."""
        from sklearn.linear_model import LinearRegression

        df = self._df
        T_col, Y_col = self._resolve_treatment_outcome()

        if not T_col or not Y_col:
            return ToolResult(status=ToolResultStatus.ERROR, output=None, error="Treatment or outcome variable not identified.")

        if T_col not in df.columns or Y_col not in df.columns:
            return ToolResult(status=ToolResultStatus.ERROR, output=None, error="Treatment or outcome variable not in dataset.")

        T = df[T_col].values
        Y = df[Y_col].values

        mask = ~(np.isnan(T) | np.isnan(Y))
        T = T[mask]
        Y = Y[mask]

        if len(T) < 50:
            return ToolResult(status=ToolResultStatus.ERROR, output=None, error="Insufficient data for specification curve.")

        # Get covariates
        covariates = []
        if self._current_state.data_profile:
            covariates = [
                c for c in self._current_state.data_profile.potential_confounders
                if c in df.columns and c != T_col and c != Y_col
            ][:15]

        estimates = []
        specs = []

        # No controls
        model = LinearRegression()
        model.fit(T.reshape(-1, 1), Y)
        estimates.append(model.coef_[0])
        specs.append("No controls")

        # Different covariate sets
        for i in range(min(n_specifications - 1, len(covariates))):
            cov_subset = covariates[: i + 1]
            try:
                cov_data = df[cov_subset].values[mask]
                valid = ~np.any(np.isnan(cov_data), axis=1)
                X_cov = np.column_stack([T[valid], cov_data[valid]])
                Y_cov = Y[valid]

                if len(X_cov) > 30:
                    model = LinearRegression()
                    model.fit(X_cov, Y_cov)
                    estimates.append(model.coef_[0])
                    specs.append(f"+{cov_subset[-1]}")
            except Exception:
                continue

        if len(estimates) < 2:
            return ToolResult(status=ToolResultStatus.ERROR, output=None, error="Could not compute multiple specifications.")

        # Summary statistics
        est_mean = float(np.mean(estimates))
        est_std = float(np.std(estimates))
        all_same_sign = all(e > 0 for e in estimates) or all(e < 0 for e in estimates)
        cv = est_std / abs(est_mean) if est_mean != 0 else float("inf")

        # Interpretation
        if cv < 0.2 and all_same_sign:
            interpretation = "HIGHLY STABLE: Estimates consistent across specifications"
        elif cv < 0.4 and all_same_sign:
            interpretation = "MODERATELY STABLE: Some variation but same direction"
        elif all_same_sign:
            interpretation = "VARIABLE: Magnitude varies but direction consistent"
        else:
            interpretation = "UNSTABLE: Sign changes across specifications"

        sens_result = SensitivityResult(
            method="Specification Curve",
            robustness_value=float(1 - min(cv, 1)),
            interpretation=f"{interpretation} (CV={cv:.2f})",
            details={
                "n_specifications": len(estimates),
                "estimate_mean": est_mean,
                "estimate_std": est_std,
                "all_same_sign": all_same_sign,
                "cv": float(cv),
            },
        )
        self._results.append(sens_result)

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "n_specifications": len(estimates),
                "mean_estimate": round(est_mean, 4),
                "std_estimate": round(est_std, 4),
                "range": [round(min(estimates), 4), round(max(estimates), 4)],
                "all_same_sign": all_same_sign,
                "cv": round(cv, 4),
                "interpretation": interpretation,
            }
        )

    async def _tool_run_placebo_test(
        self,
        state: AnalysisState,
        test_type: str = "both",
        n_placebos: int = 100,
    ) -> ToolResult:
        """Run placebo tests."""
        from sklearn.linear_model import LinearRegression

        df = self._df
        T_col, Y_col = self._resolve_treatment_outcome()

        if not T_col or not Y_col:
            return ToolResult(status=ToolResultStatus.ERROR, output=None, error="Treatment or outcome not identified.")

        T = df[T_col].dropna().values
        Y = df[Y_col].dropna().values

        n = min(len(T), len(Y))
        T = T[:n]
        Y = Y[:n]

        actual_effect = abs(self._current_state.treatment_effects[0].estimate) if self._current_state.treatment_effects else 0

        np.random.seed(42)
        results_output = {"test_type": test_type, "actual_effect": round(actual_effect, 4)}
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
            interpretation = interp_o if test_type == "placebo_outcome" else f"{interpretation}; {interp_o}"

        sens_result = SensitivityResult(
            method=f"Placebo Test ({test_type})",
            robustness_value=float(ratio),
            interpretation=interpretation,
            details={"test_type": test_type, "n_placebos": n_placebos},
        )
        self._results.append(sens_result)

        return ToolResult(status=ToolResultStatus.SUCCESS, output=results_output)

    async def _tool_run_subgroup_analysis(
        self,
        state: AnalysisState,
        subgroup_variable: str | None = None,
    ) -> ToolResult:
        """Run subgroup analysis."""
        from sklearn.linear_model import LinearRegression

        df = self._df
        T_col, Y_col = self._resolve_treatment_outcome()

        if not T_col or not Y_col:
            return ToolResult(status=ToolResultStatus.ERROR, output=None, error="Treatment or outcome not identified.")

        subgroup_var = subgroup_variable

        # Find subgroup variable
        if not subgroup_var:
            if self._current_state.data_profile:
                for col, dtype in self._current_state.data_profile.feature_types.items():
                    if dtype in ["categorical", "binary"] and col not in [T_col, Y_col]:
                        if 2 <= df[col].nunique() <= 5:
                            subgroup_var = col
                            break

        if not subgroup_var:
            subgroup_var = "_quartile"
            df = df.copy()
            df[subgroup_var] = pd.qcut(df[Y_col], 4, labels=False, duplicates="drop")

        subgroup_effects = []
        subgroup_labels = []

        for sg_val in df[subgroup_var].unique():
            mask = df[subgroup_var] == sg_val
            T_sg = df.loc[mask, T_col].values
            Y_sg = df.loc[mask, Y_col].values

            valid = ~(np.isnan(T_sg) | np.isnan(Y_sg))
            T_sg = T_sg[valid]
            Y_sg = Y_sg[valid]

            if len(T_sg) < 20:
                continue

            model = LinearRegression()
            model.fit(T_sg.reshape(-1, 1), Y_sg)
            subgroup_effects.append(float(model.coef_[0]))
            subgroup_labels.append(f"{subgroup_var}={sg_val}")

        if len(subgroup_effects) < 2:
            return ToolResult(status=ToolResultStatus.ERROR, output=None, error="Insufficient subgroups.")

        effect_mean = float(np.mean(subgroup_effects))
        effect_std = float(np.std(subgroup_effects))
        all_same_sign = all(e > 0 for e in subgroup_effects) or all(e < 0 for e in subgroup_effects)
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
            details={"subgroup_variable": subgroup_var, "n_subgroups": len(subgroup_effects), "cv": float(cv)},
        )
        self._results.append(sens_result)

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "subgroup_variable": subgroup_var,
                "n_subgroups": len(subgroup_effects),
                "subgroup_effects": [{"label": l, "effect": round(e, 4)} for l, e in zip(subgroup_labels, subgroup_effects, strict=False)],
                "mean_effect": round(effect_mean, 4),
                "std_effect": round(effect_std, 4),
                "all_same_sign": all_same_sign,
                "cv": round(cv, 4),
                "interpretation": interpretation,
            }
        )

    async def _tool_check_variance_stability(
        self,
        state: AnalysisState,
        n_bootstrap: int = 200,
    ) -> ToolResult:
        """Check variance stability via bootstrap."""
        from sklearn.linear_model import LinearRegression

        df = self._df
        T_col, Y_col = self._resolve_treatment_outcome()

        if not T_col or not Y_col:
            return ToolResult(status=ToolResultStatus.ERROR, output=None, error="Treatment or outcome not identified.")

        T = df[T_col].values
        Y = df[Y_col].values

        mask = ~(np.isnan(T) | np.isnan(Y))
        T = T[mask]
        Y = Y[mask]

        if len(T) < 50:
            return ToolResult(status=ToolResultStatus.ERROR, output=None, error="Insufficient data for bootstrap.")

        np.random.seed(42)
        bootstrap_estimates = []
        n = len(T)

        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            T_boot = T[idx]
            Y_boot = Y[idx]

            model = LinearRegression()
            model.fit(T_boot.reshape(-1, 1), Y_boot)
            bootstrap_estimates.append(model.coef_[0])

        boot_mean = float(np.mean(bootstrap_estimates))
        boot_std = float(np.std(bootstrap_estimates))
        boot_ci = (float(np.percentile(bootstrap_estimates, 2.5)), float(np.percentile(bootstrap_estimates, 97.5)))

        reported_se = self._current_state.treatment_effects[0].std_error if self._current_state.treatment_effects else boot_std
        se_ratio = boot_std / (reported_se + 0.001)

        if 0.8 <= se_ratio <= 1.2:
            interpretation = "STABLE: Bootstrap SE consistent with reported SE"
        elif se_ratio < 0.8:
            interpretation = "CONSERVATIVE: Reported SE may be larger than necessary"
        else:
            interpretation = "UNSTABLE: Bootstrap SE larger than reported"

        sens_result = SensitivityResult(
            method="Bootstrap Variance Check",
            robustness_value=float(min(se_ratio, 1 / se_ratio)),
            interpretation=interpretation,
            details={"bootstrap_se": boot_std, "reported_se": float(reported_se), "se_ratio": float(se_ratio)},
        )
        self._results.append(sens_result)

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "n_bootstrap": n_bootstrap,
                "boot_mean": round(boot_mean, 4),
                "boot_se": round(boot_std, 4),
                "boot_ci": [round(boot_ci[0], 4), round(boot_ci[1], 4)],
                "reported_se": round(float(reported_se), 4),
                "se_ratio": round(float(se_ratio), 2),
                "interpretation": interpretation,
            }
        )

    async def _tool_finalize_sensitivity(
        self,
        state: AnalysisState,
        overall_robustness: str,
        key_findings: list[str],
        reasoning: str,
        concerns: list[str] | None = None,
        recommendations: list[str] | None = None,
    ) -> ToolResult:
        """Finalize the sensitivity analysis."""
        self._finalized = True
        self._final_result = {
            "overall_robustness": overall_robustness,
            "key_findings": key_findings,
            "concerns": concerns or [],
            "recommendations": recommendations or [],
            "reasoning": reasoning,
            "n_analyses": len(self._results),
        }

        self.logger.info(
            "sensitivity_finalized",
            overall_robustness=overall_robustness,
            n_analyses=len(self._results),
        )

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "finalized": True,
                "overall_robustness": overall_robustness,
                "key_findings": key_findings,
                "concerns": concerns or [],
                "recommendations": recommendations or [],
            },
            metadata={"is_finish": True},
        )
