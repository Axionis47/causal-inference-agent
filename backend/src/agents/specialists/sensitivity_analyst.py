"""Sensitivity Analyst Agent - Truly agentic sensitivity analysis.

This agent uses the ReAct pattern to iteratively:
1. Review current treatment effect estimates
2. Run sensitivity analyses one at a time
3. Interpret results and decide if more investigation needed
4. Assess overall robustness of causal findings
"""

import pickle
import time
from typing import Any

import numpy as np
import pandas as pd

from src.agents.base import (
    AnalysisState,
    BaseAgent,
    JobStatus,
    SensitivityResult,
)
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


class SensitivityAnalystAgent(BaseAgent):
    """Truly agentic sensitivity analyst using ReAct pattern.

    The LLM iteratively:
    1. Reviews treatment effect estimates
    2. Chooses and runs sensitivity analyses one at a time
    3. Interprets results before deciding next analysis
    4. Provides final robustness assessment

    NO batch execution - each analysis is run and interpreted before the next.
    """

    AGENT_NAME = "sensitivity_analyst"

    SYSTEM_PROMPT = """You are an expert in sensitivity analysis for causal inference.
Your task is to assess how robust the causal estimates are to potential assumption violations.

You have tools to run sensitivity analyses on-demand. Use them iteratively:

1. FIRST: Review the treatment effect estimates
2. THEN: Run analyses one at a time based on what you learn
3. INTERPRET: Each result before choosing the next analysis
4. FINALLY: Provide overall robustness assessment

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

    TOOLS = [
        {
            "name": "get_estimates_summary",
            "description": "Get summary of current treatment effect estimates to understand what needs sensitivity testing.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "compute_e_value",
            "description": "Compute E-value for sensitivity to unmeasured confounding. This quantifies how strong an unmeasured confounder would need to be to explain away the observed effect.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method_index": {
                        "type": "integer",
                        "description": "Index of the method to analyze (0 for first/primary estimate)",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "compute_rosenbaum_bounds",
            "description": "Compute Rosenbaum bounds (Gamma) for sensitivity to unmeasured confounding in matched studies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "method_index": {
                        "type": "integer",
                        "description": "Index of the method to analyze",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "run_specification_curve",
            "description": "Run specification curve analysis to see how estimates vary across different model specifications.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_specifications": {
                        "type": "integer",
                        "description": "Number of specifications to try (default: 10)",
                    },
                },
                "required": [],
            },
        },
        {
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
        },
        {
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
        },
        {
            "name": "check_variance_stability",
            "description": "Check if standard errors are stable via bootstrap resampling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_bootstrap": {
                        "type": "integer",
                        "description": "Number of bootstrap iterations (default: 200)",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "finalize_sensitivity",
            "description": "Finalize the sensitivity analysis with overall robustness assessment.",
            "parameters": {
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
        },
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._df: pd.DataFrame | None = None
        self._state: AnalysisState | None = None
        self._results: list[SensitivityResult] = []
        self._treatment_var: str | None = None
        self._outcome_var: str | None = None

    def _resolve_treatment_outcome(self) -> tuple[str | None, str | None]:
        """Get treatment and outcome variables using state helper.

        Returns:
            Tuple of (treatment_var, outcome_var)
        """
        if self._treatment_var and self._outcome_var:
            return self._treatment_var, self._outcome_var

        if self._state:
            t, o = self._state.get_primary_pair()
            self._treatment_var = t
            self._outcome_var = o
            return t, o

        return None, None

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute sensitivity analysis using agentic loop.

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

            with open(state.dataframe_path, "rb") as f:
                self._df = pickle.load(f)

            self._state = state
            self._results = []

            # Build initial prompt
            initial_prompt = self._build_initial_prompt()

            # Run agentic loop
            final_result = await self._run_agentic_loop(initial_prompt, max_iterations=15)

            # Store results
            state.sensitivity_results = self._results

            # Create trace
            duration_ms = int((time.time() - start_time) * 1000)
            trace = self.create_trace(
                action="sensitivity_complete",
                reasoning=final_result.get("reasoning", "Sensitivity analysis completed"),
                outputs={
                    "overall_robustness": final_result.get("overall_robustness"),
                    "n_analyses": len(self._results),
                    "key_findings": final_result.get("key_findings", []),
                },
                duration_ms=duration_ms,
            )
            state.add_trace(trace)

            self.logger.info(
                "sensitivity_complete",
                n_results=len(self._results),
                robustness=final_result.get("overall_robustness"),
            )

        except Exception as e:
            self.logger.error("sensitivity_failed", error=str(e))
            import traceback
            traceback.print_exc()

        return state

    def _build_initial_prompt(self) -> str:
        """Build the initial prompt for the agentic loop."""
        # Get primary pair using helper method (robust to None values)
        treatment_var, outcome_var = self._state.get_primary_pair()

        # Get effects for the primary pair
        if treatment_var and outcome_var:
            effects = self._state.get_effects_for_pair(treatment_var, outcome_var)
        else:
            effects = self._state.treatment_effects

        effects_summary = ""
        for i, effect in enumerate(effects):
            effects_summary += f"{i}. {effect.method}: {effect.estimand} = {effect.estimate:.4f} (SE: {effect.std_error:.4f}, CI: [{effect.ci_lower:.4f}, {effect.ci_upper:.4f}])\n"

        return f"""You need to assess the robustness of these causal effect estimates:

Treatment Effects:
{effects_summary}
Treatment variable: {treatment_var or 'Unknown'}
Outcome variable: {outcome_var or 'Unknown'}
Sample size: {self._state.data_profile.n_samples if self._state.data_profile else 'Unknown'}

Start by reviewing the estimates, then systematically run sensitivity analyses.
Begin with E-value (most fundamental), then decide what else is needed based on results."""

    async def _run_agentic_loop(
        self,
        initial_prompt: str,
        max_iterations: int = 15,
    ) -> dict[str, Any]:
        """Run the agentic tool-calling loop."""
        messages = [{"role": "user", "content": initial_prompt}]
        final_result = {}

        for iteration in range(max_iterations):
            self.logger.info(
                "sensitivity_iteration",
                iteration=iteration,
                max_iterations=max_iterations,
                analyses_run=len(self._results),
            )

            try:
                result = await self.reason(
                    prompt=messages[-1]["content"] if messages[-1]["role"] == "user" else "Continue your analysis.",
                    context={"iteration": iteration, "n_results": len(self._results)},
                )
            except Exception as e:
                self.logger.warning("llm_call_failed", iteration=iteration, error=str(e))
                return self._auto_finalize()

            # Log LLM reasoning if provided
            response_text = result.get("response", "")
            if response_text:
                self.logger.info(
                    "sensitivity_llm_reasoning",
                    reasoning=response_text[:500] + ("..." if len(response_text) > 500 else ""),
                )

            pending_calls = result.get("pending_calls", [])

            if not pending_calls:
                self.logger.info("sensitivity_no_tool_calls", response_preview=response_text[:200])
                if "finalize" in response_text.lower() or len(self._results) >= 3:
                    return self._auto_finalize()
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": "Please call a tool to continue. If done, call finalize_sensitivity."
                })
                continue

            # Execute tool calls
            tool_results = []
            for call in pending_calls:
                tool_name = call["name"]
                tool_args = call.get("args", {})

                # Check for finalize with detailed logging
                if tool_name == "finalize_sensitivity":
                    self.logger.info(
                        "sensitivity_finalizing",
                        overall_robustness=tool_args.get("overall_robustness"),
                        num_findings=len(tool_args.get("key_findings", [])),
                        num_concerns=len(tool_args.get("concerns", [])),
                    )
                    final_result = tool_args
                    return final_result

                # Log the tool call decision
                self.logger.info(
                    "sensitivity_tool_decision",
                    tool=tool_name,
                    args=tool_args,
                )

                try:
                    tool_result = self._execute_tool(tool_name, tool_args)
                    tool_results.append(f"[{tool_name}]: {tool_result}")

                    # Log tool result summary
                    self.logger.info(
                        "sensitivity_tool_result",
                        tool=tool_name,
                        result_summary=tool_result[:200] + ("..." if len(tool_result) > 200 else ""),
                    )
                except Exception as e:
                    self.logger.warning("tool_execution_error", tool=tool_name, error=str(e))
                    tool_results.append(f"[{tool_name}]: ERROR - {str(e)}")

            messages.append({
                "role": "user",
                "content": "Tool results:\n\n" + "\n\n".join(tool_results) + "\n\nInterpret these results and decide what analysis to run next, or finalize if you have enough evidence."
            })

        return self._auto_finalize()

    def _auto_finalize(self) -> dict[str, Any]:
        """Auto-finalize when LLM doesn't explicitly finalize."""
        if not self._results:
            return {
                "overall_robustness": "uncertain",
                "key_findings": ["No sensitivity analyses completed"],
                "concerns": ["Unable to assess robustness"],
                "reasoning": "Auto-finalized: no sensitivity results obtained",
            }

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

        return {
            "overall_robustness": robustness,
            "key_findings": key_findings,
            "concerns": [],
            "reasoning": f"Auto-finalized with {len(self._results)} analyses completed",
        }

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool and return the result string."""
        if tool_name == "get_estimates_summary":
            return self._tool_get_estimates_summary()
        elif tool_name == "compute_e_value":
            return self._tool_compute_e_value(args)
        elif tool_name == "compute_rosenbaum_bounds":
            return self._tool_compute_rosenbaum_bounds(args)
        elif tool_name == "run_specification_curve":
            return self._tool_run_specification_curve(args)
        elif tool_name == "run_placebo_test":
            return self._tool_run_placebo_test(args)
        elif tool_name == "run_subgroup_analysis":
            return self._tool_run_subgroup_analysis(args)
        elif tool_name == "check_variance_stability":
            return self._tool_check_variance_stability(args)
        else:
            return f"Unknown tool: {tool_name}"

    def _tool_get_estimates_summary(self) -> str:
        """Get summary of treatment effect estimates."""
        if not self._state.treatment_effects:
            return "No treatment effects estimated yet."

        summary = "Treatment Effect Estimates Summary:\n\n"
        for i, effect in enumerate(self._state.treatment_effects):
            summary += f"""Method {i}: {effect.method}
  Estimand: {effect.estimand}
  Estimate: {effect.estimate:.4f}
  Std Error: {effect.std_error:.4f}
  95% CI: [{effect.ci_lower:.4f}, {effect.ci_upper:.4f}]
  P-value: {effect.p_value:.4f if effect.p_value else 'N/A'}
  Assumptions: {effect.assumptions_tested}

"""

        # Add summary statistics
        estimates = [e.estimate for e in self._state.treatment_effects]
        if len(estimates) > 1:
            summary += f"""Cross-Method Summary:
- Mean estimate: {np.mean(estimates):.4f}
- Std across methods: {np.std(estimates):.4f}
- Range: [{min(estimates):.4f}, {max(estimates):.4f}]
- All same sign: {'Yes' if (all(e > 0 for e in estimates) or all(e < 0 for e in estimates)) else 'No'}
"""

        return summary

    def _tool_compute_e_value(self, args: dict) -> str:
        """Compute E-value for unmeasured confounding."""
        method_index = args.get("method_index", 0)

        if method_index >= len(self._state.treatment_effects):
            return f"Invalid method index {method_index}"

        effect = self._state.treatment_effects[method_index]
        estimate = effect.estimate
        se = effect.std_error

        # Convert to approximate risk ratio for E-value calculation
        # For continuous outcomes, use approximate conversion
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
            interpretation = "ROBUST: Would need very strong unmeasured confounding to explain away"
            robustness_level = "high"
        elif e_value >= 2 and e_value_ci >= 1.2:
            interpretation = "MODERATELY ROBUST: Moderate confounding strength needed"
            robustness_level = "moderate"
        elif e_value >= 1.5:
            interpretation = "SOMEWHAT SENSITIVE: Relatively weak confounding could explain effect"
            robustness_level = "low"
        else:
            interpretation = "SENSITIVE: Even weak unmeasured confounding could explain the effect"
            robustness_level = "very low"

        # Store result
        result = SensitivityResult(
            method="E-value",
            robustness_value=float(e_value),
            interpretation=f"E-value = {e_value:.2f} (CI bound: {e_value_ci:.2f}): {interpretation}",
            details={
                "e_value_point": float(e_value),
                "e_value_ci": float(e_value_ci),
                "approximate_rr": float(rr),
                "robustness_level": robustness_level,
            },
        )
        self._results.append(result)

        return f"""E-Value Analysis for {effect.method}:

E-value (point estimate): {e_value:.2f}
E-value (confidence interval): {e_value_ci:.2f}
Approximate risk ratio: {rr:.2f}

Interpretation: {interpretation}

What this means:
- An unmeasured confounder would need to be associated with both treatment and outcome by a risk ratio of at least {e_value:.2f} to explain away the observed effect
- E-value > 2 is generally considered robust
- E-value for CI tells us minimum confounding to shift CI to include null
"""

    def _tool_compute_rosenbaum_bounds(self, args: dict) -> str:
        """Compute Rosenbaum bounds."""
        method_index = args.get("method_index", 0)

        if method_index >= len(self._state.treatment_effects):
            return f"Invalid method index {method_index}"

        effect = self._state.treatment_effects[method_index]
        estimate = effect.estimate
        se = effect.std_error

        # Approximate Gamma calculation
        z_score = abs(estimate / se) if se > 0 else 0
        gamma = 1 + z_score / 2

        # Interpretation
        if gamma >= 3:
            interpretation = "VERY ROBUST: Would need very strong hidden bias"
        elif gamma >= 2:
            interpretation = "MODERATELY ROBUST: Hidden bias would need to double odds of treatment"
        elif gamma >= 1.5:
            interpretation = "SOMEWHAT SENSITIVE: Moderate hidden bias could explain effect"
        else:
            interpretation = "SENSITIVE: Small hidden bias could explain effect"

        result = SensitivityResult(
            method="Rosenbaum Bounds (approximate)",
            robustness_value=float(gamma),
            interpretation=f"Gamma = {gamma:.2f}: {interpretation}",
            details={
                "gamma": float(gamma),
                "z_score": float(z_score),
            },
        )
        self._results.append(result)

        return f"""Rosenbaum Bounds Analysis:

Gamma (sensitivity parameter): {gamma:.2f}
Z-score: {z_score:.2f}

Interpretation: {interpretation}

What Gamma means:
- Gamma = 1 means no hidden bias
- Gamma = 2 means matched units could differ by factor of 2 in odds of treatment due to unmeasured confounder
- Higher Gamma = more robust to unmeasured confounding
"""

    def _tool_run_specification_curve(self, args: dict) -> str:
        """Run specification curve analysis."""
        n_specs = args.get("n_specifications", 10)

        from sklearn.linear_model import LinearRegression

        df = self._df
        T_col, Y_col = self._resolve_treatment_outcome()

        if not T_col or not Y_col:
            return "Treatment or outcome variable not identified."

        if T_col not in df.columns or Y_col not in df.columns:
            return "Treatment or outcome variable not in dataset."

        T = df[T_col].values
        Y = df[Y_col].values

        mask = ~(np.isnan(T) | np.isnan(Y))
        T = T[mask]
        Y = Y[mask]

        if len(T) < 50:
            return "Insufficient data for specification curve."

        # Get covariates
        covariates = []
        if self._state.data_profile:
            covariates = [
                c for c in self._state.data_profile.potential_confounders
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
        for i in range(min(n_specs - 1, len(covariates))):
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
            return "Could not compute multiple specifications."

        # Summary statistics
        est_mean = np.mean(estimates)
        est_std = np.std(estimates)
        est_range = np.max(estimates) - np.min(estimates)
        all_same_sign = all(e > 0 for e in estimates) or all(e < 0 for e in estimates)
        cv = est_std / abs(est_mean) if est_mean != 0 else float("inf")

        # Interpretation
        if cv < 0.2 and all_same_sign:
            interpretation = "HIGHLY STABLE: Estimates consistent across specifications"
            robustness = "high"
        elif cv < 0.4 and all_same_sign:
            interpretation = "MODERATELY STABLE: Some variation but same direction"
            robustness = "moderate"
        elif all_same_sign:
            interpretation = "VARIABLE: Magnitude varies but direction consistent"
            robustness = "moderate"
        else:
            interpretation = "UNSTABLE: Sign changes across specifications"
            robustness = "low"

        result = SensitivityResult(
            method="Specification Curve",
            robustness_value=float(1 - min(cv, 1)),
            interpretation=f"{interpretation} (CV={cv:.2f})",
            details={
                "n_specifications": len(estimates),
                "estimate_mean": float(est_mean),
                "estimate_std": float(est_std),
                "estimate_range": float(est_range),
                "all_same_sign": all_same_sign,
                "cv": float(cv),
            },
        )
        self._results.append(result)

        spec_results = "\n".join([f"  {s}: {e:.4f}" for s, e in zip(specs, estimates, strict=False)])

        return f"""Specification Curve Analysis ({len(estimates)} specifications):

Individual Specifications:
{spec_results}

Summary:
- Mean estimate: {est_mean:.4f}
- Std deviation: {est_std:.4f}
- Range: [{min(estimates):.4f}, {max(estimates):.4f}]
- All same sign: {'Yes' if all_same_sign else 'No'}
- Coefficient of variation: {cv:.2%}

Interpretation: {interpretation}
"""

    def _tool_run_placebo_test(self, args: dict) -> str:
        """Run placebo tests."""
        test_type = args.get("test_type", "both")
        n_placebos = args.get("n_placebos", 100)

        from sklearn.linear_model import LinearRegression

        df = self._df
        T_col, Y_col = self._resolve_treatment_outcome()

        if not T_col or not Y_col:
            return "Treatment or outcome variable not identified."

        T = df[T_col].dropna().values
        Y = df[Y_col].dropna().values

        n = min(len(T), len(Y))
        T = T[:n]
        Y = Y[:n]

        actual_effect = abs(self._state.treatment_effects[0].estimate) if self._state.treatment_effects else 0

        results_text = ""

        np.random.seed(42)

        if test_type in ["placebo_treatment", "both"]:
            # Placebo treatment test
            placebo_effects = []
            for _ in range(n_placebos):
                T_placebo = np.random.binomial(1, 0.5, size=len(Y))
                model = LinearRegression()
                model.fit(T_placebo.reshape(-1, 1), Y)
                placebo_effects.append(abs(model.coef_[0]))

            placebo_mean = np.mean(placebo_effects)
            placebo_p95 = np.percentile(placebo_effects, 95)
            ratio = actual_effect / (placebo_mean + 0.001)

            if actual_effect > 2 * placebo_p95:
                interp_t = "PASSED: Real effect far exceeds placebo distribution"
            elif actual_effect > placebo_p95:
                interp_t = "PASSED: Real effect exceeds 95th percentile of placebo"
            else:
                interp_t = "CONCERNING: Real effect within placebo distribution"

            results_text += f"""Placebo Treatment Test ({n_placebos} iterations):
- Actual effect: {actual_effect:.4f}
- Placebo mean: {placebo_mean:.4f}
- Placebo 95th percentile: {placebo_p95:.4f}
- Ratio (actual/placebo): {ratio:.2f}
- Interpretation: {interp_t}

"""

        if test_type in ["placebo_outcome", "both"]:
            # Placebo outcome test
            placebo_effects = []
            for _ in range(n_placebos):
                Y_placebo = np.random.randn(len(T))
                model = LinearRegression()
                model.fit(T.reshape(-1, 1), Y_placebo)
                placebo_effects.append(abs(model.coef_[0]))

            placebo_mean = np.mean(placebo_effects)
            placebo_p95 = np.percentile(placebo_effects, 95)
            ratio = actual_effect / (placebo_mean + 0.001)

            if actual_effect > 3 * placebo_mean:
                interp_o = "PASSED: Treatment has much larger effect on real vs placebo outcomes"
            elif actual_effect > 2 * placebo_mean:
                interp_o = "PASSED: Treatment effect notably larger on real outcome"
            else:
                interp_o = "CONCERNING: Treatment effect similar on real and placebo outcomes"

            results_text += f"""Placebo Outcome Test ({n_placebos} iterations):
- Actual effect: {actual_effect:.4f}
- Placebo mean: {placebo_mean:.4f}
- Ratio (actual/placebo): {ratio:.2f}
- Interpretation: {interp_o}

"""

        # Store result
        result = SensitivityResult(
            method=f"Placebo Test ({test_type})",
            robustness_value=float(ratio if 'ratio' in dir() else 1),
            interpretation=interp_t if test_type == "placebo_treatment" else (interp_o if test_type == "placebo_outcome" else f"{interp_t}; {interp_o}"),
            details={"test_type": test_type, "n_placebos": n_placebos},
        )
        self._results.append(result)

        return results_text

    def _tool_run_subgroup_analysis(self, args: dict) -> str:
        """Run subgroup analysis."""
        from sklearn.linear_model import LinearRegression

        subgroup_var = args.get("subgroup_variable")

        df = self._df
        T_col, Y_col = self._resolve_treatment_outcome()

        if not T_col or not Y_col:
            return "Treatment or outcome variable not identified."

        # Find subgroup variable
        if not subgroup_var:
            if self._state.data_profile:
                for col, dtype in self._state.data_profile.feature_types.items():
                    if dtype in ["categorical", "binary"] and col not in [T_col, Y_col]:
                        if 2 <= df[col].nunique() <= 5:
                            subgroup_var = col
                            break

        if not subgroup_var:
            # Create quartiles
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
            subgroup_effects.append(model.coef_[0])
            subgroup_labels.append(f"{subgroup_var}={sg_val}")

        if len(subgroup_effects) < 2:
            return "Insufficient subgroups for analysis."

        # Summary
        effect_mean = np.mean(subgroup_effects)
        effect_std = np.std(subgroup_effects)
        all_same_sign = all(e > 0 for e in subgroup_effects) or all(e < 0 for e in subgroup_effects)
        cv = effect_std / abs(effect_mean) if effect_mean != 0 else float("inf")

        if cv < 0.3 and all_same_sign:
            interpretation = "CONSISTENT: Effect similar across subgroups"
            robustness = "high"
        elif all_same_sign:
            interpretation = "DIRECTION CONSISTENT: Magnitude varies but direction stable"
            robustness = "moderate"
        else:
            interpretation = "HETEROGENEOUS: Effect varies including sign changes"
            robustness = "low"

        result = SensitivityResult(
            method="Subgroup Analysis",
            robustness_value=float(1 - min(cv, 1)),
            interpretation=interpretation,
            details={
                "subgroup_variable": subgroup_var,
                "n_subgroups": len(subgroup_effects),
                "cv": float(cv),
            },
        )
        self._results.append(result)

        sg_results = "\n".join([f"  {l}: {e:.4f}" for l, e in zip(subgroup_labels, subgroup_effects, strict=False)])

        return f"""Subgroup Analysis (by {subgroup_var}):

Subgroup Effects:
{sg_results}

Summary:
- Mean effect: {effect_mean:.4f}
- Std deviation: {effect_std:.4f}
- All same sign: {'Yes' if all_same_sign else 'No'}
- CV: {cv:.2%}

Interpretation: {interpretation}
"""

    def _tool_check_variance_stability(self, args: dict) -> str:
        """Check variance stability via bootstrap."""
        n_bootstrap = args.get("n_bootstrap", 200)

        from sklearn.linear_model import LinearRegression

        df = self._df
        T_col, Y_col = self._resolve_treatment_outcome()

        if not T_col or not Y_col:
            return "Treatment or outcome variable not identified."

        T = df[T_col].values
        Y = df[Y_col].values

        mask = ~(np.isnan(T) | np.isnan(Y))
        T = T[mask]
        Y = Y[mask]

        if len(T) < 50:
            return "Insufficient data for bootstrap."

        # Bootstrap
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

        boot_mean = np.mean(bootstrap_estimates)
        boot_std = np.std(bootstrap_estimates)
        boot_ci = (np.percentile(bootstrap_estimates, 2.5), np.percentile(bootstrap_estimates, 97.5))

        # Compare to reported SE
        reported_se = self._state.treatment_effects[0].std_error if self._state.treatment_effects else boot_std
        se_ratio = boot_std / (reported_se + 0.001)

        if 0.8 <= se_ratio <= 1.2:
            interpretation = "STABLE: Bootstrap SE consistent with reported SE"
        elif se_ratio < 0.8:
            interpretation = "CONSERVATIVE: Reported SE may be larger than necessary"
        else:
            interpretation = "UNSTABLE: Bootstrap SE larger than reported - some concern"

        result = SensitivityResult(
            method="Bootstrap Variance Check",
            robustness_value=float(min(se_ratio, 1 / se_ratio)),
            interpretation=interpretation,
            details={
                "bootstrap_se": float(boot_std),
                "reported_se": float(reported_se),
                "se_ratio": float(se_ratio),
            },
        )
        self._results.append(result)

        return f"""Bootstrap Variance Analysis ({n_bootstrap} iterations):

Bootstrap Results:
- Mean estimate: {boot_mean:.4f}
- Bootstrap SE: {boot_std:.4f}
- Bootstrap 95% CI: [{boot_ci[0]:.4f}, {boot_ci[1]:.4f}]

Comparison to Reported:
- Reported SE: {reported_se:.4f}
- Ratio (bootstrap/reported): {se_ratio:.2f}

Interpretation: {interpretation}
"""
