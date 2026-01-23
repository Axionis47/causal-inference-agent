"""ReAct Effect Estimator Agent - True agentic treatment effect estimation.

This agent uses the ReAct paradigm to autonomously:
1. Analyze data characteristics
2. Select appropriate estimation methods
3. Execute methods and observe results
4. Retry or adjust if needed
5. Synthesize findings
"""

import pickle
from typing import Any

import numpy as np
import pandas as pd

from src.agents.base import (
    AnalysisState,
    JobStatus,
    ReActAgent,
    ToolResult,
    ToolResultStatus,
    TreatmentEffectResult,
)
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


class EffectEstimatorReActAgent(ReActAgent):
    """ReAct agent for treatment effect estimation.

    This agent autonomously:
    1. Inspects the data to understand its characteristics
    2. Reasons about which methods are appropriate
    3. Executes methods one by one, observing results
    4. Handles errors by trying alternative approaches
    5. Synthesizes results and identifies the most credible estimate
    """

    AGENT_NAME = "effect_estimator_react"
    MAX_STEPS = 15

    SYSTEM_PROMPT = """You are an expert econometrician and causal inference practitioner.
Your role is to estimate treatment effects using rigorous statistical methods.

You have tools to:
1. inspect_data - Examine the dataset and its characteristics
2. analyze_treatment - Analyze the treatment variable distribution
3. check_assumptions - Check assumptions for specific methods
4. run_method - Execute a causal inference method
5. compare_results - Compare estimates across methods
6. finish - Complete the task with a summary

WORKFLOW:
1. First, ALWAYS inspect the data to understand what you're working with
2. Analyze the treatment variable to understand its distribution
3. Check assumptions for candidate methods
4. Run 2-4 appropriate methods based on data characteristics
5. Compare results and identify discrepancies
6. Finish with a synthesis of findings

CRITICAL RULES:
- NEVER run methods blindly - always check assumptions first
- If a method fails, try an alternative approach
- Run multiple methods for robustness
- Explain your reasoning at each step
- Be skeptical of estimates with very large standard errors
"""

    def __init__(self) -> None:
        """Initialize the ReAct effect estimator."""
        super().__init__()
        self._df: pd.DataFrame | None = None
        self._results: list[TreatmentEffectResult] = []
        self._register_tools()

    def _register_tools(self) -> None:
        """Register the estimation tools."""

        # Tool: Inspect data
        self.register_tool(
            name="inspect_data",
            description="Inspect the dataset to understand its characteristics for causal inference.",
            parameters={
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "enum": ["overview", "treatment", "outcome", "covariates"],
                        "description": "What aspect to focus on",
                    },
                },
                "required": ["focus"],
            },
            handler=self._inspect_data,
        )

        # Tool: Analyze treatment
        self.register_tool(
            name="analyze_treatment",
            description="Analyze the treatment variable distribution and identify potential issues.",
            parameters={
                "type": "object",
                "properties": {
                    "treatment_col": {
                        "type": "string",
                        "description": "Name of the treatment column",
                    },
                },
                "required": ["treatment_col"],
            },
            handler=self._analyze_treatment,
        )

        # Tool: Check assumptions
        self.register_tool(
            name="check_assumptions",
            description="Check assumptions for a specific causal inference method.",
            parameters={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["ols", "psm", "ipw", "aipw", "did", "iv", "rdd"],
                        "description": "Method to check assumptions for",
                    },
                    "treatment_col": {
                        "type": "string",
                        "description": "Treatment column",
                    },
                    "outcome_col": {
                        "type": "string",
                        "description": "Outcome column",
                    },
                },
                "required": ["method", "treatment_col", "outcome_col"],
            },
            handler=self._check_assumptions,
        )

        # Tool: Run method
        self.register_tool(
            name="run_method",
            description="Execute a causal inference method to estimate treatment effects.",
            parameters={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["ols", "psm", "ipw", "aipw", "did", "iv", "rdd", "s_learner", "t_learner", "causal_forest"],
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
            handler=self._run_method,
        )

        # Tool: Compare results
        self.register_tool(
            name="compare_results",
            description="Compare treatment effect estimates across methods.",
            parameters={
                "type": "object",
                "properties": {
                    "interpretation_focus": {
                        "type": "string",
                        "enum": ["magnitude", "significance", "robustness", "all"],
                        "description": "What to focus the comparison on",
                    },
                },
                "required": ["interpretation_focus"],
            },
            handler=self._compare_results,
        )

    def _get_initial_observation(self, state: AnalysisState) -> str:
        """Get initial observation from state."""
        obs = f"""Task: Estimate treatment effects for job {state.job_id}

Dataset: {state.dataset_info.name or state.dataset_info.url}
"""
        if state.data_profile:
            obs += f"""
Data Profile Available:
- Samples: {state.data_profile.n_samples}
- Features: {state.data_profile.n_features}
- Treatment candidates: {state.data_profile.treatment_candidates}
- Outcome candidates: {state.data_profile.outcome_candidates}
- Potential confounders: {state.data_profile.potential_confounders[:10]}
- Has time dimension: {state.data_profile.has_time_dimension}
- Potential instruments: {state.data_profile.potential_instruments}
"""
        if state.treatment_variable:
            obs += f"\nSelected treatment: {state.treatment_variable}"
        if state.outcome_variable:
            obs += f"\nSelected outcome: {state.outcome_variable}"

        obs += "\n\nStart by inspecting the data to understand what you're working with."

        # Load the dataframe
        if state.dataframe_path:
            try:
                with open(state.dataframe_path, "rb") as f:
                    self._df = pickle.load(f)
                obs += f"\nData loaded: {len(self._df)} rows, {len(self._df.columns)} columns"
            except Exception as e:
                obs += f"\nWarning: Could not load data: {e}"

        return obs

    async def is_task_complete(self, state: AnalysisState) -> bool:
        """Check if estimation is complete."""
        # Complete if we have at least 2 results
        return len(self._results) >= 2 and len(state.treatment_effects) >= 2

    async def _inspect_data(
        self,
        state: AnalysisState,
        focus: str,
    ) -> ToolResult:
        """Inspect the dataset."""
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="No data loaded",
            )

        df = self._df
        output = {}

        if focus == "overview":
            output = {
                "n_rows": len(df),
                "n_cols": len(df.columns),
                "columns": list(df.columns),
                "dtypes": {col: str(df[col].dtype) for col in df.columns[:20]},
                "missing_pct": {col: f"{df[col].isna().mean()*100:.1f}%" for col in df.columns if df[col].isna().any()},
            }
        elif focus == "treatment" and state.data_profile:
            candidates = state.data_profile.treatment_candidates[:5]
            output = {
                "candidates": candidates,
                "distributions": {},
            }
            for col in candidates:
                if col in df.columns:
                    output["distributions"][col] = {
                        "unique_values": int(df[col].nunique()),
                        "value_counts": df[col].value_counts().head(5).to_dict(),
                    }
        elif focus == "outcome" and state.data_profile:
            candidates = state.data_profile.outcome_candidates[:5]
            output = {
                "candidates": candidates,
                "statistics": {},
            }
            for col in candidates:
                if col in df.columns and np.issubdtype(df[col].dtype, np.number):
                    output["statistics"][col] = {
                        "mean": float(df[col].mean()),
                        "std": float(df[col].std()),
                        "min": float(df[col].min()),
                        "max": float(df[col].max()),
                    }
        elif focus == "covariates" and state.data_profile:
            confounders = state.data_profile.potential_confounders[:10]
            output = {
                "potential_confounders": confounders,
                "n_confounders": len(state.data_profile.potential_confounders),
                "numeric_covariates": [c for c in confounders if c in df.columns and np.issubdtype(df[c].dtype, np.number)],
            }

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=output,
        )

    async def _analyze_treatment(
        self,
        state: AnalysisState,
        treatment_col: str,
    ) -> ToolResult:
        """Analyze the treatment variable."""
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="No data loaded",
            )

        if treatment_col not in self._df.columns:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Column '{treatment_col}' not found",
            )

        T = self._df[treatment_col]
        n_unique = T.nunique()
        is_binary = n_unique == 2

        output = {
            "column": treatment_col,
            "n_unique": int(n_unique),
            "is_binary": is_binary,
            "value_counts": T.value_counts().to_dict(),
            "missing": int(T.isna().sum()),
        }

        if is_binary:
            output["treatment_prevalence"] = f"{T.mean()*100:.1f}%"
            output["n_treated"] = int(T.sum())
            output["n_control"] = int(len(T) - T.sum())
        else:
            output["recommendation"] = "Consider binarizing treatment (e.g., above/below median)"

        # Update state
        state.treatment_variable = treatment_col

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=output,
        )

    async def _check_assumptions(
        self,
        state: AnalysisState,
        method: str,
        treatment_col: str,
        outcome_col: str,
    ) -> ToolResult:
        """Check assumptions for a method."""
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="No data loaded",
            )

        df = self._df
        checks = {}
        warnings = []
        can_proceed = True

        # Common checks
        if treatment_col not in df.columns:
            return ToolResult(status=ToolResultStatus.ERROR, output=None, error=f"Treatment '{treatment_col}' not found")
        if outcome_col not in df.columns:
            return ToolResult(status=ToolResultStatus.ERROR, output=None, error=f"Outcome '{outcome_col}' not found")

        T = df[treatment_col]
        Y = df[outcome_col]

        # Sample size check
        n = len(df.dropna(subset=[treatment_col, outcome_col]))
        checks["sample_size"] = {"n": n, "sufficient": n >= 100}
        if n < 100:
            warnings.append("Small sample size may lead to unreliable estimates")

        # Treatment variation check
        n_treated = (T == 1).sum() if T.nunique() == 2 else (T > T.median()).sum()
        n_control = len(T) - n_treated
        checks["treatment_variation"] = {
            "n_treated": int(n_treated),
            "n_control": int(n_control),
            "ratio": f"{n_treated/n_control:.2f}" if n_control > 0 else "inf",
        }
        if n_treated < 20 or n_control < 20:
            warnings.append("Low treatment/control counts")
            can_proceed = False

        # Method-specific checks
        if method in ["psm", "ipw", "aipw"]:
            checks["overlap"] = "Requires propensity score overlap (will check during estimation)"
            checks["unconfoundedness"] = "Assumes no unmeasured confounders (untestable)"

        if method == "did":
            if not state.data_profile or not state.data_profile.has_time_dimension:
                warnings.append("DiD requires time dimension - not detected in data")
                can_proceed = False
            else:
                checks["parallel_trends"] = "Requires parallel trends assumption (partially testable)"

        if method == "iv":
            if not state.data_profile or not state.data_profile.potential_instruments:
                warnings.append("IV requires instruments - none detected")
                can_proceed = False
            else:
                checks["instruments"] = f"Found instruments: {state.data_profile.potential_instruments}"

        if method == "rdd":
            if not state.data_profile or not state.data_profile.discontinuity_candidates:
                warnings.append("RDD requires running variable - none detected")
                can_proceed = False

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "method": method,
                "checks": checks,
                "warnings": warnings,
                "can_proceed": can_proceed,
            },
        )

    async def _run_method(
        self,
        state: AnalysisState,
        method: str,
        treatment_col: str,
        outcome_col: str,
        covariates: list[str] | None = None,
    ) -> ToolResult:
        """Run a causal inference method."""
        if self._df is None:
            return ToolResult(status=ToolResultStatus.ERROR, output=None, error="No data loaded")

        df = self._df
        covariates = covariates or []

        # Filter to available columns
        available_covariates = [c for c in covariates if c in df.columns]

        try:
            from src.causal.estimators import EffectEstimatorEngine

            engine = EffectEstimatorEngine(confidence_level=0.95)

            # Prepare kwargs for special methods
            kwargs = {}
            if method == "did" and state.data_profile and state.data_profile.time_column:
                kwargs["time_col"] = state.data_profile.time_column
            if method == "iv" and state.data_profile and state.data_profile.potential_instruments:
                kwargs["instruments"] = state.data_profile.potential_instruments
            if method == "rdd" and state.data_profile and state.data_profile.discontinuity_candidates:
                kwargs["running_var"] = state.data_profile.discontinuity_candidates[0]

            # Run the method
            result = engine.run_method(
                method_name=method,
                df=df,
                treatment_col=treatment_col,
                outcome_col=outcome_col,
                covariates=available_covariates if available_covariates else None,
                **kwargs,
            )

            # Convert to TreatmentEffectResult
            effect_result = TreatmentEffectResult(
                method=result.method,
                estimand=result.estimand,
                estimate=result.estimate,
                std_error=result.std_error,
                ci_lower=result.ci_lower,
                ci_upper=result.ci_upper,
                p_value=result.p_value,
                assumptions_tested=result.assumptions_tested,
                details={
                    "n_treated": result.n_treated,
                    "n_control": result.n_control,
                    "diagnostics": result.diagnostics,
                    **result.details,
                },
            )

            # Store result
            self._results.append(effect_result)
            state.treatment_effects.append(effect_result)

            # Update state variables
            state.treatment_variable = treatment_col
            state.outcome_variable = outcome_col

            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                output={
                    "method": result.method,
                    "estimand": result.estimand,
                    "estimate": f"{result.estimate:.4f}",
                    "std_error": f"{result.std_error:.4f}",
                    "ci": f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]",
                    "p_value": f"{result.p_value:.4f}" if result.p_value else "N/A",
                    "n_treated": result.n_treated,
                    "n_control": result.n_control,
                    "significant": result.p_value < 0.05 if result.p_value else None,
                },
            )

        except Exception as e:
            self.logger.warning("method_failed", method=method, error=str(e))
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Method {method} failed: {str(e)}",
            )

    async def _compare_results(
        self,
        state: AnalysisState,
        interpretation_focus: str,
    ) -> ToolResult:
        """Compare results across methods."""
        if not self._results:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="No results to compare",
            )

        comparison = {
            "n_methods": len(self._results),
            "estimates": [],
        }

        for r in self._results:
            comparison["estimates"].append({
                "method": r.method,
                "estimate": r.estimate,
                "std_error": r.std_error,
                "ci": [r.ci_lower, r.ci_upper],
            })

        estimates = [r.estimate for r in self._results]
        comparison["mean_estimate"] = float(np.mean(estimates))
        comparison["std_across_methods"] = float(np.std(estimates))

        # Check for consistency
        ci_overlaps = all(
            r1.ci_lower <= r2.ci_upper and r2.ci_lower <= r1.ci_upper
            for i, r1 in enumerate(self._results)
            for r2 in self._results[i+1:]
        )
        comparison["ci_overlap"] = ci_overlaps

        if interpretation_focus in ["robustness", "all"]:
            comparison["robustness_assessment"] = (
                "Estimates are consistent across methods (CIs overlap)"
                if ci_overlaps
                else "CAUTION: Estimates vary substantially across methods"
            )

        if interpretation_focus in ["significance", "all"]:
            significant_count = sum(1 for r in self._results if r.p_value and r.p_value < 0.05)
            comparison["significance"] = f"{significant_count}/{len(self._results)} methods show significant effect"

        # Identify preferred estimate
        # Prefer AIPW/doubly robust, then IPW/PSM, then OLS
        preferred = None
        for r in self._results:
            if "aipw" in r.method.lower() or "doubly" in r.method.lower():
                preferred = r
                break
            if "ipw" in r.method.lower() or "psm" in r.method.lower() or "matching" in r.method.lower():
                preferred = preferred or r
        preferred = preferred or self._results[0]

        comparison["preferred_method"] = preferred.method
        comparison["preferred_estimate"] = preferred.estimate
        comparison["preferred_reasoning"] = "Doubly robust methods are preferred when available"

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=comparison,
        )

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute the ReAct estimation loop."""
        state.status = JobStatus.ESTIMATING_EFFECTS
        self._results = []  # Reset results

        # Run the ReAct loop
        state = await super().execute(state)

        self.logger.info(
            "estimation_complete",
            n_results=len(self._results),
            methods=[r.method for r in self._results],
        )

        return state
