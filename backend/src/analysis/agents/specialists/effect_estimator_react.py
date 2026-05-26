"""ReAct Effect Estimator Agent - True agentic treatment effect estimation.

This agent uses the ReAct paradigm to autonomously:
1. Analyze data characteristics
2. Select appropriate estimation methods
3. Execute methods and observe results
4. Retry or adjust if needed
5. Synthesize findings
"""


import numpy as np
import pandas as pd

from src.analysis.agents.base import (
    AnalysisState,
    JobStatus,
    ReActAgent,
    ToolResult,
    ToolResultStatus,
    TreatmentEffectResult,
)
from src.analysis.agents.registry import register_agent
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


@register_agent("effect_estimator_react")
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

    # Agent metadata (used by registry and orchestrator)
    WRITES_STATE_FIELDS = ["treatment_effects"]
    REQUIRED_STATE_FIELDS = ["data_profile", "dataframe_path"]
    JOB_STATUS = JobStatus.ESTIMATING_EFFECTS
    PROGRESS_WEIGHT = 0.15

    SYSTEM_PROMPT = """You are an expert econometrician and causal inference practitioner.
Your role is to estimate treatment effects using rigorous statistical methods.

You have tools to:
1. inspect_data - Examine the dataset and its characteristics
2. analyze_treatment - Analyze the treatment variable distribution
3. check_assumptions - Check assumptions for specific methods
4. run_method - Execute a causal inference method
5. compare_results - Compare estimates across methods

WORKFLOW (you have a 15-step budget — finalize before it runs out):
1. First, ALWAYS inspect the data to understand what you're working with
2. Analyze the treatment variable to understand its distribution
3. Check assumptions for candidate methods
4. Run 2-4 appropriate methods based on data characteristics
5. Compare results and identify discrepancies
6. Call the `finish` tool with a one-paragraph synthesis

WHEN TO FINISH:
- As soon as you have 2 successful run_method calls with valid estimates,
  prepare to finish. One more compare_results call is fine; do not keep
  retrying methods past that point.
- If the same method has failed twice with the same error, stop retrying it
  and switch to a different method or finish with what you have.
- By step 12, regardless of how many methods succeeded, call `finish`.
  An auto-finalize fallback will run an OLS baseline if you produced
  nothing, but a finish call with your actual reasoning is always better.

CRITICAL RULES:
- NEVER run methods blindly - always check assumptions first
- If a method fails, try an alternative approach (different method, not the same one)
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
        """Get LEAN initial observation - minimal context, use tools for details.

        This reduces token usage by ~500 tokens per call by not dumping
        the full data profile upfront. Agents should use context query tools
        to pull specific information as needed.
        """
        obs = f"""Task: Estimate treatment effects for job {state.job_id}
Dataset: {state.dataset_info.name or state.dataset_info.url}

Key variables:
- Treatment: {state.treatment_variable or "Use get_treatment_outcome tool to confirm"}
- Outcome: {state.outcome_variable or "Use get_treatment_outcome tool to confirm"}
"""
        # Add minimal data availability info
        if state.data_profile:
            obs += f"- Samples: {state.data_profile.n_samples}, Features: {state.data_profile.n_features}\n"

        obs += """
Start by inspecting the data overview to understand what you're working with.
Then analyze the treatment variable, check assumptions, and run estimation methods."""

        # Load the dataframe
        if state.dataframe_path:
            try:
                self._df = pd.read_parquet(state.dataframe_path)
                obs += f"\n\nData loaded: {len(self._df)} rows, {len(self._df.columns)} columns"
            except Exception as e:
                obs += f"\n\nWarning: Could not load data: {e}"

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
        """Run a causal inference method via the unified engine."""
        if self._df is None:
            return ToolResult(status=ToolResultStatus.ERROR, output=None, error="No data loaded")

        try:
            from src.causal.estimators.effect_estimator import EffectEstimatorEngine

            engine = EffectEstimatorEngine(confidence_level=0.95)
            effect_result = engine.run_method_safe(
                method=method,
                df=self._df,
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

            # Store result
            self._results.append(effect_result)
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
            self.logger.warning("method_failed", method=method, error=str(e))
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
        """Execute the ReAct estimation loop with an auto-finalize fallback.

        The 15-step ReAct budget is not always enough — on Lalonde the model
        burned all 15 calling run_method on inapplicable methods (PSM/IPW
        without overlap, IV without instruments) and finished with zero
        valid estimates. The imperative variant handles this via
        _auto_finalize() (effect_estimation/agent.py:876–921); we mirror
        the contract here so an exhausted react loop still produces a
        baseline OLS estimate that downstream stages can ground on.
        """
        self._results = []  # Reset results

        # Run the ReAct loop
        state = await super().execute(state)

        # Fallback: if the loop produced nothing, run a guaranteed-applicable
        # OLS baseline using the same engine the LLM would have called. This
        # prevents the critique from REJECT-ing the analysis purely because
        # the estimator couldn't pick a working method within its budget.
        if not state.treatment_effects:
            self._auto_finalize_ols(state)

        self.logger.info(
            "estimation_complete",
            n_results=len(self._results),
            methods=[r.method for r in self._results],
            auto_finalized=not self._results
            or all(r.method.lower().startswith("ols") for r in self._results),
        )

        return state

    def _auto_finalize_ols(self, state: AnalysisState) -> None:
        """Run an OLS baseline when the loop ended with no valid estimates.

        Resolves covariates from the same priority chain the imperative
        estimator uses (DAG adjustment set → confounder discovery →
        data profile potential confounders) and delegates to the
        EffectEstimatorEngine so encoding / sample-size gating runs
        identically to a normal run_method call. Updates self._results
        and state.treatment_effects in place; logs (and swallows) any
        exception so the surrounding orchestrator loop can continue.
        """
        if self._df is None and state.dataframe_path:
            try:
                self._df = pd.read_parquet(state.dataframe_path)
            except Exception as exc:
                self.logger.warning("auto_finalize_load_failed", error=str(exc))
                return

        if self._df is None:
            return

        treatment = state.treatment_variable
        outcome = state.outcome_variable
        if not treatment or not outcome:
            return

        covariates = self._resolve_baseline_covariates(state)

        try:
            from src.causal.estimators.effect_estimator import EffectEstimatorEngine

            engine = EffectEstimatorEngine(confidence_level=0.95)
            ols_result = engine.run_method_safe(
                method="ols",
                df=self._df,
                treatment_col=treatment,
                outcome_col=outcome,
                covariates=covariates,
                current_state=state,
            )
        except Exception as exc:
            self.logger.warning("auto_finalize_ols_raised", error=str(exc))
            state.push_decision(
                agent="effect_estimator_react",
                decision_type="auto_finalize_failed",
                choice="ols",
                reason=f"Fallback OLS raised: {exc}",
            )
            return

        if ols_result is None:
            self.logger.warning("auto_finalize_ols_no_result")
            state.push_decision(
                agent="effect_estimator_react",
                decision_type="auto_finalize_failed",
                choice="ols",
                reason="Fallback OLS returned no result (insufficient data).",
            )
            return

        ols_result.treatment_variable = treatment
        ols_result.outcome_variable = outcome
        self._results.append(ols_result)
        state.treatment_effects.append(ols_result)
        state.push_decision(
            agent="effect_estimator_react",
            decision_type="auto_finalize_baseline",
            choice="ols",
            reason=(
                f"ReAct loop exited with no valid estimates after {self.MAX_STEPS} steps; "
                f"emitted OLS baseline with {len(covariates)} covariate(s)."
            ),
        )

    def _resolve_baseline_covariates(self, state: AnalysisState) -> list[str]:
        """Same priority chain as the imperative estimator, simplified.

        Skips the one-hot-encoding shortcut: the engine's run_method_safe
        already filters to numeric columns, and at this point the pipeline
        has already done any pre-processing it was going to do.
        """
        if self._df is None:
            return []
        treatment = state.treatment_variable
        outcome = state.outcome_variable

        def _filter(cols: list[str]) -> list[str]:
            return [
                c for c in cols
                if c in self._df.columns and c != treatment and c != outcome
            ]

        if state.proposed_dag and state.proposed_dag.adjustment_set:
            covs = _filter(state.proposed_dag.adjustment_set)
            if covs:
                return covs

        if state.confounder_discovery:
            ranked = state.confounder_discovery.get("ranked_confounders", [])
            if ranked and isinstance(ranked[0], dict):
                ranked = [c.get("variable") or c.get("name") or "" for c in ranked]
            covs = _filter([c for c in ranked if c])
            if covs:
                return covs

        if state.data_profile and state.data_profile.potential_confounders:
            covs = _filter(state.data_profile.potential_confounders)
            if covs:
                return covs

        return []
