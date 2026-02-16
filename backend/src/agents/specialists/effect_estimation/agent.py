"""Effect Estimator Agent - ReAct-based treatment effect estimation.

This agent uses the ReAct pattern to iteratively:
1. Analyze data characteristics
2. Check covariate balance and propensity score overlap
3. Select and run appropriate estimation methods
4. Evaluate diagnostics and compare results
5. Iterate until confident in the estimate

Uses pull-based context - queries for information on demand.
"""

import json
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.agents.base import (
    AnalysisState,
    CausalPair,
    JobStatus,
    ToolResult,
    ToolResultStatus,
    TreatmentEffectResult,
)
from src.agents.base.context_tools import ContextTools
from src.agents.base.react_agent import ReActAgent
from src.agents.registry import register_agent
from src.logging_config.structured import get_logger

from .diagnostics import (
    check_influence_diagnostics,
    check_residual_diagnostics,
    check_specification_diagnostics,
)
from .estimation_methods import run_method
from .method_selector import SampleSizeThresholds, find_closest_column

logger = get_logger(__name__)


@register_agent("effect_estimator")
class EffectEstimatorAgent(ReActAgent, ContextTools):
    """ReAct-based treatment effect estimator.

    This agent iteratively:
    1. Queries domain knowledge and prior agent findings
    2. Investigates data characteristics
    3. Checks balance and overlap
    4. Runs estimation methods one at a time
    5. Evaluates diagnostics
    6. Compares results across methods
    7. Decides when to stop and which estimate to trust

    Uses pull-based context - queries for information on demand.
    """

    AGENT_NAME = "effect_estimator"
    MAX_STEPS = 20

    SYSTEM_PROMPT = """You are an expert econometrician and causal inference practitioner.
Your task is to estimate treatment effects using rigorous statistical methods.

CRITICAL WORKFLOW - Follow this order strictly:
1. Steps 1-3: Quick data gathering (get_treatment_outcome, get_data_summary, get_dag_adjustment_set)
2. Steps 4-5: Run your FIRST estimation method using run_estimation_method (start with "ols")
3. Steps 6+: Run additional methods (ipw, aipw, matching) and compare results
4. Final step: Call finalize_estimation or finish

⚠️ CRITICAL RULE: You MUST call run_estimation_method by step 5 at the latest.
Do NOT spend more than 3 steps on information gathering. The primary goal is to
produce treatment effect estimates, not to gather information indefinitely.

ESTIMATION TOOLS (use these - they are your primary tools):
- run_estimation_method: Run OLS, IPW, AIPW, matching, etc. ALWAYS start with "ols"
- check_method_diagnostics: Check quality of last estimate
- compare_estimates: Compare results across methods
- finalize_estimation: Finalize with best estimate

CONTEXT TOOLS (use sparingly, max 2-3 calls total):
- get_treatment_outcome: Get treatment/outcome variable names
- get_data_summary: Quick data overview
- get_dag_adjustment_set: Get confounders from causal DAG

Method selection: Start with OLS (always works), then try IPW and AIPW if covariates available.

Call tools iteratively. After running each method, decide whether to run another or finalize."""

    REQUIRED_STATE_FIELDS = ["data_profile", "dataframe_path"]
    WRITES_STATE_FIELDS = ["treatment_effects"]
    JOB_STATUS = "estimating_effects"
    PROGRESS_WEIGHT = 60

    def __init__(self) -> None:
        """Initialize the effect estimator agent."""
        super().__init__()

        # Register context query tools from mixin
        self.register_context_tools()

        # Internal state
        self._df: pd.DataFrame | None = None
        self._treatment_var: str | None = None
        self._outcome_var: str | None = None
        self._covariates: list[str] = []
        self._results: list[TreatmentEffectResult] = []
        self._propensity_scores: np.ndarray | None = None
        self._last_method_result: TreatmentEffectResult | None = None
        self._current_state: AnalysisState | None = None
        self._finalized: bool = False
        self._final_result: dict[str, Any] = {}

        # Register estimation-specific tools
        self._register_estimation_tools()

    def _register_estimation_tools(self) -> None:
        """Register tools for effect estimation."""
        self.register_tool(
            name="get_data_summary",
            description="Get summary statistics of the dataset including sample sizes, treatment/control split, outcome distribution, and available covariates.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=self._tool_get_data_summary,
        )

        self.register_tool(
            name="check_covariate_balance",
            description="Check balance of covariates between treatment and control groups. Returns standardized mean differences.",
            parameters={
                "type": "object",
                "properties": {
                    "covariates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of covariate names to check. If empty, checks all available covariates.",
                    },
                },
                "required": [],
            },
            handler=self._tool_check_covariate_balance,
        )

        self.register_tool(
            name="estimate_propensity_scores",
            description="Estimate propensity scores and check overlap between treatment groups.",
            parameters={
                "type": "object",
                "properties": {
                    "covariates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Covariates to use for propensity model. If empty, uses discovered confounders or all available.",
                    },
                },
                "required": [],
            },
            handler=self._tool_estimate_propensity_scores,
        )

        self.register_tool(
            name="run_estimation_method",
            description="Run a specific causal estimation method and get the treatment effect estimate.",
            parameters={
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": [
                            "ols",
                            "ipw",
                            "aipw",
                            "matching",
                            "s_learner",
                            "t_learner",
                            "x_learner",
                            "causal_forest",
                            "double_ml",
                            "did",
                            "iv",
                            "rdd",
                        ],
                        "description": "The estimation method to run",
                    },
                    "covariates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Covariates to adjust for. If empty, uses discovered confounders.",
                    },
                },
                "required": ["method"],
            },
            handler=self._tool_run_estimation_method,
        )

        self.register_tool(
            name="check_method_diagnostics",
            description="Check diagnostics for the most recently run method (e.g., residual analysis, influence points).",
            parameters={
                "type": "object",
                "properties": {
                    "diagnostic_type": {
                        "type": "string",
                        "enum": ["residuals", "influence", "specification", "all"],
                        "description": "Type of diagnostic to run",
                    },
                },
                "required": ["diagnostic_type"],
            },
            handler=self._tool_check_method_diagnostics,
        )

        self.register_tool(
            name="compare_estimates",
            description="Compare all estimates obtained so far across methods.",
            parameters={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=self._tool_compare_estimates,
        )

        self.register_tool(
            name="finalize_estimation",
            description="Finalize the analysis with your conclusions. Call this when you have enough evidence.",
            parameters={
                "type": "object",
                "properties": {
                    "preferred_method": {
                        "type": "string",
                        "description": "Which method's estimate you consider most credible",
                    },
                    "preferred_estimate": {
                        "type": "number",
                        "description": "The treatment effect estimate you recommend",
                    },
                    "confidence_level": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Your confidence in this estimate",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Detailed reasoning for your conclusions",
                    },
                    "caveats": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Important caveats about the estimate",
                    },
                },
                "required": ["preferred_method", "preferred_estimate", "confidence_level", "reasoning"],
            },
            handler=self._tool_finalize_estimation,
        )

    # =========================================================================
    # ReAct Required Methods
    # =========================================================================

    def _get_initial_observation(self, state: AnalysisState) -> str:
        """Get the initial observation for the ReAct loop."""
        obs = f"""You are estimating treatment effects for a causal analysis.

Treatment variable: {self._treatment_var}
Outcome variable: {self._outcome_var}
Available covariates: {self._covariates[:20]}{'...' if len(self._covariates) > 20 else ''}
Number of samples: {len(self._df) if self._df is not None else 'unknown'}

⚠️ ACTION PLAN - Follow this exactly:
Step 1: Call run_estimation_method with method="ols"
Step 2: Call run_estimation_method with method="ipw"
Step 3: Call run_estimation_method with method="aipw"
Step 4: Call compare_estimates to see results
Step 5: Call finalize_estimation with your best estimate

You MUST attempt ALL three methods (OLS, IPW, AIPW). If a method fails due to
missing covariates or other issues, the tool will return an error - that is fine,
just proceed to the next method.

START NOW: Call run_estimation_method with method="ols"."""

        # Add context from state if available
        if state.confounder_discovery:
            confounders = state.confounder_discovery.get("ranked_confounders", [])
            if confounders:
                obs += f"\n\nConfounders identified: {confounders[:5]}"

        if state.data_profile:
            if state.data_profile.has_time_dimension:
                obs += " Has time dimension (DiD may be applicable)."
            if state.data_profile.potential_instruments:
                obs += f" Potential instruments: {state.data_profile.potential_instruments}"

        return obs

    def _build_react_prompt(
        self,
        state: AnalysisState,
        observation: str,
        step_num: int,
    ) -> str:
        """Build the prompt for a ReAct step with urgency for estimation."""
        prompt = f"""Step {step_num}/{self.MAX_STEPS}

OBSERVATION:
{observation}

"""
        # Add previous steps context (last 3 steps)
        if self._steps:
            prompt += "PREVIOUS STEPS:\n"
            for step in self._steps[-3:]:
                prompt += f"Step {step.step_number}: Action={step.action}, "
                prompt += f"Result={'Success' if step.result and step.result.status == ToolResultStatus.SUCCESS else 'Failed'}\n"
            prompt += "\n"

        # Add urgency based on step number and whether estimation has been done
        has_run_estimation = any(
            s.action == "run_estimation_method" for s in self._steps
        )

        if not has_run_estimation and step_num >= 3:
            prompt += """⚠️ URGENT: You have NOT called run_estimation_method yet!
You MUST call run_estimation_method with method="ols" RIGHT NOW.
Do NOT call any more information-gathering tools.

"""
        elif not has_run_estimation and step_num >= 2:
            prompt += """REMINDER: Call run_estimation_method with method="ols" on this step.
Information gathering is complete enough to proceed.

"""
        elif has_run_estimation and len(self._results) >= 2:
            prompt += """You have multiple estimates. Consider calling finalize_estimation or compare_estimates.

"""

        prompt += """Based on the observation, think step by step:
1. What does this observation tell me?
2. What should I do next to make progress?
3. Which tool should I use?

Then call the appropriate tool."""

        return prompt

    async def is_task_complete(self, state: AnalysisState) -> bool:
        """Check if estimation task is complete."""
        return self._finalized

    # =========================================================================
    # Multi-Pair Causal Analysis: LLM-driven pair selection
    # =========================================================================

    async def _identify_valid_causal_pairs(
        self,
        state: AnalysisState,
    ) -> list[tuple[str, str, str]]:
        """Identify valid treatment-outcome pairs using LLM filtering.

        Prioritizes:
        1. User-specified variables (if both provided)
        2. Single candidates (no LLM needed)
        3. LLM selection from multiple candidates

        Returns:
            List of (treatment, outcome, rationale) tuples
        """
        profile = state.data_profile

        # Priority 1: User specified both variables - use those
        if state.treatment_variable and state.outcome_variable:
            treatment = state.treatment_variable
            outcome = state.outcome_variable

            if self._df is not None:
                matched_treatment = find_closest_column(treatment, list(self._df.columns))
                matched_outcome = find_closest_column(outcome, list(self._df.columns))
                if matched_treatment:
                    treatment = matched_treatment
                if matched_outcome:
                    outcome = matched_outcome

            self.logger.info(
                "using_user_specified_variables",
                treatment=treatment,
                outcome=outcome,
            )
            return [(
                treatment,
                outcome,
                "User specified"
            )]

        # Priority 2: No profile - can't infer
        if profile is None:
            return []

        treatment_candidates = profile.treatment_candidates or []
        outcome_candidates = profile.outcome_candidates or []

        # Priority 3: No candidates found
        if not treatment_candidates or not outcome_candidates:
            return []

        # Priority 4: Single candidate each - no LLM needed
        if len(treatment_candidates) == 1 and len(outcome_candidates) == 1:
            self.logger.info(
                "single_candidates",
                treatment=treatment_candidates[0],
                outcome=outcome_candidates[0],
            )
            return [(
                treatment_candidates[0],
                outcome_candidates[0],
                "Single candidates identified"
            )]

        # Priority 5: Multiple candidates - use LLM to filter
        self.logger.info(
            "multiple_candidates_llm_filtering",
            n_treatments=len(treatment_candidates),
            n_outcomes=len(outcome_candidates),
        )

        prompt = self._build_pair_selection_prompt(profile)
        try:
            result = await self.llm.generate(
                prompt=prompt,
                system_instruction="You are an expert in causal inference. Return valid JSON only.",
            )
            pairs = self._parse_pair_selection({"response": result.text}, profile)
            if pairs:
                return pairs
        except Exception as e:
            self.logger.warning("pair_selection_llm_failed", error=str(e))

        # Fallback: Use first candidates
        self.logger.info("fallback_to_first_candidates")
        return [(
            treatment_candidates[0],
            outcome_candidates[0],
            "Fallback to first candidates"
        )]

    def _build_pair_selection_prompt(self, profile) -> str:
        """Build LLM prompt for causal pair selection."""
        return f"""You are evaluating potential causal relationships in a dataset.

Dataset Profile:
- Total features: {profile.n_features}
- Feature names: {profile.feature_names[:30]}{"..." if len(profile.feature_names) > 30 else ""}
- Treatment candidates: {profile.treatment_candidates}
- Outcome candidates: {profile.outcome_candidates}
- Potential confounders: {profile.potential_confounders[:10]}{"..." if len(profile.potential_confounders) > 10 else ""}

Your task: Identify which treatment-outcome pairs represent VALID causal questions.

A valid causal pair requires:
1. Temporal ordering: Treatment could plausibly precede outcome
2. Manipulability: Treatment is something that could be intervened upon
3. Non-identity: Treatment and outcome measure different concepts
4. Plausible mechanism: There's a reasonable pathway for effect

INVALID pairs include:
- Demographic → Demographic (age cannot cause gender)
- Outcome → Treatment (reverse causality)
- Proxies of each other (revenue_usd → revenue_eur)
- Treatment cannot affect outcome by design

Return your analysis as JSON (no markdown, just raw JSON):
{{
    "valid_pairs": [
        {{"treatment": "var_name", "outcome": "var_name", "rationale": "brief explanation", "priority": 1}}
    ],
    "rejected_pairs": [
        {{"treatment": "var_name", "outcome": "var_name", "reason": "why invalid"}}
    ]
}}

IMPORTANT:
- Limit to at most 3 valid pairs (prioritize the most scientifically interesting)
- Priority 1 = most important, 2 = secondary, 3 = exploratory
- Only include pairs where BOTH variables are in the candidates lists above
"""

    def _parse_pair_selection(
        self,
        result: dict,
        profile,
    ) -> list[tuple[str, str, str]]:
        """Parse LLM response to extract valid pairs."""
        response_text = result.get("response", "")

        try:
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                else:
                    return []

            data = json.loads(json_str)
            valid_pairs = data.get("valid_pairs", [])

            pairs = []
            valid_treatments = set(profile.treatment_candidates or [])
            valid_outcomes = set(profile.outcome_candidates or [])

            for p in valid_pairs[:3]:
                treatment = p.get("treatment")
                outcome = p.get("outcome")
                rationale = p.get("rationale", "LLM selected")

                if treatment in valid_treatments and outcome in valid_outcomes:
                    pairs.append((treatment, outcome, rationale))
                    self.logger.info(
                        "valid_pair_identified",
                        treatment=treatment,
                        outcome=outcome,
                        rationale=rationale[:100],
                    )

            return pairs

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            self.logger.warning("pair_selection_parse_failed", error=str(e))
            return []

    def _get_covariates_for_pair(
        self,
        state: AnalysisState,
        treatment: str,
        outcome: str,
    ) -> list[str]:
        """Get covariates for a specific treatment-outcome pair.

        Handles both numeric and categorical confounders. Categorical columns
        are automatically one-hot encoded (drop_first) and the dummy columns
        are added to self._df so downstream estimation tools can use them.
        """
        raw_confounders: list[str] = []

        # Priority 1: Use discovered confounders
        if state.confounder_discovery and state.confounder_discovery.get("ranked_confounders"):
            raw_confounders = [
                c for c in state.confounder_discovery["ranked_confounders"]
                if c in self._df.columns and c != treatment and c != outcome
            ]

        # Priority 2: Use profile's potential confounders
        elif state.data_profile and state.data_profile.potential_confounders:
            raw_confounders = [
                c for c in state.data_profile.potential_confounders
                if c in self._df.columns and c != treatment and c != outcome
            ]

        # Priority 3: Use all non-ID columns except treatment and outcome
        else:
            raw_confounders = [
                c for c in self._df.columns
                if c != treatment and c != outcome
            ]

        # Separate numeric vs categorical confounders
        numeric_covariates = [
            c for c in raw_confounders
            if pd.api.types.is_numeric_dtype(self._df[c])
        ]

        cat_confounders = [
            c for c in raw_confounders
            if c in self._df.columns
            and not pd.api.types.is_numeric_dtype(self._df[c])
            and self._df[c].nunique() <= 20
        ]

        # One-hot encode categorical confounders and add to working dataframe
        dummy_cols: list[str] = []
        if cat_confounders:
            dummies = pd.get_dummies(
                self._df[cat_confounders], prefix_sep="_", drop_first=True,
            )
            for col in dummies.columns:
                self._df[col] = dummies[col].astype(float)
            dummy_cols = list(dummies.columns)

        return numeric_covariates + dummy_cols

    # =========================================================================
    # Main execution
    # =========================================================================

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute treatment effect estimation using ReAct loop.

        Args:
            state: Current analysis state

        Returns:
            Updated state with treatment effect results
        """
        self.logger.info(
            "estimation_start",
            job_id=state.job_id,
            has_profile=state.data_profile is not None,
        )

        state.status = JobStatus.ESTIMATING_EFFECTS
        start_time = time.time()

        try:
            # Load data
            if state.dataframe_path is None:
                state.mark_failed("No data available for estimation", self.AGENT_NAME)
                return state

            self._df = pd.read_parquet(state.dataframe_path)

            # Store state reference
            self._current_state = state
            all_results = []

            # Identify valid causal pairs (LLM-driven when multiple candidates)
            pairs = await self._identify_valid_causal_pairs(state)

            if not pairs:
                state.mark_failed("No valid treatment-outcome pairs identified", self.AGENT_NAME)
                return state

            self.logger.info("causal_pairs_identified", count=len(pairs))

            # Store analyzed pairs in state for downstream agents
            state.analyzed_pairs = [
                CausalPair(treatment=t, outcome=o, rationale=r, priority=i + 1)
                for i, (t, o, r) in enumerate(pairs)
            ]

            # Backfill state variables from primary pair for backward compatibility
            if pairs and not state.treatment_variable:
                state.treatment_variable = pairs[0][0]
                state.outcome_variable = pairs[0][1]

            # Analyze each valid pair
            for treatment, outcome, rationale in pairs:
                # Use fuzzy matching to find actual column names
                actual_treatment = find_closest_column(treatment, list(self._df.columns))
                actual_outcome = find_closest_column(outcome, list(self._df.columns))

                # Validate variables exist in data
                if actual_treatment is None or actual_outcome is None:
                    self.logger.warning(
                        "skipping_invalid_pair",
                        treatment=treatment,
                        outcome=outcome,
                        reason="variable not in data"
                    )
                    continue

                # Log if fuzzy matching was used
                if actual_treatment != treatment or actual_outcome != outcome:
                    self.logger.info(
                        "fuzzy_matched_variables",
                        original_treatment=treatment,
                        matched_treatment=actual_treatment,
                        original_outcome=outcome,
                        matched_outcome=actual_outcome,
                    )
                    treatment = actual_treatment
                    outcome = actual_outcome

                self.logger.info(
                    "analyzing_causal_pair",
                    treatment=treatment,
                    outcome=outcome,
                    rationale=rationale[:50],
                )

                # Clear any previous results for this pair (from prior iterations)
                state.treatment_effects = [
                    e for e in state.treatment_effects
                    if not (e.treatment == treatment and e.outcome == outcome)
                ]

                # Set current pair
                self._treatment_var = treatment
                self._outcome_var = outcome
                self._results = []
                self._finalized = False

                # Get covariates for this pair
                self._covariates = self._get_covariates_for_pair(state, treatment, outcome)

                # Run ReAct loop for this pair
                await super().execute(state)

                # If not finalized via tool, auto-finalize
                if not self._finalized:
                    self.logger.warning("estimation_auto_finalize")
                    self._auto_finalize()
                    self._finalized = True

                # Tag results with pair info
                for result in self._results:
                    result.treatment_variable = treatment
                    result.outcome_variable = outcome

                all_results.extend(self._results)

            # Update state with all results
            state.treatment_effects = all_results

            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.info(
                "estimation_complete",
                n_pairs=len(pairs),
                n_results=len(all_results),
                duration_ms=duration_ms,
            )

        except Exception as e:
            self.logger.exception("estimation_failed", error=str(e))
            state.mark_failed(f"Effect estimation failed: {str(e)}", self.AGENT_NAME)

        return state

    def _auto_finalize(self) -> dict[str, Any]:
        """Auto-finalize when LLM doesn't explicitly finalize."""
        if not self._results:
            # Emergency fallback: run at least OLS as a baseline estimation
            try:
                covariates = self._covariates or []
                ols_result = run_method(
                    "ols", self._treatment_var, self._outcome_var,
                    covariates, self._df, self._current_state,
                )
                if ols_result:
                    self._results.append(ols_result)
                    self.logger.info(
                        "auto_finalize_ols_fallback",
                        estimate=ols_result.estimate,
                    )
            except Exception as e:
                self.logger.warning("auto_finalize_ols_failed", error=str(e))

        if not self._results:
            return {
                "preferred_method": "none",
                "preferred_estimate": 0.0,
                "confidence_level": "low",
                "reasoning": "No estimation methods succeeded",
                "results": [],
            }

        # Pick the doubly robust or AIPW estimate if available, else median
        preferred = None
        for r in self._results:
            if "doubly" in r.method.lower() or "aipw" in r.method.lower():
                preferred = r
                break

        if not preferred:
            estimates = sorted(self._results, key=lambda x: x.estimate)
            preferred = estimates[len(estimates) // 2]

        return {
            "preferred_method": preferred.method,
            "preferred_estimate": preferred.estimate,
            "confidence_level": "medium" if len(self._results) >= 2 else "low",
            "reasoning": f"Auto-finalized with {len(self._results)} methods. Preferred: {preferred.method}",
            "results": self._results,
        }

    # =========================================================================
    # Tool Handlers (async, return ToolResult)
    # =========================================================================

    async def _tool_get_data_summary(
        self,
        state: AnalysisState,
    ) -> ToolResult:
        """Get summary statistics with sample size warnings and method recommendations."""
        df = self._df
        T = df[self._treatment_var]
        Y = df[self._outcome_var]

        n_total = len(df)
        n_treated = int(T.sum())
        n_control = n_total - n_treated

        # Add sample size warnings
        warning = SampleSizeThresholds.get_sample_size_warning(n_treated, n_control)
        recommended = SampleSizeThresholds.get_recommended_methods(n_treated, n_control)

        # Build guidance based on sample size
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
                "n_covariates": len(self._covariates),
                "covariates_sample": self._covariates[:10],
                "recommended_methods": recommended,
                "guidance": guidance,
                "warning": warning,
            }
        )

    async def _tool_check_covariate_balance(
        self,
        state: AnalysisState,
        covariates: list[str] | None = None,
    ) -> ToolResult:
        """Check covariate balance between treatment groups."""
        df = self._df
        T = df[self._treatment_var].values.astype(float)

        # Binarize continuous treatment using median split
        if len(np.unique(T)) > 2:
            T = (T > np.median(T)).astype(int)

        if not covariates:
            covariates = self._covariates[:15]

        # Filter to numeric covariates only
        numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
        covariates = [c for c in covariates if c in numeric_cols]

        balance_results = []
        imbalanced = []

        for cov in covariates:
            if cov not in df.columns:
                continue

            try:
                x = df[cov].values.astype(float)
                treated_mean = np.nanmean(x[T == 1])
                control_mean = np.nanmean(x[T == 0])

                if np.isnan(treated_mean) or np.isnan(control_mean):
                    continue

                pooled_std = np.sqrt(
                    (np.nanvar(x[T == 1]) + np.nanvar(x[T == 0])) / 2
                )

                if pooled_std > 0 and not np.isnan(pooled_std):
                    smd = (treated_mean - control_mean) / pooled_std
                else:
                    smd = 0

                status = "BALANCED" if abs(smd) < 0.1 else ("MODERATE" if abs(smd) < 0.25 else "IMBALANCED")
                balance_results.append({
                    "covariate": cov,
                    "smd": round(smd, 3),
                    "status": status,
                })

                if abs(smd) >= 0.1:
                    imbalanced.append(cov)
            except Exception:
                pass

        recommendation = "Adjustment needed for imbalanced covariates" if imbalanced else "Good balance, simple methods may suffice"

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "covariates_checked": len(balance_results),
                "imbalanced_count": len(imbalanced),
                "imbalanced_covariates": imbalanced,
                "balance_details": balance_results[:10],
                "recommendation": recommendation,
            }
        )

    async def _tool_estimate_propensity_scores(
        self,
        state: AnalysisState,
        covariates: list[str] | None = None,
    ) -> ToolResult:
        """Estimate propensity scores and check overlap."""
        df = self._df
        T = df[self._treatment_var].values.astype(float)

        # Binarize continuous treatment using median split
        if len(np.unique(T)) > 2:
            T = (T > np.median(T)).astype(int)

        if not covariates:
            covariates = self._covariates[:15]

        # Filter to valid numeric covariates
        numeric_cols = set(df.select_dtypes(include=[np.number]).columns)
        valid_covs = [c for c in covariates if c in df.columns and c in numeric_cols]
        if not valid_covs:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="No valid numeric covariates for propensity model."
            )

        X = df[valid_covs].values.astype(float)

        # Handle missing values
        mask = ~np.any(np.isnan(X), axis=1)
        X_clean = X[mask]
        T_clean = T[mask]

        if len(X_clean) < 50:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="Insufficient data after removing missing values."
            )

        # Fit propensity model
        try:
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_clean, T_clean)
            ps = model.predict_proba(X_clean)[:, 1]
        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Propensity model failed: {str(e)}"
            )

        # Store for later use
        self._propensity_scores = np.full(len(T), np.nan)
        self._propensity_scores[mask] = ps

        # Check overlap
        ps_treated = ps[T_clean == 1]
        ps_control = ps[T_clean == 0]

        overlap_min = max(ps_control.min(), ps_treated.min())
        overlap_max = min(ps_control.max(), ps_treated.max())

        in_support = float(np.mean((ps >= overlap_min) & (ps <= overlap_max)))
        overlap_quality = "GOOD" if in_support > 0.9 else ("MODERATE" if in_support > 0.7 else "POOR")

        if in_support > 0.9:
            recommendation = "Good overlap supports IPW and matching methods"
        elif in_support > 0.7:
            recommendation = "Consider trimming extreme PS; AIPW may be more robust than IPW"
        else:
            recommendation = "Poor overlap is concerning - results may be extrapolation"

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "treated_ps": {
                    "mean": round(float(ps_treated.mean()), 3),
                    "std": round(float(ps_treated.std()), 3),
                    "min": round(float(ps_treated.min()), 3),
                    "max": round(float(ps_treated.max()), 3),
                },
                "control_ps": {
                    "mean": round(float(ps_control.mean()), 3),
                    "std": round(float(ps_control.std()), 3),
                    "min": round(float(ps_control.min()), 3),
                    "max": round(float(ps_control.max()), 3),
                },
                "common_support": [round(overlap_min, 3), round(overlap_max, 3)],
                "proportion_in_support": round(in_support, 3),
                "overlap_quality": overlap_quality,
                "recommendation": recommendation,
            }
        )

    async def _tool_run_estimation_method(
        self,
        state: AnalysisState,
        method: str,
        covariates: list[str] | None = None,
    ) -> ToolResult:
        """Run a specific estimation method."""
        if not covariates:
            covariates = self._covariates

        # Filter covariates
        valid_covs = [c for c in covariates if c in self._df.columns]

        try:
            result = run_method(
                method, self._treatment_var, self._outcome_var,
                valid_covs, self._df, self._current_state,
            )
            if result:
                self._results.append(result)
                self._last_method_result = result

                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    output={
                        "method": result.method,
                        "estimand": result.estimand,
                        "estimate": round(result.estimate, 4),
                        "std_error": round(result.std_error, 4),
                        "ci_lower": round(result.ci_lower, 4),
                        "ci_upper": round(result.ci_upper, 4),
                        "p_value": round(result.p_value, 4) if result.p_value else None,
                        "assumptions": result.assumptions_tested,
                        "details": result.details,
                    }
                )
            else:
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    output=None,
                    error=f"Method {method} failed to produce results."
                )
        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Method {method} failed with error: {str(e)}"
            )

    async def _tool_check_method_diagnostics(
        self,
        state: AnalysisState,
        diagnostic_type: str = "all",
    ) -> ToolResult:
        """Check diagnostics for the last method."""
        if not self._last_method_result:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="No method has been run yet. Run a method first."
            )

        result = self._last_method_result
        diagnostics = {}

        if diagnostic_type in ["residuals", "all"]:
            diagnostics["residuals"] = check_residual_diagnostics(
                self._df, self._treatment_var, self._outcome_var, self._covariates
            )

        if diagnostic_type in ["influence", "all"]:
            diagnostics["influence"] = check_influence_diagnostics(
                self._df, self._outcome_var
            )

        if diagnostic_type in ["specification", "all"]:
            diagnostics["specification"] = check_specification_diagnostics(result)

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "method": result.method,
                "diagnostics": diagnostics,
            }
        )

    async def _tool_compare_estimates(
        self,
        state: AnalysisState,
    ) -> ToolResult:
        """Compare estimates across methods with reliability weighting."""
        if not self._results:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error="No estimates to compare yet. Run some methods first."
            )

        estimates = [r.estimate for r in self._results]

        # Simple statistics
        mean_est = float(np.mean(estimates))
        std_est = float(np.std(estimates))
        cv = std_est / (abs(mean_est) + 1e-10)

        # Median and MAD (more robust to outliers)
        median_est = float(np.median(estimates))
        mad = float(np.median(np.abs(np.array(estimates) - median_est)))

        # Track reliability for weighting
        reliability_weights = {"high": 3, "medium": 2, "low": 1, None: 1}
        weighted_sum = 0.0
        weight_total = 0.0

        individual_estimates = []
        for r in self._results:
            reliability = r.details.get("reliability", "unknown") if r.details else "unknown"
            weight = reliability_weights.get(reliability, 1)
            weighted_sum += r.estimate * weight
            weight_total += weight
            individual_estimates.append({
                "method": r.method,
                "estimate": round(r.estimate, 4),
                "std_error": round(r.std_error, 4),
                "reliability": reliability,
            })

        weighted_avg = weighted_sum / weight_total if weight_total > 0 else mean_est

        # Identify potential outliers using MAD
        outliers = []
        for r in self._results:
            if mad > 0 and abs(r.estimate - median_est) > 3 * mad:
                outliers.append(r.method)

        consistency = "CONSISTENT" if cv < 0.2 else ("MODERATE" if cv < 0.5 else "INCONSISTENT")

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "n_methods": len(self._results),
                "individual_estimates": individual_estimates,
                "statistics": {
                    "mean": round(mean_est, 4),
                    "median": round(median_est, 4),
                    "weighted_mean": round(weighted_avg, 4),
                    "std": round(std_est, 4),
                    "mad": round(mad, 4),
                    "cv": round(cv, 4),
                    "range": [round(min(estimates), 4), round(max(estimates), 4)],
                },
                "consistency": consistency,
                "outlier_methods": outliers,
                "recommendation": f"Use {'mean' if consistency == 'CONSISTENT' else 'median'} estimate: {round(mean_est if consistency == 'CONSISTENT' else median_est, 4)}",
            }
        )

    async def _tool_finalize_estimation(
        self,
        state: AnalysisState,
        preferred_method: str,
        preferred_estimate: float,
        confidence_level: str,
        reasoning: str,
        caveats: list[str] | None = None,
    ) -> ToolResult:
        """Finalize the estimation with conclusions."""
        self._finalized = True
        self._final_result = {
            "preferred_method": preferred_method,
            "preferred_estimate": preferred_estimate,
            "confidence_level": confidence_level,
            "reasoning": reasoning,
            "caveats": caveats or [],
            "n_methods_run": len(self._results),
        }

        self.logger.info(
            "estimation_finalized",
            preferred_method=preferred_method,
            preferred_estimate=preferred_estimate,
            confidence_level=confidence_level,
            n_methods=len(self._results),
        )

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "finalized": True,
                "preferred_method": preferred_method,
                "preferred_estimate": preferred_estimate,
                "confidence_level": confidence_level,
                "reasoning": reasoning,
                "caveats": caveats or [],
            },
            metadata={"is_finish": True},
        )
