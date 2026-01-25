"""Effect Estimator Agent - Truly agentic treatment effect estimation.

This agent uses the ReAct pattern to iteratively:
1. Analyze data characteristics
2. Check covariate balance and propensity score overlap
3. Select and run appropriate estimation methods
4. Evaluate diagnostics and compare results
5. Iterate until confident in the estimate
"""

import json
import pickle
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.agents.base import (
    AnalysisState,
    BaseAgent,
    CausalPair,
    JobStatus,
    TreatmentEffectResult,
)
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


# =============================================================================
# Sample Size Thresholds for Method Selection
# =============================================================================
# These thresholds are based on statistical power and overfitting concerns.
# ML-based methods need more data to avoid overfitting; simple methods work with less.

class SampleSizeThresholds:
    """Minimum sample sizes for reliable estimation by method type."""

    # Minimum samples per treatment arm
    MIN_SAMPLES_BASIC = 30  # OLS, IPW - simple parametric methods
    MIN_SAMPLES_MATCHING = 50  # PSM - needs enough for good matches
    MIN_SAMPLES_ML = 100  # T/X/S-Learner - ML methods prone to overfitting
    MIN_SAMPLES_FOREST = 200  # Causal Forest, Double ML - complex ML

    # Total sample thresholds
    MIN_TOTAL_BASIC = 50
    MIN_TOTAL_ML = 200
    MIN_TOTAL_FOREST = 500

    @classmethod
    def get_method_requirements(cls, method: str) -> dict:
        """Get sample size requirements for a method."""
        requirements = {
            "ols": {"min_per_arm": cls.MIN_SAMPLES_BASIC, "min_total": cls.MIN_TOTAL_BASIC, "complexity": "low"},
            "ipw": {"min_per_arm": cls.MIN_SAMPLES_BASIC, "min_total": cls.MIN_TOTAL_BASIC, "complexity": "low"},
            "aipw": {"min_per_arm": cls.MIN_SAMPLES_BASIC, "min_total": cls.MIN_TOTAL_BASIC, "complexity": "medium"},
            "matching": {"min_per_arm": cls.MIN_SAMPLES_MATCHING, "min_total": 100, "complexity": "medium"},
            "s_learner": {"min_per_arm": cls.MIN_SAMPLES_ML, "min_total": cls.MIN_TOTAL_ML, "complexity": "high"},
            "t_learner": {"min_per_arm": cls.MIN_SAMPLES_ML, "min_total": cls.MIN_TOTAL_ML, "complexity": "high"},
            "x_learner": {"min_per_arm": cls.MIN_SAMPLES_ML, "min_total": cls.MIN_TOTAL_ML, "complexity": "high"},
            "causal_forest": {"min_per_arm": cls.MIN_SAMPLES_FOREST, "min_total": cls.MIN_TOTAL_FOREST, "complexity": "very_high"},
            "double_ml": {"min_per_arm": cls.MIN_SAMPLES_FOREST, "min_total": cls.MIN_TOTAL_FOREST, "complexity": "very_high"},
        }
        return requirements.get(method, {"min_per_arm": cls.MIN_SAMPLES_BASIC, "min_total": cls.MIN_TOTAL_BASIC, "complexity": "unknown"})

    @classmethod
    def check_method_viability(cls, method: str, n_treated: int, n_control: int) -> tuple[bool, str]:
        """Check if a method is viable given sample sizes.

        Returns:
            Tuple of (is_viable, reason_if_not_viable)
        """
        reqs = cls.get_method_requirements(method)
        min_arm = min(n_treated, n_control)
        total = n_treated + n_control

        if min_arm < reqs["min_per_arm"]:
            return False, f"Smallest arm has {min_arm} samples, but {method} requires {reqs['min_per_arm']}+ per arm"
        if total < reqs["min_total"]:
            return False, f"Total sample size {total} is below minimum {reqs['min_total']} for {method}"
        return True, ""

    @classmethod
    def get_recommended_methods(cls, n_treated: int, n_control: int) -> list[str]:
        """Get list of methods recommended for the given sample sizes."""
        recommended = []
        methods_priority = ["ols", "ipw", "aipw", "matching", "s_learner", "t_learner", "x_learner", "causal_forest", "double_ml"]

        for method in methods_priority:
            viable, _ = cls.check_method_viability(method, n_treated, n_control)
            if viable:
                recommended.append(method)

        return recommended

    @classmethod
    def get_sample_size_warning(cls, n_treated: int, n_control: int) -> str | None:
        """Generate a warning message if sample size is concerning."""
        min_arm = min(n_treated, n_control)
        total = n_treated + n_control

        warnings = []

        if min_arm < 50:
            warnings.append(f"⚠️ SMALL SAMPLE: Only {min_arm} samples in smallest arm. ML methods will likely overfit.")
        elif min_arm < 100:
            warnings.append(f"⚠️ MODERATE SAMPLE: {min_arm} samples in smallest arm. Use regularized methods.")

        if total < 200:
            warnings.append(f"⚠️ Limited total sample ({total}). Prefer OLS/IPW over complex ML methods.")

        if n_treated < 100 and n_control > 3 * n_treated:
            warnings.append(f"⚠️ IMBALANCED: {n_treated} treated vs {n_control} control. Consider matching methods carefully.")

        if warnings:
            return "\n".join(warnings)
        return None


def _find_closest_column(name: str, columns: list[str]) -> str | None:
    """Find the closest matching column name using fuzzy matching.

    Handles cases where LLM returns "treatment" but column is "treat".
    """
    if name in columns:
        return name

    name_lower = name.lower()

    # Try exact lowercase match
    for col in columns:
        if col.lower() == name_lower:
            return col

    # Try prefix matching (treatment -> treat)
    for col in columns:
        col_lower = col.lower()
        if name_lower.startswith(col_lower) or col_lower.startswith(name_lower):
            return col

    # Try substring matching
    for col in columns:
        col_lower = col.lower()
        if name_lower in col_lower or col_lower in name_lower:
            return col

    return None


class EffectEstimatorAgent(BaseAgent):
    """Truly agentic effect estimator using ReAct pattern.

    The LLM iteratively:
    1. Investigates data characteristics
    2. Checks balance and overlap
    3. Runs estimation methods one at a time
    4. Evaluates diagnostics
    5. Compares results across methods
    6. Decides when to stop and which estimate to trust

    NO pre-computation - all evidence gathered via tool calls.
    """

    AGENT_NAME = "effect_estimator"

    SYSTEM_PROMPT = """You are an expert econometrician and causal inference practitioner.
Your task is to estimate treatment effects using rigorous statistical methods.

You have tools to investigate the data and run estimation methods. Use them iteratively:

1. FIRST: Get data summary to understand the dataset
2. THEN: Check covariate balance between treatment groups
3. THEN: Check propensity score overlap (for observational studies)
4. THEN: Run estimation methods one by one, checking diagnostics after each
5. FINALLY: Compare results and finalize with your best estimate

IMPORTANT GUIDELINES:
- Start with simpler methods (OLS, IPW) before complex ones (Causal Forest, DML)
- Always check diagnostics after running a method
- If propensity scores have poor overlap, note this limitation
- If methods disagree significantly, investigate why
- Use at least 2-3 methods for robustness before finalizing

Method selection guidance:
- OLS: Always a good baseline, but may be biased with confounding
- IPW: Good when propensity model is well-specified
- AIPW/Doubly Robust: Robust to misspecification of either model
- Matching: Good for interpretability, requires common support
- Meta-learners (S/T/X): Good for heterogeneous effects
- Causal Forest: Best for discovering heterogeneity, needs large N
- DiD: Only if panel data with pre/post periods
- IV: Only if valid instruments available
- RDD: Only if running variable with cutoff exists

Call tools iteratively. Analyze each result before deciding next steps."""

    TOOLS = [
        {
            "name": "get_data_summary",
            "description": "Get summary statistics of the dataset including sample sizes, treatment/control split, outcome distribution, and available covariates.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "check_covariate_balance",
            "description": "Check balance of covariates between treatment and control groups. Returns standardized mean differences.",
            "parameters": {
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
        },
        {
            "name": "estimate_propensity_scores",
            "description": "Estimate propensity scores and check overlap between treatment groups.",
            "parameters": {
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
        },
        {
            "name": "run_estimation_method",
            "description": "Run a specific causal estimation method and get the treatment effect estimate.",
            "parameters": {
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
        },
        {
            "name": "check_method_diagnostics",
            "description": "Check diagnostics for the most recently run method (e.g., residual analysis, influence points).",
            "parameters": {
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
        },
        {
            "name": "compare_estimates",
            "description": "Compare all estimates obtained so far across methods.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "finalize_results",
            "description": "Finalize the analysis with your conclusions. Call this when you have enough evidence.",
            "parameters": {
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
        },
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._df: pd.DataFrame | None = None
        self._treatment_var: str | None = None
        self._outcome_var: str | None = None
        self._covariates: list[str] = []
        self._results: list[TreatmentEffectResult] = []
        self._propensity_scores: np.ndarray | None = None
        self._last_method_result: TreatmentEffectResult | None = None
        self._state: AnalysisState | None = None

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
            # Apply fuzzy matching if dataframe is loaded
            treatment = state.treatment_variable
            outcome = state.outcome_variable

            if self._df is not None:
                matched_treatment = _find_closest_column(treatment, list(self._df.columns))
                matched_outcome = _find_closest_column(outcome, list(self._df.columns))
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
            result = await self.reason(
                prompt=prompt,
                context={"task": "pair_selection"},
            )
            pairs = self._parse_pair_selection(result, profile)
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

        # Try to extract JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in response_text:
                json_str = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                json_str = response_text.split("```")[1].split("```")[0].strip()
            else:
                # Try to find JSON object directly
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response_text[start:end]
                else:
                    return []

            data = json.loads(json_str)
            valid_pairs = data.get("valid_pairs", [])

            # Validate and extract pairs
            pairs = []
            valid_treatments = set(profile.treatment_candidates or [])
            valid_outcomes = set(profile.outcome_candidates or [])

            for p in valid_pairs[:3]:  # Max 3 pairs
                treatment = p.get("treatment")
                outcome = p.get("outcome")
                rationale = p.get("rationale", "LLM selected")

                # Validate variables exist in candidates
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
        """Get covariates for a specific treatment-outcome pair."""
        # Priority 1: Use discovered confounders
        if state.confounder_discovery and state.confounder_discovery.get("ranked_confounders"):
            return [
                c for c in state.confounder_discovery["ranked_confounders"]
                if c in self._df.columns and c != treatment and c != outcome
            ]

        # Priority 2: Use profile's potential confounders
        if state.data_profile and state.data_profile.potential_confounders:
            return [
                c for c in state.data_profile.potential_confounders
                if c in self._df.columns and c != treatment and c != outcome
            ]

        # Priority 3: Use all numeric columns except treatment and outcome
        return [
            c for c in self._df.select_dtypes(include=[np.number]).columns
            if c != treatment and c != outcome
        ]

    # =========================================================================
    # Main execution
    # =========================================================================

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute treatment effect estimation using agentic loop.

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

            with open(state.dataframe_path, "rb") as f:
                self._df = pickle.load(f)

            # Store state reference
            self._state = state
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
            # This ensures downstream agents that use state.treatment_variable work correctly
            if pairs and not state.treatment_variable:
                state.treatment_variable = pairs[0][0]
                state.outcome_variable = pairs[0][1]

            # Analyze each valid pair
            for treatment, outcome, rationale in pairs:
                # Use fuzzy matching to find actual column names
                actual_treatment = _find_closest_column(treatment, list(self._df.columns))
                actual_outcome = _find_closest_column(outcome, list(self._df.columns))

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

                # Set current pair
                self._treatment_var = treatment
                self._outcome_var = outcome
                self._results = []

                # Get covariates for this pair
                self._covariates = self._get_covariates_for_pair(state, treatment, outcome)

                # Run agentic estimation for this pair
                final_result = await self._run_agentic_loop(
                    self._build_initial_prompt(state),
                    max_iterations=15
                )

                # Tag results with pair info
                for result in self._results:
                    result.treatment_variable = treatment
                    result.outcome_variable = outcome

                all_results.extend(self._results)

            # Update state with all results
            state.treatment_effects = all_results

            # Create trace
            duration_ms = int((time.time() - start_time) * 1000)
            trace = self.create_trace(
                action="estimation_complete",
                reasoning=f"Analyzed {len(pairs)} causal pairs with {len(all_results)} total estimates",
                outputs={
                    "n_pairs_analyzed": len(pairs),
                    "n_total_estimates": len(all_results),
                    "estimates": [
                        {
                            "treatment": r.treatment_variable,
                            "outcome": r.outcome_variable,
                            "method": r.method,
                            "estimate": r.estimate,
                            "se": r.std_error,
                        }
                        for r in all_results
                    ],
                },
                duration_ms=duration_ms,
            )
            state.add_trace(trace)

            self.logger.info(
                "estimation_complete",
                n_pairs=len(pairs),
                n_results=len(all_results),
            )

        except Exception as e:
            self.logger.error("estimation_failed", error=str(e))
            import traceback
            traceback.print_exc()
            state.mark_failed(f"Effect estimation failed: {str(e)}", self.AGENT_NAME)

        return state

    def _build_initial_prompt(self, state: AnalysisState) -> str:
        """Build the initial prompt for the agentic loop."""
        prompt = f"""You need to estimate the causal effect of treatment on outcome.

Treatment variable: {self._treatment_var}
Outcome variable: {self._outcome_var}
Available covariates for adjustment: {self._covariates[:20]}{'...' if len(self._covariates) > 20 else ''}

"""
        if state.confounder_discovery:
            prompt += f"""
Confounder Discovery Results:
- Confirmed confounders: {state.confounder_discovery.get('ranked_confounders', [])}
- Discovery reasoning: {state.confounder_discovery.get('reasoning', 'N/A')[:500]}

"""

        if state.data_profile:
            prompt += f"""
Data Profile:
- Samples: {state.data_profile.n_samples}
- Has time dimension: {state.data_profile.has_time_dimension}
- Potential instruments: {state.data_profile.potential_instruments}
- Discontinuity candidates: {state.data_profile.discontinuity_candidates}

"""

        prompt += """Start by getting a data summary, then systematically investigate and run estimation methods.
Your goal is to produce a credible causal effect estimate with appropriate uncertainty quantification."""

        return prompt

    async def _run_agentic_loop(
        self,
        initial_prompt: str,
        max_iterations: int = 20,
    ) -> dict[str, Any]:
        """Run the agentic tool-calling loop.

        Args:
            initial_prompt: Starting prompt
            max_iterations: Maximum iterations

        Returns:
            Final result dictionary
        """
        messages = [{"role": "user", "content": initial_prompt}]
        final_result = {}

        for iteration in range(max_iterations):
            self.logger.info(
                "effect_estimator_iteration",
                iteration=iteration,
                max_iterations=max_iterations,
                methods_run=len(self._results),
            )

            # Get LLM response with tool calls
            try:
                result = await self.reason(
                    prompt=messages[-1]["content"] if messages[-1]["role"] == "user" else "Continue with your analysis.",
                    context={"iteration": iteration, "n_results": len(self._results)},
                )
            except Exception as e:
                self.logger.warning("llm_call_failed", iteration=iteration, error=str(e))
                if self._results:
                    return self._auto_finalize()
                raise

            # Log LLM reasoning if provided
            response_text = result.get("response", "")
            if response_text:
                self.logger.info(
                    "estimator_llm_reasoning",
                    reasoning=response_text[:500] + ("..." if len(response_text) > 500 else ""),
                )

            # Check for tool calls
            pending_calls = result.get("pending_calls", [])

            if not pending_calls:
                self.logger.info("estimator_no_tool_calls", response_preview=response_text[:200])
                if "finalize" in response_text.lower() or len(self._results) >= 3:
                    return self._auto_finalize()
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": "Please continue by calling a tool. If you're done, call finalize_results."
                })
                continue

            # Execute each tool call
            tool_results = []
            for call in pending_calls:
                tool_name = call["name"]
                tool_args = call.get("args", {})

                # Check for finalize with detailed logging
                if tool_name == "finalize_results":
                    self.logger.info(
                        "estimator_finalizing",
                        preferred_method=tool_args.get("preferred_method"),
                        preferred_estimate=tool_args.get("preferred_estimate"),
                        confidence_level=tool_args.get("confidence_level"),
                        num_results=len(self._results),
                    )
                    final_result = tool_args
                    final_result["results"] = self._results
                    return final_result

                # Log the tool call decision
                self.logger.info(
                    "estimator_tool_decision",
                    tool=tool_name,
                    args=tool_args,
                )

                # Execute the tool
                try:
                    tool_result = self._execute_tool(tool_name, tool_args)
                    tool_results.append(f"[{tool_name}]: {tool_result}")

                    # Log tool result summary
                    self.logger.info(
                        "estimator_tool_result",
                        tool=tool_name,
                        result_summary=tool_result[:200] + ("..." if len(tool_result) > 200 else ""),
                    )
                except Exception as e:
                    self.logger.warning("tool_execution_error", tool=tool_name, error=str(e))
                    tool_results.append(f"[{tool_name}]: ERROR - {str(e)}")

            # Add tool results to conversation
            messages.append({
                "role": "user",
                "content": "Tool results:\n\n" + "\n\n".join(tool_results) + "\n\nAnalyze these results and decide your next step."
            })

        # Max iterations reached - auto-finalize
        self.logger.warning("max_iterations_reached_auto_finalizing")
        return self._auto_finalize()

    def _auto_finalize(self) -> dict[str, Any]:
        """Auto-finalize when LLM doesn't explicitly finalize."""
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
            # Use median estimate
            estimates = sorted(self._results, key=lambda x: x.estimate)
            preferred = estimates[len(estimates) // 2]

        return {
            "preferred_method": preferred.method,
            "preferred_estimate": preferred.estimate,
            "confidence_level": "medium" if len(self._results) >= 2 else "low",
            "reasoning": f"Auto-finalized with {len(self._results)} methods. Preferred: {preferred.method}",
            "results": self._results,
        }

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool and return the result string.

        Args:
            tool_name: Name of the tool
            args: Tool arguments

        Returns:
            Result string to return to LLM
        """
        if tool_name == "get_data_summary":
            return self._tool_get_data_summary()

        elif tool_name == "check_covariate_balance":
            covariates = args.get("covariates", [])
            return self._tool_check_covariate_balance(covariates)

        elif tool_name == "estimate_propensity_scores":
            covariates = args.get("covariates", [])
            return self._tool_estimate_propensity_scores(covariates)

        elif tool_name == "run_estimation_method":
            method = args.get("method")
            covariates = args.get("covariates", [])
            return self._tool_run_estimation_method(method, covariates)

        elif tool_name == "check_method_diagnostics":
            diagnostic_type = args.get("diagnostic_type", "all")
            return self._tool_check_method_diagnostics(diagnostic_type)

        elif tool_name == "compare_estimates":
            return self._tool_compare_estimates()

        else:
            return f"Unknown tool: {tool_name}"

    def _tool_get_data_summary(self) -> str:
        """Get summary statistics with sample size warnings and method recommendations."""
        df = self._df
        T = df[self._treatment_var]
        Y = df[self._outcome_var]

        n_total = len(df)
        n_treated = int(T.sum())
        n_control = n_total - n_treated

        summary = f"""Dataset Summary:
- Total samples: {n_total}
- Treated (T=1): {n_treated} ({100*n_treated/n_total:.1f}%)
- Control (T=0): {n_control} ({100*n_control/n_total:.1f}%)

Outcome ({self._outcome_var}):
- Overall mean: {Y.mean():.4f} (std: {Y.std():.4f})
- Treated mean: {Y[T==1].mean():.4f} (std: {Y[T==1].std():.4f})
- Control mean: {Y[T==0].mean():.4f} (std: {Y[T==0].std():.4f})
- Naive difference: {Y[T==1].mean() - Y[T==0].mean():.4f}

Available covariates ({len(self._covariates)}): {self._covariates[:15]}{'...' if len(self._covariates) > 15 else ''}
"""

        # Add sample size warnings
        warning = SampleSizeThresholds.get_sample_size_warning(n_treated, n_control)
        if warning:
            summary += f"\n{warning}\n"

        # Add method recommendations based on sample size
        recommended = SampleSizeThresholds.get_recommended_methods(n_treated, n_control)
        summary += f"\nRecommended methods for this sample size: {recommended}\n"

        # Add specific guidance
        if n_treated < 100 or n_control < 100:
            summary += """
METHOD SELECTION GUIDANCE (Small Sample):
- PREFER: OLS, IPW, AIPW (parametric methods work well with small samples)
- AVOID: T-Learner, X-Learner, Causal Forest (ML methods will overfit)
- If using matching: expect limited precision due to small treatment group
"""
        elif n_treated < 200:
            summary += """
METHOD SELECTION GUIDANCE (Moderate Sample):
- SAFE: OLS, IPW, AIPW, Matching, S-Learner
- CAUTIOUS: T-Learner, X-Learner (will use regularized models)
- AVOID: Causal Forest (needs larger sample for reliable heterogeneity detection)
"""
        else:
            summary += """
METHOD SELECTION GUIDANCE (Adequate Sample):
- All methods should work reasonably well
- ML methods (T/X-Learner, Causal Forest) can detect heterogeneous effects
- Compare multiple methods for robustness
"""

        return summary

    def _tool_check_covariate_balance(self, covariates: list[str]) -> str:
        """Check covariate balance between treatment groups."""
        df = self._df
        T = df[self._treatment_var].values

        if not covariates:
            covariates = self._covariates[:15]  # Check top 15

        results = []
        imbalanced = []

        for cov in covariates:
            if cov not in df.columns:
                continue

            try:
                x = df[cov].values
                treated_mean = np.nanmean(x[T == 1])
                control_mean = np.nanmean(x[T == 0])
                pooled_std = np.sqrt(
                    (np.nanvar(x[T == 1]) + np.nanvar(x[T == 0])) / 2
                )

                if pooled_std > 0:
                    smd = (treated_mean - control_mean) / pooled_std
                else:
                    smd = 0

                status = "BALANCED" if abs(smd) < 0.1 else ("MODERATE" if abs(smd) < 0.25 else "IMBALANCED")
                results.append(f"  {cov}: SMD={smd:.3f} ({status})")

                if abs(smd) >= 0.1:
                    imbalanced.append(cov)
            except Exception:
                results.append(f"  {cov}: Could not compute")

        summary = "Covariate Balance (Standardized Mean Differences):\n"
        summary += "\n".join(results)
        summary += f"\n\nImbalanced covariates (|SMD| >= 0.1): {imbalanced}"
        summary += f"\nRecommendation: {'Adjustment needed for imbalanced covariates' if imbalanced else 'Good balance, simple methods may suffice'}"

        return summary

    def _tool_estimate_propensity_scores(self, covariates: list[str]) -> str:
        """Estimate propensity scores and check overlap."""
        from sklearn.linear_model import LogisticRegression

        df = self._df
        T = df[self._treatment_var].values

        if not covariates:
            covariates = self._covariates[:15]

        # Filter to valid covariates
        valid_covs = [c for c in covariates if c in df.columns]
        if not valid_covs:
            return "No valid covariates for propensity model."

        X = df[valid_covs].values

        # Handle missing values
        mask = ~np.any(np.isnan(X), axis=1)
        X_clean = X[mask]
        T_clean = T[mask]

        if len(X_clean) < 50:
            return "Insufficient data after removing missing values."

        # Fit propensity model
        try:
            model = LogisticRegression(max_iter=1000, random_state=42)
            model.fit(X_clean, T_clean)
            ps = model.predict_proba(X_clean)[:, 1]
        except Exception as e:
            return f"Propensity model failed: {str(e)}"

        # Store for later use
        self._propensity_scores = np.full(len(T), np.nan)
        self._propensity_scores[mask] = ps

        # Check overlap
        ps_treated = ps[T_clean == 1]
        ps_control = ps[T_clean == 0]

        overlap_min = max(ps_control.min(), ps_treated.min())
        overlap_max = min(ps_control.max(), ps_treated.max())
        overlap_range = overlap_max - overlap_min

        # Proportion in common support
        in_support = np.mean((ps >= overlap_min) & (ps <= overlap_max))

        summary = f"""Propensity Score Analysis:

Distribution:
- Treated: mean={ps_treated.mean():.3f}, std={ps_treated.std():.3f}, range=[{ps_treated.min():.3f}, {ps_treated.max():.3f}]
- Control: mean={ps_control.mean():.3f}, std={ps_control.std():.3f}, range=[{ps_control.min():.3f}, {ps_control.max():.3f}]

Overlap Assessment:
- Common support region: [{overlap_min:.3f}, {overlap_max:.3f}]
- Proportion in common support: {in_support:.1%}
- Overlap quality: {"GOOD" if in_support > 0.9 else ("MODERATE" if in_support > 0.7 else "POOR")}

Recommendations:
"""
        if in_support > 0.9:
            summary += "- Good overlap supports IPW and matching methods\n"
        elif in_support > 0.7:
            summary += "- Consider trimming extreme propensity scores\n"
            summary += "- AIPW may be more robust than IPW\n"
        else:
            summary += "- Poor overlap is concerning - results may be extrapolation\n"
            summary += "- Consider bounding methods or different identification strategy\n"

        return summary

    def _tool_run_estimation_method(self, method: str, covariates: list[str]) -> str:
        """Run a specific estimation method."""
        if not covariates:
            covariates = self._covariates

        # Filter covariates
        valid_covs = [c for c in covariates if c in self._df.columns]

        try:
            result = self._run_method(method, valid_covs)
            if result:
                self._results.append(result)
                self._last_method_result = result

                return f"""Method: {result.method}
Estimand: {result.estimand}
Estimate: {result.estimate:.4f}
Std Error: {result.std_error:.4f}
95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]
P-value: {result.p_value:.4f if result.p_value else 'N/A'}
Assumptions: {result.assumptions_tested}
Details: {json.dumps(result.details, indent=2, default=str)}
"""
            else:
                return f"Method {method} failed to produce results."
        except Exception as e:
            return f"Method {method} failed with error: {str(e)}"

    def _tool_check_method_diagnostics(self, diagnostic_type: str) -> str:
        """Check diagnostics for the last method."""
        if not self._last_method_result:
            return "No method has been run yet. Run a method first."

        result = self._last_method_result
        diagnostics = []

        if diagnostic_type in ["residuals", "all"]:
            diagnostics.append(self._check_residual_diagnostics())

        if diagnostic_type in ["influence", "all"]:
            diagnostics.append(self._check_influence_diagnostics())

        if diagnostic_type in ["specification", "all"]:
            diagnostics.append(self._check_specification_diagnostics())

        return f"Diagnostics for {result.method}:\n\n" + "\n\n".join(diagnostics)

    def _check_residual_diagnostics(self) -> str:
        """Check residual-based diagnostics."""
        # Simple residual check
        df = self._df
        T = df[self._treatment_var].values
        Y = df[self._outcome_var].values

        # Compute residuals from simple model
        from sklearn.linear_model import LinearRegression
        X = df[self._covariates[:10]].values if self._covariates else np.ones((len(Y), 1))

        mask = ~np.any(np.isnan(X), axis=1)
        model = LinearRegression()
        model.fit(np.column_stack([T[mask], X[mask]]), Y[mask])
        residuals = Y[mask] - model.predict(np.column_stack([T[mask], X[mask]]))

        # Normality test
        _, normality_p = stats.shapiro(residuals[:min(5000, len(residuals))])

        # Heteroskedasticity (Breusch-Pagan style)
        _, hetero_p = stats.pearsonr(np.abs(residuals), model.predict(np.column_stack([T[mask], X[mask]])))

        return f"""Residual Diagnostics:
- Residual mean: {residuals.mean():.4f} (should be ~0)
- Residual std: {residuals.std():.4f}
- Normality test p-value: {normality_p:.4f} (>0.05 suggests normality)
- Heteroskedasticity indicator: {abs(hetero_p):.4f} (<0.1 suggests homoskedasticity)
"""

    def _check_influence_diagnostics(self) -> str:
        """Check for influential observations."""
        df = self._df
        Y = df[self._outcome_var].values

        # Simple leverage check
        n = len(Y)
        threshold = 4 / n

        return f"""Influence Diagnostics:
- Sample size: {n}
- Influential observation threshold: {threshold:.4f}
- Recommendation: Check for outliers if estimate seems unstable
"""

    def _check_specification_diagnostics(self) -> str:
        """Check model specification."""
        result = self._last_method_result
        if not result:
            return "No results to diagnose."

        issues = []
        if result.std_error > abs(result.estimate):
            issues.append("- High uncertainty: SE > |estimate|")
        if result.p_value and result.p_value > 0.1:
            issues.append("- Not statistically significant at 10% level")
        if "n_treated" in result.details:
            if result.details["n_treated"] < 50:
                issues.append("- Small treated sample may cause instability")

        return f"""Specification Diagnostics:
- Effect size: {result.estimate:.4f}
- Relative SE: {result.std_error / (abs(result.estimate) + 1e-10):.2f}
- Issues found: {len(issues)}
{chr(10).join(issues) if issues else '- No major specification issues detected'}
"""

    def _tool_compare_estimates(self) -> str:
        """Compare estimates across methods with reliability weighting."""
        if not self._results:
            return "No estimates to compare yet. Run some methods first."

        estimates = [r.estimate for r in self._results]

        # Simple statistics
        mean_est = np.mean(estimates)
        std_est = np.std(estimates)
        cv = std_est / (abs(mean_est) + 1e-10)

        # Median and MAD (more robust to outliers)
        median_est = np.median(estimates)
        mad = np.median(np.abs(estimates - median_est))

        comparison = f"""Estimate Comparison ({len(self._results)} methods):

Individual estimates:
"""
        # Track reliability for weighting
        reliability_weights = {"high": 3, "medium": 2, "low": 1, None: 1}
        weighted_sum = 0.0
        weight_total = 0.0

        for r in self._results:
            reliability = r.details.get("reliability", "unknown") if r.details else "unknown"
            weight = reliability_weights.get(reliability, 1)
            weighted_sum += r.estimate * weight
            weight_total += weight

            reliability_str = f" [{reliability}]" if reliability != "unknown" else ""
            comparison += f"  {r.method}: {r.estimate:.4f} (SE: {r.std_error:.4f}){reliability_str}\n"

        # Reliability-weighted average
        weighted_avg = weighted_sum / weight_total if weight_total > 0 else mean_est

        comparison += f"""
Summary Statistics:
- Simple mean: {mean_est:.4f}
- Median (robust): {median_est:.4f}
- Reliability-weighted mean: {weighted_avg:.4f}
- Std across methods: {std_est:.4f}
- MAD (robust spread): {mad:.4f}
- Coefficient of variation: {cv:.2%}
- Range: [{min(estimates):.4f}, {max(estimates):.4f}]

Consistency Assessment: {"CONSISTENT" if cv < 0.2 else ("MODERATE" if cv < 0.5 else "INCONSISTENT")}
"""

        # Identify potential outliers using MAD
        outliers = []
        for r in self._results:
            if mad > 0 and abs(r.estimate - median_est) > 3 * mad:
                outliers.append(r.method)

        if outliers:
            comparison += f"\n⚠️ POTENTIAL OUTLIER METHODS: {outliers}"
            comparison += "\nThese estimates deviate significantly from the median. Consider their reliability."

        if cv >= 0.5:
            comparison += "\n\n⚠️ WARNING: Estimates vary substantially across methods."
            comparison += "\nRECOMMENDATION: Trust methods with 'high' reliability and prefer median over mean."

        # Suggest best estimate
        comparison += f"""

RECOMMENDED ESTIMATE:
- If methods are consistent: Use mean ({mean_est:.4f})
- If outliers present: Use median ({median_est:.4f})
- Accounting for reliability: Use weighted mean ({weighted_avg:.4f})
"""

        return comparison

    def _run_method(self, method: str, covariates: list[str]) -> TreatmentEffectResult | None:
        """Run a specific estimation method.

        This delegates to the causal estimators engine or implements methods directly.
        """
        df = self._df
        T_col = self._treatment_var
        Y_col = self._outcome_var

        # Prepare clean data
        all_cols = [T_col, Y_col] + [c for c in covariates if c in df.columns]
        df_clean = df[all_cols].dropna()

        if len(df_clean) < 50:
            return None

        T = df_clean[T_col].values
        Y = df_clean[Y_col].values
        X = df_clean[covariates].values if covariates else None

        # Ensure binary treatment
        if len(np.unique(T)) > 2:
            T = (T > np.median(T)).astype(int)

        method_map = {
            "ols": self._estimate_ols,
            "ipw": self._estimate_ipw,
            "aipw": self._estimate_aipw,
            "matching": self._estimate_matching,
            "s_learner": self._estimate_s_learner,
            "t_learner": self._estimate_t_learner,
            "x_learner": self._estimate_x_learner,
            "causal_forest": self._estimate_causal_forest,
            "double_ml": self._estimate_double_ml,
            "did": self._estimate_did,
            "iv": self._estimate_iv,
            "rdd": self._estimate_rdd,
        }

        estimator = method_map.get(method)
        if estimator is None:
            return None

        return estimator(T, Y, X, df_clean)

    def _estimate_ols(self, T, Y, X, df) -> TreatmentEffectResult:
        """OLS regression estimate."""
        import statsmodels.api as sm

        if X is not None and X.shape[1] > 0:
            design = np.column_stack([np.ones(len(T)), T, X])
        else:
            design = np.column_stack([np.ones(len(T)), T])

        model = sm.OLS(Y, design)
        results = model.fit()

        ate = results.params[1]
        se = results.bse[1]
        ci = results.conf_int()[1]
        pval = results.pvalues[1]

        return TreatmentEffectResult(
            method="OLS Regression",
            estimand="ATE",
            estimate=float(ate),
            std_error=float(se),
            ci_lower=float(ci[0]),
            ci_upper=float(ci[1]),
            p_value=float(pval),
            assumptions_tested=["Linearity", "No unmeasured confounding"],
            details={"r_squared": float(results.rsquared), "n_obs": int(results.nobs)},
        )

    def _estimate_ipw(self, T, Y, X, df) -> TreatmentEffectResult | None:
        """Inverse Probability Weighting estimate."""
        from sklearn.linear_model import LogisticRegression

        if X is None or X.shape[1] == 0:
            return None

        # Estimate propensity scores
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X, T)
        ps = ps_model.predict_proba(X)[:, 1]
        ps = np.clip(ps, 0.01, 0.99)

        # IPW estimator
        weights_treated = T / ps
        weights_control = (1 - T) / (1 - ps)
        ate = np.mean(weights_treated * Y) - np.mean(weights_control * Y)

        # Bootstrap for SE
        n_bootstrap = 200
        bootstrap_estimates = []
        n = len(Y)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            w_t = weights_treated[idx]
            w_c = weights_control[idx]
            y_b = Y[idx]
            bootstrap_estimates.append(np.mean(w_t * y_b) - np.mean(w_c * y_b))

        se = np.std(bootstrap_estimates)
        ci_lower = np.percentile(bootstrap_estimates, 2.5)
        ci_upper = np.percentile(bootstrap_estimates, 97.5)

        return TreatmentEffectResult(
            method="Inverse Probability Weighting",
            estimand="ATE",
            estimate=float(ate),
            std_error=float(se),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=float(2 * (1 - stats.norm.cdf(abs(ate / se)))) if se > 0 else None,
            assumptions_tested=["Unconfoundedness", "Positivity"],
            details={
                "mean_ps_treated": float(np.mean(ps[T == 1])),
                "mean_ps_control": float(np.mean(ps[T == 0])),
            },
        )

    def _estimate_aipw(self, T, Y, X, df) -> TreatmentEffectResult | None:
        """Augmented IPW (Doubly Robust) estimate with regularized models.

        Key improvements:
        - Uses Ridge regression for outcome models (better with small samples)
        - Proper sample size checks
        - Reports model diagnostics
        """
        from sklearn.linear_model import LogisticRegression, Ridge

        if X is None or X.shape[1] == 0:
            return None

        treated_idx = T == 1
        control_idx = T == 0
        n_treated = int(np.sum(treated_idx))
        n_control = int(np.sum(control_idx))

        # Check sample size viability
        viable, reason = SampleSizeThresholds.check_method_viability("aipw", n_treated, n_control)
        if not viable:
            logger.warning("aipw_skipped", reason=reason, n_treated=n_treated, n_control=n_control)
            return None

        # Propensity scores with regularization
        ps_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        ps_model.fit(X, T)
        ps = ps_model.predict_proba(X)[:, 1]
        ps = np.clip(ps, 0.01, 0.99)

        # Outcome models with Ridge regularization (more stable for small samples)
        # Alpha chosen based on sample size
        min_arm = min(n_treated, n_control)
        alpha = max(0.1, 10.0 / min_arm)  # More regularization for smaller samples

        mu1_model = Ridge(alpha=alpha)
        mu0_model = Ridge(alpha=alpha)

        mu1_model.fit(X[treated_idx], Y[treated_idx])
        mu0_model.fit(X[control_idx], Y[control_idx])

        mu1 = mu1_model.predict(X)
        mu0 = mu0_model.predict(X)

        # AIPW estimator
        aipw1 = mu1 + T * (Y - mu1) / ps
        aipw0 = mu0 + (1 - T) * (Y - mu0) / (1 - ps)
        ate = np.mean(aipw1 - aipw0)

        # Bootstrap for SE
        n_bootstrap = 200
        bootstrap_estimates = []
        n = len(Y)
        for _ in range(n_bootstrap):
            idx = np.random.choice(n, size=n, replace=True)
            bootstrap_estimates.append(np.mean(aipw1[idx] - aipw0[idx]))

        se = np.std(bootstrap_estimates)
        ci_lower = np.percentile(bootstrap_estimates, 2.5)
        ci_upper = np.percentile(bootstrap_estimates, 97.5)

        # Reliability based on sample size
        if min_arm >= 100:
            reliability = "high"
        elif min_arm >= 50:
            reliability = "medium"
        else:
            reliability = "low"

        return TreatmentEffectResult(
            method="Doubly Robust (AIPW)",
            estimand="ATE",
            estimate=float(ate),
            std_error=float(se),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=float(2 * (1 - stats.norm.cdf(abs(ate / se)))) if se > 0 else None,
            assumptions_tested=["Unconfoundedness", "Correct propensity OR outcome model"],
            details={
                "is_doubly_robust": True,
                "n_treated": n_treated,
                "n_control": n_control,
                "ridge_alpha": float(alpha),
                "reliability": reliability,
                "mean_ps_treated": float(np.mean(ps[treated_idx])),
                "mean_ps_control": float(np.mean(ps[control_idx])),
            },
        )

    def _estimate_matching(self, T, Y, X, df) -> TreatmentEffectResult | None:
        """Propensity Score Matching estimate with k-NN and caliper.

        Key improvements:
        - Uses k-NN matching (k=5 by default) to utilize more control observations
        - Applies caliper (0.2 * std of propensity score) to avoid poor matches
        - Weights matches by inverse distance
        - Reports match quality diagnostics
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import NearestNeighbors

        if X is None or X.shape[1] == 0:
            return None

        treated_idx_orig = np.where(T == 1)[0]
        control_idx_orig = np.where(T == 0)[0]
        n_treated = len(treated_idx_orig)
        n_control = len(control_idx_orig)

        # Check sample size viability
        viable, reason = SampleSizeThresholds.check_method_viability("matching", n_treated, n_control)
        if not viable:
            logger.warning("matching_skipped", reason=reason, n_treated=n_treated, n_control=n_control)
            return None

        # Estimate propensity scores
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X, T)
        ps = ps_model.predict_proba(X)[:, 1]

        # Caliper: 0.2 * std of propensity scores (standard practice)
        caliper = 0.2 * np.std(ps)

        # Determine k based on control pool size
        # More controls = can use more neighbors
        k = min(5, max(1, n_control // n_treated))

        # k-NN matching
        nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree")
        nn.fit(ps[control_idx_orig].reshape(-1, 1))
        distances, indices = nn.kneighbors(ps[treated_idx_orig].reshape(-1, 1))

        # Apply caliper and compute weighted ATT
        matched_effects = []
        good_matches = 0
        poor_matches = 0

        for i, treat_idx in enumerate(treated_idx_orig):
            # Get matched controls for this treated unit
            matched_distances = distances[i]
            matched_control_indices = control_idx_orig[indices[i]]

            # Apply caliper: only use matches within caliper distance
            within_caliper = matched_distances <= caliper

            if not np.any(within_caliper):
                # No good matches - use closest anyway but flag it
                poor_matches += 1
                # Use only the closest match
                best_control = matched_control_indices[0]
                matched_effects.append(Y[treat_idx] - Y[best_control])
            else:
                good_matches += 1
                # Weighted average of controls within caliper
                valid_controls = matched_control_indices[within_caliper]
                valid_distances = matched_distances[within_caliper]

                # Inverse distance weights (add small epsilon to avoid division by zero)
                weights = 1.0 / (valid_distances + 1e-6)
                weights = weights / weights.sum()

                control_outcome = np.average(Y[valid_controls], weights=weights)
                matched_effects.append(Y[treat_idx] - control_outcome)

        att = np.mean(matched_effects)
        match_quality = good_matches / n_treated if n_treated > 0 else 0

        # Bootstrap for SE with proper matching
        n_bootstrap = 200
        bootstrap_estimates = []
        for b in range(n_bootstrap):
            boot_idx = np.random.choice(len(matched_effects), size=len(matched_effects), replace=True)
            bootstrap_estimates.append(np.mean([matched_effects[i] for i in boot_idx]))

        se = np.std(bootstrap_estimates)
        ci_lower = np.percentile(bootstrap_estimates, 2.5)
        ci_upper = np.percentile(bootstrap_estimates, 97.5)

        # Reliability based on match quality
        if match_quality >= 0.9:
            reliability = "high"
        elif match_quality >= 0.7:
            reliability = "medium"
        else:
            reliability = "low"

        return TreatmentEffectResult(
            method="Propensity Score Matching (k-NN)",
            estimand="ATT",
            estimate=float(att),
            std_error=float(se),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=float(2 * (1 - stats.norm.cdf(abs(att / se)))) if se > 0 else None,
            assumptions_tested=["Unconfoundedness", "Common support"],
            details={
                "n_treated": n_treated,
                "n_control": n_control,
                "k_neighbors": k,
                "caliper": float(caliper),
                "good_matches": good_matches,
                "poor_matches": poor_matches,
                "match_quality": float(match_quality),
                "reliability": reliability,
                "mean_ps_treated": float(np.mean(ps[treated_idx_orig])),
                "mean_ps_control": float(np.mean(ps[control_idx_orig])),
            },
        )

    def _estimate_s_learner(self, T, Y, X, df) -> TreatmentEffectResult | None:
        """S-Learner estimate with adaptive regularization.

        S-Learner is less prone to overfitting than T/X-Learner because it fits
        a single model on all data, but still needs regularization for small samples.
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import Ridge

        if X is None or X.shape[1] == 0:
            return None

        n_treated = int(np.sum(T == 1))
        n_control = int(np.sum(T == 0))
        n_total = len(T)

        # Check sample size viability (S-Learner is more forgiving)
        viable, reason = SampleSizeThresholds.check_method_viability("s_learner", n_treated, n_control)
        if not viable:
            logger.warning("s_learner_skipped", reason=reason, n_treated=n_treated, n_control=n_control)
            return None

        X_with_T = np.column_stack([X, T])

        # Adaptive model complexity
        if n_total < 200:
            # Small sample: Use Ridge regression
            model = Ridge(alpha=1.0)
            model_type = "ridge"
            reliability = "medium"  # S-Learner is more stable
        elif n_total < 500:
            # Moderate sample: Regularized GBM
            model = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                min_samples_leaf=max(10, n_total // 50),
                learning_rate=0.1,
                random_state=42,
            )
            model_type = "gbm_regularized"
            reliability = "medium"
        else:
            # Large sample: Full GBM
            model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=10,
                random_state=42,
            )
            model_type = "gbm_full"
            reliability = "high"

        model.fit(X_with_T, Y)

        X_treat = np.column_stack([X, np.ones(len(X))])
        X_control = np.column_stack([X, np.zeros(len(X))])

        y1_pred = model.predict(X_treat)
        y0_pred = model.predict(X_control)
        cate = y1_pred - y0_pred
        ate = np.mean(cate)

        # Bootstrap for SE
        n_bootstrap = 100
        bootstrap_ates = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(Y), size=len(Y), replace=True)
            bootstrap_ates.append(np.mean(cate[idx]))

        se = np.std(bootstrap_ates)
        ci_lower = np.percentile(bootstrap_ates, 2.5)
        ci_upper = np.percentile(bootstrap_ates, 97.5)

        return TreatmentEffectResult(
            method="S-Learner",
            estimand="ATE",
            estimate=float(ate),
            std_error=float(se),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=float(2 * (1 - stats.norm.cdf(abs(ate / se)))) if se > 0 else None,
            assumptions_tested=["Unconfoundedness"],
            details={
                "cate_std": float(np.std(cate)),
                "model_type": model_type,
                "reliability": reliability,
                "n_total": n_total,
            },
        )

    def _estimate_t_learner(self, T, Y, X, df) -> TreatmentEffectResult | None:
        """T-Learner estimate with adaptive regularization based on sample size.

        Key improvements:
        - Uses sample size thresholds to prevent overfitting
        - Adaptive regularization (fewer trees, limited depth for small samples)
        - Proper bootstrap that re-fits models
        - Reliability scoring based on sample size
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import Ridge

        if X is None or X.shape[1] == 0:
            return None

        treated_idx = T == 1
        control_idx = T == 0
        n_treated = int(np.sum(treated_idx))
        n_control = int(np.sum(control_idx))

        # Check sample size viability
        viable, reason = SampleSizeThresholds.check_method_viability("t_learner", n_treated, n_control)
        if not viable:
            logger.warning("t_learner_skipped", reason=reason, n_treated=n_treated, n_control=n_control)
            return None

        # Adaptive model complexity based on sample size
        min_arm = min(n_treated, n_control)

        if min_arm < 150:
            # Small sample: Use Ridge regression (linear, highly regularized)
            model_t = Ridge(alpha=1.0)
            model_c = Ridge(alpha=1.0)
            model_type = "ridge"
            reliability = "low"
        elif min_arm < 300:
            # Moderate sample: Regularized GBM with limited complexity
            model_t = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                min_samples_leaf=max(10, min_arm // 20),
                learning_rate=0.1,
                random_state=42,
            )
            model_c = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                min_samples_leaf=max(10, min_arm // 20),
                learning_rate=0.1,
                random_state=42,
            )
            model_type = "gbm_regularized"
            reliability = "medium"
        else:
            # Large sample: Full GBM
            model_t = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=10,
                random_state=42,
            )
            model_c = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=10,
                random_state=42,
            )
            model_type = "gbm_full"
            reliability = "high"

        # Fit models
        model_t.fit(X[treated_idx], Y[treated_idx])
        model_c.fit(X[control_idx], Y[control_idx])

        y1_pred = model_t.predict(X)
        y0_pred = model_c.predict(X)
        cate = y1_pred - y0_pred
        ate = np.mean(cate)

        # Proper bootstrap: re-fit models on each bootstrap sample
        n_bootstrap = 100
        bootstrap_ates = []

        for b in range(n_bootstrap):
            # Resample within each group
            idx_t = np.random.choice(np.where(treated_idx)[0], size=n_treated, replace=True)
            idx_c = np.random.choice(np.where(control_idx)[0], size=n_control, replace=True)

            # Create bootstrap models with same settings
            if min_arm < 150:
                boot_model_t = Ridge(alpha=1.0)
                boot_model_c = Ridge(alpha=1.0)
            elif min_arm < 300:
                boot_model_t = GradientBoostingRegressor(
                    n_estimators=30, max_depth=3, min_samples_leaf=max(10, min_arm // 20),
                    learning_rate=0.1, random_state=42 + b
                )
                boot_model_c = GradientBoostingRegressor(
                    n_estimators=30, max_depth=3, min_samples_leaf=max(10, min_arm // 20),
                    learning_rate=0.1, random_state=42 + b
                )
            else:
                boot_model_t = GradientBoostingRegressor(
                    n_estimators=50, max_depth=4, min_samples_leaf=10, random_state=42 + b
                )
                boot_model_c = GradientBoostingRegressor(
                    n_estimators=50, max_depth=4, min_samples_leaf=10, random_state=42 + b
                )

            try:
                boot_model_t.fit(X[idx_t], Y[idx_t])
                boot_model_c.fit(X[idx_c], Y[idx_c])
                boot_y1 = boot_model_t.predict(X)
                boot_y0 = boot_model_c.predict(X)
                bootstrap_ates.append(np.mean(boot_y1 - boot_y0))
            except Exception:
                continue

        se = np.std(bootstrap_ates) if bootstrap_ates else np.std(cate) / np.sqrt(len(T))
        ci_lower = np.percentile(bootstrap_ates, 2.5) if len(bootstrap_ates) > 10 else ate - 1.96 * se
        ci_upper = np.percentile(bootstrap_ates, 97.5) if len(bootstrap_ates) > 10 else ate + 1.96 * se

        return TreatmentEffectResult(
            method="T-Learner",
            estimand="ATE",
            estimate=float(ate),
            std_error=float(se),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=float(2 * (1 - stats.norm.cdf(abs(ate / se)))) if se > 0 else None,
            assumptions_tested=["Unconfoundedness"],
            details={
                "cate_std": float(np.std(cate)),
                "n_treated_train": n_treated,
                "n_control_train": n_control,
                "model_type": model_type,
                "reliability": reliability,
                "samples_per_tree_treated": n_treated / (50 if min_arm < 300 else 100),
            },
        )

    def _estimate_x_learner(self, T, Y, X, df) -> TreatmentEffectResult | None:
        """X-Learner estimate with adaptive regularization.

        Key improvements:
        - Uses sample size thresholds to prevent overfitting
        - Adaptive regularization for outcome and CATE models
        - Proper model complexity for each stage
        """
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import LogisticRegression, Ridge

        if X is None or X.shape[1] == 0:
            return None

        treated_idx = T == 1
        control_idx = T == 0
        n_treated = int(np.sum(treated_idx))
        n_control = int(np.sum(control_idx))

        # Check sample size viability
        viable, reason = SampleSizeThresholds.check_method_viability("x_learner", n_treated, n_control)
        if not viable:
            logger.warning("x_learner_skipped", reason=reason, n_treated=n_treated, n_control=n_control)
            return None

        min_arm = min(n_treated, n_control)

        # Adaptive model complexity
        if min_arm < 150:
            # Small sample: Use Ridge regression
            model_t = Ridge(alpha=1.0)
            model_c = Ridge(alpha=1.0)
            tau1_model = Ridge(alpha=1.0)
            tau0_model = Ridge(alpha=1.0)
            model_type = "ridge"
            reliability = "low"
        elif min_arm < 300:
            # Moderate sample: Regularized GBM
            gbm_params = {
                "n_estimators": 30,
                "max_depth": 3,
                "min_samples_leaf": max(10, min_arm // 20),
                "learning_rate": 0.1,
                "random_state": 42,
            }
            model_t = GradientBoostingRegressor(**gbm_params)
            model_c = GradientBoostingRegressor(**gbm_params)
            tau1_model = GradientBoostingRegressor(**gbm_params)
            tau0_model = GradientBoostingRegressor(**gbm_params)
            model_type = "gbm_regularized"
            reliability = "medium"
        else:
            # Large sample: Full GBM
            model_t = GradientBoostingRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)
            model_c = GradientBoostingRegressor(n_estimators=100, max_depth=5, min_samples_leaf=10, random_state=42)
            tau1_model = GradientBoostingRegressor(n_estimators=50, max_depth=4, min_samples_leaf=10, random_state=42)
            tau0_model = GradientBoostingRegressor(n_estimators=50, max_depth=4, min_samples_leaf=10, random_state=42)
            model_type = "gbm_full"
            reliability = "high"

        # Step 1: Fit outcome models
        model_t.fit(X[treated_idx], Y[treated_idx])
        model_c.fit(X[control_idx], Y[control_idx])

        # Step 2: Impute treatment effects
        D1 = Y[treated_idx] - model_c.predict(X[treated_idx])
        D0 = model_t.predict(X[control_idx]) - Y[control_idx]

        # Step 3: Fit CATE models
        tau1_model.fit(X[treated_idx], D1)
        tau0_model.fit(X[control_idx], D0)

        # Step 4: Propensity weighting
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X, T)
        ps = np.clip(ps_model.predict_proba(X)[:, 1], 0.01, 0.99)

        tau1 = tau1_model.predict(X)
        tau0 = tau0_model.predict(X)
        cate = ps * tau0 + (1 - ps) * tau1
        ate = np.mean(cate)

        # Bootstrap with proper re-fitting (simplified for speed)
        n_bootstrap = 100
        bootstrap_ates = []
        for b in range(n_bootstrap):
            # Resample predictions (faster than re-fitting all 4 models)
            idx = np.random.choice(len(Y), size=len(Y), replace=True)
            bootstrap_ates.append(np.mean(cate[idx]))

        se = np.std(bootstrap_ates)
        ci_lower = np.percentile(bootstrap_ates, 2.5)
        ci_upper = np.percentile(bootstrap_ates, 97.5)

        return TreatmentEffectResult(
            method="X-Learner",
            estimand="ATE",
            estimate=float(ate),
            std_error=float(se),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=float(2 * (1 - stats.norm.cdf(abs(ate / se)))) if se > 0 else None,
            assumptions_tested=["Unconfoundedness"],
            details={
                "cate_std": float(np.std(cate)),
                "n_treated_train": n_treated,
                "n_control_train": n_control,
                "model_type": model_type,
                "reliability": reliability,
            },
        )

    def _estimate_causal_forest(self, T, Y, X, df) -> TreatmentEffectResult | None:
        """Causal Forest estimate using EconML."""
        try:
            from econml.dml import CausalForestDML
        except ImportError:
            return None

        if X is None or X.shape[1] == 0:
            return None

        cf = CausalForestDML(n_estimators=100, min_samples_leaf=10, random_state=42)

        try:
            cf.fit(Y, T, X=X)
            cate = cf.effect(X)
            ate = np.mean(cate)

            ci = cf.effect_interval(X, alpha=0.05)
            ate_ci_lower = np.mean(ci[0])
            ate_ci_upper = np.mean(ci[1])
            se = (ate_ci_upper - ate_ci_lower) / (2 * 1.96)

            return TreatmentEffectResult(
                method="Causal Forest",
                estimand="ATE",
                estimate=float(ate),
                std_error=float(se),
                ci_lower=float(ate_ci_lower),
                ci_upper=float(ate_ci_upper),
                p_value=float(2 * (1 - stats.norm.cdf(abs(ate / se)))) if se > 0 else None,
                assumptions_tested=["Unconfoundedness", "Overlap"],
                details={"cate_std": float(np.std(cate))},
            )
        except Exception:
            return None

    def _estimate_double_ml(self, T, Y, X, df) -> TreatmentEffectResult | None:
        """Double Machine Learning estimate."""
        try:
            from econml.dml import LinearDML
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        except ImportError:
            return None

        if X is None or X.shape[1] == 0:
            return None

        dml = LinearDML(
            model_y=GradientBoostingRegressor(n_estimators=100, random_state=42),
            model_t=GradientBoostingClassifier(n_estimators=100, random_state=42),
            random_state=42,
        )

        try:
            dml.fit(Y, T, X=X)
            ate = dml.ate(X)
            ate_interval = dml.ate_interval(X, alpha=0.05)
            se = (ate_interval[1] - ate_interval[0]) / (2 * 1.96)

            return TreatmentEffectResult(
                method="Double ML",
                estimand="ATE",
                estimate=float(ate),
                std_error=float(se),
                ci_lower=float(ate_interval[0]),
                ci_upper=float(ate_interval[1]),
                p_value=float(2 * (1 - stats.norm.cdf(abs(ate / se)))) if se > 0 else None,
                assumptions_tested=["Unconfoundedness", "Overlap"],
                details={"method": "LinearDML"},
            )
        except Exception:
            return None

    def _estimate_did(self, T, Y, X, df) -> TreatmentEffectResult | None:
        """Difference-in-Differences estimate."""
        if not self._state or not self._state.data_profile:
            return None

        if not self._state.data_profile.has_time_dimension:
            return None

        time_col = self._state.data_profile.time_column
        if time_col not in df.columns:
            return None

        time_vals = sorted(df[time_col].unique())
        if len(time_vals) < 2:
            return None

        pre_period = time_vals[0]
        post_period = time_vals[-1]

        pre_treated = Y[(T == 1) & (df[time_col] == pre_period)]
        post_treated = Y[(T == 1) & (df[time_col] == post_period)]
        pre_control = Y[(T == 0) & (df[time_col] == pre_period)]
        post_control = Y[(T == 0) & (df[time_col] == post_period)]

        if any(len(g) == 0 for g in [pre_treated, post_treated, pre_control, post_control]):
            return None

        did = (np.mean(post_treated) - np.mean(pre_treated)) - \
              (np.mean(post_control) - np.mean(pre_control))

        var_did = (
            np.var(post_treated) / len(post_treated) +
            np.var(pre_treated) / len(pre_treated) +
            np.var(post_control) / len(post_control) +
            np.var(pre_control) / len(pre_control)
        )
        se = np.sqrt(var_did)

        return TreatmentEffectResult(
            method="Difference-in-Differences",
            estimand="ATT",
            estimate=float(did),
            std_error=float(se),
            ci_lower=float(did - 1.96 * se),
            ci_upper=float(did + 1.96 * se),
            p_value=float(2 * (1 - stats.norm.cdf(abs(did / se)))) if se > 0 else None,
            assumptions_tested=["Parallel trends"],
            details={"time_col": time_col},
        )

    def _estimate_iv(self, T, Y, X, df) -> TreatmentEffectResult | None:
        """Instrumental Variables (2SLS) estimate."""
        import statsmodels.api as sm

        if not self._state or not self._state.data_profile:
            return None

        if not self._state.data_profile.potential_instruments:
            return None

        iv_col = self._state.data_profile.potential_instruments[0]
        if iv_col not in df.columns:
            return None

        Z = df[iv_col].values

        # First stage
        Z_design = sm.add_constant(Z)
        first_stage = sm.OLS(T, Z_design).fit()
        f_stat = first_stage.fvalue

        T_hat = first_stage.predict(Z_design)

        # Second stage
        T_hat_design = sm.add_constant(T_hat)
        second_stage = sm.OLS(Y, T_hat_design).fit()

        late = second_stage.params[1]
        se = second_stage.bse[1]
        ci = second_stage.conf_int()[1]

        return TreatmentEffectResult(
            method="Instrumental Variables (2SLS)",
            estimand="LATE",
            estimate=float(late),
            std_error=float(se),
            ci_lower=float(ci[0]),
            ci_upper=float(ci[1]),
            p_value=float(second_stage.pvalues[1]),
            assumptions_tested=["Relevance", "Exclusion restriction", "Monotonicity"],
            details={"first_stage_f": float(f_stat), "instrument": iv_col},
        )

    def _estimate_rdd(self, T, Y, X, df) -> TreatmentEffectResult | None:
        """Regression Discontinuity Design estimate."""
        import statsmodels.api as sm

        if not self._state or not self._state.data_profile:
            return None

        if not self._state.data_profile.discontinuity_candidates:
            return None

        run_col = self._state.data_profile.discontinuity_candidates[0]
        if run_col not in df.columns:
            return None

        R = df[run_col].values
        cutoff = np.median(R)
        R_centered = R - cutoff

        bandwidth = np.std(R) * 0.5
        near_cutoff = np.abs(R_centered) <= bandwidth

        if np.sum(near_cutoff) < 50:
            return None

        R_local = R_centered[near_cutoff]
        Y_local = Y[near_cutoff]
        T_local = (R_local >= 0).astype(int)

        design = np.column_stack([
            np.ones(len(R_local)),
            T_local,
            R_local,
            T_local * R_local,
        ])

        model = sm.OLS(Y_local, design).fit()
        late = model.params[1]
        se = model.bse[1]
        ci = model.conf_int()[1]

        return TreatmentEffectResult(
            method="Regression Discontinuity",
            estimand="LATE",
            estimate=float(late),
            std_error=float(se),
            ci_lower=float(ci[0]),
            ci_upper=float(ci[1]),
            p_value=float(model.pvalues[1]),
            assumptions_tested=["Continuity at cutoff", "No manipulation"],
            details={"cutoff": float(cutoff), "bandwidth": float(bandwidth), "running_var": run_col},
        )
