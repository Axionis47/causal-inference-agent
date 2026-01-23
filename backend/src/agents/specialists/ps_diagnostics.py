"""Propensity Score Diagnostics Agent - Truly agentic PS validation.

This agent uses the ReAct pattern to iteratively:
1. Estimate propensity scores
2. Assess overlap between treatment groups
3. Check covariate balance before/after weighting
4. Test model calibration
5. Try alternative specifications if needed
6. Provide actionable recommendations
"""

import pickle
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

from src.agents.base import AnalysisState, BaseAgent
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


class PSDiagnosticsAgent(BaseAgent):
    """Truly agentic propensity score diagnostics using ReAct pattern.

    The LLM iteratively:
    1. Estimates propensity scores with different specifications
    2. Checks overlap and identifies positivity violations
    3. Assesses covariate balance with/without weighting
    4. Tests model calibration
    5. Tries improvements if diagnostics are poor
    6. Provides final recommendations

    NO pre-computation - all diagnostics computed on-demand via tool calls.
    """

    AGENT_NAME = "ps_diagnostics"

    SYSTEM_PROMPT = """You are an expert in propensity score methods for causal inference.
Your task is to diagnose and validate propensity score models.

You have tools to compute diagnostics on-demand. Use them iteratively:

1. FIRST: Estimate propensity scores with the baseline specification
2. CHECK: Overlap between treatment groups (look for positivity violations)
3. CHECK: Covariate balance (SMD should be < 0.1 for good balance)
4. CHECK: Model calibration
5. IF ISSUES: Try alternative specifications (add interactions, polynomial terms)
6. FINALLY: Provide recommendations based on all diagnostics

KEY THRESHOLDS:
- SMD < 0.1: Good balance
- SMD 0.1-0.25: Moderate imbalance
- SMD > 0.25: Severe imbalance
- Overlap > 90%: Good
- Overlap 70-90%: Moderate
- Overlap < 70%: Poor (consider trimming)

WHAT TO LOOK FOR:
- Extreme propensity scores (near 0 or 1) indicate positivity violations
- High SMD after weighting means PS model is misspecified
- Poor calibration suggests model is not well-fitted

Investigate issues before making final recommendations."""

    TOOLS = [
        {
            "name": "estimate_propensity_scores",
            "description": "Estimate propensity scores using logistic regression with specified covariates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "covariates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Covariates to include in PS model. If empty, uses all available confounders.",
                    },
                    "add_interactions": {
                        "type": "boolean",
                        "description": "Whether to add pairwise interactions",
                    },
                    "add_polynomials": {
                        "type": "boolean",
                        "description": "Whether to add quadratic terms for numeric covariates",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "check_overlap",
            "description": "Check overlap/common support between treatment groups based on estimated propensity scores.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "check_covariate_balance",
            "description": "Check standardized mean differences for all covariates between treatment groups.",
            "parameters": {
                "type": "object",
                "properties": {
                    "weighted": {
                        "type": "boolean",
                        "description": "If true, compute balance after IPW weighting",
                    },
                    "specific_covariates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific covariates to check. If empty, checks all.",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "check_calibration",
            "description": "Check calibration of the propensity score model (how well predicted probabilities match observed rates).",
            "parameters": {
                "type": "object",
                "properties": {
                    "n_bins": {
                        "type": "integer",
                        "description": "Number of bins for calibration curve",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "compute_trimmed_stats",
            "description": "Compute statistics after trimming extreme propensity scores.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lower_bound": {
                        "type": "number",
                        "description": "Lower bound for trimming (e.g., 0.01)",
                    },
                    "upper_bound": {
                        "type": "number",
                        "description": "Upper bound for trimming (e.g., 0.99)",
                    },
                },
                "required": ["lower_bound", "upper_bound"],
            },
        },
        {
            "name": "get_covariate_summary",
            "description": "Get summary statistics for a specific covariate to understand imbalance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "covariate": {
                        "type": "string",
                        "description": "Name of the covariate to examine",
                    },
                },
                "required": ["covariate"],
            },
        },
        {
            "name": "finalize_diagnostics",
            "description": "Finalize PS diagnostics with recommendations. Call this when you have enough evidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_quality": {
                        "type": "string",
                        "enum": ["excellent", "good", "acceptable", "needs_improvement", "unreliable"],
                        "description": "Overall quality of the PS model",
                    },
                    "proceed_with_analysis": {
                        "type": "boolean",
                        "description": "Whether to proceed with PS-based estimation",
                    },
                    "recommended_method": {
                        "type": "string",
                        "enum": ["ipw", "matching", "aipw", "doubly_robust", "other"],
                        "description": "Recommended estimation method given diagnostics",
                    },
                    "trimming_bounds": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "[lower, upper] bounds for PS trimming, or null if no trimming",
                    },
                    "warnings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Important warnings about using PS in this analysis",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Detailed reasoning for recommendations",
                    },
                },
                "required": ["model_quality", "proceed_with_analysis", "recommended_method", "reasoning"],
            },
        },
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._df: pd.DataFrame | None = None
        self._treatment_var: str | None = None
        self._outcome_var: str | None = None
        self._confounders: list[str] = []
        self._propensity_scores: np.ndarray | None = None
        self._treatment: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._state: AnalysisState | None = None

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute propensity score diagnostics using agentic loop.

        Args:
            state: Current analysis state

        Returns:
            Updated state with PS diagnostics
        """
        self.logger.info("ps_diagnostics_start", job_id=state.job_id)
        start_time = time.time()

        try:
            # Load data
            df = self._load_dataframe(state)
            if df is None:
                self.logger.warning("no_dataframe_for_ps_diagnostics")
                return state

            self._df = df
            self._state = state
            self._treatment_var = state.treatment_variable
            self._outcome_var = state.outcome_variable

            if not self._treatment_var or not self._outcome_var:
                self.logger.warning("no_treatment_outcome_vars")
                return state

            # Get confounders
            if state.confounder_discovery and state.confounder_discovery.get("ranked_confounders"):
                self._confounders = state.confounder_discovery["ranked_confounders"]
            elif state.data_profile and state.data_profile.potential_confounders:
                self._confounders = state.data_profile.potential_confounders
            else:
                self._confounders = [
                    c for c in df.select_dtypes(include=[np.number]).columns
                    if c != self._treatment_var and c != self._outcome_var
                ]

            # Filter to valid columns
            self._confounders = [
                c for c in self._confounders
                if c in df.columns and c != self._treatment_var and c != self._outcome_var
            ]

            if not self._confounders:
                self.logger.warning("no_confounders_for_ps")
                return state

            # Build initial prompt
            initial_prompt = self._build_initial_prompt()

            # Run agentic loop
            final_result = await self._run_agentic_loop(initial_prompt, max_iterations=15)

            # Store results
            state.ps_diagnostics = final_result

            # Create trace
            duration_ms = int((time.time() - start_time) * 1000)
            trace = self.create_trace(
                action="ps_diagnostics_complete",
                reasoning=final_result.get("reasoning", "PS diagnostics completed"),
                outputs={
                    "model_quality": final_result.get("model_quality"),
                    "proceed": final_result.get("proceed_with_analysis"),
                    "recommended_method": final_result.get("recommended_method"),
                },
                duration_ms=duration_ms,
            )
            state.add_trace(trace)

            self.logger.info(
                "ps_diagnostics_complete",
                model_quality=final_result.get("model_quality"),
            )

        except Exception as e:
            self.logger.error("ps_diagnostics_failed", error=str(e))
            import traceback
            traceback.print_exc()

        return state

    def _load_dataframe(self, state: AnalysisState) -> pd.DataFrame | None:
        """Load dataframe from state."""
        if state.dataframe_path:
            try:
                if state.dataframe_path.endswith(".pkl"):
                    with open(state.dataframe_path, "rb") as f:
                        return pickle.load(f)
                else:
                    return pd.read_csv(state.dataframe_path)
            except Exception as e:
                self.logger.error("load_failed", error=str(e))
        return None

    def _build_initial_prompt(self) -> str:
        """Build the initial prompt for the agentic loop."""
        return f"""You need to validate the propensity score model for this causal analysis.

Treatment variable: {self._treatment_var}
Outcome variable: {self._outcome_var}
Available confounders ({len(self._confounders)}): {self._confounders[:15]}{'...' if len(self._confounders) > 15 else ''}

Start by estimating propensity scores, then systematically check:
1. Overlap between treatment groups
2. Covariate balance (with and without IPW weighting)
3. Model calibration

If diagnostics are poor, try alternative specifications (interactions, polynomials, trimming).
Provide final recommendations when you have sufficient evidence."""

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
                "ps_diagnostics_iteration",
                iteration=iteration,
                max_iterations=max_iterations,
                has_ps_scores=self._propensity_scores is not None,
            )

            try:
                result = await self.reason(
                    prompt=messages[-1]["content"] if messages[-1]["role"] == "user" else "Continue your analysis.",
                    context={"iteration": iteration},
                )
            except Exception as e:
                self.logger.warning("llm_call_failed", iteration=iteration, error=str(e))
                return self._auto_finalize()

            # Log LLM reasoning if provided
            response_text = result.get("response", "")
            if response_text:
                self.logger.info(
                    "ps_llm_reasoning",
                    reasoning=response_text[:500] + ("..." if len(response_text) > 500 else ""),
                )

            pending_calls = result.get("pending_calls", [])

            if not pending_calls:
                self.logger.info("ps_no_tool_calls", response_preview=response_text[:200])
                if "finalize" in response_text.lower():
                    return self._auto_finalize()
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": "Please call a tool to continue. If done, call finalize_diagnostics."
                })
                continue

            # Execute tool calls
            tool_results = []
            for call in pending_calls:
                tool_name = call["name"]
                tool_args = call.get("args", {})

                # Check for finalize with detailed logging
                if tool_name == "finalize_diagnostics":
                    self.logger.info(
                        "ps_finalizing",
                        model_quality=tool_args.get("model_quality"),
                        proceed_with_analysis=tool_args.get("proceed_with_analysis"),
                        recommended_method=tool_args.get("recommended_method"),
                    )
                    final_result = tool_args
                    return final_result

                # Log the tool call decision
                self.logger.info(
                    "ps_tool_decision",
                    tool=tool_name,
                    args=tool_args,
                )

                try:
                    tool_result = self._execute_tool(tool_name, tool_args)
                    tool_results.append(f"[{tool_name}]: {tool_result}")

                    # Log tool result summary
                    self.logger.info(
                        "ps_tool_result",
                        tool=tool_name,
                        result_summary=tool_result[:200] + ("..." if len(tool_result) > 200 else ""),
                    )
                except Exception as e:
                    self.logger.warning("tool_execution_error", tool=tool_name, error=str(e))
                    tool_results.append(f"[{tool_name}]: ERROR - {str(e)}")

            messages.append({
                "role": "user",
                "content": "Tool results:\n\n" + "\n\n".join(tool_results) + "\n\nAnalyze and decide next step."
            })

        return self._auto_finalize()

    def _auto_finalize(self) -> dict[str, Any]:
        """Auto-finalize when LLM doesn't explicitly finalize."""
        if self._propensity_scores is None:
            return {
                "model_quality": "unreliable",
                "proceed_with_analysis": False,
                "recommended_method": "other",
                "warnings": ["Could not estimate propensity scores"],
                "reasoning": "PS estimation failed",
            }

        # Compute basic diagnostics
        T = self._treatment
        ps = self._propensity_scores

        treated_idx = T == 1
        control_idx = T == 0

        overlap_min = max(ps[control_idx].min(), ps[treated_idx].min())
        overlap_max = min(ps[control_idx].max(), ps[treated_idx].max())
        pct_overlap = np.mean((ps >= overlap_min) & (ps <= overlap_max)) * 100

        positivity_violations = np.sum((ps < 0.01) | (ps > 0.99))

        if pct_overlap > 85 and positivity_violations < 10:
            quality = "good"
            proceed = True
            method = "aipw"
        elif pct_overlap > 70:
            quality = "acceptable"
            proceed = True
            method = "aipw"
        else:
            quality = "needs_improvement"
            proceed = True
            method = "doubly_robust"

        return {
            "model_quality": quality,
            "proceed_with_analysis": proceed,
            "recommended_method": method,
            "trimming_bounds": [0.01, 0.99] if positivity_violations > 5 else None,
            "warnings": ["Auto-finalized due to incomplete analysis"] if positivity_violations > 10 else [],
            "reasoning": f"Auto-finalized: {pct_overlap:.1f}% overlap, {positivity_violations} positivity violations",
        }

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool and return the result string."""
        if tool_name == "estimate_propensity_scores":
            return self._tool_estimate_ps(args)
        elif tool_name == "check_overlap":
            return self._tool_check_overlap()
        elif tool_name == "check_covariate_balance":
            return self._tool_check_balance(args)
        elif tool_name == "check_calibration":
            return self._tool_check_calibration(args)
        elif tool_name == "compute_trimmed_stats":
            return self._tool_compute_trimmed(args)
        elif tool_name == "get_covariate_summary":
            return self._tool_get_covariate_summary(args)
        else:
            return f"Unknown tool: {tool_name}"

    def _tool_estimate_ps(self, args: dict) -> str:
        """Estimate propensity scores."""
        df = self._df
        covariates = args.get("covariates", []) or self._confounders
        add_interactions = args.get("add_interactions", False)
        add_polynomials = args.get("add_polynomials", False)

        # Filter to valid covariates
        covariates = [c for c in covariates if c in df.columns]
        if not covariates:
            return "No valid covariates specified."

        # Prepare features
        X = df[covariates].values
        T = df[self._treatment_var].values

        # Handle missing
        mask = ~(np.any(np.isnan(X), axis=1) | np.isnan(T))
        X = X[mask]
        T = T[mask]

        if len(X) < 50:
            return "Insufficient data after removing missing values."

        # Add polynomial terms
        if add_polynomials:
            X_poly = []
            for i in range(X.shape[1]):
                X_poly.append(X[:, i])
                X_poly.append(X[:, i] ** 2)
            X = np.column_stack(X_poly)

        # Add interactions (top pairs only to avoid explosion)
        if add_interactions and X.shape[1] <= 10:
            interactions = []
            for i in range(min(X.shape[1], 5)):
                for j in range(i + 1, min(X.shape[1], 5)):
                    interactions.append(X[:, i] * X[:, j])
            if interactions:
                X = np.column_stack([X] + interactions)

        # Fit model
        try:
            model = LogisticRegression(max_iter=1000, random_state=42, penalty='l2', C=1.0)
            model.fit(X, T)
            ps = model.predict_proba(X)[:, 1]
        except Exception as e:
            return f"Model fitting failed: {str(e)}"

        # Store results
        self._propensity_scores = ps
        self._treatment = T
        self._mask = mask

        # Summary
        n_treated = int(np.sum(T == 1))
        n_control = int(np.sum(T == 0))

        return f"""Propensity Score Model Estimated:
- Covariates used: {len(covariates)} (interactions={add_interactions}, polynomials={add_polynomials})
- N treated: {n_treated}
- N control: {n_control}
- PS range (treated): [{ps[T==1].min():.3f}, {ps[T==1].max():.3f}], mean={ps[T==1].mean():.3f}
- PS range (control): [{ps[T==0].min():.3f}, {ps[T==0].max():.3f}], mean={ps[T==0].mean():.3f}
- PS near 0 (<0.01): {np.sum(ps < 0.01)}
- PS near 1 (>0.99): {np.sum(ps > 0.99)}
"""

    def _tool_check_overlap(self) -> str:
        """Check overlap between treatment groups."""
        if self._propensity_scores is None:
            return "No propensity scores estimated yet. Call estimate_propensity_scores first."

        ps = self._propensity_scores
        T = self._treatment

        treated_idx = T == 1
        control_idx = T == 0

        ps_treated = ps[treated_idx]
        ps_control = ps[control_idx]

        # Overlap region
        overlap_min = max(ps_control.min(), ps_treated.min())
        overlap_max = min(ps_control.max(), ps_treated.max())

        # Proportion in common support
        in_overlap = (ps >= overlap_min) & (ps <= overlap_max)
        pct_overlap = np.mean(in_overlap) * 100

        # Distribution comparison
        ks_stat, ks_pval = stats.ks_2samp(ps_treated, ps_control)

        # Positivity assessment
        near_zero = np.sum(ps < 0.01)
        near_one = np.sum(ps > 0.99)
        positivity_violations = near_zero + near_one

        # Quality assessment
        if pct_overlap > 95 and positivity_violations < 5:
            quality = "EXCELLENT"
        elif pct_overlap > 85 and positivity_violations < 20:
            quality = "GOOD"
        elif pct_overlap > 70:
            quality = "MODERATE"
        elif pct_overlap > 50:
            quality = "POOR"
        else:
            quality = "CRITICAL"

        return f"""Overlap Assessment:

PS Distribution Comparison:
- Treated: mean={ps_treated.mean():.3f}, std={ps_treated.std():.3f}, median={np.median(ps_treated):.3f}
- Control: mean={ps_control.mean():.3f}, std={ps_control.std():.3f}, median={np.median(ps_control):.3f}
- KS statistic: {ks_stat:.3f} (p={ks_pval:.4f})

Overlap Region:
- Common support: [{overlap_min:.3f}, {overlap_max:.3f}]
- Percent in overlap: {pct_overlap:.1f}%
- Overlap quality: {quality}

Positivity Violations:
- PS < 0.01: {near_zero} observations
- PS > 0.99: {near_one} observations
- Total violations: {positivity_violations}

Recommendation: {"Good overlap, proceed with PS methods" if quality in ["EXCELLENT", "GOOD"] else "Consider trimming extreme PS values or using doubly robust methods"}
"""

    def _tool_check_balance(self, args: dict) -> str:
        """Check covariate balance."""
        if self._propensity_scores is None:
            return "No propensity scores estimated yet."

        weighted = args.get("weighted", False)
        specific_covariates = args.get("specific_covariates", [])

        df = self._df
        ps = self._propensity_scores
        T = self._treatment
        mask = self._mask

        # Get covariates to check
        covariates = specific_covariates if specific_covariates else self._confounders[:15]
        covariates = [c for c in covariates if c in df.columns]

        results = []
        smd_values = []
        imbalanced = []
        severely_imbalanced = []

        df_subset = df[mask].copy()

        for cov in covariates:
            x = df_subset[cov].values

            if weighted:
                # IPW weights
                weights_t = T / ps
                weights_c = (1 - T) / (1 - ps)

                # Clip extreme weights
                weights_t = np.clip(weights_t, 0, 10)
                weights_c = np.clip(weights_c, 0, 10)

                treated_mean = np.average(x[T == 1], weights=weights_t[T == 1])
                control_mean = np.average(x[T == 0], weights=weights_c[T == 0])
            else:
                treated_mean = np.mean(x[T == 1])
                control_mean = np.mean(x[T == 0])

            pooled_std = np.sqrt((np.var(x[T == 1]) + np.var(x[T == 0])) / 2)
            smd = (treated_mean - control_mean) / pooled_std if pooled_std > 0 else 0

            smd_values.append(abs(smd))

            if abs(smd) >= 0.25:
                status = "SEVERE"
                severely_imbalanced.append(cov)
                imbalanced.append(cov)
            elif abs(smd) >= 0.1:
                status = "IMBALANCED"
                imbalanced.append(cov)
            else:
                status = "OK"

            results.append(f"  {cov}: SMD={smd:.3f} ({status})")

        mean_smd = np.mean(smd_values)
        max_smd = np.max(smd_values)

        # Overall assessment
        if mean_smd < 0.05 and len(imbalanced) == 0:
            quality = "EXCELLENT"
        elif mean_smd < 0.1 and len(severely_imbalanced) == 0:
            quality = "GOOD"
        elif mean_smd < 0.15:
            quality = "MODERATE"
        else:
            quality = "POOR"

        return f"""Covariate Balance {'(IPW Weighted)' if weighted else '(Unweighted)'}:

{chr(10).join(results)}

Summary:
- Mean |SMD|: {mean_smd:.3f}
- Max |SMD|: {max_smd:.3f}
- Imbalanced (SMD > 0.1): {len(imbalanced)} covariates
- Severely imbalanced (SMD > 0.25): {len(severely_imbalanced)} covariates
- Overall balance quality: {quality}

Imbalanced covariates: {imbalanced if imbalanced else 'None'}
"""

    def _tool_check_calibration(self, args: dict) -> str:
        """Check model calibration."""
        if self._propensity_scores is None:
            return "No propensity scores estimated yet."

        n_bins = args.get("n_bins", 10)
        ps = self._propensity_scores
        T = self._treatment

        try:
            prob_true, prob_pred = calibration_curve(T, ps, n_bins=n_bins, strategy='uniform')
            calibration_error = np.mean(np.abs(prob_true - prob_pred))

            # Hosmer-Lemeshow style assessment
            well_calibrated = calibration_error < 0.1

            bin_results = []
            for i, (true, pred) in enumerate(zip(prob_true, prob_pred, strict=False)):
                bin_results.append(f"  Bin {i+1}: predicted={pred:.3f}, observed={true:.3f}, diff={abs(true-pred):.3f}")

            return f"""Calibration Assessment:

Calibration Curve ({n_bins} bins):
{chr(10).join(bin_results)}

Summary:
- Mean calibration error: {calibration_error:.4f}
- Well calibrated: {"Yes" if well_calibrated else "No"}
- Interpretation: {"Model probabilities are reliable" if well_calibrated else "Model may be misspecified - consider adding terms or using different link"}
"""
        except Exception as e:
            return f"Calibration check failed: {str(e)}"

    def _tool_compute_trimmed(self, args: dict) -> str:
        """Compute statistics after trimming."""
        if self._propensity_scores is None:
            return "No propensity scores estimated yet."

        lower = args.get("lower_bound", 0.01)
        upper = args.get("upper_bound", 0.99)

        ps = self._propensity_scores
        T = self._treatment

        # Apply trimming
        keep = (ps >= lower) & (ps <= upper)
        n_trimmed = np.sum(~keep)
        pct_trimmed = (n_trimmed / len(ps)) * 100

        ps_trimmed = ps[keep]
        T_trimmed = T[keep]

        n_treated_trimmed = np.sum(T_trimmed == 1)
        n_control_trimmed = np.sum(T_trimmed == 0)

        # Check overlap after trimming
        ps_treated = ps_trimmed[T_trimmed == 1]
        ps_control = ps_trimmed[T_trimmed == 0]

        overlap_min = max(ps_control.min(), ps_treated.min())
        overlap_max = min(ps_control.max(), ps_treated.max())
        pct_overlap = np.mean((ps_trimmed >= overlap_min) & (ps_trimmed <= overlap_max)) * 100

        return f"""Trimmed Sample Statistics (bounds=[{lower}, {upper}]):

Trimming Impact:
- Observations trimmed: {n_trimmed} ({pct_trimmed:.1f}%)
- Remaining sample: {len(ps_trimmed)}
- Remaining treated: {n_treated_trimmed}
- Remaining control: {n_control_trimmed}

After Trimming:
- PS range (treated): [{ps_treated.min():.3f}, {ps_treated.max():.3f}]
- PS range (control): [{ps_control.min():.3f}, {ps_control.max():.3f}]
- Overlap: {pct_overlap:.1f}%

Assessment: {"Trimming improves overlap" if pct_overlap > 90 else "Consider different bounds"}
"""

    def _tool_get_covariate_summary(self, args: dict) -> str:
        """Get detailed summary for a specific covariate."""
        cov = args.get("covariate")
        if not cov or cov not in self._df.columns:
            return f"Covariate '{cov}' not found."

        df = self._df
        T = df[self._treatment_var].values
        x = df[cov].values

        # Remove missing
        valid = ~(np.isnan(x) | np.isnan(T))
        x = x[valid]
        T = T[valid]

        treated_vals = x[T == 1]
        control_vals = x[T == 0]

        return f"""Covariate Summary: {cov}

Treated Group:
- N: {len(treated_vals)}
- Mean: {np.mean(treated_vals):.4f}
- Std: {np.std(treated_vals):.4f}
- Median: {np.median(treated_vals):.4f}
- Range: [{np.min(treated_vals):.4f}, {np.max(treated_vals):.4f}]

Control Group:
- N: {len(control_vals)}
- Mean: {np.mean(control_vals):.4f}
- Std: {np.std(control_vals):.4f}
- Median: {np.median(control_vals):.4f}
- Range: [{np.min(control_vals):.4f}, {np.max(control_vals):.4f}]

Difference:
- Mean difference: {np.mean(treated_vals) - np.mean(control_vals):.4f}
- SMD: {(np.mean(treated_vals) - np.mean(control_vals)) / np.sqrt((np.var(treated_vals) + np.var(control_vals)) / 2):.4f}
"""
