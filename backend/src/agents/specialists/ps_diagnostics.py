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

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

from src.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.agents.base.context_tools import ContextTools
from src.agents.base.react_agent import ReActAgent
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


class PSDiagnosticsAgent(ReActAgent, ContextTools):
    """Truly agentic propensity score diagnostics using ReAct pattern.

    The LLM iteratively:
    1. Estimates propensity scores with different specifications
    2. Checks overlap and identifies positivity violations
    3. Assesses covariate balance with/without weighting
    4. Tests model calibration
    5. Tries improvements if diagnostics are poor
    6. Provides final recommendations

    Uses ReAct pattern with pull-based context tools.
    """

    AGENT_NAME = "ps_diagnostics"
    MAX_STEPS = 15

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

    def __init__(self) -> None:
        super().__init__()
        self._df: pd.DataFrame | None = None
        self._treatment_var: str | None = None
        self._outcome_var: str | None = None
        self._confounders: list[str] = []
        self._propensity_scores: np.ndarray | None = None
        self._treatment: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._current_state: AnalysisState | None = None
        self._finalized: bool = False

        # Register context tools from mixin
        self.register_context_tools()
        # Register PS-specific tools
        self._register_ps_tools()

    def _register_ps_tools(self) -> None:
        """Register propensity score diagnostic tools."""
        self.register_tool(
            name="estimate_propensity_scores",
            description="Estimate propensity scores using logistic regression with specified covariates.",
            handler=self._tool_estimate_ps,
            parameters={
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
        )

        self.register_tool(
            name="check_overlap",
            description="Check overlap/common support between treatment groups based on estimated propensity scores.",
            handler=self._tool_check_overlap,
            parameters={},
        )

        self.register_tool(
            name="check_covariate_balance",
            description="Check standardized mean differences for all covariates between treatment groups.",
            handler=self._tool_check_balance,
            parameters={
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
        )

        self.register_tool(
            name="check_calibration",
            description="Check calibration of the propensity score model (how well predicted probabilities match observed rates).",
            handler=self._tool_check_calibration,
            parameters={
                "n_bins": {
                    "type": "integer",
                    "description": "Number of bins for calibration curve",
                },
            },
        )

        self.register_tool(
            name="compute_trimmed_stats",
            description="Compute statistics after trimming extreme propensity scores.",
            handler=self._tool_compute_trimmed,
            parameters={
                "lower_bound": {
                    "type": "number",
                    "description": "Lower bound for trimming (e.g., 0.01)",
                },
                "upper_bound": {
                    "type": "number",
                    "description": "Upper bound for trimming (e.g., 0.99)",
                },
            },
        )

        self.register_tool(
            name="get_covariate_summary",
            description="Get summary statistics for a specific covariate to understand imbalance.",
            handler=self._tool_get_covariate_summary,
            parameters={
                "covariate": {
                    "type": "string",
                    "description": "Name of the covariate to examine",
                },
            },
        )

        self.register_tool(
            name="finalize_diagnostics",
            description="Finalize PS diagnostics with recommendations. Call this when you have enough evidence.",
            handler=self._tool_finalize_diagnostics,
            parameters={
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
        )

    def _get_initial_observation(self, state: AnalysisState) -> str:
        """Build initial observation for PS diagnostics."""
        return f"""PROPENSITY SCORE DIAGNOSTICS TASK
Treatment variable: {self._treatment_var}
Outcome variable: {self._outcome_var}
Available confounders ({len(self._confounders)}): {self._confounders[:15]}{'...' if len(self._confounders) > 15 else ''}

Start by estimating propensity scores, then systematically check:
1. Overlap between treatment groups
2. Covariate balance (with and without IPW weighting)
3. Model calibration

If diagnostics are poor, try alternative specifications (interactions, polynomials, trimming).
Call finalize_diagnostics when you have sufficient evidence."""

    async def is_task_complete(self, state: AnalysisState) -> bool:
        """Check if PS diagnostics is complete."""
        return self._finalized

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
            self._current_state = state
            self._finalized = False
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

            # Run ReAct loop
            await super().execute(state)

            # Auto-finalize if LLM didn't call finalize_diagnostics
            if not self._finalized:
                final_result = self._auto_finalize()
                state.ps_diagnostics = final_result

            # Create trace
            duration_ms = int((time.time() - start_time) * 1000)
            ps_diag = state.ps_diagnostics or {}
            trace = self.create_trace(
                action="ps_diagnostics_complete",
                reasoning=ps_diag.get("reasoning", "PS diagnostics completed"),
                outputs={
                    "model_quality": ps_diag.get("model_quality"),
                    "proceed": ps_diag.get("proceed_with_analysis"),
                    "recommended_method": ps_diag.get("recommended_method"),
                },
                duration_ms=duration_ms,
            )
            state.add_trace(trace)

            self.logger.info(
                "ps_diagnostics_complete",
                model_quality=ps_diag.get("model_quality"),
            )

        except Exception as e:
            self.logger.exception("ps_diagnostics_failed", error=str(e))

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

    def _auto_finalize(self) -> dict:
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

    # -------------------------------------------------------------------------
    # Tool Handlers (async, return ToolResult)
    # -------------------------------------------------------------------------

    async def _tool_estimate_ps(
        self,
        state: AnalysisState,
        covariates: list[str] | None = None,
        add_interactions: bool = False,
        add_polynomials: bool = False,
    ) -> ToolResult:
        """Estimate propensity scores."""
        df = self._df
        cov_list = covariates if covariates else self._confounders

        # Filter to valid covariates
        cov_list = [c for c in cov_list if c in df.columns]
        if not cov_list:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error="No valid covariates specified.",
            )

        # Prepare features
        X = df[cov_list].values
        T = df[self._treatment_var].values

        # Handle missing
        mask = ~(np.any(np.isnan(X), axis=1) | np.isnan(T))
        X = X[mask]
        T = T[mask]

        if len(X) < 50:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error="Insufficient data after removing missing values.",
            )

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
            model = LogisticRegression(max_iter=1000, random_state=42, penalty="l2", C=1.0)
            model.fit(X, T)
            ps = model.predict_proba(X)[:, 1]
        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error=f"Model fitting failed: {str(e)}",
            )

        # Store results
        self._propensity_scores = ps
        self._treatment = T
        self._mask = mask

        # Summary
        n_treated = int(np.sum(T == 1))
        n_control = int(np.sum(T == 0))

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "covariates_used": len(cov_list),
                "add_interactions": add_interactions,
                "add_polynomials": add_polynomials,
                "n_treated": n_treated,
                "n_control": n_control,
                "ps_treated_range": [float(ps[T == 1].min()), float(ps[T == 1].max())],
                "ps_treated_mean": float(ps[T == 1].mean()),
                "ps_control_range": [float(ps[T == 0].min()), float(ps[T == 0].max())],
                "ps_control_mean": float(ps[T == 0].mean()),
                "ps_near_zero": int(np.sum(ps < 0.01)),
                "ps_near_one": int(np.sum(ps > 0.99)),
            },
        )

    async def _tool_check_overlap(self, state: AnalysisState) -> ToolResult:
        """Check overlap between treatment groups."""
        if self._propensity_scores is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error="No propensity scores estimated yet. Call estimate_propensity_scores first.",
            )

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
        near_zero = int(np.sum(ps < 0.01))
        near_one = int(np.sum(ps > 0.99))
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

        recommendation = (
            "Good overlap, proceed with PS methods"
            if quality in ["EXCELLENT", "GOOD"]
            else "Consider trimming extreme PS values or using doubly robust methods"
        )

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "treated_stats": {
                    "mean": float(ps_treated.mean()),
                    "std": float(ps_treated.std()),
                    "median": float(np.median(ps_treated)),
                },
                "control_stats": {
                    "mean": float(ps_control.mean()),
                    "std": float(ps_control.std()),
                    "median": float(np.median(ps_control)),
                },
                "ks_statistic": float(ks_stat),
                "ks_pvalue": float(ks_pval),
                "common_support": [float(overlap_min), float(overlap_max)],
                "pct_overlap": float(pct_overlap),
                "quality": quality,
                "ps_near_zero": near_zero,
                "ps_near_one": near_one,
                "positivity_violations": positivity_violations,
                "recommendation": recommendation,
            },
        )

    async def _tool_check_balance(
        self,
        state: AnalysisState,
        weighted: bool = False,
        specific_covariates: list[str] | None = None,
    ) -> ToolResult:
        """Check covariate balance."""
        if self._propensity_scores is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error="No propensity scores estimated yet.",
            )

        df = self._df
        ps = self._propensity_scores
        T = self._treatment
        mask = self._mask

        # Get covariates to check
        cov_list = specific_covariates if specific_covariates else self._confounders[:15]
        cov_list = [c for c in cov_list if c in df.columns]

        smd_results = {}
        smd_values = []
        imbalanced = []
        severely_imbalanced = []

        df_subset = df[mask].copy()

        for cov in cov_list:
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

            smd_results[cov] = {"smd": float(smd), "status": status}

        mean_smd = float(np.mean(smd_values))
        max_smd = float(np.max(smd_values))

        # Overall assessment
        if mean_smd < 0.05 and len(imbalanced) == 0:
            quality = "EXCELLENT"
        elif mean_smd < 0.1 and len(severely_imbalanced) == 0:
            quality = "GOOD"
        elif mean_smd < 0.15:
            quality = "MODERATE"
        else:
            quality = "POOR"

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "weighted": weighted,
                "covariate_balance": smd_results,
                "mean_smd": mean_smd,
                "max_smd": max_smd,
                "n_imbalanced": len(imbalanced),
                "n_severely_imbalanced": len(severely_imbalanced),
                "imbalanced_covariates": imbalanced,
                "quality": quality,
            },
        )

    async def _tool_check_calibration(
        self, state: AnalysisState, n_bins: int = 10
    ) -> ToolResult:
        """Check model calibration."""
        if self._propensity_scores is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error="No propensity scores estimated yet.",
            )

        ps = self._propensity_scores
        T = self._treatment

        try:
            prob_true, prob_pred = calibration_curve(T, ps, n_bins=n_bins, strategy="uniform")
            calibration_error = float(np.mean(np.abs(prob_true - prob_pred)))

            # Hosmer-Lemeshow style assessment
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

    async def _tool_compute_trimmed(
        self, state: AnalysisState, lower_bound: float, upper_bound: float
    ) -> ToolResult:
        """Compute statistics after trimming."""
        if self._propensity_scores is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error="No propensity scores estimated yet.",
            )

        ps = self._propensity_scores
        T = self._treatment

        # Apply trimming
        keep = (ps >= lower_bound) & (ps <= upper_bound)
        n_trimmed = int(np.sum(~keep))
        pct_trimmed = float((n_trimmed / len(ps)) * 100)

        ps_trimmed = ps[keep]
        T_trimmed = T[keep]

        n_treated_trimmed = int(np.sum(T_trimmed == 1))
        n_control_trimmed = int(np.sum(T_trimmed == 0))

        # Check overlap after trimming
        ps_treated = ps_trimmed[T_trimmed == 1]
        ps_control = ps_trimmed[T_trimmed == 0]

        overlap_min = max(ps_control.min(), ps_treated.min())
        overlap_max = min(ps_control.max(), ps_treated.max())
        pct_overlap = float(np.mean((ps_trimmed >= overlap_min) & (ps_trimmed <= overlap_max)) * 100)

        assessment = "Trimming improves overlap" if pct_overlap > 90 else "Consider different bounds"

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "bounds": [lower_bound, upper_bound],
                "n_trimmed": n_trimmed,
                "pct_trimmed": pct_trimmed,
                "remaining_sample": len(ps_trimmed),
                "remaining_treated": n_treated_trimmed,
                "remaining_control": n_control_trimmed,
                "ps_treated_range": [float(ps_treated.min()), float(ps_treated.max())],
                "ps_control_range": [float(ps_control.min()), float(ps_control.max())],
                "pct_overlap": pct_overlap,
                "assessment": assessment,
            },
        )

    async def _tool_get_covariate_summary(
        self, state: AnalysisState, covariate: str
    ) -> ToolResult:
        """Get detailed summary for a specific covariate."""
        if not covariate or covariate not in self._df.columns:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error=f"Covariate '{covariate}' not found.",
            )

        df = self._df
        T = df[self._treatment_var].values
        x = df[covariate].values

        # Remove missing
        valid = ~(np.isnan(x) | np.isnan(T))
        x = x[valid]
        T = T[valid]

        treated_vals = x[T == 1]
        control_vals = x[T == 0]

        pooled_var = (np.var(treated_vals) + np.var(control_vals)) / 2
        smd = (np.mean(treated_vals) - np.mean(control_vals)) / np.sqrt(pooled_var) if pooled_var > 0 else 0

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "covariate": covariate,
                "treated": {
                    "n": len(treated_vals),
                    "mean": float(np.mean(treated_vals)),
                    "std": float(np.std(treated_vals)),
                    "median": float(np.median(treated_vals)),
                    "range": [float(np.min(treated_vals)), float(np.max(treated_vals))],
                },
                "control": {
                    "n": len(control_vals),
                    "mean": float(np.mean(control_vals)),
                    "std": float(np.std(control_vals)),
                    "median": float(np.median(control_vals)),
                    "range": [float(np.min(control_vals)), float(np.max(control_vals))],
                },
                "mean_difference": float(np.mean(treated_vals) - np.mean(control_vals)),
                "smd": float(smd),
            },
        )

    async def _tool_finalize_diagnostics(
        self,
        state: AnalysisState,
        model_quality: str,
        proceed_with_analysis: bool,
        recommended_method: str,
        reasoning: str,
        trimming_bounds: list[float] | None = None,
        warnings: list[str] | None = None,
    ) -> ToolResult:
        """Finalize PS diagnostics with recommendations."""
        self.logger.info(
            "ps_finalizing",
            model_quality=model_quality,
            proceed_with_analysis=proceed_with_analysis,
            recommended_method=recommended_method,
        )

        result = {
            "model_quality": model_quality,
            "proceed_with_analysis": proceed_with_analysis,
            "recommended_method": recommended_method,
            "trimming_bounds": trimming_bounds,
            "warnings": warnings or [],
            "reasoning": reasoning,
        }

        state.ps_diagnostics = result
        self._finalized = True

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=result,
        )
