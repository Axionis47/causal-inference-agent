"""Critique Agent - Truly Agentic Analysis Review with Data Investigation.

This agent reviews causal analysis quality using multi-perspective debate AND data investigation:
- Advocate, Skeptic, and Methodologist perspectives debate
- Each perspective can INVESTIGATE the actual data using tools
- Critics can verify claims, check diagnostics, and find issues
- Final synthesis is grounded in actual evidence

Not just summaries - critics can actually look at the data!
"""

import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.agents.base import (
    AnalysisState,
    BaseAgent,
    CritiqueDecision,
    CritiqueFeedback,
    JobStatus,
)
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


@dataclass
class PerspectiveArgument:
    """An argument from a specific perspective."""
    perspective: str
    position: str  # "support", "challenge", "neutral"
    key_points: list[str]
    evidence: list[str]
    concerns: list[str]
    confidence: float  # 0-1


class CritiqueAgent(BaseAgent):
    """Truly agentic critique agent with data investigation capabilities.

    This agent implements a "debate" pattern where perspectives can investigate:

    1. ADVOCATE: Argues the analysis is valid, investigates evidence FOR
    2. SKEPTIC: Challenges claims, investigates potential problems
    3. METHODOLOGIST: Neutral arbiter, verifies technical quality

    Unlike summary-only critique, this agent can:
    - Check actual covariate balance
    - Verify treatment effect estimates
    - Investigate outliers and influential observations
    - Run diagnostic checks
    - Compare estimates across subgroups

    Evidence-based debate leads to better critique.
    """

    AGENT_NAME = "critique"

    SYSTEM_PROMPT = """You are an expert methodologist reviewing causal inference analyses.
Your role is to provide rigorous critique by INVESTIGATING the actual data and results.

CRITICAL: You have tools to investigate claims. Use them to gather evidence before
forming your critique. Do NOT just rely on summaries - verify claims yourself!

Investigation workflow:
1. Review the analysis summary to understand what was done
2. Use tools to verify key claims and check for problems
3. Look for issues that summaries might miss
4. Form your critique based on EVIDENCE from your investigation

When evaluating, consider:
- STATISTICAL VALIDITY: Are methods appropriate? Sample sizes adequate?
- ASSUMPTION CHECKING: Were key assumptions tested?
- METHOD SELECTION: Right methods for the data structure?
- COMPLETENESS: Sensitivity analysis done? Multiple methods?
- ROBUSTNESS: Do results hold across subgroups and specifications?

Use the investigation tools to gather evidence, then call finalize_critique."""

    TOOLS = [
        {
            "name": "get_analysis_summary",
            "description": "Get overview of the analysis including methods used, estimates, and diagnostics.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "check_covariate_balance",
            "description": "Check actual covariate balance between treatment groups. Critical for verifying selection bias.",
            "parameters": {
                "type": "object",
                "properties": {
                    "covariates": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Covariates to check (empty for all)",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "check_estimate_consistency",
            "description": "Check consistency of treatment effect estimates across different methods.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "check_subgroup_effects",
            "description": "Check if treatment effect varies across subgroups (heterogeneity).",
            "parameters": {
                "type": "object",
                "properties": {
                    "subgroup_variable": {
                        "type": "string",
                        "description": "Variable to split by (empty to auto-select)",
                    },
                },
                "required": [],
            },
        },
        {
            "name": "check_influential_observations",
            "description": "Identify observations that strongly influence the treatment effect estimate.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "verify_propensity_scores",
            "description": "Verify propensity score quality (overlap, calibration).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "check_sensitivity_robustness",
            "description": "Check sensitivity analysis results for robustness to unmeasured confounding.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "finalize_critique",
            "description": "Finalize your critique with scores and decision. Call this after investigation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "scores": {
                        "type": "object",
                        "properties": {
                            "statistical_validity": {"type": "integer", "minimum": 1, "maximum": 5},
                            "assumption_checking": {"type": "integer", "minimum": 1, "maximum": 5},
                            "method_selection": {"type": "integer", "minimum": 1, "maximum": 5},
                            "completeness": {"type": "integer", "minimum": 1, "maximum": 5},
                            "reproducibility": {"type": "integer", "minimum": 1, "maximum": 5},
                            "interpretation": {"type": "integer", "minimum": 1, "maximum": 5},
                        },
                        "description": "Scores for each dimension (1-5)",
                    },
                    "decision": {
                        "type": "string",
                        "enum": ["APPROVE", "ITERATE", "REJECT"],
                        "description": "Overall decision",
                    },
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific issues found",
                    },
                    "improvements": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Actionable improvements",
                    },
                    "evidence_summary": {
                        "type": "string",
                        "description": "Summary of evidence gathered through investigation",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Detailed reasoning for decision",
                    },
                },
                "required": ["scores", "decision", "reasoning"],
            },
        },
    ]

    def __init__(self):
        """Initialize the critique agent."""
        super().__init__()
        self._df: pd.DataFrame | None = None
        self._state: AnalysisState | None = None
        self._investigation_evidence: list[str] = []

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute critique through investigation and debate.

        Args:
            state: Current analysis state with results

        Returns:
            Updated state with critique feedback
        """
        self.logger.info(
            "critique_start",
            job_id=state.job_id,
            iteration=state.iteration_count,
            n_effects=len(state.treatment_effects),
        )

        state.status = JobStatus.CRITIQUE_REVIEW
        start_time = time.time()

        try:
            # Load data for investigation
            self._df = self._load_dataframe(state)
            self._state = state
            self._investigation_evidence = []

            # Build initial prompt
            initial_prompt = self._build_initial_prompt()

            # Run the agentic investigation loop
            final_result = await self._run_agentic_loop(initial_prompt, max_iterations=10)

            # Create feedback from result
            feedback = self._create_feedback(final_result, state.iteration_count + 1)

            # Add feedback to state
            state.critique_history.append(feedback)

            # Create trace
            duration_ms = int((time.time() - start_time) * 1000)
            trace = self.create_trace(
                action="critique_complete",
                reasoning=feedback.reasoning,
                outputs={
                    "decision": feedback.decision.value,
                    "scores": feedback.scores,
                    "n_issues": len(feedback.issues),
                    "evidence_gathered": len(self._investigation_evidence),
                },
                duration_ms=duration_ms,
            )
            state.add_trace(trace)

            self.logger.info(
                "critique_complete",
                decision=feedback.decision.value,
                avg_score=sum(feedback.scores.values()) / len(feedback.scores),
            )

        except Exception as e:
            self.logger.error("critique_failed", error=str(e))
            import traceback
            traceback.print_exc()
            # Fallback to heuristic critique
            feedback = self._heuristic_critique(state, str(e))
            state.critique_history.append(feedback)

        return state

    def _load_dataframe(self, state: AnalysisState) -> pd.DataFrame | None:
        """Load dataframe for investigation."""
        if state.dataframe_path and Path(state.dataframe_path).exists():
            try:
                with open(state.dataframe_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return None

    def _build_initial_prompt(self) -> str:
        """Build the initial prompt for investigation."""
        state = self._state

        n_methods = len(state.treatment_effects)
        n_sens = len(state.sensitivity_results) if state.sensitivity_results else 0

        return f"""You are critically reviewing a causal inference analysis.

Treatment: {state.treatment_variable}
Outcome: {state.outcome_variable}
Methods used: {n_methods}
Sensitivity analyses: {n_sens}

Your task is to INVESTIGATE the analysis quality by using the available tools.
Do not just accept summaries - verify claims yourself!

Start by calling get_analysis_summary to understand what was done,
then use investigation tools to verify:
1. Is covariate balance adequate?
2. Are estimates consistent across methods?
3. Are there influential observations?
4. Do results hold across subgroups?
5. Are sensitivity results robust?

After gathering evidence, call finalize_critique with your assessment."""

    async def _run_agentic_loop(
        self, initial_prompt: str, max_iterations: int = 10
    ) -> dict[str, Any]:
        """Run the agentic investigation loop."""
        messages = [{"role": "user", "content": initial_prompt}]

        for iteration in range(max_iterations):
            self.logger.info(
                "critique_iteration",
                iteration=iteration,
                max_iterations=max_iterations,
                evidence_gathered=len(self._investigation_evidence),
            )

            # Get LLM response with tool calls
            result = await self.reason(
                prompt=messages[-1]["content"],
                context={"iteration": iteration, "max_iterations": max_iterations},
            )

            # Log LLM reasoning if provided
            response_text = result.get("response", "")
            if response_text:
                self.logger.info(
                    "critique_llm_reasoning",
                    reasoning=response_text[:500] + ("..." if len(response_text) > 500 else ""),
                )

            pending_calls = result.get("pending_calls", [])

            if not pending_calls:
                self.logger.info("critique_no_tool_calls", response_preview=response_text[:200])
                if "finalize" in response_text.lower() or iteration > 7:
                    return self._auto_finalize()
                messages.append({
                    "role": "user",
                    "content": "Please investigate using tools, then call finalize_critique.",
                })
                continue

            # Execute tool calls
            tool_results = []
            for call in pending_calls:
                tool_name = call.get("name")
                tool_args = call.get("args", {})

                # Check for finalize with detailed logging
                if tool_name == "finalize_critique":
                    scores = tool_args.get("scores", {})
                    decision = tool_args.get("decision")
                    self.logger.info(
                        "critique_finalizing",
                        decision=decision,
                        scores=scores,
                        avg_score=sum(scores.values()) / len(scores) if scores else 0,
                        reasoning=tool_args.get("reasoning", "")[:200],
                    )
                    return tool_args

                # Log the tool call decision
                self.logger.info(
                    "critique_tool_decision",
                    tool=tool_name,
                    args=tool_args,
                )

                # Execute the tool
                tool_result = self._execute_tool(tool_name, tool_args)
                tool_results.append(f"## {tool_name}\n{tool_result}")

                # Log tool result summary
                self.logger.info(
                    "critique_tool_result",
                    tool=tool_name,
                    result_summary=tool_result[:200] + ("..." if len(tool_result) > 200 else ""),
                )

            # Feed results back to LLM
            results_text = "\n\n".join(tool_results)
            messages.append({
                "role": "user",
                "content": f"Tool results:\n\n{results_text}\n\nContinue investigation or call finalize_critique.",
            })

        # Max iterations - auto-finalize
        self.logger.warning("critique_max_iterations")
        return self._auto_finalize()

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Execute a tool and return results as string."""
        try:
            if tool_name == "get_analysis_summary":
                return self._tool_get_analysis_summary()
            elif tool_name == "check_covariate_balance":
                return self._tool_check_covariate_balance(args.get("covariates", []))
            elif tool_name == "check_estimate_consistency":
                return self._tool_check_estimate_consistency()
            elif tool_name == "check_subgroup_effects":
                return self._tool_check_subgroup_effects(args.get("subgroup_variable"))
            elif tool_name == "check_influential_observations":
                return self._tool_check_influential_observations()
            elif tool_name == "verify_propensity_scores":
                return self._tool_verify_propensity_scores()
            elif tool_name == "check_sensitivity_robustness":
                return self._tool_check_sensitivity_robustness()
            else:
                return f"Unknown tool: {tool_name}"
        except Exception as e:
            self.logger.error(f"tool_execution_error", tool=tool_name, error=str(e))
            return f"Error executing {tool_name}: {str(e)}"

    def _tool_get_analysis_summary(self) -> str:
        """Get overview of the analysis."""
        state = self._state
        output = "Analysis Summary:\n"
        output += "=" * 50 + "\n"

        # Dataset info
        if state.data_profile:
            output += f"\nDataset:\n"
            output += f"  Samples: {state.data_profile.n_samples}\n"
            output += f"  Features: {state.data_profile.n_features}\n"
            output += f"  Treatment: {state.treatment_variable}\n"
            output += f"  Outcome: {state.outcome_variable}\n"

        # Causal graph
        if state.proposed_dag:
            output += f"\nCausal Graph:\n"
            output += f"  Method: {state.proposed_dag.discovery_method}\n"
            output += f"  Nodes: {len(state.proposed_dag.nodes)}\n"
            output += f"  Edges: {len(state.proposed_dag.edges)}\n"

        # Treatment effects
        output += f"\nTreatment Effect Estimates ({len(state.treatment_effects)}):\n"
        for effect in state.treatment_effects:
            sig = "***" if effect.p_value and effect.p_value < 0.001 else "**" if effect.p_value and effect.p_value < 0.01 else "*" if effect.p_value and effect.p_value < 0.05 else ""
            output += f"  {effect.method}: {effect.estimate:.4f} (SE={effect.std_error:.4f}) {sig}\n"
            output += f"    95% CI: [{effect.ci_lower:.4f}, {effect.ci_upper:.4f}]\n"

        # Sensitivity
        if state.sensitivity_results:
            output += f"\nSensitivity Analysis ({len(state.sensitivity_results)}):\n"
            for sens in state.sensitivity_results:
                output += f"  {sens.method}: {sens.interpretation[:80]}...\n"

        # Previous critiques
        if state.critique_history:
            output += f"\nPrevious Critiques ({len(state.critique_history)}):\n"
            for fb in state.critique_history:
                output += f"  Iteration {fb.iteration}: {fb.decision.value}\n"

        self._investigation_evidence.append("Reviewed analysis summary")
        return output

    def _tool_check_covariate_balance(self, covariates: list[str]) -> str:
        """Check actual covariate balance."""
        if self._df is None:
            return "No data available for balance check."

        state = self._state
        df = self._df
        treatment_col = state.treatment_variable

        if treatment_col not in df.columns:
            return f"Treatment variable '{treatment_col}' not found."

        # Get groups
        try:
            treated = df[df[treatment_col] == 1]
            control = df[df[treatment_col] == 0]
        except Exception:
            treated = df[df[treatment_col] == df[treatment_col].max()]
            control = df[df[treatment_col] == df[treatment_col].min()]

        # Select covariates
        if not covariates:
            if state.data_profile and state.data_profile.potential_confounders:
                covariates = state.data_profile.potential_confounders[:10]
            else:
                covariates = df.select_dtypes(include=[np.number]).columns.tolist()
                covariates = [c for c in covariates if c not in [treatment_col, state.outcome_variable]][:10]

        output = "Covariate Balance Check:\n"
        output += "=" * 50 + "\n"
        output += f"Treated: n={len(treated)}, Control: n={len(control)}\n\n"

        imbalanced = []
        for cov in covariates:
            if cov not in df.columns or not pd.api.types.is_numeric_dtype(df[cov]):
                continue

            t_vals = treated[cov].dropna()
            c_vals = control[cov].dropna()

            if len(t_vals) < 5 or len(c_vals) < 5:
                continue

            # SMD
            mean_diff = t_vals.mean() - c_vals.mean()
            pooled_std = np.sqrt((t_vals.std() ** 2 + c_vals.std() ** 2) / 2)
            smd = abs(mean_diff / pooled_std) if pooled_std > 0 else 0

            status = ""
            if smd > 0.25:
                status = " *** SEVERE ***"
                imbalanced.append((cov, smd))
            elif smd > 0.1:
                status = " ** IMBALANCED **"
                imbalanced.append((cov, smd))

            output += f"  {cov}: SMD={smd:.3f}{status}\n"

        output += f"\nSummary: {len(imbalanced)} of {len(covariates)} covariates imbalanced (SMD > 0.1)\n"

        if imbalanced:
            output += "\nIMBALANCED COVARIATES (potential confounding):\n"
            for cov, smd in sorted(imbalanced, key=lambda x: x[1], reverse=True)[:5]:
                output += f"  - {cov}: SMD={smd:.3f}\n"
            self._investigation_evidence.append(f"Found {len(imbalanced)} imbalanced covariates")
        else:
            self._investigation_evidence.append("Covariates are well-balanced")

        return output

    def _tool_check_estimate_consistency(self) -> str:
        """Check consistency of estimates across methods."""
        state = self._state

        if len(state.treatment_effects) < 2:
            return "Only 1 method used - cannot check consistency across methods."

        output = "Estimate Consistency Check:\n"
        output += "=" * 50 + "\n"

        estimates = []
        methods = []
        for effect in state.treatment_effects:
            estimates.append(effect.estimate)
            methods.append(effect.method)

        estimates = np.array(estimates)
        mean_est = np.mean(estimates)
        std_est = np.std(estimates)
        cv = std_est / (abs(mean_est) + 1e-10) * 100

        output += f"Mean estimate: {mean_est:.4f}\n"
        output += f"Std across methods: {std_est:.4f}\n"
        output += f"Coefficient of variation: {cv:.1f}%\n\n"

        output += "Individual estimates:\n"
        for i, (method, est) in enumerate(zip(methods, estimates)):
            deviation = abs(est - mean_est) / (std_est + 1e-10)
            flag = " (outlier)" if deviation > 2 else ""
            output += f"  {method}: {est:.4f}{flag}\n"

        # Assess consistency
        if cv < 10:
            output += "\nConsistency: EXCELLENT - estimates agree within 10%\n"
            self._investigation_evidence.append("Estimates highly consistent across methods")
        elif cv < 25:
            output += "\nConsistency: GOOD - reasonable agreement across methods\n"
            self._investigation_evidence.append("Estimates moderately consistent")
        elif cv < 50:
            output += "\nConsistency: MODERATE - some variation across methods\n"
            self._investigation_evidence.append("Estimates show notable variation")
        else:
            output += "\nConsistency: POOR - estimates vary significantly\n"
            self._investigation_evidence.append("CONCERN: Estimates inconsistent across methods")

        # Check sign consistency
        positive = sum(1 for e in estimates if e > 0)
        if positive == len(estimates) or positive == 0:
            output += "Sign: All estimates have same sign (good)\n"
        else:
            output += f"Sign: WARNING - {positive}/{len(estimates)} positive (inconsistent)\n"
            self._investigation_evidence.append("CONCERN: Estimates have inconsistent signs")

        return output

    def _tool_check_subgroup_effects(self, subgroup_variable: str | None) -> str:
        """Check treatment effect heterogeneity."""
        if self._df is None:
            return "No data available for subgroup analysis."

        state = self._state
        df = self._df
        treatment_col = state.treatment_variable
        outcome_col = state.outcome_variable

        if treatment_col not in df.columns or outcome_col not in df.columns:
            return "Treatment or outcome variable not found."

        # Select subgroup variable
        if not subgroup_variable:
            # Auto-select a binary or categorical variable
            for col in df.columns:
                if col not in [treatment_col, outcome_col]:
                    if df[col].nunique() <= 5 and df[col].nunique() >= 2:
                        subgroup_variable = col
                        break

        if not subgroup_variable:
            return "No suitable subgroup variable found."

        output = f"Subgroup Analysis ({subgroup_variable}):\n"
        output += "=" * 50 + "\n"

        subgroup_effects = []
        for val in df[subgroup_variable].dropna().unique():
            subgroup = df[df[subgroup_variable] == val]
            treated = subgroup[subgroup[treatment_col] == subgroup[treatment_col].max()]
            control = subgroup[subgroup[treatment_col] == subgroup[treatment_col].min()]

            if len(treated) >= 10 and len(control) >= 10:
                effect = treated[outcome_col].mean() - control[outcome_col].mean()
                subgroup_effects.append((val, effect, len(treated), len(control)))

        if not subgroup_effects:
            return "Not enough data in subgroups for analysis."

        output += f"Variable: {subgroup_variable}\n\n"
        effects = []
        for val, effect, n_t, n_c in subgroup_effects:
            output += f"  {val}: Effect={effect:.4f} (n_t={n_t}, n_c={n_c})\n"
            effects.append(effect)

        # Check heterogeneity
        if len(effects) > 1:
            effect_std = np.std(effects)
            effect_mean = np.mean(effects)
            heterogeneity = effect_std / (abs(effect_mean) + 1e-10)

            output += f"\nHeterogeneity: {heterogeneity:.2%}\n"

            if heterogeneity > 0.5:
                output += "WARNING: Substantial heterogeneity - effect varies across subgroups\n"
                self._investigation_evidence.append(f"Substantial heterogeneity in {subgroup_variable}")
            else:
                output += "Effect is relatively consistent across subgroups\n"
                self._investigation_evidence.append(f"Effects consistent across {subgroup_variable}")

        return output

    def _tool_check_influential_observations(self) -> str:
        """Check for influential observations."""
        if self._df is None:
            return "No data available for influence check."

        state = self._state
        df = self._df
        treatment_col = state.treatment_variable
        outcome_col = state.outcome_variable

        if treatment_col not in df.columns or outcome_col not in df.columns:
            return "Treatment or outcome variable not found."

        output = "Influential Observation Check:\n"
        output += "=" * 50 + "\n"

        # Simple leverage/influence check using outcome residuals
        try:
            from sklearn.linear_model import LinearRegression

            # Prepare data
            covariates = []
            if state.data_profile and state.data_profile.potential_confounders:
                covariates = [c for c in state.data_profile.potential_confounders[:5]
                              if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

            X = df[[treatment_col] + covariates].dropna()
            y = df.loc[X.index, outcome_col]

            if len(X) < 50:
                return "Not enough complete cases for influence analysis."

            # Fit model and get residuals
            model = LinearRegression()
            model.fit(X, y)
            y_pred = model.predict(X)
            residuals = y - y_pred

            # Standardized residuals
            std_residuals = (residuals - residuals.mean()) / residuals.std()

            # Count influential
            n_high_leverage = (np.abs(std_residuals) > 3).sum()
            pct_influential = n_high_leverage / len(residuals) * 100

            output += f"Total observations: {len(residuals)}\n"
            output += f"High-leverage (|resid| > 3): {n_high_leverage} ({pct_influential:.1f}%)\n"

            if pct_influential > 5:
                output += "\nWARNING: Many influential observations - results may be unstable\n"
                self._investigation_evidence.append(f"Found {n_high_leverage} influential observations")
            elif pct_influential > 1:
                output += "\nSome influential observations present - consider robust methods\n"
                self._investigation_evidence.append("Some influential observations present")
            else:
                output += "\nFew influential observations - results likely stable\n"
                self._investigation_evidence.append("Few influential observations")

            # Show treatment effect excluding outliers
            non_outlier_mask = np.abs(std_residuals) <= 3
            X_clean = X[non_outlier_mask]
            y_clean = y[non_outlier_mask]

            treated_clean = y_clean[X_clean[treatment_col] == X_clean[treatment_col].max()]
            control_clean = y_clean[X_clean[treatment_col] == X_clean[treatment_col].min()]

            effect_all = df.groupby(treatment_col)[outcome_col].mean().diff().iloc[-1]
            effect_clean = treated_clean.mean() - control_clean.mean()

            output += f"\nEffect (all data): {effect_all:.4f}\n"
            output += f"Effect (excl. outliers): {effect_clean:.4f}\n"
            output += f"Change: {((effect_clean - effect_all) / effect_all * 100):.1f}%\n"

            if abs((effect_clean - effect_all) / effect_all) > 0.2:
                output += "\nWARNING: Effect changes >20% when excluding outliers!\n"
                self._investigation_evidence.append("Effect sensitive to outliers")

        except Exception as e:
            output += f"\nCould not complete influence check: {str(e)}\n"

        return output

    def _tool_verify_propensity_scores(self) -> str:
        """Verify propensity score quality."""
        state = self._state

        output = "Propensity Score Verification:\n"
        output += "=" * 50 + "\n"

        # Check if PS diagnostics exist
        has_ps_results = False
        for effect in state.treatment_effects:
            if effect.details and "propensity_scores" in str(effect.details):
                has_ps_results = True
                break

        if not has_ps_results:
            output += "No propensity score results found in analysis.\n"
            output += "Consider: Were PS-based methods used (IPW, AIPW, Matching)?\n"
            self._investigation_evidence.append("PS methods may not have been used")
            return output

        # Check PS diagnostics from state
        if hasattr(state, 'ps_diagnostics') and state.ps_diagnostics:
            diag = state.ps_diagnostics
            output += f"PS Diagnostics Available:\n"

            if hasattr(diag, 'overlap'):
                output += f"  Overlap: {diag.overlap}\n"
            if hasattr(diag, 'balance_achieved'):
                output += f"  Balance achieved: {diag.balance_achieved}\n"

            self._investigation_evidence.append("Verified PS diagnostics")
        else:
            output += "PS diagnostics not stored separately.\n"
            output += "Check treatment effect details for PS quality metrics.\n"

        # Look for PS-related info in treatment effects
        for effect in state.treatment_effects:
            if effect.method in ["IPW", "AIPW", "Matching"]:
                output += f"\n{effect.method} details:\n"
                if effect.assumptions_tested:
                    output += f"  Assumptions: {effect.assumptions_tested}\n"
                if effect.details:
                    for key, val in effect.details.items():
                        if any(x in key.lower() for x in ["overlap", "balance", "ps", "propensity"]):
                            output += f"  {key}: {val}\n"

        return output

    def _tool_check_sensitivity_robustness(self) -> str:
        """Check sensitivity analysis results."""
        state = self._state

        output = "Sensitivity Analysis Review:\n"
        output += "=" * 50 + "\n"

        if not state.sensitivity_results:
            output += "No sensitivity analysis performed.\n"
            output += "RECOMMENDATION: Add sensitivity analysis to assess robustness.\n"
            self._investigation_evidence.append("No sensitivity analysis - robustness unknown")
            return output

        output += f"Analyses performed: {len(state.sensitivity_results)}\n\n"

        robust = True
        for sens in state.sensitivity_results:
            output += f"{sens.method}:\n"
            output += f"  Result: {sens.robustness_value}\n"
            output += f"  Interpretation: {sens.interpretation}\n\n"

            # Check for concerning keywords
            interp_lower = sens.interpretation.lower()
            if any(x in interp_lower for x in ["not robust", "concern", "sensitive", "weak"]):
                robust = False

        if robust:
            output += "Overall: Results appear robust to unmeasured confounding.\n"
            self._investigation_evidence.append("Sensitivity analysis supports robustness")
        else:
            output += "WARNING: Sensitivity analysis raises concerns about robustness.\n"
            self._investigation_evidence.append("CONCERN: Sensitivity analysis raises robustness concerns")

        return output

    def _auto_finalize(self) -> dict[str, Any]:
        """Auto-generate final critique."""
        self.logger.info("critique_auto_finalize")

        state = self._state

        # Start with base scores
        scores = {
            "statistical_validity": 3,
            "assumption_checking": 3,
            "method_selection": 3,
            "completeness": 3,
            "reproducibility": 3,
            "interpretation": 3,
        }

        issues = []
        improvements = []

        # Adjust based on evidence
        n_methods = len(state.treatment_effects)
        if n_methods >= 3:
            scores["method_selection"] = 4
        elif n_methods == 1:
            scores["method_selection"] = 2
            improvements.append("Use multiple estimation methods")

        if state.sensitivity_results:
            scores["completeness"] = 4
        else:
            improvements.append("Add sensitivity analysis")

        # Check investigation evidence for concerns
        for evidence in self._investigation_evidence:
            if "CONCERN" in evidence or "WARNING" in evidence:
                issues.append(evidence)

        # Determine decision
        avg_score = sum(scores.values()) / len(scores)
        if avg_score >= 4 and len(issues) == 0:
            decision = "APPROVE"
        elif avg_score >= 3 or len(issues) <= 2:
            decision = "ITERATE"
        else:
            decision = "REJECT"

        return {
            "scores": scores,
            "decision": decision,
            "issues": issues,
            "improvements": improvements,
            "evidence_summary": " | ".join(self._investigation_evidence[:5]),
            "reasoning": f"Auto-critique based on investigation. Methods: {n_methods}, Issues: {len(issues)}",
        }

    def _create_feedback(self, result: dict[str, Any], iteration: int) -> CritiqueFeedback:
        """Create CritiqueFeedback from finalize result."""
        scores = result.get("scores", {
            "statistical_validity": 3,
            "assumption_checking": 3,
            "method_selection": 3,
            "completeness": 3,
            "reproducibility": 3,
            "interpretation": 3,
        })

        decision_str = result.get("decision", "ITERATE")
        try:
            decision = CritiqueDecision(decision_str)
        except ValueError:
            decision = CritiqueDecision.ITERATE

        reasoning = result.get("reasoning", "")
        evidence = result.get("evidence_summary", "")
        if evidence:
            reasoning = f"{reasoning}\n\nEvidence: {evidence}"

        return CritiqueFeedback(
            decision=decision,
            iteration=iteration,
            scores=scores,
            issues=result.get("issues", []),
            improvements=result.get("improvements", []),
            reasoning=reasoning,
        )

    def _heuristic_critique(self, state: AnalysisState, error: str) -> CritiqueFeedback:
        """Generate heuristic critique when LLM unavailable."""
        issues = []
        improvements = []
        scores = {
            "statistical_validity": 3,
            "assumption_checking": 3,
            "method_selection": 3,
            "completeness": 3,
            "reproducibility": 3,
            "interpretation": 3,
        }

        # Evaluate based on available results
        if state.treatment_effects:
            n_methods = len(state.treatment_effects)
            if n_methods >= 3:
                scores["method_selection"] = 4
                scores["statistical_validity"] = 4
            elif n_methods == 1:
                scores["method_selection"] = 2
                improvements.append("Use multiple estimation methods for robustness")

            # Check consistency
            estimates = [e.estimate for e in state.treatment_effects]
            if estimates:
                cv = np.std(estimates) / (np.mean(estimates) + 1e-10)
                if cv < 0.2:
                    scores["statistical_validity"] += 1
                elif cv > 0.5:
                    issues.append("Estimates vary significantly across methods")
                    scores["statistical_validity"] -= 1

        if state.sensitivity_results:
            scores["completeness"] = 4
        else:
            improvements.append("Add sensitivity analysis to test robustness")

        # Determine decision
        avg_score = sum(scores.values()) / len(scores)
        if avg_score >= 4:
            decision = CritiqueDecision.APPROVE
        elif avg_score >= 3:
            decision = CritiqueDecision.ITERATE
        else:
            decision = CritiqueDecision.REJECT

        return CritiqueFeedback(
            decision=decision,
            iteration=state.iteration_count + 1,
            scores=scores,
            issues=issues,
            improvements=improvements,
            reasoning=f"Heuristic evaluation (LLM error: {error[:100]}). Avg score: {avg_score:.1f}/5.",
        )
