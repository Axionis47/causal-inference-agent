"""Confounder Discovery Agent - Autonomously identifies confounders using causal reasoning.

This agent uses TRUE agentic tool-use: the LLM decides what evidence to compute,
calls tools to gather that evidence, and iteratively reasons about confounders.
"""

import pickle
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from src.agents.base import AnalysisState, BaseAgent
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


class ConfounderDiscoveryAgent(BaseAgent):
    """Agent that discovers confounders through iterative tool-based reasoning.

    This agent is TRULY AGENTIC - the LLM:
    1. Decides what statistical tests to run
    2. Calls tools to compute evidence
    3. Iterates based on results
    4. Makes final decisions based on gathered evidence

    NO pre-computation of evidence - the LLM drives the investigation.
    """

    AGENT_NAME = "confounder_discovery"

    SYSTEM_PROMPT = """You are an expert causal inference researcher investigating confounders.

Your task is to identify variables that confound the treatment-outcome relationship.
A CONFOUNDER is a variable that:
1. Affects the TREATMENT (causes or is associated with treatment assignment)
2. Affects the OUTCOME (causes or is associated with the outcome)
3. Is NOT caused by the treatment (not a mediator or collider)

You have tools to gather statistical evidence. USE THEM to investigate each variable:
- compute_correlation: Check association between any two variables
- compute_partial_correlation: Check if association remains after controlling for another variable
- test_confounder_criteria: Test if a variable meets confounder criteria
- finalize_confounders: Submit your final list of confounders

INVESTIGATION STRATEGY:
1. First, get the list of candidate variables
2. For each candidate, compute its correlation with BOTH treatment AND outcome
3. If a variable correlates with both, it's a potential confounder - investigate further
4. Use partial correlations to distinguish confounders from mediators
5. Rank confounders by the strength of their confounding effect

Be THOROUGH - missing a true confounder leads to biased causal estimates.
Call tools to gather evidence, don't just guess."""

    TOOLS = [
        {
            "name": "get_candidate_variables",
            "description": "Get the list of candidate variables to investigate as potential confounders",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "compute_correlation",
            "description": "Compute Pearson correlation between two variables. Use this to check if a variable is associated with treatment or outcome.",
            "parameters": {
                "type": "object",
                "properties": {
                    "var1": {"type": "string", "description": "First variable name"},
                    "var2": {"type": "string", "description": "Second variable name"},
                },
                "required": ["var1", "var2"],
            },
        },
        {
            "name": "compute_partial_correlation",
            "description": "Compute partial correlation between var1 and var2, controlling for control_var. Helps distinguish confounders from mediators.",
            "parameters": {
                "type": "object",
                "properties": {
                    "var1": {"type": "string", "description": "First variable"},
                    "var2": {"type": "string", "description": "Second variable"},
                    "control_var": {"type": "string", "description": "Variable to control for"},
                },
                "required": ["var1", "var2", "control_var"],
            },
        },
        {
            "name": "test_confounder_criteria",
            "description": "Test if a variable meets the statistical criteria for being a confounder (associated with both T and Y)",
            "parameters": {
                "type": "object",
                "properties": {
                    "variable": {"type": "string", "description": "Variable to test"},
                },
                "required": ["variable"],
            },
        },
        {
            "name": "finalize_confounders",
            "description": "Submit your final list of identified confounders after investigation",
            "parameters": {
                "type": "object",
                "properties": {
                    "confounders": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of confirmed confounders, ranked by importance",
                    },
                    "excluded": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Variables investigated but excluded (mediators, colliders, noise)",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Explanation of your confounder identification process",
                    },
                },
                "required": ["confounders", "reasoning"],
            },
        },
    ]

    def __init__(self):
        super().__init__()
        self._df: pd.DataFrame | None = None
        self._treatment_var: str | None = None
        self._outcome_var: str | None = None
        self._investigation_log: list[dict] = []

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute confounder discovery through iterative tool-based reasoning."""
        self.logger.info("confounder_discovery_start", job_id=state.job_id)
        start_time = time.time()

        try:
            # Load data
            self._df = self._load_dataframe(state)
            if self._df is None:
                self.logger.warning("no_dataframe_for_confounder_discovery")
                return state

            self._treatment_var = state.treatment_variable
            self._outcome_var = state.outcome_variable

            if not self._treatment_var or not self._outcome_var:
                self.logger.warning("no_treatment_or_outcome_specified")
                return state

            # Get candidate variables
            candidates = self._get_candidates()

            # Start agentic investigation
            prompt = f"""Investigate potential confounders for estimating the causal effect of '{self._treatment_var}' on '{self._outcome_var}'.

Available candidate variables: {candidates}

Start by calling get_candidate_variables, then systematically investigate each candidate:
1. For each candidate, compute its correlation with the treatment variable
2. Compute its correlation with the outcome variable
3. If it correlates with BOTH (|r| > 0.05), it's a potential confounder
4. Use partial correlations to verify it's not a mediator

After investigating, call finalize_confounders with your ranked list.

BEGIN YOUR INVESTIGATION NOW by calling the appropriate tools."""

            # Run agentic loop - LLM calls tools iteratively
            result = await self._run_agentic_loop(prompt, max_iterations=15)

            # Extract final confounders from result
            if result:
                confounders = result.get("confounders", [])
                excluded = result.get("excluded", [])
                reasoning = result.get("reasoning", "")

                # Update state
                if state.data_profile:
                    state.data_profile.potential_confounders = confounders

                state.confounder_discovery = {
                    "ranked_confounders": confounders,
                    "excluded_variables": excluded,
                    "adjustment_strategy": reasoning,
                    "investigation_log": self._investigation_log,
                }

                self.logger.info(
                    "confounder_discovery_complete",
                    n_confounders=len(confounders),
                    n_excluded=len(excluded),
                )
            else:
                # Fallback if agentic loop didn't complete
                self.logger.info("using_fallback_confounder_identification")
                state = self._fallback_confounder_identification(state)

            # Create trace
            duration_ms = int((time.time() - start_time) * 1000)
            trace = self.create_trace(
                action="confounder_discovery_complete",
                reasoning="Identified confounders through agentic tool-based investigation",
                outputs=state.confounder_discovery if state.confounder_discovery else {},
                duration_ms=duration_ms,
            )
            state.add_trace(trace)

        except Exception as e:
            self.logger.error("confounder_discovery_failed", error=str(e))
            state = self._fallback_confounder_identification(state)

        return state

    async def _run_agentic_loop(
        self,
        initial_prompt: str,
        max_iterations: int = 15,
    ) -> dict[str, Any] | None:
        """Run the agentic tool-calling loop.

        The LLM calls tools, we execute them and return results,
        LLM continues until it calls finalize_confounders.
        """
        messages = [{"role": "user", "content": initial_prompt}]
        final_result = None

        for iteration in range(max_iterations):
            self.logger.info(
                "confounder_iteration",
                iteration=iteration,
                max_iterations=max_iterations,
                investigations=len(self._investigation_log),
            )

            try:
                # Get LLM response with tool calls
                result = await self.reason(messages[-1]["content"] if iteration == 0 else "Continue your investigation based on the tool results.")

                # Log LLM reasoning if provided
                response_text = result.get("response", "")
                if response_text:
                    self.logger.info(
                        "confounder_llm_reasoning",
                        reasoning=response_text[:500] + ("..." if len(response_text) > 500 else ""),
                    )

                pending_calls = result.get("pending_calls", [])

                if not pending_calls:
                    self.logger.info("confounder_no_tool_calls", response_preview=response_text[:200])
                    break

                # Process each tool call
                tool_results = []
                for call in pending_calls:
                    tool_name = call["name"]
                    tool_args = call.get("args", {})

                    # Check if this is the final tool
                    if tool_name == "finalize_confounders":
                        confounders = tool_args.get("confounders", [])
                        self.logger.info(
                            "confounder_finalizing",
                            n_confounders=len(confounders),
                            top_confounders=confounders[:5] if confounders else [],
                            strategy=tool_args.get("adjustment_strategy", "")[:100],
                        )
                        return tool_args

                    # Log the tool call decision
                    self.logger.info(
                        "confounder_tool_decision",
                        tool=tool_name,
                        args=tool_args,
                    )

                    # Execute the tool
                    tool_result = self._execute_tool(tool_name, tool_args)

                    # Log the investigation
                    self._investigation_log.append({
                        "tool": tool_name,
                        "args": tool_args,
                        "result": tool_result,
                    })

                    tool_results.append({
                        "tool": tool_name,
                        "result": tool_result,
                    })

                    # Log tool result summary
                    self.logger.info(
                        "confounder_tool_result",
                        tool=tool_name,
                        result_summary=tool_result[:200] + ("..." if len(tool_result) > 200 else ""),
                    )

                # Build context for next iteration
                results_str = "\n".join([
                    f"Tool: {tr['tool']}\nResult: {tr['result']}"
                    for tr in tool_results
                ])
                messages.append({
                    "role": "assistant",
                    "content": f"Tool results:\n{results_str}\n\nContinue investigating or call finalize_confounders if done.",
                })

            except Exception as e:
                self.logger.warning("confounder_iteration_failed", error=str(e))
                break

        return final_result

    def _execute_tool(self, tool_name: str, args: dict) -> str:
        """Execute a tool and return the result as a string."""
        if self._df is None:
            return "Error: No data loaded"

        try:
            if tool_name == "get_candidate_variables":
                candidates = self._get_candidates()
                return f"Candidate variables: {candidates}"

            elif tool_name == "compute_correlation":
                var1 = args.get("var1")
                var2 = args.get("var2")
                if var1 not in self._df.columns or var2 not in self._df.columns:
                    return f"Error: Variable not found. Available: {list(self._df.columns)}"

                corr, pval = stats.pearsonr(self._df[var1], self._df[var2])
                significance = "significant" if pval < 0.05 else "not significant"
                strength = "strong" if abs(corr) > 0.3 else ("moderate" if abs(corr) > 0.1 else "weak")
                return f"Correlation({var1}, {var2}): r={corr:.4f}, p={pval:.4f} ({strength}, {significance})"

            elif tool_name == "compute_partial_correlation":
                var1 = args.get("var1")
                var2 = args.get("var2")
                control_var = args.get("control_var")

                for v in [var1, var2, control_var]:
                    if v not in self._df.columns:
                        return f"Error: Variable '{v}' not found"

                # Compute partial correlation
                from sklearn.linear_model import LinearRegression

                X = self._df[[var1, var2, control_var]].dropna()
                v1 = X[var1].values
                v2 = X[var2].values
                ctrl = X[control_var].values.reshape(-1, 1)

                # Residualize
                v1_resid = v1 - LinearRegression().fit(ctrl, v1).predict(ctrl)
                v2_resid = v2 - LinearRegression().fit(ctrl, v2).predict(ctrl)

                partial_corr, pval = stats.pearsonr(v1_resid, v2_resid)
                return f"Partial correlation({var1}, {var2} | {control_var}): r={partial_corr:.4f}, p={pval:.4f}"

            elif tool_name == "test_confounder_criteria":
                variable = args.get("variable")
                if variable not in self._df.columns:
                    return f"Error: Variable '{variable}' not found"

                # Test association with treatment
                corr_t, pval_t = stats.pearsonr(self._df[variable], self._df[self._treatment_var])
                # Test association with outcome
                corr_y, pval_y = stats.pearsonr(self._df[variable], self._df[self._outcome_var])

                affects_treatment = abs(corr_t) > 0.05 and pval_t < 0.1
                affects_outcome = abs(corr_y) > 0.05 and pval_y < 0.1
                is_confounder = affects_treatment and affects_outcome

                confounder_strength = abs(corr_t) * abs(corr_y)

                return f"""Confounder test for '{variable}':
- Correlation with treatment ({self._treatment_var}): r={corr_t:.4f}, p={pval_t:.4f} → {'YES' if affects_treatment else 'NO'}
- Correlation with outcome ({self._outcome_var}): r={corr_y:.4f}, p={pval_y:.4f} → {'YES' if affects_outcome else 'NO'}
- IS CONFOUNDER: {'YES (strength={:.4f})'.format(confounder_strength) if is_confounder else 'NO'}"""

            elif tool_name == "finalize_confounders":
                # This is handled in the caller
                return "Confounders finalized"

            else:
                return f"Error: Unknown tool '{tool_name}'"

        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def _get_candidates(self) -> list[str]:
        """Get candidate variables for confounder investigation."""
        if self._df is None:
            return []

        # All numeric columns except treatment and outcome
        numeric_cols = self._df.select_dtypes(include=[np.number]).columns.tolist()
        candidates = [
            c for c in numeric_cols
            if c != self._treatment_var and c != self._outcome_var
        ]
        return candidates

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

    def _fallback_confounder_identification(self, state: AnalysisState) -> AnalysisState:
        """Fallback when agentic loop fails - use statistical heuristics."""
        self.logger.info("using_statistical_fallback")

        if self._df is None:
            return state

        candidates = self._get_candidates()
        confounders = []

        for col in candidates:
            try:
                corr_t = abs(stats.pearsonr(self._df[col], self._df[self._treatment_var])[0])
                corr_y = abs(stats.pearsonr(self._df[col], self._df[self._outcome_var])[0])

                # Include if associated with both
                if corr_t > 0.03 and corr_y > 0.03:
                    confounders.append((col, corr_t * corr_y))
            except Exception:
                pass

        # Sort by confounder strength
        confounders.sort(key=lambda x: x[1], reverse=True)
        ranked = [c for c, _ in confounders]

        if state.data_profile:
            state.data_profile.potential_confounders = ranked

        state.confounder_discovery = {
            "ranked_confounders": ranked,
            "excluded_variables": [],
            "adjustment_strategy": "Statistical fallback - variables correlated with both T and Y",
            "investigation_log": [],
        }

        return state
