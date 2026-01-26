"""Confounder Discovery Agent - Autonomously identifies confounders using causal reasoning.

This agent uses TRUE agentic tool-use: the LLM decides what evidence to compute,
calls tools to gather that evidence, and iteratively reasons about confounders.
"""

import pickle
import time

import numpy as np
import pandas as pd
from scipy import stats

from src.agents.base import AnalysisState, ToolResult, ToolResultStatus
from src.agents.base.context_tools import ContextTools
from src.agents.base.react_agent import ReActAgent
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


class ConfounderDiscoveryAgent(ReActAgent, ContextTools):
    """Agent that discovers confounders through iterative tool-based reasoning.

    This agent is TRULY AGENTIC - the LLM:
    1. Decides what statistical tests to run
    2. Calls tools to compute evidence
    3. Iterates based on results
    4. Makes final decisions based on gathered evidence

    Uses ReAct pattern with pull-based context tools.
    """

    AGENT_NAME = "confounder_discovery"
    MAX_STEPS = 15

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

    def __init__(self) -> None:
        super().__init__()
        self._df: pd.DataFrame | None = None
        self._current_state: AnalysisState | None = None
        self._treatment_var: str | None = None
        self._outcome_var: str | None = None
        self._investigation_log: list[dict] = []
        self._finalized: bool = False

        # Register context tools from mixin
        self.register_context_tools()
        # Register discovery-specific tools
        self._register_discovery_tools()

    def _register_discovery_tools(self) -> None:
        """Register confounder discovery tools."""
        self.register_tool(
            name="get_candidate_variables",
            description="Get the list of candidate variables to investigate as potential confounders",
            handler=self._tool_get_candidates,
            parameters={},
        )

        self.register_tool(
            name="compute_correlation",
            description="Compute Pearson correlation between two variables. Use this to check if a variable is associated with treatment or outcome.",
            handler=self._tool_compute_correlation,
            parameters={
                "var1": {"type": "string", "description": "First variable name"},
                "var2": {"type": "string", "description": "Second variable name"},
            },
        )

        self.register_tool(
            name="compute_partial_correlation",
            description="Compute partial correlation between var1 and var2, controlling for control_var. Helps distinguish confounders from mediators.",
            handler=self._tool_compute_partial_correlation,
            parameters={
                "var1": {"type": "string", "description": "First variable"},
                "var2": {"type": "string", "description": "Second variable"},
                "control_var": {"type": "string", "description": "Variable to control for"},
            },
        )

        self.register_tool(
            name="test_confounder_criteria",
            description="Test if a variable meets the statistical criteria for being a confounder (associated with both T and Y)",
            handler=self._tool_test_confounder_criteria,
            parameters={
                "variable": {"type": "string", "description": "Variable to test"},
            },
        )

        self.register_tool(
            name="finalize_confounders",
            description="Submit your final list of identified confounders after investigation",
            handler=self._tool_finalize_confounders,
            parameters={
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
        )

    def _resolve_treatment_outcome(self) -> tuple[str | None, str | None]:
        """Get treatment and outcome variables using state helper.

        Returns:
            Tuple of (treatment_var, outcome_var)
        """
        if self._treatment_var and self._outcome_var:
            return self._treatment_var, self._outcome_var

        if self._current_state:
            t, o = self._current_state.get_primary_pair()
            self._treatment_var = t
            self._outcome_var = o
            return t, o

        return None, None

    def _get_initial_observation(self, state: AnalysisState) -> str:
        """Build initial observation for confounder discovery."""
        candidates = self._get_candidates()
        treatment = self._treatment_var or "unknown"
        outcome = self._outcome_var or "unknown"

        return f"""CONFOUNDER DISCOVERY TASK
Treatment variable: {treatment}
Outcome variable: {outcome}
Candidate variables to investigate: {candidates}

Your goal is to identify which candidates are confounders (affect both treatment and outcome).
Use the tools to systematically investigate each candidate:
1. Start by getting the candidate list
2. For each candidate, test if it meets confounder criteria
3. Use correlations and partial correlations to verify
4. Call finalize_confounders when done with your ranked list."""

    async def is_task_complete(self, state: AnalysisState) -> bool:
        """Check if confounder discovery is complete."""
        return self._finalized

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

            # Store state reference for helper methods
            self._current_state = state
            self._investigation_log = []
            self._finalized = False

            # Use helper to resolve treatment/outcome (falls back to analyzed_pairs)
            self._treatment_var, self._outcome_var = self._resolve_treatment_outcome()

            if not self._treatment_var or not self._outcome_var:
                self.logger.warning("no_treatment_or_outcome_specified")
                return state

            # Run ReAct loop - LLM calls tools iteratively
            await super().execute(state)

            # Auto-finalize if LLM didn't call finalize_confounders
            if not self._finalized:
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

    # -------------------------------------------------------------------------
    # Tool Handlers (async, return ToolResult)
    # -------------------------------------------------------------------------

    async def _tool_get_candidates(self, state: AnalysisState) -> ToolResult:
        """Get candidate variables for confounder investigation."""
        candidates = self._get_candidates()
        self._investigation_log.append({
            "tool": "get_candidate_variables",
            "args": {},
            "result": candidates,
        })
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "candidates": candidates,
                "treatment": self._treatment_var,
                "outcome": self._outcome_var,
            },
        )

    async def _tool_compute_correlation(
        self, state: AnalysisState, var1: str, var2: str
    ) -> ToolResult:
        """Compute Pearson correlation between two variables."""
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error="No data loaded",
            )

        if var1 not in self._df.columns or var2 not in self._df.columns:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error=f"Variable not found. Available: {list(self._df.columns)}",
            )

        corr, pval = stats.pearsonr(self._df[var1], self._df[var2])
        significance = "significant" if pval < 0.05 else "not significant"
        strength = "strong" if abs(corr) > 0.3 else ("moderate" if abs(corr) > 0.1 else "weak")

        result = {
            "var1": var1,
            "var2": var2,
            "correlation": float(corr),
            "p_value": float(pval),
            "strength": strength,
            "significance": significance,
        }

        self._investigation_log.append({
            "tool": "compute_correlation",
            "args": {"var1": var1, "var2": var2},
            "result": result,
        })

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=result,
        )

    async def _tool_compute_partial_correlation(
        self, state: AnalysisState, var1: str, var2: str, control_var: str
    ) -> ToolResult:
        """Compute partial correlation controlling for a third variable."""
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error="No data loaded",
            )

        for v in [var1, var2, control_var]:
            if v not in self._df.columns:
                return ToolResult(
                    status=ToolResultStatus.ERROR,
                    error=f"Variable '{v}' not found",
                )

        from sklearn.linear_model import LinearRegression

        X = self._df[[var1, var2, control_var]].dropna()
        v1_vals = X[var1].values
        v2_vals = X[var2].values
        ctrl = X[control_var].values.reshape(-1, 1)

        # Residualize
        v1_resid = v1_vals - LinearRegression().fit(ctrl, v1_vals).predict(ctrl)
        v2_resid = v2_vals - LinearRegression().fit(ctrl, v2_vals).predict(ctrl)

        partial_corr, pval = stats.pearsonr(v1_resid, v2_resid)

        result = {
            "var1": var1,
            "var2": var2,
            "control_var": control_var,
            "partial_correlation": float(partial_corr),
            "p_value": float(pval),
        }

        self._investigation_log.append({
            "tool": "compute_partial_correlation",
            "args": {"var1": var1, "var2": var2, "control_var": control_var},
            "result": result,
        })

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=result,
        )

    async def _tool_test_confounder_criteria(
        self, state: AnalysisState, variable: str
    ) -> ToolResult:
        """Test if a variable meets confounder criteria."""
        if self._df is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error="No data loaded",
            )

        if variable not in self._df.columns:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                error=f"Variable '{variable}' not found",
            )

        # Test association with treatment
        corr_t, pval_t = stats.pearsonr(self._df[variable], self._df[self._treatment_var])
        # Test association with outcome
        corr_y, pval_y = stats.pearsonr(self._df[variable], self._df[self._outcome_var])

        affects_treatment = abs(corr_t) > 0.05 and pval_t < 0.1
        affects_outcome = abs(corr_y) > 0.05 and pval_y < 0.1
        is_confounder = affects_treatment and affects_outcome
        confounder_strength = abs(corr_t) * abs(corr_y)

        result = {
            "variable": variable,
            "treatment_var": self._treatment_var,
            "outcome_var": self._outcome_var,
            "corr_with_treatment": float(corr_t),
            "pval_treatment": float(pval_t),
            "affects_treatment": affects_treatment,
            "corr_with_outcome": float(corr_y),
            "pval_outcome": float(pval_y),
            "affects_outcome": affects_outcome,
            "is_confounder": is_confounder,
            "confounder_strength": float(confounder_strength) if is_confounder else 0.0,
        }

        self._investigation_log.append({
            "tool": "test_confounder_criteria",
            "args": {"variable": variable},
            "result": result,
        })

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output=result,
        )

    async def _tool_finalize_confounders(
        self,
        state: AnalysisState,
        confounders: list[str],
        reasoning: str,
        excluded: list[str] | None = None,
    ) -> ToolResult:
        """Finalize the list of identified confounders."""
        excluded = excluded or []

        self.logger.info(
            "confounder_finalizing",
            n_confounders=len(confounders),
            top_confounders=confounders[:5] if confounders else [],
        )

        # Update state
        if state.data_profile:
            state.data_profile.potential_confounders = confounders

        state.confounder_discovery = {
            "ranked_confounders": confounders,
            "excluded_variables": excluded,
            "adjustment_strategy": reasoning,
            "investigation_log": self._investigation_log,
        }

        self._finalized = True

        self.logger.info(
            "confounder_discovery_complete",
            n_confounders=len(confounders),
            n_excluded=len(excluded),
        )

        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "confounders": confounders,
                "excluded": excluded,
                "reasoning": reasoning,
            },
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

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
