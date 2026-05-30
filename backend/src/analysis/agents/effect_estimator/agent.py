"""Effect Estimator Agent - ReAct-based treatment effect estimation.

This agent uses the ReAct pattern to iteratively:
1. Analyze data characteristics
2. Check covariate balance and propensity score overlap
3. Select and run appropriate estimation methods
4. Evaluate diagnostics and compare results
5. Iterate until confident in the estimate

Uses pull-based context - queries for information on demand.
"""

import time
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.agents.base import (
    AnalysisState,
    CausalPair,
    ToolResult,
    TreatmentEffectResult,
)
from src.analysis.agents.base.context_tools import ContextTools
from src.analysis.agents.base.react_agent import ReActAgent
from src.analysis.agents.registry import register_agent
from src.logging_config.structured import get_logger

from . import tools
from .covariates import get_covariates_for_pair
from .helpers import (
    auto_finalize,
    build_initial_observation,
    build_react_prompt,
    load_dataframe,
)
from .method_selector import find_closest_column
from .pair_selection import identify_valid_causal_pairs
from .prompt import SYSTEM_PROMPT

logger = get_logger(__name__)


@register_agent("effect_estimator")
class EffectEstimatorAgent(ReActAgent, ContextTools):
    """ReAct-based treatment effect estimator.

    Iteratively queries upstream findings, investigates the data,
    checks balance and overlap, runs estimation methods, evaluates
    diagnostics, compares results, and finalizes a preferred estimate.
    """

    AGENT_NAME = "effect_estimator"
    MAX_STEPS = 20

    SYSTEM_PROMPT = SYSTEM_PROMPT

    REQUIRED_STATE_FIELDS = ["data_profile", "dataframe_path"]
    WRITES_STATE_FIELDS = ["treatment_effects", "analyzed_pairs", "treatment_binarization_threshold"]
    JOB_STATUS = "estimating_effects"
    PROGRESS_WEIGHT = 60

    def __init__(self) -> None:
        super().__init__()
        self.register_context_tools()

        self._df: pd.DataFrame | None = None
        self._treatment_var: str | None = None
        self._outcome_var: str | None = None
        self._covariates: list[str] = []
        self._results: list[TreatmentEffectResult] = []
        self._propensity_scores: np.ndarray | None = None
        self._last_method_result: TreatmentEffectResult | None = None
        self._overlap_quality: str | None = None  # L6: PS overlap for method gating
        self._current_state: AnalysisState | None = None
        self._finalized: bool = False
        self._final_result: dict[str, Any] = {}

        for tool_module in tools.MODULES:
            self._register(tool_module)

    def _register(self, tool_module) -> None:
        async def _handler(state: AnalysisState, *args, **kwargs) -> ToolResult:
            return await tool_module.handle(self, state, *args, **kwargs)
        self.register_tool(
            name=tool_module.SCHEMA["name"],
            description=tool_module.SCHEMA["description"],
            parameters=tool_module.SCHEMA["parameters"],
            handler=_handler,
        )
        short_name = tool_module.__name__.rsplit(".", 1)[-1]
        setattr(self, f"_tool_{short_name}", _handler)

    def _get_initial_observation(self, state: AnalysisState) -> str:
        return build_initial_observation(
            self._treatment_var,
            self._outcome_var,
            self._covariates,
            len(self._df) if self._df is not None else "unknown",
        )

    def _build_react_prompt(
        self,
        state: AnalysisState,
        observation: str,
        step_num: int,
    ) -> str:
        return build_react_prompt(
            steps=self._steps,
            step_num=step_num,
            max_steps=self.MAX_STEPS,
            observation=observation,
            n_results=len(self._results),
        )

    async def is_task_complete(self, state: AnalysisState) -> bool:
        return self._finalized

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Iterate over identified causal pairs, run the ReAct loop per pair."""
        self.logger.info(
            "estimation_start",
            job_id=state.job_id,
            has_profile=state.data_profile is not None,
        )
        start_time = time.time()

        try:
            if state.dataframe_path is None:
                state.mark_failed("No data available for estimation", self.AGENT_NAME)
                return state

            self._df = load_dataframe(state.dataframe_path)
            self._current_state = state
            all_results: list[TreatmentEffectResult] = []

            pairs = await identify_valid_causal_pairs(self, state)

            if not pairs:
                state.mark_failed("No valid treatment-outcome pairs identified", self.AGENT_NAME)
                return state

            self.logger.info("causal_pairs_identified", count=len(pairs))

            state.analyzed_pairs = [
                CausalPair(treatment=t, outcome=o, rationale=r, priority=i + 1)
                for i, (t, o, r) in enumerate(pairs)
            ]

            for treatment, outcome, rationale in pairs:
                actual_treatment = find_closest_column(treatment, list(self._df.columns))
                actual_outcome = find_closest_column(outcome, list(self._df.columns))

                if actual_treatment is None or actual_outcome is None:
                    self.logger.warning(
                        "skipping_invalid_pair",
                        treatment=treatment,
                        outcome=outcome,
                        reason="variable not in data",
                    )
                    continue

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

                # Clear prior results for this pair (in case of re-iteration after critique).
                state.treatment_effects = [
                    e for e in state.treatment_effects
                    if not (
                        e.treatment_variable == treatment
                        and e.outcome_variable == outcome
                    )
                ]

                self._treatment_var = treatment
                self._outcome_var = outcome
                self._results = []
                self._finalized = False

                self._covariates = get_covariates_for_pair(self, state, treatment, outcome)

                await super().execute(state)

                if not self._finalized:
                    self.logger.warning("estimation_auto_finalize")
                    self._auto_finalize()
                    self._finalized = True

                for result in self._results:
                    result.treatment_variable = treatment
                    result.outcome_variable = outcome

                all_results.extend(self._results)

            state.treatment_effects = all_results

            # CR4: All-methods-failed across all pairs is a hard failure.
            if not all_results:
                state.mark_failed(
                    "All estimation methods failed across all causal pairs. Cannot produce results.",
                    self.AGENT_NAME,
                )
                return state

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
        return auto_finalize(
            self._df,
            self._current_state,
            self._treatment_var,
            self._outcome_var,
            self._covariates,
            self._results,
            self.logger,
        )
