"""Sensitivity Analyst Agent - ReAct-based sensitivity analysis.

This agent uses the ReAct pattern to iteratively:
1. Review current treatment effect estimates
2. Run sensitivity analyses one at a time
3. Interpret results and decide if more investigation needed
4. Assess overall robustness of causal findings

Uses pull-based context - queries for information on demand.
"""

import time
from typing import Any

import pandas as pd

from src.analysis.agents.base import (
    AnalysisState,
    JobStatus,
    SensitivityResult,
    ToolResult,
)
from src.analysis.agents.base.context_tools import ContextTools
from src.analysis.agents.base.react_agent import ReActAgent
from src.analysis.agents.registry import register_agent
from src.logging_config.structured import get_logger

from . import tools
from .helpers import auto_finalize, initial_observation_text, load_dataframe
from .prompt import SYSTEM_PROMPT

logger = get_logger(__name__)


@register_agent("sensitivity_analyst")
class SensitivityAnalystAgent(ReActAgent, ContextTools):
    """ReAct-based sensitivity analyst.

    Iteratively queries upstream findings, picks one sensitivity test
    at a time, interprets, and finally writes a robustness assessment.
    """

    AGENT_NAME = "sensitivity_analyst"
    MAX_STEPS = 15

    WRITES_STATE_FIELDS = ["sensitivity_results"]
    REQUIRED_STATE_FIELDS = ["treatment_effects"]
    JOB_STATUS = JobStatus.SENSITIVITY_ANALYSIS
    PROGRESS_WEIGHT = 0.08

    SYSTEM_PROMPT = SYSTEM_PROMPT

    def __init__(self) -> None:
        super().__init__()
        self.register_context_tools()

        self._df: pd.DataFrame | None = None
        self._current_state: AnalysisState | None = None
        self._results: list[SensitivityResult] = []
        self._treatment_var: str | None = None
        self._outcome_var: str | None = None
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
        treatment_var, outcome_var = state.get_primary_pair()
        return initial_observation_text(treatment_var, outcome_var)

    async def is_task_complete(self, state: AnalysisState) -> bool:
        return self._finalized

    def _resolve_treatment_outcome(self) -> tuple[str | None, str | None]:
        """Cache primary pair on first call so tool handlers can read it cheaply."""
        if self._treatment_var and self._outcome_var:
            return self._treatment_var, self._outcome_var

        if self._current_state:
            t, o = self._current_state.get_primary_pair()
            self._treatment_var = t
            self._outcome_var = o
            return t, o

        return None, None

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute the ReAct sensitivity loop."""
        self.logger.info(
            "sensitivity_start",
            job_id=state.job_id,
            n_effects=len(state.treatment_effects),
        )
        start_time = time.time()

        try:
            if not state.treatment_effects:
                self.logger.warning("no_effects_for_sensitivity")
                return state

            if state.dataframe_path is None:
                return state

            self._df = load_dataframe(state.dataframe_path)
            if self._df is None:
                self.logger.warning("no_dataframe_for_sensitivity")
                return state

            self._current_state = state
            self._results = []
            self._finalized = False

            await super().execute(state)

            if not self._finalized:
                self.logger.warning("sensitivity_auto_finalize")
                self._final_result = self._auto_finalize()
                self._finalized = True

            state.sensitivity_results = self._results

            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.info(
                "sensitivity_complete",
                n_results=len(self._results),
                robustness=self._final_result.get("overall_robustness"),
                duration_ms=duration_ms,
            )

        except Exception as e:
            self.logger.exception("sensitivity_failed", error=str(e))

        return state

    def _auto_finalize(self) -> dict[str, Any]:
        return auto_finalize(self._results)
