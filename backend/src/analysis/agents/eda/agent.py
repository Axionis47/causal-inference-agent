"""EDA Agent - ReAct-based exploratory data analysis for causal inference."""

import time
from typing import Any

import pandas as pd

from src.analysis.agents.base import (
    AnalysisState,
    JobStatus,
    ToolResult,
)
from src.analysis.agents.base.context_tools import ContextTools
from src.analysis.agents.base.react_agent import ReActAgent
from src.analysis.agents.registry import register_agent
from src.logging_config.structured import get_logger

from . import tools
from .brief import CAPABILITY as EDA_CAPABILITY, build_brief, preflight
from .helpers import (
    auto_finalize,
    initial_observation_text,
    load_dataframe,
    populate_eda_result,
)
from .output import EDAResult
from .prompt import SYSTEM_PROMPT

logger = get_logger(__name__)


@register_agent("eda_agent")
class EDAAgent(ReActAgent, ContextTools):
    """ReAct-based EDA agent that investigates data quality for causal inference."""

    AGENT_NAME = "eda_agent"
    MAX_STEPS = 15

    WRITES_STATE_FIELDS = ["eda_result"]
    REQUIRED_STATE_FIELDS = ["data_profile", "dataframe_path"]
    JOB_STATUS = JobStatus.EXPLORATORY_ANALYSIS
    PROGRESS_WEIGHT = 0.08
    CAPABILITY = EDA_CAPABILITY

    SYSTEM_PROMPT = SYSTEM_PROMPT

    def __init__(self) -> None:
        super().__init__()
        self.register_context_tools()

        self._df: pd.DataFrame | None = None
        self._eda_result: EDAResult | None = None
        self._treatment_var: str | None = None
        self._outcome_var: str | None = None
        self._finalized: bool = False
        self._final_result: dict[str, Any] = {}

        self._analyzed_distributions: dict[str, dict] = {}
        self._outlier_results: dict[str, dict] = {}
        self._correlation_results: dict | None = None
        self._vif_results: dict[str, float] = {}
        self._balance_results: dict[str, dict] = {}
        self._missing_analysis: dict | None = None

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
        return initial_observation_text(state)

    async def is_task_complete(self, state: AnalysisState) -> bool:
        return self._finalized and state.eda_result is not None

    async def execute(self, state: AnalysisState) -> AnalysisState:
        refusal = preflight(state)
        if refusal is not None:
            state.agent_briefs[refusal.agent] = refusal
            self.logger.info(
                "eda_refused",
                flag=refusal.flags[0].value,
                headline=refusal.headline,
            )
            return state

        self.logger.info(
            "eda_agent_start",
            job_id=state.job_id,
            dataset=state.dataset_info.name or state.dataset_info.url,
        )
        start_time = time.time()

        try:
            self._df = load_dataframe(state.dataframe_path)
            if self._df is None:
                state.mark_failed("Failed to load dataset for EDA", self.AGENT_NAME)
                return state

            self._eda_result = EDAResult()
            self._treatment_var, self._outcome_var = state.get_primary_pair()

            self._analyzed_distributions = {}
            self._outlier_results = {}
            self._correlation_results = None
            self._vif_results = {}
            self._balance_results = {}
            self._missing_analysis = None
            self._finalized = False

            self._current_state = state

            state = await super().execute(state)

            if not self._finalized:
                self.logger.warning("eda_auto_finalize")
                self._final_result = self._auto_finalize()
                self._finalized = True

            self._populate_eda_result(self._final_result)
            state.eda_result = self._eda_result

            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.info(
                "eda_complete",
                quality_score=self._eda_result.data_quality_score,
                issues_found=len(self._eda_result.data_quality_issues),
                duration_ms=duration_ms,
            )
        except Exception as e:
            self.logger.exception("eda_failed", error=str(e))
            state.mark_failed(f"EDA failed: {str(e)}", self.AGENT_NAME)
        finally:
            if "eda_agent" not in state.agent_briefs:
                brief = build_brief(state)
                state.agent_briefs[brief.agent] = brief
                self.logger.info(
                    "eda_brief",
                    status=brief.status,
                    flags=[f.value for f in brief.flags],
                )

        return state

    def _load_dataframe(self, state: AnalysisState) -> pd.DataFrame | None:
        return load_dataframe(state.dataframe_path)

    def _auto_finalize(self) -> dict[str, Any]:
        return auto_finalize(
            self._missing_analysis,
            self._outlier_results,
            self._vif_results,
            self._balance_results,
        )

    def _populate_eda_result(self, final_result: dict[str, Any]) -> None:
        populate_eda_result(
            self._eda_result,
            final_result,
            self._analyzed_distributions,
            self._outlier_results,
            self._vif_results,
            self._balance_results,
        )
