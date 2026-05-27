"""Data Repair Agent - iterative LLM-driven data quality repair."""

import time

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
from .helpers import (
    auto_finalize,
    count_outliers,
    initial_observation_text,
    load_dataframe,
    save_dataframe,
)
from .prompt import SYSTEM_PROMPT

logger = get_logger(__name__)


@register_agent("data_repair")
class DataRepairAgent(ReActAgent, ContextTools):
    """Truly agentic data repair agent that iteratively fixes data quality issues."""

    AGENT_NAME = "data_repair"
    MAX_STEPS = 20
    WRITES_STATE_FIELDS = ["data_repairs"]
    REQUIRED_STATE_FIELDS = ["data_profile", "dataframe_path"]
    JOB_STATUS = JobStatus.PROFILING
    PROGRESS_WEIGHT = 0.06

    SYSTEM_PROMPT = SYSTEM_PROMPT

    def __init__(self) -> None:
        super().__init__()
        self._df: pd.DataFrame | None = None
        self._df_original: pd.DataFrame | None = None
        self._current_state: AnalysisState | None = None
        self._repairs_applied: list[dict] = []
        self._finalized: bool = False

        self.register_context_tools()

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
        shape = self._df.shape if self._df is not None else (0, 0)
        return initial_observation_text(state, shape)

    async def is_task_complete(self, state: AnalysisState) -> bool:
        return self._finalized

    async def execute(self, state: AnalysisState) -> AnalysisState:
        self.logger.info("data_repair_start", job_id=state.job_id)
        start_time = time.time()

        try:
            self._df = self._load_dataframe(state)
            if self._df is None:
                state.mark_failed("No dataframe available for repair", self.AGENT_NAME)
                return state

            self._df_original = self._df.copy()
            self._current_state = state
            self._repairs_applied = []
            self._finalized = False

            self.logger.info("data_loaded", shape=self._df.shape)

            await super().execute(state)

            if not self._finalized:
                self._auto_finalize()

            self._save_dataframe(self._df, state)
            state.data_repairs = self._repairs_applied

            duration_ms = int((time.time() - start_time) * 1000)
            trace = self.create_trace(
                action="data_repair_complete",
                reasoning="Data repair completed",
                outputs={
                    "repairs_applied": len(self._repairs_applied),
                    "original_shape": list(self._df_original.shape),
                    "final_shape": list(self._df.shape),
                },
                duration_ms=duration_ms,
            )
            state.add_trace(trace)

            self.logger.info(
                "data_repair_complete",
                repairs_applied=len(self._repairs_applied),
                original_shape=self._df_original.shape,
                new_shape=self._df.shape,
            )

        except Exception as e:
            self.logger.exception("data_repair_failed", error=str(e))
            state.add_trace(self.create_trace(
                action="repair_skipped",
                reasoning=f"Repair failed but continuing: {str(e)}",
                outputs={"error": str(e)},
            ))

        return state

    def _load_dataframe(self, state: AnalysisState) -> pd.DataFrame | None:
        return load_dataframe(state.dataframe_path, self.logger)

    def _save_dataframe(self, df: pd.DataFrame, state: AnalysisState) -> None:
        save_dataframe(df, state.dataframe_path, self.logger)

    def _count_outliers(self, data) -> int:
        return count_outliers(data)

    def _auto_finalize(self) -> dict:
        return auto_finalize(self._df, self._df_original, self._repairs_applied)
