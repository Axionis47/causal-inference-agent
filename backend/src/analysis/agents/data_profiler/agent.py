"""DataProfilerAgent: ReAct-driven dataset profiling for causal inference.

Loads the dataset, computes a basic statistical profile, and runs a ReAct loop
where the LLM iteratively investigates columns and finalizes the causal
structure (treatment / outcome / confounders / instruments / encoding).
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from src.analysis.agents.base import (
    AnalysisState,
    JobStatus,
    ReActAgent,
    ToolResult,
)
from src.analysis.agents.base.context_tools import ContextTools
from src.analysis.agents.registry import register_agent
from src.logging_config.structured import get_logger

from . import tools
from .encoding import auto_encode, build_encoding
from .helpers import (
    auto_finalize,
    compute_basic_profile,
    exclude_oracle_columns,
    initial_observation_text,
    populate_profile,
    save_dataframe,
)
from .loading import fetch_kaggle_metadata, load_dataset
from .output import DataProfile
from .prompt import SYSTEM_PROMPT

logger = get_logger(__name__)


@register_agent("data_profiler")
class DataProfilerAgent(ReActAgent, ContextTools):
    """ReAct-based data profiler that identifies causal structure through investigation."""

    AGENT_NAME = "data_profiler"
    MAX_STEPS = 15

    WRITES_STATE_FIELDS = ["data_profile", "dataframe_path", "treatment_encoding"]
    REQUIRED_STATE_FIELDS = ["dataset_info"]
    JOB_STATUS = JobStatus.PROFILING
    PROGRESS_WEIGHT = 0.08

    SYSTEM_PROMPT = SYSTEM_PROMPT

    def __init__(self) -> None:
        super().__init__()
        self.register_context_tools()

        self._df: pd.DataFrame | None = None
        self._profile: DataProfile | None = None
        self._finalized: bool = False
        self._final_result: dict[str, Any] = {}

        for tool_module in tools.MODULES:
            self._register(tool_module)

    def _register(self, tool_module) -> None:
        """Register one tool module and expose its handler as _<tool_name> so
        existing tests that call the handler directly keep working.
        """
        async def _handler(state: AnalysisState, *args, **kwargs) -> ToolResult:
            return await tool_module.handle(self, state, *args, **kwargs)

        self.register_tool(
            name=tool_module.SCHEMA["name"],
            description=tool_module.SCHEMA["description"],
            parameters=tool_module.SCHEMA["parameters"],
            handler=_handler,
        )
        setattr(self, f"_tool_{tool_module.__name__.rsplit('.', 1)[-1]}", _handler)

    def _get_initial_observation(self, state: AnalysisState) -> str:
        return initial_observation_text(state)

    async def is_task_complete(self, state: AnalysisState) -> bool:
        return self._finalized and state.data_profile is not None

    # Method shims preserving the legacy private-method surface so tests that
    # call the helpers directly (rather than through execute()) keep working.

    def _compute_basic_profile(self, df: pd.DataFrame) -> DataProfile:
        return compute_basic_profile(df)

    def _auto_finalize(self) -> dict[str, Any]:
        return auto_finalize(self._df, self._profile)

    def _populate_profile(self, final_result: dict[str, Any]) -> None:
        populate_profile(self._profile, final_result)

    def _save_dataframe(self, df: pd.DataFrame, job_id: str) -> str:
        return save_dataframe(df, job_id)

    def _exclude_oracle_columns(self, df: pd.DataFrame, state: AnalysisState) -> pd.DataFrame:
        return exclude_oracle_columns(df, state, self.logger)

    @staticmethod
    def _is_oracle_column(col_name: str) -> bool:
        from .helpers import is_oracle_column
        return is_oracle_column(col_name)

    async def execute(self, state: AnalysisState) -> AnalysisState:
        self.logger.info(
            "profiling_start",
            job_id=state.job_id,
            dataset=state.dataset_info.name or state.dataset_info.url,
        )
        start_time = time.time()

        try:
            if "kaggle.com" in state.dataset_info.url and not state.raw_metadata:
                await fetch_kaggle_metadata(state)

            self._df, load_error = await load_dataset(state)
            if self._df is None:
                state.mark_failed(load_error or "Failed to load dataset", self.AGENT_NAME)
                return state

            self._df = exclude_oracle_columns(self._df, state, self.logger)
            self._profile = compute_basic_profile(self._df)

            state.dataframe_path = save_dataframe(self._df, state.job_id)
            self._current_state = state

            state = await super().execute(state)

            if not self._finalized:
                self.logger.warning("profiler_auto_finalize")
                self._final_result = auto_finalize(self._df, self._profile)
                self._finalized = True

            populate_profile(self._profile, self._final_result)

            state.data_profile = self._profile
            state.dataset_info.n_samples = self._profile.n_samples
            state.dataset_info.n_features = self._profile.n_features
            state.push_sse_event(
                "data_profile_ready",
                {
                    "n_samples": self._profile.n_samples,
                    "n_features": self._profile.n_features,
                    "treatment_candidates": list(self._profile.treatment_candidates),
                    "outcome_candidates": list(self._profile.outcome_candidates),
                    "potential_confounders": list(self._profile.potential_confounders),
                    "potential_instruments": list(self._profile.potential_instruments),
                    "feature_types": dict(self._profile.feature_types),
                    "missing_values": dict(self._profile.missing_values),
                },
            )

            self._apply_treatment_encoding(state)

            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.info(
                "profiling_complete",
                samples=self._profile.n_samples,
                features=self._profile.n_features,
                treatment_candidates=self._profile.treatment_candidates,
                outcome_candidates=self._profile.outcome_candidates,
                duration_ms=duration_ms,
            )

        except Exception as e:
            self.logger.exception("profiling_failed", error=str(e))
            state.mark_failed(f"Data profiling failed: {str(e)}", self.AGENT_NAME)

        return state

    def _apply_treatment_encoding(self, state: AnalysisState) -> None:
        """Set state.treatment_encoding from the LLM's finalize_profile choice;
        fall back to auto-detection for string-typed treatments.
        """
        enc_strategy = self._final_result.get("treatment_encoding_strategy")
        if enc_strategy and enc_strategy != "none":
            control_val = self._final_result.get("treatment_control_value")
            treatment_col = (
                self._final_result.get("treatment_candidates", [None])[0]
                if self._final_result.get("treatment_candidates")
                else None
            )
            unique_vals: list = []
            if treatment_col and self._df is not None and treatment_col in self._df.columns:
                unique_vals = self._df[treatment_col].dropna().unique().tolist()
            state.treatment_encoding = build_encoding(unique_vals, enc_strategy, control_val)
            if state.treatment_encoding:
                self.logger.info(
                    "treatment_encoding_set",
                    strategy=enc_strategy,
                    control_value=control_val,
                    mapping=state.treatment_encoding.value_mapping,
                )

        if (
            state.treatment_encoding is None
            and self._df is not None
            and self._profile is not None
            and self._profile.treatment_candidates
        ):
            tc = state.treatment_variable or self._profile.treatment_candidates[0]
            if tc in self._df.columns and self._df[tc].dtype == object:
                unique_vals = self._df[tc].dropna().unique().tolist()
                state.treatment_encoding = auto_encode(unique_vals, self._df[tc].value_counts())
                if state.treatment_encoding:
                    self.logger.info(
                        "treatment_encoding_auto_detected",
                        strategy=state.treatment_encoding.strategy,
                        control_value=state.treatment_encoding.control_value,
                        mapping=state.treatment_encoding.value_mapping,
                    )
