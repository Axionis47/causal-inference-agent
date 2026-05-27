"""Causal Discovery Agent - ReAct-based graph structure learning."""

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
from .helpers import (
    auto_finalize,
    check_path,
    create_simple_dag,
    initial_observation_text,
    load_dataframe,
)
from .output import CausalDAG
from .prompt import SYSTEM_PROMPT

logger = get_logger(__name__)


@register_agent("causal_discovery")
class CausalDiscoveryAgent(ReActAgent, ContextTools):
    """ReAct-based causal discovery agent."""

    AGENT_NAME = "causal_discovery"
    MAX_STEPS = 15

    WRITES_STATE_FIELDS = ["discovered_dag"]
    REQUIRED_STATE_FIELDS = ["data_profile", "dataframe_path"]
    JOB_STATUS = JobStatus.DISCOVERING_CAUSAL
    PROGRESS_WEIGHT = 0.10

    SYSTEM_PROMPT = SYSTEM_PROMPT

    def __init__(self) -> None:
        super().__init__()
        self.register_context_tools()

        self._df: pd.DataFrame | None = None
        self._current_state: AnalysisState | None = None
        self._discovered_graphs: dict[str, CausalDAG] = {}
        self._current_graph: CausalDAG | None = None
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
        return initial_observation_text(state)

    async def is_task_complete(self, state: AnalysisState) -> bool:
        return self._finalized and state.discovered_dag is not None

    async def execute(self, state: AnalysisState) -> AnalysisState:
        self.logger.info(
            "discovery_start",
            job_id=state.job_id,
            dataset=state.dataset_info.name or state.dataset_info.url,
        )
        start_time = time.time()

        try:
            self._df = self._load_dataframe(state)
            if self._df is None:
                state.mark_failed("Failed to load dataset for discovery", self.AGENT_NAME)
                return state

            self._current_state = state
            self._discovered_graphs = {}
            self._current_graph = None
            self._finalized = False

            self._treatment_var, self._outcome_var = state.get_primary_pair()

            state = await super().execute(state)

            if not self._finalized:
                self.logger.warning("discovery_auto_finalize")
                self._final_result = self._auto_finalize()
                self._finalized = True

            chosen_alg = self._final_result.get("chosen_algorithm", "")
            if chosen_alg and chosen_alg in self._discovered_graphs:
                state.discovered_dag = self._discovered_graphs[chosen_alg]
            elif self._current_graph:
                state.discovered_dag = self._current_graph
            elif self._discovered_graphs:
                state.discovered_dag = list(self._discovered_graphs.values())[0]
            else:
                state.discovered_dag = self._create_simple_dag()

            if state.discovered_dag:
                state.discovered_dag.interpretation = self._final_result.get("interpretation", "")

            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.info(
                "discovery_complete",
                has_dag=state.discovered_dag is not None,
                algorithm=self._final_result.get("chosen_algorithm", "unknown"),
                n_nodes=len(state.discovered_dag.nodes) if state.discovered_dag else 0,
                n_edges=len(state.discovered_dag.edges) if state.discovered_dag else 0,
                duration_ms=duration_ms,
            )

        except Exception as e:
            self.logger.exception("discovery_failed", error=str(e))
            state.discovered_dag = self._create_simple_dag()

        return state

    def _load_dataframe(self, state: AnalysisState) -> pd.DataFrame | None:
        return load_dataframe(state.dataframe_path)

    def _create_simple_dag(self) -> CausalDAG:
        profile = self._current_state.data_profile if self._current_state else None
        return create_simple_dag(self._treatment_var, self._outcome_var, profile)

    def _auto_finalize(self) -> dict[str, Any]:
        return auto_finalize(self._discovered_graphs, self._treatment_var, self._outcome_var)

    def _check_path(self, dag: CausalDAG, source: str, target: str) -> bool:
        return check_path(dag, source, target)
