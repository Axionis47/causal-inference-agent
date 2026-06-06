"""Confounder Discovery Agent - identifies confounders via iterative tool use."""

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
from .brief import CAPABILITY as CD_CAPABILITY, build_brief, preflight
from .helpers import (
    candidates_from_df,
    fallback_confounder_identification,
    initial_observation_text,
    load_dataframe,
)
from .prompt import SYSTEM_PROMPT

logger = get_logger(__name__)


@register_agent("confounder_discovery")
class ConfounderDiscoveryAgent(ReActAgent, ContextTools):
    """Agent that discovers confounders through iterative tool-based reasoning."""

    AGENT_NAME = "confounder_discovery"
    MAX_STEPS = 15
    WRITES_STATE_FIELDS = ["confounder_discovery"]
    REQUIRED_STATE_FIELDS = ["data_profile", "dataframe_path"]
    JOB_STATUS = JobStatus.DISCOVERING_CAUSAL
    PROGRESS_WEIGHT = 0.06
    CAPABILITY = CD_CAPABILITY

    SYSTEM_PROMPT = SYSTEM_PROMPT

    def __init__(self) -> None:
        super().__init__()
        self._df: pd.DataFrame | None = None
        self._current_state: AnalysisState | None = None
        self._treatment_var: str | None = None
        self._outcome_var: str | None = None
        self._investigation_log: list[dict] = []
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

    def _resolve_treatment_outcome(self) -> tuple[str | None, str | None]:
        if self._treatment_var and self._outcome_var:
            return self._treatment_var, self._outcome_var
        if self._current_state:
            t, o = self._current_state.get_primary_pair()
            self._treatment_var = t
            self._outcome_var = o
            return t, o
        return None, None

    def _get_initial_observation(self, state: AnalysisState) -> str:
        candidates = candidates_from_df(self._df, self._treatment_var, self._outcome_var)
        dag_available = (
            state.proposed_dag is not None
            and hasattr(state.proposed_dag, "edges")
            and len(state.proposed_dag.edges) > 0
        )
        return initial_observation_text(
            candidates, self._treatment_var, self._outcome_var, dag_available
        )

    async def is_task_complete(self, state: AnalysisState) -> bool:
        return self._finalized

    async def execute(self, state: AnalysisState) -> AnalysisState:
        refusal = preflight(state)
        if refusal is not None:
            state.agent_briefs[refusal.agent] = refusal
            self.logger.info(
                "confounder_discovery_refused",
                flag=refusal.flags[0].value,
                headline=refusal.headline,
            )
            return state

        self.logger.info("confounder_discovery_start", job_id=state.job_id)
        start_time = time.time()

        try:
            self._df = self._load_dataframe(state)
            if self._df is None:
                self.logger.warning("no_dataframe_for_confounder_discovery")
                return state

            self._current_state = state
            self._investigation_log = []
            self._finalized = False

            self._treatment_var, self._outcome_var = self._resolve_treatment_outcome()

            if not self._treatment_var or not self._outcome_var:
                self.logger.warning("no_treatment_or_outcome_specified")
                return state

            await super().execute(state)

            if not self._finalized:
                self.logger.info("using_fallback_confounder_identification")
                state = self._fallback_confounder_identification(state)

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
        finally:
            if "confounder_discovery" not in state.agent_briefs:
                brief = build_brief(state)
                state.agent_briefs[brief.agent] = brief
                self.logger.info(
                    "confounder_discovery_brief",
                    status=brief.status,
                    flags=[f.value for f in brief.flags],
                )

        return state

    def _load_dataframe(self, state: AnalysisState) -> pd.DataFrame | None:
        return load_dataframe(state.dataframe_path, self.logger)

    def _get_candidates(self) -> list[str]:
        return candidates_from_df(self._df, self._treatment_var, self._outcome_var)

    def _fallback_confounder_identification(self, state: AnalysisState) -> AnalysisState:
        if self._df is None:
            return state

        ranked, excluded = fallback_confounder_identification(
            df=self._df,
            treatment_var=self._treatment_var,
            outcome_var=self._outcome_var,
            dag=state.proposed_dag,
            domain_knowledge=state.domain_knowledge if hasattr(state, "domain_knowledge") else None,
            logger=self.logger,
        )

        if state.data_profile:
            state.data_profile.potential_confounders = ranked

        state.confounder_discovery = {
            "ranked_confounders": ranked,
            "excluded_variables": excluded,
            "adjustment_strategy": (
                "Statistical fallback - three criteria (classic confounder, prognostic variable, "
                "balance indicator), filtered by causal role"
            ),
            "investigation_log": [],
        }

        self.logger.info(
            "fallback_confounder_identification_complete",
            n_confounders=len(ranked),
            n_excluded=len(excluded),
            top_confounders=ranked[:5],
        )
        return state
