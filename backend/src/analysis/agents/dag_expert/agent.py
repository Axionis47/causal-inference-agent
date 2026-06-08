"""DAG Expert Agent - domain-informed causal graph construction."""

from __future__ import annotations

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
from .brief import CAPABILITY as DE_CAPABILITY, build_brief, preflight
from .helpers import (
    adjustment_set_from_dag,
    build_fallback_dag,
    initial_observation_text,
)
from .prompt import SYSTEM_PROMPT

logger = get_logger(__name__)


@register_agent("dag_expert")
class DAGExpertAgent(ReActAgent, ContextTools):
    """Domain expert agent that constructs validated causal DAGs."""

    AGENT_NAME = "dag_expert"
    # The procedure needs room to gather roles AND build: classify several roles,
    # pull discovery edges, propose domain edges, fuse, then get the adjustment
    # set. 12 was too tight, the loop spent it all on context tools and never
    # committed a DAG. 25 leaves headroom for the build phase.
    MAX_STEPS = 25
    WRITES_STATE_FIELDS = ["refined_dag"]
    REQUIRED_STATE_FIELDS = ["dataset_info", "discovered_dag"]
    JOB_STATUS = JobStatus.DISCOVERING_CAUSAL
    PROGRESS_WEIGHT = 0.08
    CAPABILITY = DE_CAPABILITY

    SYSTEM_PROMPT = SYSTEM_PROMPT

    def __init__(self) -> None:
        super().__init__()
        self.register_context_tools()

        self._domain_edges: list[dict] = []
        self._data_edges: list[dict] = []
        self._forbidden_edges: list[tuple[str, str, str]] = []
        self._variable_roles: dict[str, str] = {}

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
        return (
            state.refined_dag is not None
            and state.refined_dag.variable_roles is not None
            and state.refined_dag.adjustment_set is not None
        )

    async def execute(self, state: AnalysisState) -> AnalysisState:
        refusal = preflight(state)
        if refusal is not None:
            state.agent_briefs[refusal.agent] = refusal
            self.logger.info(
                "dag_expert_refused",
                flag=refusal.flags[0].value,
                headline=refusal.headline,
            )
            return state

        self.logger.info(
            "dag_expert_start",
            job_id=state.job_id,
            has_discovery_dag=state.discovered_dag is not None,
        )

        self._domain_edges = []
        self._data_edges = []
        self._forbidden_edges = []
        self._variable_roles = {}

        try:
            state = await super().execute(state)

            # Robustness: the ReAct loop sometimes classifies roles but runs out
            # of steps before calling fuse_and_validate, leaving refined_dag None
            # (a hard halt) or without an adjustment set. Never hand back an
            # unusable DAG: build the canonical confounder DAG so estimation
            # always has a valid backdoor set.
            if state.refined_dag is None:
                state.refined_dag = build_fallback_dag(state, self._variable_roles)
                self.logger.info(
                    "dag_expert_fallback_dag",
                    reason="loop produced no refined_dag",
                    adjustment_set=state.refined_dag.adjustment_set,
                )
            elif not (state.refined_dag.adjustment_set or []):
                info = adjustment_set_from_dag(
                    state.refined_dag,
                    state.treatment_variable,
                    state.outcome_variable,
                )
                if info["adjustment_set"]:
                    state.refined_dag.adjustment_set = info["adjustment_set"]
                else:
                    state.refined_dag = build_fallback_dag(state, self._variable_roles)
                    self.logger.info(
                        "dag_expert_fallback_dag",
                        reason="empty adjustment set after fusion",
                        adjustment_set=state.refined_dag.adjustment_set,
                    )

            self.logger.info(
                "dag_expert_complete",
                n_edges=len(state.refined_dag.edges) if state.refined_dag else 0,
                n_roles_classified=len(self._variable_roles),
            )
        finally:
            if "dag_expert" not in state.agent_briefs:
                brief = build_brief(state)
                state.agent_briefs[brief.agent] = brief
                self.logger.info(
                    "dag_expert_brief",
                    status=brief.status,
                    flags=[f.value for f in brief.flags],
                )

        return state
