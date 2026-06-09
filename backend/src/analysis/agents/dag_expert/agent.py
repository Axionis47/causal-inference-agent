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
    build_canonical_dag,
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

            # The canonical backdoor decides the adjustment set, not the thin,
            # noisily-oriented fusion DAG the ReAct loop builds. Default to
            # adjusting for every pre-treatment covariate; a real confounder can
            # no longer vanish just because the LLM or discovery never drew an
            # edge for it. The loop's role classifications are carried over as
            # advisory context (a later step turns them into justified
            # exclusions). Data discovery is preserved untouched in
            # state.discovered_dag as the advisory record.
            canonical = build_canonical_dag(state)
            if self._variable_roles:
                merged = dict(canonical.variable_roles or {})
                for var, role in self._variable_roles.items():
                    merged.setdefault(var, role)
                canonical.variable_roles = merged
            state.refined_dag = canonical

            self.logger.info(
                "dag_expert_complete",
                n_covariates=len(canonical.adjustment_set or []),
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
