"""DomainKnowledgeAgent: investigates dataset metadata to seed causal context.

ReAct loop over a metadata-only tool surface. The agent does NOT touch the
dataframe; it reads description, columns, tags, and forms hypotheses about
treatment, outcome, confounders, temporal ordering, and immutable variables.

The typed output (DomainKnowledge) lives at AnalysisState.domain_knowledge and
is what downstream specialists consume via the ask_domain_knowledge context tool.
"""

from __future__ import annotations

from typing import Any

from src.analysis.agents.base import AnalysisState, JobStatus, ReActAgent
from src.analysis.agents.registry import register_agent
from src.logging_config.structured import get_logger

from . import tools
from .helpers import has_treatment_and_outcome_hypotheses, initial_observation_text
from .output import DomainKnowledge, Hypothesis, Uncertainty
from .prompt import SYSTEM_PROMPT

logger = get_logger(__name__)


@register_agent("domain_knowledge")
class DomainKnowledgeAgent(ReActAgent):
    """Investigates dataset metadata to build causal understanding.

    Does NOT look at the actual data, only metadata. Forms hypotheses about
    treatment, outcome, confounders; identifies immutables and temporal order.
    """

    AGENT_NAME = "domain_knowledge"
    MAX_STEPS = 12

    WRITES_STATE_FIELDS = ["domain_knowledge"]
    REQUIRED_STATE_FIELDS = ["dataset_info"]
    JOB_STATUS = JobStatus.PROFILING
    PROGRESS_WEIGHT = 0.04

    SYSTEM_PROMPT = SYSTEM_PROMPT

    def __init__(self) -> None:
        super().__init__()
        self._hypotheses: list[dict[str, Any]] = []
        self._uncertainties: list[dict[str, Any]] = []
        self._temporal_understanding: str | None = None
        self._immutable_vars: list[str] = []

        for tool_module in tools.MODULES:
            self._register(tool_module)

    def _register(self, tool_module) -> None:
        """Register one tool module and expose its handler as a private method.

        Binding to `_<tool_name>` keeps the legacy call surface (`agent._read_description(state)`)
        working for tests that exercise tool handlers directly.
        """
        async def _handler(state: AnalysisState, *args, **kwargs):
            return await tool_module.handle(self, state, *args, **kwargs)

        self.register_tool(
            name=tool_module.SCHEMA["name"],
            description=tool_module.SCHEMA["description"],
            parameters=tool_module.SCHEMA["parameters"],
            handler=_handler,
        )
        setattr(self, f"_{tool_module.SCHEMA['name']}", _handler)

    def _get_initial_observation(self, state: AnalysisState) -> str:
        return initial_observation_text(state)

    async def is_task_complete(self, state: AnalysisState) -> bool:
        return has_treatment_and_outcome_hypotheses(self._hypotheses)

    async def execute(self, state: AnalysisState) -> AnalysisState:
        self._hypotheses = []
        self._uncertainties = []
        self._temporal_understanding = None
        self._immutable_vars = []

        state = await super().execute(state)

        all_hypotheses = [Hypothesis(**h) for h in self._hypotheses]
        live_hypotheses = [h for h in all_hypotheses if not h.revised]
        uncertainties = [Uncertainty(**u) for u in self._uncertainties]

        state.domain_knowledge = DomainKnowledge(
            hypotheses=live_hypotheses,
            all_hypotheses=all_hypotheses,
            uncertainties=uncertainties,
            temporal_understanding=self._temporal_understanding,
            immutable_vars=list(self._immutable_vars),
            investigation_complete=True,
        )

        self.logger.info(
            "domain_knowledge_complete",
            num_hypotheses=len(live_hypotheses),
            num_uncertainties=len(uncertainties),
            num_immutable=len(self._immutable_vars),
        )

        return state
