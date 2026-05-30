"""ReAct Effect Estimator Agent - True agentic treatment effect estimation.

This agent uses the ReAct paradigm to autonomously:
1. Analyze data characteristics
2. Select appropriate estimation methods
3. Execute methods and observe results
4. Retry or adjust if needed
5. Synthesize findings
"""

import pandas as pd

from src.analysis.agents.base import (
    AnalysisState,
    JobStatus,
    ReActAgent,
    ToolResult,
    TreatmentEffectResult,
)
from src.analysis.agents.registry import register_agent
from src.logging_config.structured import get_logger

from . import tools
from .helpers import (
    auto_finalize_ols,
    build_initial_observation,
    load_dataframe,
    resolve_baseline_covariates,
)
from .prompt import SYSTEM_PROMPT

logger = get_logger(__name__)


@register_agent("effect_estimator_react")
class EffectEstimatorReActAgent(ReActAgent):
    """ReAct agent for treatment effect estimation.

    This agent autonomously:
    1. Inspects the data to understand its characteristics
    2. Reasons about which methods are appropriate
    3. Executes methods one by one, observing results
    4. Handles errors by trying alternative approaches
    5. Synthesizes results and identifies the most credible estimate
    """

    AGENT_NAME = "effect_estimator_react"
    MAX_STEPS = 15

    WRITES_STATE_FIELDS = ["treatment_effects"]
    REQUIRED_STATE_FIELDS = ["data_profile", "dataframe_path"]
    JOB_STATUS = JobStatus.ESTIMATING_EFFECTS
    PROGRESS_WEIGHT = 0.15

    SYSTEM_PROMPT = SYSTEM_PROMPT

    def __init__(self) -> None:
        super().__init__()
        self._df: pd.DataFrame | None = None
        self._results: list[TreatmentEffectResult] = []
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
        load_error: str | None = None
        if state.dataframe_path:
            try:
                self._df = load_dataframe(state.dataframe_path)
                if self._df is None:
                    load_error = f"file not found: {state.dataframe_path}"
            except Exception as e:
                load_error = str(e)
        return build_initial_observation(state, self._df, load_error)

    async def is_task_complete(self, state: AnalysisState) -> bool:
        return len(self._results) >= 2 and len(state.treatment_effects) >= 2

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute the ReAct estimation loop with an auto-finalize fallback.

        The 15-step ReAct budget is not always enough; on Lalonde the model
        burned all 15 calling run_method on inapplicable methods (PSM/IPW
        without overlap, IV without instruments) and finished with zero
        valid estimates. The imperative variant handles this via
        _auto_finalize() (effect_estimator/agent.py); we mirror the contract
        here so an exhausted react loop still produces a baseline OLS
        estimate that downstream stages can ground on.
        """
        self._results = []

        state = await super().execute(state)

        if not state.treatment_effects:
            self._auto_finalize_ols(state)

        self.logger.info(
            "estimation_complete",
            n_results=len(self._results),
            methods=[r.method for r in self._results],
            auto_finalized=not self._results
            or all(r.method.lower().startswith("ols") for r in self._results),
        )

        return state

    def _auto_finalize_ols(self, state: AnalysisState) -> None:
        self._df = auto_finalize_ols(
            self._df,
            state,
            self._results,
            self.logger,
            self.MAX_STEPS,
        )

    def _resolve_baseline_covariates(self, state: AnalysisState) -> list[str]:
        return resolve_baseline_covariates(self._df, state)
