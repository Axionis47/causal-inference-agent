"""Propensity Score Diagnostics Agent - Truly agentic PS validation.

This agent uses the ReAct pattern to iteratively:
1. Estimate propensity scores
2. Assess overlap between treatment groups
3. Check covariate balance before/after weighting
4. Test model calibration
5. Try alternative specifications if needed
6. Provide actionable recommendations
"""

import time

import numpy as np
import pandas as pd

from src.analysis.agents.base import AnalysisState, JobStatus, ToolResult
from src.analysis.agents.base.context_tools import ContextTools
from src.analysis.agents.base.react_agent import ReActAgent
from src.analysis.agents.registry import register_agent
from src.logging_config.structured import get_logger

from . import tools
from .brief import (
    CAPABILITY as PS_CAPABILITY,
    Metrics,
    build_brief,
    preflight,
)
from .helpers import (
    auto_finalize,
    initial_observation_text,
    load_dataframe,
    resolve_confounders,
)
from .prompt import SYSTEM_PROMPT

logger = get_logger(__name__)


@register_agent("ps_diagnostics")
class PSDiagnosticsAgent(ReActAgent, ContextTools):
    """Truly agentic propensity score diagnostics using ReAct pattern.

    The LLM iteratively:
    1. Estimates propensity scores with different specifications
    2. Checks overlap and identifies positivity violations
    3. Assesses covariate balance with/without weighting
    4. Tests model calibration
    5. Tries improvements if diagnostics are poor
    6. Provides final recommendations

    Uses ReAct pattern with pull-based context tools.
    """

    AGENT_NAME = "ps_diagnostics"
    MAX_STEPS = 15
    WRITES_STATE_FIELDS = ["ps_diagnostics"]
    REQUIRED_STATE_FIELDS = ["data_profile", "dataframe_path"]
    JOB_STATUS = JobStatus.ESTIMATING_EFFECTS
    PROGRESS_WEIGHT = 0.06
    CAPABILITY = PS_CAPABILITY

    SYSTEM_PROMPT = SYSTEM_PROMPT

    def __init__(self) -> None:
        super().__init__()
        self._df: pd.DataFrame | None = None
        self._treatment_var: str | None = None
        self._outcome_var: str | None = None
        self._confounders: list[str] = []
        self._propensity_scores: np.ndarray | None = None
        self._treatment: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._current_state: AnalysisState | None = None
        self._finalized: bool = False
        self._metrics: Metrics = Metrics()

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
        return initial_observation_text(
            self._treatment_var, self._outcome_var, self._confounders
        )

    async def is_task_complete(self, state: AnalysisState) -> bool:
        return self._finalized

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute propensity score diagnostics using the ReAct loop."""
        # Reset per-execution metrics so re-dispatch (critique iteration)
        # does not see stale values from a prior run on the same instance.
        self._metrics = Metrics()

        refusal = preflight(state)
        if refusal is not None:
            state.agent_briefs[refusal.agent] = refusal
            self.logger.info(
                "ps_diagnostics_refused",
                flag=refusal.flags[0].value,
                headline=refusal.headline,
            )
            return state

        self.logger.info("ps_diagnostics_start", job_id=state.job_id)
        start_time = time.time()

        try:
            df = self._load_dataframe(state)
            if df is None:
                self.logger.warning("no_dataframe_for_ps_diagnostics")
                return state

            self._df = df
            self._current_state = state
            self._finalized = False
            self._treatment_var = state.treatment_variable
            self._outcome_var = state.outcome_variable

            if not self._treatment_var or not self._outcome_var:
                self.logger.warning("no_treatment_outcome_vars")
                return state

            self._confounders = resolve_confounders(
                state, df, self._treatment_var, self._outcome_var
            )
            if (
                state.proposed_dag
                and state.proposed_dag.adjustment_set
                and len(state.proposed_dag.adjustment_set) > 0
            ):
                self.logger.info(
                    "using_dag_expert_adjustment_set",
                    n_vars=len(self._confounders),
                )

            if not self._confounders:
                self.logger.warning("no_confounders_for_ps")
                return state

            await super().execute(state)

            if not self._finalized:
                state.ps_diagnostics = self._auto_finalize()

            duration_ms = int((time.time() - start_time) * 1000)
            ps_diag = state.ps_diagnostics or {}
            trace = self.create_trace(
                action="ps_diagnostics_complete",
                reasoning=ps_diag.get("reasoning", "PS diagnostics completed"),
                outputs={
                    "model_quality": ps_diag.get("model_quality"),
                    "proceed": ps_diag.get("proceed_with_analysis"),
                    "recommended_method": ps_diag.get("recommended_method"),
                },
                duration_ms=duration_ms,
            )
            state.add_trace(trace)

            self.logger.info(
                "ps_diagnostics_complete",
                model_quality=ps_diag.get("model_quality"),
            )

        except Exception as e:
            self.logger.exception("ps_diagnostics_failed", error=str(e))
        finally:
            # Always emit a brief so the orchestrator has a typed signal,
            # regardless of which path the body took (early return on
            # missing df / no confounders, exception, or normal finish).
            if "ps_diagnostics" not in state.agent_briefs:
                brief = build_brief(state, self._metrics)
                state.agent_briefs[brief.agent] = brief
                self.logger.info(
                    "ps_diagnostics_brief",
                    status=brief.status,
                    flags=[f.value for f in brief.flags],
                )

        return state

    def _load_dataframe(self, state: AnalysisState) -> pd.DataFrame | None:
        return load_dataframe(state.dataframe_path, self.logger)

    def _auto_finalize(self) -> dict:
        return auto_finalize(self._propensity_scores, self._treatment)
