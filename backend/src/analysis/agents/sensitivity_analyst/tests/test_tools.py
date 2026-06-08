"""Tool-handler tests for sensitivity_analyst tools."""
import types

import pytest

from src.analysis.agents.base import AnalysisState, DatasetInfo, ToolResultStatus
from src.analysis.agents.effect_estimator.output import TreatmentEffectResult
from src.analysis.agents.sensitivity_analyst.tools import (
    compute_e_value,
    compute_rosenbaum_bounds,
)


def _agent_with_effect():
    state = AnalysisState(
        job_id="s",
        dataset_info=DatasetInfo(url="x"),
        treatment_variable="treat",
        outcome_variable="re78",
    )
    state.treatment_effects = [
        TreatmentEffectResult(
            method="OLS",
            estimand="ATE",
            estimate=1.2,
            std_error=0.4,
            ci_lower=0.4,
            ci_upper=2.0,
            p_value=0.01,
        )
    ]
    return types.SimpleNamespace(_current_state=state, _df=None, _results=[])


@pytest.mark.asyncio
async def test_compute_e_value_accepts_float_method_index():
    # Regression: Gemini passes method_index as a float (0.0). list indexing
    # crashed with "list indices must be integers, not float", which made the
    # tool error on every call and the agent loop without producing results.
    agent = _agent_with_effect()
    result = await compute_e_value.handle(
        agent, agent._current_state, method_index=0.0
    )
    assert result.status == ToolResultStatus.SUCCESS
    assert result.output["method_analyzed"] == "OLS"


@pytest.mark.asyncio
async def test_compute_rosenbaum_bounds_accepts_float_method_index():
    agent = _agent_with_effect()
    result = await compute_rosenbaum_bounds.handle(
        agent, agent._current_state, method_index=0.0
    )
    assert result.status == ToolResultStatus.SUCCESS
