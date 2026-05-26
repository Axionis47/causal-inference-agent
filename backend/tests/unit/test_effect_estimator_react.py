"""Tests for the ReAct effect estimator's auto-finalize fallback.

The ReAct loop has a 15-step budget. When the model picks methods that
fail (e.g. PSM/IPW with poor overlap, IV without an instrument), the
loop can exit with `state.treatment_effects = []`. Before this fix, the
critique then correctly REJECTed the analysis but the user got no
estimates at all and no notebook. The auto-finalize fallback runs an
OLS baseline so the loop always emits at least one valid estimate.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analysis.agents import (
    AnalysisState,
    DataProfile,
    DatasetInfo,
    TreatmentEffectResult,
)
from src.analysis.agents.specialists.effect_estimator_react import EffectEstimatorReActAgent


def _make_synthetic_data(tmp_path: Path) -> Path:
    """Lalonde-shaped data the engine can fit OLS on."""
    rng = np.random.default_rng(42)
    n = 400
    age = rng.normal(40, 10, n)
    income = rng.normal(50, 15, n)
    treat = (rng.uniform(size=n) < 0.4).astype(int)
    outcome = 2.0 * treat + 0.05 * age + 0.03 * income + rng.normal(0, 1, n)
    df = pd.DataFrame(
        {"treat": treat, "outcome": outcome, "age": age, "income": income}
    )
    path = tmp_path / "lalonde-like.parquet"
    df.to_parquet(path)
    return path


def _make_state(data_path: Path) -> AnalysisState:
    state = AnalysisState(
        job_id="test-react-finalize",
        dataset_info=DatasetInfo(url="t://t", name="t"),
        treatment_variable="treat",
        outcome_variable="outcome",
    )
    state.dataframe_path = str(data_path)
    state.data_profile = DataProfile(
        n_samples=400,
        n_features=4,
        feature_names=["treat", "outcome", "age", "income"],
        feature_types={
            "treat": "binary",
            "outcome": "numeric",
            "age": "numeric",
            "income": "numeric",
        },
        missing_values={"treat": 0, "outcome": 0, "age": 0, "income": 0},
        numeric_stats={},
        categorical_stats={},
        potential_confounders=["age", "income"],
    )
    return state


@pytest.mark.asyncio
async def test_auto_finalize_runs_ols_when_loop_ends_empty(tmp_path: Path):
    """If the ReAct loop exits with no estimates, _auto_finalize_ols must
    run a baseline so downstream stages have something to ground on."""
    data_path = _make_synthetic_data(tmp_path)
    state = _make_state(data_path)
    agent = EffectEstimatorReActAgent()

    # Skip the actual ReAct loop — simulate a budget-exhausted exit by
    # patching super().execute to a no-op that returns state unchanged.
    async def fake_react_loop(self_state):
        return self_state

    with patch(
        "src.analysis.agents.base.react_agent.ReActAgent.execute",
        new=lambda self, state: fake_react_loop(state),
    ):
        result_state = await agent.execute(state)

    assert len(result_state.treatment_effects) >= 1
    baseline = result_state.treatment_effects[0]
    assert isinstance(baseline, TreatmentEffectResult)
    assert "ols" in baseline.method.lower()
    # The auto-finalize decision must show up in the audit trail.
    auto_decisions = [
        d for d in result_state.decisions
        if d.decision_type == "auto_finalize_baseline"
    ]
    assert len(auto_decisions) == 1


@pytest.mark.asyncio
async def test_auto_finalize_skipped_when_loop_already_produced_results(
    tmp_path: Path,
):
    """If the ReAct loop produced ≥1 estimate, the fallback should not run
    (we don't want to double-count or override a more sophisticated method)."""
    data_path = _make_synthetic_data(tmp_path)
    state = _make_state(data_path)

    pre_existing = TreatmentEffectResult(
        method="AIPW",
        estimand="ATE",
        estimate=1.95,
        std_error=0.2,
        ci_lower=1.5,
        ci_upper=2.4,
        p_value=0.001,
        treatment_variable="treat",
        outcome_variable="outcome",
    )

    agent = EffectEstimatorReActAgent()

    async def fake_react_loop(self_state):
        # Simulate a loop that did produce a result.
        self_state.treatment_effects.append(pre_existing)
        return self_state

    with patch(
        "src.analysis.agents.base.react_agent.ReActAgent.execute",
        new=lambda self, state: fake_react_loop(state),
    ):
        result_state = await agent.execute(state)

    # Only the AIPW result; no OLS fallback was added.
    assert len(result_state.treatment_effects) == 1
    assert result_state.treatment_effects[0].method == "AIPW"
    auto_decisions = [
        d for d in result_state.decisions
        if d.decision_type == "auto_finalize_baseline"
    ]
    assert auto_decisions == []


def test_resolve_baseline_covariates_priority_chain(tmp_path: Path):
    """The covariate priority chain matches the imperative estimator:
    DAG adjustment set → confounder discovery → data profile."""
    data_path = _make_synthetic_data(tmp_path)
    state = _make_state(data_path)
    agent = EffectEstimatorReActAgent()
    agent._df = pd.read_parquet(data_path)

    # No DAG, no confounder discovery → falls through to data_profile.
    covs = agent._resolve_baseline_covariates(state)
    assert set(covs) == {"age", "income"}
