"""Shared fixtures for confounder_discovery tests."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.agents.base import AnalysisState, DatasetInfo
from src.analysis.agents.confounder_discovery.agent import ConfounderDiscoveryAgent


@pytest.fixture
def agent():
    return ConfounderDiscoveryAgent()


@pytest.fixture
def confounder_dataframe():
    """DataFrame where `age` confounds (treat, outcome) and `noise` does not.

    age -> treat, age -> outcome, treat -> outcome (with effect), noise ~ uniform.
    Sample size kept >= 80 so the t-tests have enough power.
    """
    rng = np.random.default_rng(7)
    n = 200
    age = rng.normal(40, 10, n)
    treat = (age + rng.normal(0, 3, n) > 40).astype(int)
    outcome = 2 * treat + 0.5 * age + rng.normal(0, 2, n)
    noise = rng.normal(0, 1, n)
    return pd.DataFrame({"treat": treat, "age": age, "outcome": outcome, "noise": noise})


@pytest.fixture
def state(tmp_path, confounder_dataframe):
    path = tmp_path / "data.parquet"
    confounder_dataframe.to_parquet(path)
    return AnalysisState(
        job_id="conf-job",
        dataset_info=DatasetInfo(url="https://kaggle.com/x/y", name="d", local_path=str(path)),
        dataframe_path=str(path),
        treatment_variable="treat",
        outcome_variable="outcome",
    )


@pytest.fixture
def primed_agent(agent, confounder_dataframe, state):
    """Agent with _df / _current_state populated, treatment/outcome resolved."""
    agent._df = confounder_dataframe.copy()
    agent._current_state = state
    agent._treatment_var = "treat"
    agent._outcome_var = "outcome"
    return agent
