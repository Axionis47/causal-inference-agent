"""Shared fixtures for data_repair tests."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.agents.base import AnalysisState, DatasetInfo
from src.analysis.agents.data_repair.agent import DataRepairAgent


@pytest.fixture
def agent():
    return DataRepairAgent()


@pytest.fixture
def sample_dataframe():
    """A small DataFrame with a binary treatment, numeric covariates, and one column with missing values."""
    np.random.seed(7)
    n = 80
    df = pd.DataFrame({
        "treat": np.random.binomial(1, 0.4, n),
        "age": np.random.normal(40, 10, n),
        "income": np.random.normal(50000, 15000, n),
        "education": np.random.randint(10, 20, n).astype(float),
        "outcome": np.random.normal(100, 20, n),
    })
    df.loc[df.index[:8], "education"] = np.nan
    return df


@pytest.fixture
def state(tmp_path, sample_dataframe):
    """State that points at a real parquet file so the loader works."""
    path = tmp_path / "data.parquet"
    sample_dataframe.to_parquet(path)
    return AnalysisState(
        job_id="repair-job",
        dataset_info=DatasetInfo(url="https://kaggle.com/x/y", name="d", local_path=str(path)),
        dataframe_path=str(path),
        treatment_variable="treat",
        outcome_variable="outcome",
    )


@pytest.fixture
def primed_agent(agent, sample_dataframe, state):
    """An agent with _df / _df_original / _current_state populated (ReAct-loop scaffolding)."""
    agent._df = sample_dataframe.copy()
    agent._df_original = sample_dataframe.copy()
    agent._current_state = state
    return agent
