"""Shared fixtures for data_profiler tests."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.agents.base import AnalysisState, DatasetInfo
from src.analysis.agents.data_profiler.agent import DataProfilerAgent


@pytest.fixture
def agent():
    return DataProfilerAgent()


@pytest.fixture
def sample_dataframe():
    """A small DataFrame shaped like a job-training RCT (binary treat + numeric covariates)."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "treat": np.random.binomial(1, 0.3, n),
        "age": np.random.normal(40, 10, n),
        "income": np.random.normal(50000, 15000, n),
        "education": np.random.randint(10, 20, n),
        "outcome": np.random.normal(100, 20, n),
        "gender": np.random.choice(["M", "F"], n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
    })


@pytest.fixture
def state_with_dataframe(sample_dataframe, tmp_path):
    """State pointing at a real CSV on disk so loading paths can be exercised."""
    csv_path = tmp_path / "test_data.csv"
    sample_dataframe.to_csv(csv_path, index=False)

    return AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(
            url="https://kaggle.com/test/dataset",
            name="test_dataset",
            local_path=str(csv_path),
        ),
    )
