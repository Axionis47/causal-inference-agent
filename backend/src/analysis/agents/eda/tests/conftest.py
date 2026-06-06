"""Shared fixtures for eda tests."""

import pickle

import numpy as np
import pandas as pd
import pytest

from src.analysis.agents.base import AnalysisState, DataProfile, DatasetInfo
from src.analysis.agents.eda.agent import EDAAgent


@pytest.fixture
def agent():
    return EDAAgent()


@pytest.fixture
def sample_dataframe():
    """A small DataFrame shaped like an RCT-style covariate set."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "treat": np.random.binomial(1, 0.4, n),
        "age": np.random.normal(40, 10, n),
        "income": np.random.normal(50000, 15000, n),
        "education": np.random.randint(10, 20, n),
        "outcome": np.random.normal(100, 20, n),
        "gender": np.random.choice(["M", "F"], n),
        "region": np.random.choice(["North", "South", "East", "West"], n),
    })


@pytest.fixture
def sample_profile(sample_dataframe):
    return DataProfile(
        n_samples=100,
        n_features=7,
        feature_names=list(sample_dataframe.columns),
        feature_types={
            "treat": "binary",
            "age": "numeric",
            "income": "numeric",
            "education": "numeric",
            "outcome": "numeric",
            "gender": "categorical",
            "region": "categorical",
        },
        missing_values={col: 0 for col in sample_dataframe.columns},
        numeric_stats={
            "treat": {"mean": 0.4, "std": 0.49, "min": 0.0, "max": 1.0},
            "age": {"mean": 40.0, "std": 10.0, "min": 20.0, "max": 60.0},
            "income": {"mean": 50000.0, "std": 15000.0, "min": 20000.0, "max": 80000.0},
            "education": {"mean": 15.0, "std": 3.0, "min": 10.0, "max": 20.0},
            "outcome": {"mean": 100.0, "std": 20.0, "min": 60.0, "max": 140.0},
        },
        categorical_stats={
            "gender": {"M": 50, "F": 50},
            "region": {"North": 25, "South": 25, "East": 25, "West": 25},
        },
        treatment_candidates=["treat"],
        outcome_candidates=["outcome", "income"],
        potential_confounders=["age", "education"],
    )


@pytest.fixture
def state_with_dataframe(sample_dataframe, sample_profile, tmp_path):
    """State pointing at a pickled DataFrame; tests inject _df directly."""
    pkl_path = tmp_path / "test_data.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(sample_dataframe, f)

    return AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(
            url="https://kaggle.com/test/dataset",
            name="test_dataset",
            local_path=str(pkl_path),
        ),
        dataframe_path=str(pkl_path),
        data_profile=sample_profile,
    )
