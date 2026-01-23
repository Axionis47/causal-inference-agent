"""Pytest configuration and shared fixtures."""

import os
import sys
from pathlib import Path

import pytest

# Add src to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Set test environment
os.environ["ENVIRONMENT"] = "development"
os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY", "test-key")
os.environ["KAGGLE_KEY"] = os.environ.get("KAGGLE_KEY", "test-key")
os.environ["CLAUDE_API_KEY"] = os.environ.get("CLAUDE_API_KEY", "test-key")


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    import numpy as np
    import pandas as pd

    np.random.seed(42)
    n = 200

    # Generate confounders
    age = np.random.normal(40, 10, n)
    income = np.random.normal(50000, 15000, n)

    # Treatment depends on confounders
    propensity = 1 / (1 + np.exp(-(0.02 * (age - 40) + 0.00001 * (income - 50000))))
    treatment = np.random.binomial(1, propensity)

    # Outcome depends on treatment and confounders
    outcome = (
        5000  # base
        + 2000 * treatment  # TRUE ATE = 2000
        + 50 * age
        + 0.1 * income
        + np.random.normal(0, 1000, n)
    )

    return pd.DataFrame({
        "age": age,
        "income": income,
        "treatment": treatment,
        "outcome": outcome,
    })


@pytest.fixture
def analysis_state():
    """Create a sample AnalysisState for testing."""
    from src.agents import AnalysisState, DatasetInfo

    return AnalysisState(
        job_id="test-job-123",
        dataset_info=DatasetInfo(
            url="https://kaggle.com/test",
            name="test_dataset",
        ),
        treatment_variable="treatment",
        outcome_variable="outcome",
    )


@pytest.fixture
def mock_gemini_response():
    """Create mock Gemini API response."""
    return {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": "This is a test response from Gemini."
                }]
            }
        }]
    }
