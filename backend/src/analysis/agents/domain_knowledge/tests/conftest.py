"""Shared fixtures for domain_knowledge tests."""

import pytest

from src.analysis.agents.base import AnalysisState, DatasetInfo
from src.analysis.agents.domain_knowledge.agent import DomainKnowledgeAgent


@pytest.fixture
def agent():
    return DomainKnowledgeAgent()


@pytest.fixture
def state_with_metadata():
    """State seeded with rich Kaggle metadata (LaLonde shape)."""
    return AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(url="https://kaggle.com/test", name="lalonde"),
        raw_metadata={
            "title": "LaLonde NSW Job Training Program",
            "description": (
                "This dataset contains data from the National Supported Work (NSW) "
                "demonstration, a labor training program implemented in the mid-1970s. "
                "Participants were randomly assigned to receive job training. "
                "The outcome of interest is earnings in 1978 (re78). "
                "Baseline characteristics include age, education, race (black, hispanic), "
                "marital status (married), degree status (nodegree), and pre-program earnings (re74, re75)."
            ),
            "subtitle": "Causal inference benchmark dataset",
            "tags": ["economics", "causal-inference", "employment"],
            "keywords": ["treatment effect", "job training"],
            "column_descriptions": {
                "treat": "Treatment indicator: 1 if received job training, 0 otherwise",
                "age": "Age in years",
                "education": "Years of education",
                "re78": "Earnings in 1978 (outcome)",
            },
            "metadata_quality": "high",
        },
    )


@pytest.fixture
def state_with_minimal_metadata():
    """State with thin metadata (low-quality Kaggle source)."""
    return AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(url="https://kaggle.com/test", name="test_data"),
        raw_metadata={
            "title": "Test Dataset",
            "description": "",
            "tags": [],
            "column_descriptions": {},
            "metadata_quality": "low",
        },
    )
