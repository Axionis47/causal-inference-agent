"""Tests for the pure helpers: basic profile + auto-finalize."""

import pandas as pd

from src.analysis.agents.data_profiler.helpers import (
    compute_deterministic_profile,
    detect_time_column,
)


class TestComputeBasicProfile:
    def test_computes_profile(self, agent, sample_dataframe):
        """Type-detection: binary, numeric, and categorical columns get the right labels."""
        profile = agent._compute_basic_profile(sample_dataframe)

        assert profile.n_samples == 100
        assert profile.n_features == 7
        assert "treat" in profile.feature_names
        assert profile.feature_types["treat"] == "binary"
        assert profile.feature_types["income"] == "numeric"
        assert profile.feature_types["gender"] == "categorical"

    def test_handles_missing_values(self, agent):
        """Per-column NaN counts roll up correctly."""
        df = pd.DataFrame({"a": [1, 2, None, 4], "b": [None, None, 3, 4]})
        profile = agent._compute_basic_profile(df)

        assert profile.missing_values["a"] == 1
        assert profile.missing_values["b"] == 2


class TestAutoFinalize:
    """Heuristic fallback when the agent never calls finalize_profile."""

    def test_auto_finalize_finds_treatment(self, agent, sample_dataframe):
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)
        result = agent._auto_finalize()
        assert "treat" in result["treatment_candidates"]

    def test_auto_finalize_finds_outcome(self, agent, sample_dataframe):
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)
        result = agent._auto_finalize()
        assert len(result["outcome_candidates"]) > 0

    def test_auto_finalize_finds_confounders(self, agent, sample_dataframe):
        agent._df = sample_dataframe
        agent._profile = agent._compute_basic_profile(sample_dataframe)
        result = agent._auto_finalize()
        assert len(result["potential_confounders"]) > 0


class TestDetectTimeColumn:
    def test_detects_datetime_dtype(self):
        df = pd.DataFrame(
            {"ts": pd.to_datetime(["2021-01-01", "2021-02-01"]), "y": [1, 2]}
        )
        types = {"ts": "datetime", "y": "binary"}
        assert detect_time_column(df, types) == (True, "ts")

    def test_detects_time_keyword(self):
        df = pd.DataFrame({"year": [2019, 2020, 2021], "y": [1.0, 2.0, 3.0]})
        types = {"year": "ordinal", "y": "numeric"}
        assert detect_time_column(df, types) == (True, "year")

    def test_none_when_no_time_column(self):
        df = pd.DataFrame({"age": [40, 41], "income": [50000, 60000]})
        types = {"age": "numeric", "income": "numeric"}
        assert detect_time_column(df, types) == (False, None)


class TestComputeDeterministicProfile:
    def test_facts_match_the_frame(self, sample_dataframe):
        profile = compute_deterministic_profile(sample_dataframe)
        assert profile.n_samples == 100
        assert profile.feature_types["treat"] == "binary"
        assert profile.feature_types["income"] == "numeric"

    def test_leaves_role_candidates_empty(self, sample_dataframe):
        """The pre-gate profile stores facts only: no machine role guesses."""
        profile = compute_deterministic_profile(sample_dataframe)
        assert profile.treatment_candidates == []
        assert profile.outcome_candidates == []
        assert profile.potential_confounders == []
        assert profile.potential_instruments == []

    def test_sets_time_tag_when_date_present(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2021-01-01", "2021-02-01", "2021-03-01"]),
                "sales": [10.0, 12.0, 11.0],
            }
        )
        profile = compute_deterministic_profile(df)
        assert profile.has_time_dimension is True
        assert profile.time_column == "date"

    def test_no_time_tag_for_cross_section(self, sample_dataframe):
        profile = compute_deterministic_profile(sample_dataframe)
        assert profile.has_time_dimension is False
        assert profile.time_column is None
