"""Tests for the pure helpers: basic profile + auto-finalize."""

import pandas as pd


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
