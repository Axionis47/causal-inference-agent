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


class TestCaptureFileSamples:
    def test_samples_each_tabular_file_and_skips_others(self, tmp_path):
        """One preview per csv/parquet, with real rows and a true row count;
        non-tabular files are ignored."""
        from src.analysis.agents.data_profiler.helpers import capture_file_samples

        pd.DataFrame({"treat": [0, 1, 0], "re78": [0.0, 9930.05, 3.5]}).to_csv(
            tmp_path / "train.csv", index=False
        )
        pd.DataFrame({"x": range(25)}).to_parquet(tmp_path / "extra.parquet")
        (tmp_path / "readme.txt").write_text("not tabular")

        samples = capture_file_samples(tmp_path)

        by_name = {s.name: s for s in samples}
        assert set(by_name) == {"train.csv", "extra.parquet"}  # txt skipped
        assert by_name["train.csv"].columns == ["treat", "re78"]
        assert by_name["train.csv"].total_rows == 3
        assert by_name["train.csv"].rows[1]["re78"] == 9930.05
        assert by_name["extra.parquet"].total_rows == 25

    def test_caps_rows_and_isolates_a_bad_file(self, tmp_path):
        """head() caps the rows; a corrupt file is recorded as an error
        without losing the good ones."""
        from src.analysis.agents.data_profiler.helpers import capture_file_samples

        pd.DataFrame({"x": range(50)}).to_csv(tmp_path / "big.csv", index=False)
        (tmp_path / "broken.parquet").write_text("not a parquet")

        samples = capture_file_samples(tmp_path)
        by_name = {s.name: s for s in samples}

        assert len(by_name["big.csv"].rows) == 10  # head cap
        assert by_name["big.csv"].total_rows == 50
        assert by_name["broken.parquet"].error is not None
        assert by_name["broken.parquet"].rows == []
