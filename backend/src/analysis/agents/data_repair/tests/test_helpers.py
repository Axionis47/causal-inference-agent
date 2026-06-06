"""Tests for the pure helpers: count_outliers and auto_finalize."""

import pandas as pd

from src.analysis.agents.data_repair.helpers import auto_finalize, count_outliers


class TestCountOutliers:
    def test_counts_iqr_outliers(self):
        """Two extreme values land outside the 1.5*IQR fence."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100, -100])
        assert count_outliers(data) == 2

    def test_no_outliers_in_uniform_data(self):
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert count_outliers(data) == 0


class TestAutoFinalize:
    def test_no_repairs_summary_says_so(self, sample_dataframe):
        result = auto_finalize(sample_dataframe, sample_dataframe, [])
        assert result["repairs_summary"] == ["No repairs applied"]
        assert "missing" in result["quality_assessment"].lower() or "good" in result["quality_assessment"].lower()

    def test_summarises_repairs(self, sample_dataframe):
        repairs = [
            {"type": "missing", "strategy": "median", "columns": ["education"]},
            {"type": "outliers", "strategy": "clip", "columns": ["age", "income"]},
        ]
        result = auto_finalize(sample_dataframe, sample_dataframe, repairs)
        assert len(result["repairs_summary"]) == 2
        assert "missing" in result["repairs_summary"][0]
        assert "outliers" in result["repairs_summary"][1]

    def test_warns_on_significant_row_loss(self, sample_dataframe):
        """Dropping >20% of rows should surface a selection-bias caution."""
        trimmed = sample_dataframe.iloc[: int(len(sample_dataframe) * 0.5)]
        result = auto_finalize(trimmed, sample_dataframe, [])
        assert any("row loss" in c.lower() for c in result["cautions"])
