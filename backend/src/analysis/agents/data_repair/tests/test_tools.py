"""Tool-handler tests covering the diagnostic and finalize paths."""

import pytest

from src.analysis.agents.base import ToolResultStatus


class TestToolGetDataSummary:
    @pytest.mark.asyncio
    async def test_reports_shape_and_treatment_info(self, primed_agent, state):
        result = await primed_agent._tool_get_data_summary(state)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["shape"] == [80, 5]
        assert result.output["treatment_variable"] == "treat"
        assert "values" in result.output["treatment_info"]


class TestToolCheckMissingValues:
    @pytest.mark.asyncio
    async def test_finds_missing_education(self, primed_agent, state):
        result = await primed_agent._tool_check_missing_values(state)
        assert result.status == ToolResultStatus.SUCCESS
        cols = [r["column"] for r in result.output["columns_with_missing"]]
        assert "education" in cols

    @pytest.mark.asyncio
    async def test_clean_dataset_returns_no_missing(self, primed_agent, state):
        primed_agent._df = primed_agent._df.dropna()
        result = await primed_agent._tool_check_missing_values(state)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output.get("columns") == [] or result.output.get("columns_with_missing") == []


class TestToolCheckOutliers:
    @pytest.mark.asyncio
    async def test_excludes_treatment(self, primed_agent, state):
        """Treatment is filtered out of the default 'all numeric' column list."""
        result = await primed_agent._tool_check_outliers(state, method="iqr")
        assert result.status == ToolResultStatus.SUCCESS
        cols = [r["column"] for r in result.output["outliers_detected"]]
        assert "treat" not in cols

    @pytest.mark.asyncio
    async def test_zscore_path(self, primed_agent, state):
        result = await primed_agent._tool_check_outliers(state, method="zscore")
        assert result.status == ToolResultStatus.SUCCESS


class TestToolCheckSkewness:
    @pytest.mark.asyncio
    async def test_runs_without_skew(self, primed_agent, state):
        """Roughly normal columns yield no skewness flags."""
        result = await primed_agent._tool_check_skewness(state)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_skewed_variables"] is False


class TestToolRepairMissing:
    @pytest.mark.asyncio
    async def test_median_imputation_records_before_after(self, primed_agent, state):
        result = await primed_agent._tool_repair_missing(
            state, strategy="median", columns=["education"]
        )
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["after_missing"] == 0
        assert primed_agent._repairs_applied[-1]["type"] == "missing"
        assert primed_agent._repairs_applied[-1]["strategy"] == "median"


class TestToolValidateCurrentState:
    @pytest.mark.asyncio
    async def test_reports_quality_score_and_status(self, primed_agent, state):
        result = await primed_agent._tool_validate_current_state(state)
        assert result.status == ToolResultStatus.SUCCESS
        assert "quality_score" in result.output
        assert result.output["treatment_variable_present"] is True


class TestToolFinalizeRepairs:
    @pytest.mark.asyncio
    async def test_flips_finalized_and_records_assessment(self, primed_agent, state):
        result = await primed_agent._tool_finalize_repairs(
            state,
            quality_assessment="Good",
            repairs_summary=["median impute on education"],
            cautions=[],
        )
        assert result.status == ToolResultStatus.SUCCESS
        assert primed_agent._finalized is True
        assert result.output["quality_assessment"] == "Good"
