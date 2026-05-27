"""Tool-handler tests: one class per registered EDA tool."""

import pandas as pd
import pytest

from src.analysis.agents.base import EDAResult, ToolResultStatus


class TestToolGetOverview:
    @pytest.mark.asyncio
    async def test_returns_overview(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        result = await agent._tool_get_overview(state_with_dataframe)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["n_samples"] == 100
        assert result.output["n_features"] == 7
        assert "numeric_columns" in result.output
        assert "categorical_columns" in result.output
        assert "binary_columns" in result.output

    @pytest.mark.asyncio
    async def test_includes_treatment_info(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        result = await agent._tool_get_overview(state_with_dataframe)
        assert result.output["treatment"] is not None
        assert result.output["treatment"]["variable"] == "treat"

    @pytest.mark.asyncio
    async def test_includes_outcome_info(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        result = await agent._tool_get_overview(state_with_dataframe)
        assert result.output["outcome"] is not None
        assert result.output["outcome"]["variable"] == "outcome"

    @pytest.mark.asyncio
    async def test_error_when_no_dataframe(self, agent, state_with_dataframe):
        agent._df = None
        result = await agent._tool_get_overview(state_with_dataframe)
        assert result.status == ToolResultStatus.ERROR
        assert "not loaded" in result.error


class TestToolAnalyzeVariable:
    @pytest.mark.asyncio
    async def test_analyzes_numeric_variable(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        result = await agent._tool_analyze_variable(state_with_dataframe, variable="income")
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["type"] == "numeric"
        assert "mean" in result.output
        assert "std" in result.output
        assert "skewness" in result.output

    @pytest.mark.asyncio
    async def test_includes_normality_tests(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        result = await agent._tool_analyze_variable(
            state_with_dataframe, variable="income", include_normality_tests=True
        )
        assert result.status == ToolResultStatus.SUCCESS
        assert "normality_tests" in result.output

    @pytest.mark.asyncio
    async def test_analyzes_categorical_variable(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        result = await agent._tool_analyze_variable(state_with_dataframe, variable="gender")
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["type"] == "categorical"
        assert "top_values" in result.output

    @pytest.mark.asyncio
    async def test_variable_not_found(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        result = await agent._tool_analyze_variable(state_with_dataframe, variable="nonexistent")
        assert result.status == ToolResultStatus.ERROR
        assert "not found" in result.error


class TestToolDetectOutliers:
    @pytest.mark.asyncio
    async def test_detects_outliers(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        result = await agent._tool_detect_outliers(state_with_dataframe, method="both")
        assert result.status == ToolResultStatus.SUCCESS
        assert "variables_checked" in result.output

    @pytest.mark.asyncio
    async def test_iqr_method(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        result = await agent._tool_detect_outliers(state_with_dataframe, method="iqr")
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["method"] == "iqr"

    @pytest.mark.asyncio
    async def test_zscore_method(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        result = await agent._tool_detect_outliers(state_with_dataframe, method="zscore")
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["method"] == "zscore"


class TestToolComputeCorrelations:
    @pytest.mark.asyncio
    async def test_computes_correlations(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._eda_result = EDAResult()
        result = await agent._tool_compute_correlations(state_with_dataframe)
        assert result.status == ToolResultStatus.SUCCESS
        assert "n_variables" in result.output
        assert "high_correlations_count" in result.output

    @pytest.mark.asyncio
    async def test_spearman_method(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._eda_result = EDAResult()
        result = await agent._tool_compute_correlations(state_with_dataframe, method="spearman")
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["method"] == "spearman"

    @pytest.mark.asyncio
    async def test_custom_threshold(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._eda_result = EDAResult()
        result = await agent._tool_compute_correlations(state_with_dataframe, threshold=0.5)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["threshold"] == 0.5


class TestToolComputeVIF:
    @pytest.mark.asyncio
    async def test_computes_vif(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        agent._eda_result = EDAResult()
        result = await agent._tool_compute_vif(state_with_dataframe)
        assert result.status == ToolResultStatus.SUCCESS
        assert "n_covariates" in result.output
        assert "top_vif" in result.output

    @pytest.mark.asyncio
    async def test_vif_with_specific_covariates(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._eda_result = EDAResult()
        result = await agent._tool_compute_vif(
            state_with_dataframe, covariates=["age", "income", "education"]
        )
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["n_covariates"] == 3


class TestToolCheckBalance:
    @pytest.mark.asyncio
    async def test_checks_balance(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        agent._outcome_var = "outcome"
        agent._eda_result = EDAResult()
        result = await agent._tool_check_balance(state_with_dataframe)
        assert result.status == ToolResultStatus.SUCCESS
        assert "treatment_variable" in result.output
        assert "n_treated" in result.output
        assert "n_control" in result.output
        assert "n_imbalanced" in result.output

    @pytest.mark.asyncio
    async def test_balance_with_specific_covariates(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._treatment_var = "treat"
        agent._eda_result = EDAResult()
        result = await agent._tool_check_balance(
            state_with_dataframe, covariates=["age", "income"]
        )
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["n_covariates_checked"] == 2

    @pytest.mark.asyncio
    async def test_balance_no_treatment(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        agent._treatment_var = None
        agent._eda_result = EDAResult()
        result = await agent._tool_check_balance(state_with_dataframe)
        assert result.status == ToolResultStatus.SUCCESS
        assert "error" in result.output


class TestToolCheckMissing:
    @pytest.mark.asyncio
    async def test_no_missing(self, agent, sample_dataframe, state_with_dataframe):
        agent._df = sample_dataframe
        result = await agent._tool_check_missing(state_with_dataframe)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_missing"] is False

    @pytest.mark.asyncio
    async def test_with_missing(self, agent, state_with_dataframe):
        df = pd.DataFrame({
            "a": [1, 2, None, 4, 5],
            "b": [None, None, 3, 4, 5],
            "treat": [0, 1, 0, 1, 0],
        })
        agent._df = df
        agent._treatment_var = "treat"
        result = await agent._tool_check_missing(state_with_dataframe)
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["has_missing"] is True
        assert result.output["n_cols_with_missing"] == 2


class TestToolFinalize:
    @pytest.mark.asyncio
    async def test_finalizes_eda(self, agent, sample_dataframe, state_with_dataframe):
        """Finalize records the LLM's assessment and flips agent._finalized."""
        agent._df = sample_dataframe
        agent._eda_result = EDAResult()
        result = await agent._tool_finalize(
            state_with_dataframe,
            data_quality_score=85.0,
            key_findings=["Good data quality", "Some outliers"],
            data_quality_issues=["Minor outliers in income"],
            recommendations=["Use robust methods"],
            causal_readiness="ready",
        )
        assert result.status == ToolResultStatus.SUCCESS
        assert result.output["eda_finalized"] is True
        assert agent._finalized is True
        assert agent._final_result["data_quality_score"] == 85.0
        assert agent._final_result["causal_readiness"] == "ready"
