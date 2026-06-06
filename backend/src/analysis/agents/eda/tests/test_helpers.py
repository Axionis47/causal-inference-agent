"""Tests for the pure helpers: auto-finalize and EDAResult population."""

from src.analysis.agents.base import EDAResult


class TestAutoFinalize:
    """Heuristic fallback when the agent never calls finalize_eda."""

    def test_auto_finalize_no_issues(self, agent, sample_dataframe):
        agent._df = sample_dataframe
        agent._missing_analysis = {"has_missing": False}
        agent._outlier_results = {}
        agent._vif_results = {}
        agent._balance_results = {}
        result = agent._auto_finalize()
        assert result["data_quality_score"] >= 80
        assert result["causal_readiness"] == "ready"

    def test_auto_finalize_with_missing_data(self, agent, sample_dataframe):
        agent._df = sample_dataframe
        agent._missing_analysis = {"has_missing": True, "total_missing_pct": 15.0}
        agent._outlier_results = {}
        agent._vif_results = {}
        agent._balance_results = {}
        result = agent._auto_finalize()
        assert result["data_quality_score"] < 100
        assert "missing data" in " ".join(result["data_quality_issues"]).lower()

    def test_auto_finalize_with_outliers(self, agent, sample_dataframe):
        agent._df = sample_dataframe
        agent._missing_analysis = {"has_missing": False}
        agent._outlier_results = {
            "col1": {"iqr_outliers": 5},
            "col2": {"iqr_outliers": 3},
            "col3": {"iqr_outliers": 4},
        }
        agent._vif_results = {}
        agent._balance_results = {}
        result = agent._auto_finalize()
        assert result["data_quality_score"] < 100
        assert "outlier" in " ".join(result["recommendations"]).lower()

    def test_auto_finalize_with_multicollinearity(self, agent, sample_dataframe):
        agent._df = sample_dataframe
        agent._missing_analysis = {"has_missing": False}
        agent._outlier_results = {}
        agent._vif_results = {"var1": 12.0, "var2": 8.0}
        agent._balance_results = {}
        result = agent._auto_finalize()
        assert result["data_quality_score"] < 100
        assert "multicollinearity" in " ".join(result["data_quality_issues"]).lower()

    def test_auto_finalize_with_imbalance(self, agent, sample_dataframe):
        agent._df = sample_dataframe
        agent._missing_analysis = {"has_missing": False}
        agent._outlier_results = {}
        agent._vif_results = {}
        agent._balance_results = {
            "age": {"is_balanced": False, "smd": 0.3},
            "income": {"is_balanced": False, "smd": 0.25},
            "education": {"is_balanced": True, "smd": 0.05},
        }
        result = agent._auto_finalize()
        assert result["data_quality_score"] < 100
        assert "imbalanced" in " ".join(result["data_quality_issues"]).lower()


class TestPopulateEDAResult:
    def test_populates_from_final_result(self, agent):
        """Populate stitches finalize output and per-tool evidence into EDAResult."""
        agent._eda_result = EDAResult()
        agent._analyzed_distributions = {"age": {"mean": 40}}
        agent._outlier_results = {"income": {"iqr_outliers": 5}}
        agent._vif_results = {"age": 1.5}
        agent._balance_results = {"age": {"smd": 0.05, "is_balanced": True}}

        final_result = {
            "data_quality_score": 75.0,
            "key_findings": ["Finding 1"],
            "data_quality_issues": ["Issue 1"],
            "recommendations": ["Rec 1"],
            "causal_readiness": "needs_attention",
        }

        agent._populate_eda_result(final_result)

        assert agent._eda_result.data_quality_score == 75.0
        assert agent._eda_result.data_quality_issues == ["Issue 1"]
        assert agent._eda_result.distribution_stats == {"age": {"mean": 40}}

    def test_populates_plot_captions_from_final_result(self, agent):
        """Captions written by finalize_eda land on the EDAResult slot."""
        agent._eda_result = EDAResult()
        agent._analyzed_distributions = {}
        agent._outlier_results = {}
        agent._vif_results = {}
        agent._balance_results = {}

        agent._populate_eda_result({
            "data_quality_score": 80.0,
            "key_findings": [],
            "data_quality_issues": [],
            "recommendations": [],
            "causal_readiness": "ready",
            "plot_captions": {
                "love_plot": "Two covariates exceed SMD 0.1.",
                "correlation_heatmap": "Highest |r| is 0.42 (age, educ).",
            },
        })

        assert agent._eda_result.plot_captions["love_plot"].startswith("Two")
        assert "0.42" in agent._eda_result.plot_captions["correlation_heatmap"]

    def test_populates_empty_captions_when_finalize_omitted_them(self, agent):
        """When finalize was called without plot_captions, the slot stays an empty dict."""
        agent._eda_result = EDAResult()
        agent._analyzed_distributions = {}
        agent._outlier_results = {}
        agent._vif_results = {}
        agent._balance_results = {}

        agent._populate_eda_result({
            "data_quality_score": 80.0,
            "key_findings": [],
            "data_quality_issues": [],
            "recommendations": [],
            "causal_readiness": "ready",
        })

        assert agent._eda_result.plot_captions == {}
