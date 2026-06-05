"""Validation tests for the EDAResult typed output model."""

from src.analysis.agents.eda import EDAResult


class TestEDAResult:
    def test_defaults(self):
        """Every field has a sensible default so the agent can write incrementally."""
        r = EDAResult()
        assert r.correlation_matrix == {}
        assert r.high_correlations == []
        assert r.distribution_stats == {}
        assert r.outliers == {}
        assert r.vif_scores == {}
        assert r.multicollinearity_warnings == []
        assert r.covariate_balance == {}
        assert r.balance_summary == ""
        assert r.data_quality_score == 0.0
        assert r.data_quality_issues == []
        assert r.summary_table == {}
        assert r.plot_captions == {}

    def test_plot_captions_round_trip(self):
        """plot_captions persist through model_dump / model_validate."""
        r1 = EDAResult(
            plot_captions={
                "love_plot": "Three covariates exceed SMD 0.1; the rest are balanced.",
                "correlation_heatmap": "age and educ correlate at 0.42; no |r| > 0.7.",
            }
        )
        r2 = EDAResult.model_validate(r1.model_dump())
        assert r2.plot_captions == r1.plot_captions

    def test_full_construction(self):
        r = EDAResult(
            data_quality_score=87.5,
            data_quality_issues=["minor outliers"],
            balance_summary="All covariates well-balanced",
            vif_scores={"age": 1.4, "income": 1.2},
        )
        assert r.data_quality_score == 87.5
        assert r.vif_scores["age"] == 1.4
        assert r.balance_summary == "All covariates well-balanced"

    def test_round_trip(self):
        """model_dump / model_validate must round-trip cleanly for persistence."""
        r1 = EDAResult(
            data_quality_score=72.0,
            data_quality_issues=["mild skew"],
            high_correlations=[{"var1": "age", "var2": "income", "correlation": 0.81}],
        )
        r2 = EDAResult.model_validate(r1.model_dump())
        assert r2 == r1
