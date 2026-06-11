"""EstimateResult, EDA, and profile summaries: shape and bounds."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis_v2.spec import (
    ColumnProfile,
    EDACheck,
    EDACheckStatus,
    EDAPlan,
    EDASummary,
    EffectEstimate,
    EstimateResult,
    MethodLane,
    ProfileSummary,
)


def test_estimate_result_requires_at_least_one_effect():
    with pytest.raises(ValidationError):
        EstimateResult(
            lane=MethodLane.OBSERVATIONAL,
            estimator="ols",
            effects=[],
            n_rows_used=100,
            outcome="re78",
        )


def test_p_value_outside_unit_interval_is_rejected():
    with pytest.raises(ValidationError):
        EffectEstimate(estimand="ate", estimate=1794.34, p_value=1.5)


def test_primary_is_the_first_effect():
    result = EstimateResult(
        lane=MethodLane.MEDIATION,
        estimator="product of coefficients",
        effects=[
            EffectEstimate(estimand="total", estimate=0.60),
            EffectEstimate(estimand="indirect", estimate=0.30),
            EffectEstimate(estimand="direct", estimate=0.30),
        ],
        n_rows_used=2000,
        outcome="y",
        treatment="t",
    )
    assert result.primary.estimand == "total"


def test_eda_summary_lookup_and_did_recipe_shape():
    summary = EDASummary(
        plan=EDAPlan(
            target_lane=MethodLane.DID,
            base_checks=["missingness", "outcome_distribution"],
            targeted_checks=["pre_period_trends", "group_sizes_by_time"],
        ),
        checks=[
            EDACheck(
                name="pre_period_trends",
                status=EDACheckStatus.WARNING,
                detail="pre-trends diverge slightly in 1991",
                metrics={"pre_trend_gap": 0.4},
            )
        ],
        usable_sample_size=384,
    )
    assert summary.check("pre_period_trends").status == EDACheckStatus.WARNING
    assert summary.check("absent") is None
    assert "group_sizes_by_time" in summary.plan.targeted_checks


def test_profile_summary_column_lookup_and_missing_fraction_bounds():
    with pytest.raises(ValidationError):
        ColumnProfile(
            name="re78",
            dtype="float64",
            semantic_type="numeric",
            missing_count=3,
            missing_fraction=1.2,
            n_unique=500,
        )
    profile = ProfileSummary(
        n_rows=614,
        n_columns=2,
        columns=[
            ColumnProfile(
                name="treat",
                dtype="int64",
                semantic_type="binary",
                missing_count=0,
                missing_fraction=0.0,
                n_unique=2,
            )
        ],
        id_like_columns=["Unnamed: 0"],
    )
    assert profile.column("treat").semantic_type == "binary"
    assert profile.column("absent") is None
    assert profile.column_names() == ["treat"]
