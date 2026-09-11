"""New approvals describe numerical work; historical displays stay read-only."""
from __future__ import annotations

from typing import Any

import polars as pl
import pytest
from pydantic import ValidationError

from causal.analysis import interface as api
from causal.analysis.contracts import (
    AnalysisEvidence,
    ApprovedPlan,
    BoundaryError,
    CompiledPlan,
    PlotPlan,
)
from causal.analysis.integration.numerical import ExecutionContext, prepare
from causal.analysis.methods.randomized.estimation import RandomizedExperimentAdapter
from causal.analysis.tests.support.interface import approve, design


def approved() -> tuple[ApprovedPlan, pl.DataFrame]:
    data = pl.DataFrame({"id": list(range(24)), "arm": ["control", "treated"] * 12,
        "y": [2.0 + 3.0 * (i % 2) + ((i * 7) % 11) / 10 for i in range(24)]})
    plan = approve(data, design("randomized", {"assignment_mechanism": "individual_randomized"}),
        {"method": "randomized", "treatment_column": "arm", "unit_column": "id",
         "treated_value": "treated", "comparator_value": "control", "estimator": "difference_in_means"})
    return plan, data


def old_plot() -> PlotPlan:
    return PlotPlan(plot_id="old-effect-chart", dependency="", applicability="applicable",
                    explanation="Historical display prescription.")


def test_new_public_execution_has_no_display_builder_and_preserves_the_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, data = approved()

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a display builder ran inside numerical analysis")

    monkeypatch.setattr(RandomizedExperimentAdapter, "figures", forbidden, raising=False)
    result = api.execute(plan, data)
    assert plan.plan.schema_version == "analysis-plan.v3" and plan.plan.plots == ()
    assert result.schema_version == "analysis-evidence.v2"
    assert result.plots == result.plotting_data == () and result.supporting_data
    assert result.primary.status == "computed"
    assert result.provenance.seed == prepare(plan.plan, data).legacy_plan.seed == 17
    assert prepare(plan.plan, data).legacy_plan.figure_builder_ids == ()


def test_raw_support_values_and_failed_support_remain_visible_without_display_coordinates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, data = approved()
    monkeypatch.setattr(ExecutionContext, "collect", lambda self, result: (
        {"scientific_summary": {"mean": 3.125, "n": 24, "unavailable": float("nan")}},
        {"failed_support": "The supporting calculation failed."}))
    result = api.execute(plan, data)
    support = {row.computation_id: row for row in result.supporting_data}
    assert result.primary.status == "computed"
    assert {item.name: item.value for item in support["scientific_summary"].measurements} == {
        "mean": 3.125, "n": 24, "unavailable": None}
    assert support["scientific_summary"].status == "unavailable"
    assert support["failed_support"].status == "failed"
    assert "supporting calculation failed" in support["failed_support"].explanation
    assert "NaN" not in result.model_dump_json()


def test_historical_plan_can_be_read_but_requires_fresh_approval_before_execution() -> None:
    plan, data = approved()
    historical = CompiledPlan.model_validate({**plan.plan.model_dump(),
        "schema_version": "analysis-plan.v1", "plots": (old_plot(),)})
    assert historical.plots == (old_plot(),)
    old_approval = ApprovedPlan(plan=historical, approved_hash=historical.plan_hash,
                                approved_by="reviewer", approval_reference="approval:historical")
    with pytest.raises(BoundaryError, match="Historical plans"):
        api.execute(old_approval, data)
    with pytest.raises(ValidationError, match="historical v1"):
        CompiledPlan.model_validate({**plan.plan.model_dump(), "plots": (old_plot(),)})


def test_historical_evidence_stays_readable_but_new_outputs_cannot_author_plot_points() -> None:
    plan, data = approved()
    result = api.execute(plan, data)
    legacy_points = [{"plot_id": "old-effect-chart", "columns": ("x", "y"), "rows": ((0, 3.0),)}]
    historical = AnalysisEvidence.model_validate({**result.model_dump(),
        "schema_version": "analysis-evidence.v1", "plotting_data": legacy_points})
    assert historical.plotting_data[0].rows == ((0, 3.0),)
    with pytest.raises(ValidationError, match="historical v1"):
        AnalysisEvidence.model_validate({**result.model_dump(), "plotting_data": legacy_points})
