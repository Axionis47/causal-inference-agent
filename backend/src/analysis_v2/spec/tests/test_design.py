"""Lanes, eligibility, and the frozen method plan."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis_v2.spec import (
    DesignCandidate,
    EligibilityState,
    LaneEligibility,
    MethodLane,
    MethodPlan,
    ToolEligibility,
)


def test_method_lane_has_exactly_the_eight_lanes():
    assert {lane.value for lane in MethodLane} == {
        "observational",
        "matching",
        "did",
        "rdd",
        "iv",
        "time_series",
        "mediation",
        "survival",
    }


def test_lane_eligibility_requires_a_reason():
    with pytest.raises(ValidationError):
        LaneEligibility(lane=MethodLane.RDD, state=EligibilityState.DISABLED, reason="")
    ok = LaneEligibility(
        lane=MethodLane.RDD,
        state=EligibilityState.DISABLED,
        reason="no running variable or cutoff detected",
        missing_requirements=["running_variable", "cutoff_value"],
    )
    assert ok.missing_requirements == ["running_variable", "cutoff_value"]


def test_tool_eligibility_lookup_and_enabled_lanes_include_conditional():
    table = ToolEligibility(
        lanes=[
            LaneEligibility(
                lane=MethodLane.OBSERVATIONAL,
                state=EligibilityState.ENABLED,
                reason="outcome and treatment resolved",
            ),
            LaneEligibility(
                lane=MethodLane.MATCHING,
                state=EligibilityState.CONDITIONAL,
                reason="enabled if treatment is binary",
            ),
            LaneEligibility(
                lane=MethodLane.IV,
                state=EligibilityState.DISABLED,
                reason="no instrument column",
                missing_requirements=["instrument"],
            ),
        ]
    )
    assert table.for_lane(MethodLane.IV).state == EligibilityState.DISABLED
    assert table.for_lane(MethodLane.SURVIVAL) is None
    assert table.enabled_lanes() == [MethodLane.OBSERVATIONAL, MethodLane.MATCHING]


def test_design_candidate_and_method_plan_carry_lane_specific_settings():
    candidate = DesignCandidate(
        lane=MethodLane.RDD,
        design_label="sharp RDD at the $1000 recovery threshold",
        rationale="treatment assigned deterministically above the cutoff",
        required_fields={"running_variable": "expected_recovery_amount"},
        missing_requirements=["cutoff confirmation"],
    )
    assert candidate.lane == MethodLane.RDD

    plan = MethodPlan(
        lane=MethodLane.RDD,
        estimator="local linear regression",
        estimand="late",
        outcome="actual_recovery_amount",
        settings={"cutoff": 1000.0, "bandwidth": 250.0},
    )
    loaded = MethodPlan.model_validate(plan.model_dump(mode="json"))
    assert loaded.settings["cutoff"] == 1000.0
