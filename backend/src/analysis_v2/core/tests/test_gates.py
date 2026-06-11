"""GateResult invariants: BACK/FAIL must justify themselves."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis_v2.core import AnalysisStage, GateResult, GateStatus


def test_back_gate_requires_back_to_and_hard_failures():
    with pytest.raises(ValidationError):
        GateResult(status=GateStatus.BACK, hard_failures=["bad column"])
    with pytest.raises(ValidationError):
        GateResult(status=GateStatus.BACK, back_to=AnalysisStage.S1_INTAKE_PARSED)
    ok = GateResult(
        status=GateStatus.BACK,
        back_to=AnalysisStage.S1_INTAKE_PARSED,
        hard_failures=["selected outcome column missing"],
    )
    assert ok.back_to == AnalysisStage.S1_INTAKE_PARSED


def test_fail_gate_requires_hard_failures():
    with pytest.raises(ValidationError):
        GateResult(status=GateStatus.FAIL)
    ok = GateResult.fail(["treatment has no variation"])
    assert ok.status == GateStatus.FAIL


def test_back_to_is_rejected_on_non_back_results():
    with pytest.raises(ValidationError):
        GateResult(status=GateStatus.ADVANCE, back_to=AnalysisStage.S1_INTAKE_PARSED)


def test_back_target_must_be_strictly_earlier_than_current_stage():
    gate = GateResult.back(
        AnalysisStage.S7_METHOD_EXECUTED, ["notebook execution failure"]
    )
    with pytest.raises(ValueError):
        gate.validate_back_target(AnalysisStage.S7_METHOD_EXECUTED)
    with pytest.raises(ValueError):
        gate.validate_back_target(AnalysisStage.S5_PLAN_CRITIQUED)
    gate.validate_back_target(AnalysisStage.S11_NOTEBOOK_VERIFIED)  # earlier: ok


def test_factory_helpers_build_valid_results():
    assert GateResult.advance().status == GateStatus.ADVANCE
    warned = GateResult.advance(soft_warnings=["poor overlap"])
    assert warned.soft_warnings == ["poor overlap"]
    asked = GateResult.needs_user(["RDD cutoff needs confirmation"])
    assert asked.status == GateStatus.NEEDS_USER
    assert asked.reasons == ["RDD cutoff needs confirmation"]
