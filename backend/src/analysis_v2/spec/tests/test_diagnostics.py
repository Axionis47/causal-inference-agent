"""Diagnostics and sensitivity: rerun only for invalid setups."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis_v2.spec import (
    CheckStatus,
    DiagnosticCheck,
    DiagnosticsResult,
    RobustnessStatus,
    SensitivityResult,
)


def _overlap_check(status: CheckStatus = CheckStatus.WARNING) -> DiagnosticCheck:
    return DiagnosticCheck(
        name="propensity_overlap",
        status=status,
        detail="3% of treated units outside common support",
        metrics={"share_off_support": 0.03},
    )


def test_rerun_required_must_state_the_invalid_setup_reason():
    with pytest.raises(ValidationError):
        SensitivityResult(
            robustness=RobustnessStatus.FRAGILE,
            confidence_reason="estimate flips sign under small confounding",
            rerun_required=True,
        )
    ok = SensitivityResult(
        robustness=RobustnessStatus.NOT_SUPPORTED,
        confidence_reason="post-treatment variable was used as a control",
        rerun_required=True,
        rerun_reason="bad control: post-treatment variable in the adjustment set",
    )
    assert ok.rerun_required


def test_rerun_reason_without_rerun_required_is_rejected():
    with pytest.raises(ValidationError):
        SensitivityResult(
            robustness=RobustnessStatus.ROBUST,
            confidence_reason="stable across bandwidths",
            rerun_reason="leftover reason",
        )


def test_a_fragile_result_does_not_need_a_rerun():
    fragile = SensitivityResult(
        checks=[_overlap_check()],
        robustness=RobustnessStatus.FRAGILE,
        confidence_reason="E-value 1.3: weak unmeasured confounding could explain it",
    )
    assert not fragile.rerun_required
    assert fragile.robustness == RobustnessStatus.FRAGILE


def test_diagnostics_result_lookup_by_name():
    result = DiagnosticsResult(
        checks=[_overlap_check(), _overlap_check(CheckStatus.PASS).model_copy(
            update={"name": "covariate_balance"}
        )],
        overall=CheckStatus.WARNING,
    )
    assert result.check("propensity_overlap").metrics["share_off_support"] == 0.03
    assert result.check("nonexistent") is None
