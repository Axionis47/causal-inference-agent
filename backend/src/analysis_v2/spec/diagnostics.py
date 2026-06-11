"""Diagnostics and sensitivity contracts (S8).

Sensitivity asks: if the assumptions are slightly wrong, does the result
still hold? The answer changes confidence and language, never the workflow
direction. `rerun_required` exists only for invalid setups (bad control,
leakage, wrong treatment construction), not weak results.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class CheckStatus(StrEnum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class DiagnosticCheck(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    status: CheckStatus
    detail: str = Field(default="", max_length=1000)
    metrics: dict[str, float] = Field(default_factory=dict)
    artifact_ids: list[str] = Field(default_factory=list)


class DiagnosticsResult(BaseModel):
    checks: list[DiagnosticCheck] = Field(default_factory=list)
    overall: CheckStatus = CheckStatus.PASS
    summary: str = Field(default="", max_length=2000)

    def check(self, name: str) -> DiagnosticCheck | None:
        for c in self.checks:
            if c.name == name:
                return c
        return None


class RobustnessStatus(StrEnum):
    ROBUST = "robust"
    FRAGILE = "fragile"
    NOT_SUPPORTED = "not_supported"


class SensitivityResult(BaseModel):
    checks: list[DiagnosticCheck] = Field(default_factory=list)
    robustness: RobustnessStatus
    confidence_reason: str = Field(min_length=1, max_length=1000)
    rerun_required: bool = False
    rerun_reason: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def _rerun_only_for_invalid_setup(self) -> "SensitivityResult":
        if self.rerun_required and not self.rerun_reason:
            raise ValueError("rerun_required must state the invalid-setup reason")
        if not self.rerun_required and self.rerun_reason:
            raise ValueError("rerun_reason is only valid with rerun_required")
        return self
