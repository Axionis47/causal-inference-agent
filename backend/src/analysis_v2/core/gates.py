"""Gate results: how a stage tells the orchestrator what happens next.

Hard gates control workflow progression. Soft gates control interpretation
strength only; they ride along as warnings and never move the workflow.
A weak, fragile, or inconclusive result is reported honestly, not rerun.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from .stages import AnalysisStage, is_earlier


class GateStatus(StrEnum):
    ADVANCE = "advance"  # move to the next stage on the spine
    NEEDS_USER = "needs_user"  # park the job; emit a confirmation card
    FAIL = "fail"  # terminal failure; hard_failures says why
    BACK = "back"  # deterministic hard-gate failure; rerun from back_to


class GateResult(BaseModel):
    """One stage's verdict on workflow progression.

    `hard_failures` are the deterministic reasons a FAIL or BACK happened.
    `soft_warnings` never change `status`; downstream agents read them to
    downgrade interpretation strength.
    """

    status: GateStatus = GateStatus.ADVANCE
    hard_failures: list[str] = Field(default_factory=list)
    soft_warnings: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    back_to: AnalysisStage | None = None

    @model_validator(mode="after")
    def _coherent(self) -> "GateResult":
        if self.status == GateStatus.BACK and self.back_to is None:
            raise ValueError("BACK gate result requires back_to")
        if self.status != GateStatus.BACK and self.back_to is not None:
            raise ValueError("back_to is only valid with status BACK")
        if self.status in (GateStatus.FAIL, GateStatus.BACK) and not self.hard_failures:
            raise ValueError(f"{self.status} gate result requires hard_failures")
        return self

    def validate_back_target(self, current: AnalysisStage) -> None:
        """A BACK jump must land strictly earlier on the spine."""
        if self.back_to is not None and not is_earlier(self.back_to, current):
            raise ValueError(
                f"back_to {self.back_to} is not earlier than current stage {current}"
            )

    @classmethod
    def advance(cls, *, soft_warnings: list[str] | None = None) -> "GateResult":
        return cls(status=GateStatus.ADVANCE, soft_warnings=soft_warnings or [])

    @classmethod
    def fail(cls, hard_failures: list[str], *, reasons: list[str] | None = None) -> "GateResult":
        return cls(status=GateStatus.FAIL, hard_failures=hard_failures, reasons=reasons or [])

    @classmethod
    def back(cls, back_to: AnalysisStage, hard_failures: list[str]) -> "GateResult":
        return cls(status=GateStatus.BACK, back_to=back_to, hard_failures=hard_failures)

    @classmethod
    def needs_user(cls, reasons: list[str]) -> "GateResult":
        return cls(status=GateStatus.NEEDS_USER, reasons=reasons)
