"""Targeted EDA contracts (S4): the plan, per-check results, the summary.

EDA describes the data's story for the detected design. It never makes a
causal effect claim; checks that look like estimates (raw mean difference,
pre-trends) are descriptive inputs to the plan gate and diagnostics.
"""
from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from .design import MethodLane


class EDACheckStatus(StrEnum):
    OK = "ok"
    WARNING = "warning"
    FAILED = "failed"  # the check itself could not run
    SKIPPED = "skipped"  # not applicable; reason says why


class EDACheck(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    status: EDACheckStatus
    detail: str = Field(default="", max_length=1000)
    metrics: dict[str, float] = Field(default_factory=dict)
    artifact_ids: list[str] = Field(default_factory=list)


class EDAPlan(BaseModel):
    """Which checks run: the base recipe plus the design-targeted recipe."""

    target_lane: MethodLane | None = None
    base_checks: list[str] = Field(default_factory=list)
    targeted_checks: list[str] = Field(default_factory=list)
    rationale: str = Field(default="", max_length=1000)


class EDASummary(BaseModel):
    plan: EDAPlan
    checks: list[EDACheck] = Field(default_factory=list)
    story: str = Field(default="", max_length=4000)  # public narrative summary
    warnings: list[str] = Field(default_factory=list)
    usable_sample_size: int | None = Field(default=None, ge=0)

    def check(self, name: str) -> EDACheck | None:
        for c in self.checks:
            if c.name == name:
                return c
        return None
