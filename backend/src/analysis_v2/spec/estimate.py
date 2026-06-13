"""Estimate results (S7): the typed output of the one executed method lane."""
from __future__ import annotations

from pydantic import BaseModel, Field

from .design import MethodLane

# Estimators whose identification IS backdoor adjustment, so the DAG's
# identifiability verdict is the right honesty criterion for them. DiD, RDD, IV,
# survival, and mediation carry their own identifying assumptions and are not
# bound by the backdoor verdict. Shared by claim_critic (caps the claim) and the
# flow audit (checks the claim) so the two cannot disagree.
BACKDOOR_IDENTIFIED_ESTIMATORS = frozenset({
    "regression_adjustment",
    "propensity_matching",
})


class EffectEstimate(BaseModel):
    """One estimand from the executed lane (a lane may emit a small set,
    e.g. mediation's direct/indirect/total, never competing methods)."""

    estimand: str = Field(min_length=1, max_length=60)
    estimate: float
    std_error: float | None = Field(default=None, ge=0.0)
    ci_lower: float | None = None
    ci_upper: float | None = None
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    unit: str | None = Field(default=None, max_length=120)
    interpretation: str = Field(default="", max_length=1000)


class EstimateResult(BaseModel):
    lane: MethodLane
    estimator: str = Field(min_length=1, max_length=120)
    effects: list[EffectEstimate] = Field(min_length=1)
    n_rows_used: int = Field(ge=0)
    outcome: str = Field(min_length=1)
    treatment: str | None = None
    covariates_used: list[str] = Field(default_factory=list)
    model_summary_artifact_id: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def primary(self) -> EffectEstimate:
        return self.effects[0]
