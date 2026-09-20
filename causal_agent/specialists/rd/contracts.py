"""Discontinuity lane contracts. What each node writes.

Lane-invariant artifacts (Contrast, Checks, Estimate, Refutation, Interpretation, Feasibility) live in
causal_agent.common.contracts. These are the ones only this lane needs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from causal_agent.common.contracts import Checks, Cited, Contrast, Interpretation

Estimand = Literal["effect_at_cutoff", "complier_effect_at_cutoff", "itt_at_cutoff"]


class Score(BaseModel):
    """The running variable, the cutoff, which side got the change, and who actually took it up."""

    column: str | None = Field(description="the column holding the score the cutoff was applied to; null if the notes state no cutoff rule on a numeric score")
    cutoff: float | None = Field(default=None, description="the cutoff value, in the score's own units, exactly as the notes state it")
    treated_side: Literal["above", "below"] = Field(default="above", description="which side of the cutoff got the change")
    cutoff_value_treated: bool = Field(
        default=True,
        description="true if a unit whose score equals the cutoff exactly got the change (an 'at or above' or 'at or below' rule); false if the rule is strict",
    )
    takeup_column: str | None = Field(
        default=None,
        description="the column recording whether the unit actually received the change, or null when the data records none and treatment is the cutoff rule itself",
    )
    takeup_level: str | None = Field(
        default=None, description="the exact value of the take-up column meaning the unit received the change; null when takeup_column is null"
    )
    reason: str
    cites: list[str]


class ShapeFacts(BaseModel):
    rows_file: int
    rows_score: int = Field(description="rows with a finite score; the density test's population")
    rows_primary: int = Field(description="rows complete on outcome, score, and take-up; every primary fit and falsification uses these")
    rows_covariates: int = Field(description="rows also complete on the balance-tested covariates; the adjusted fit and the continuity checks use these")
    n_left: int = Field(description="primary rows on the control side")
    n_right: int = Field(description="primary rows on the treated side")
    takeup_left: float | None = None
    takeup_right: float | None = None
    kind: Literal["sharp", "fuzzy"]
    distinct_scores: int
    duplicate_share_left: float
    duplicate_share_right: float
    rows_at_cutoff: int
    cutoff_shift: float = Field(
        description="how far the effective cutoff moved, in the score's units, so that units exactly at the cutoff fall on the side the notes declare; 0 when none sit there"
    )
    score_min: float
    score_max: float
    cluster_column: str | None = None


class CovariateRelation(BaseModel):
    """One column's standing at the cutoff. Three claims, each cited when true."""

    column: str
    predetermined: bool = Field(
        description="the column's value was fixed before the score was set and the change decided, so it must be continuous at the cutoff"
    )
    affected_by_treatment: bool = Field(
        description="the column's value could have been changed by the treatment; adjusting for it would remove part of the effect"
    )
    is_outcome_measure: bool = Field(description="another measure of the outcome, or a later outcome, not a covariate")
    reasons: list[Cited] = Field(description="one entry per claim marked true, each citing the card that supports it")


class Excluded(BaseModel):
    column: str
    why: str


class Covariates(BaseModel):
    balance_tested: list[str] = Field(default_factory=list, description="predetermined and numeric: each is tested for continuity at the cutoff")
    adjusted: list[str] = Field(default_factory=list, description="balance-tested and not affected: entered as covariates in the adjusted secondary fit")
    excluded: list[Excluded] = Field(default_factory=list)

    def render(self) -> str:
        lines = [f"balance-tested: {', '.join(self.balance_tested) or 'none'}", f"adjusted: {', '.join(self.adjusted) or 'none'}"]
        lines += [f"  excluded {x.column}: {x.why}" for x in self.excluded]
        return "\n".join(lines)


class DesignAssessment(BaseModel):
    action: Literal["proceed", "stop"]
    reason: str = Field(
        description="one or two sentences citing the flag numbers and, for density or covariate flags, the note that argues for or against them"
    )
    cites: list[str] = Field(default_factory=list, description="every flagged check address, plus the pack addresses the argument rests on")


class EstimatorPick(BaseModel):
    name: str = Field(description="one of the names offered")
    reason: str
    cites: list[str] = Field(default_factory=list)


class Bandwidths(BaseModel):
    h: float = Field(description="the estimation bandwidth of the primary spec")
    b: float = Field(description="the bias bandwidth of the primary spec")
    h_cer: float | None = Field(default=None, description="the coverage-error-optimal bandwidth, used for falsification; null under the support-points rule")
    rule: Literal["mse", "support_points"] = Field(
        default="mse",
        description="mse: the library's MSE-optimal choice; support_points: the score has few distinct values, so h is the distance that keeps a declared number of support points on each side",
    )
    n_h_left: int = 0
    n_h_right: int = 0


class Design(BaseModel):
    """The frozen design. Everything after this is deterministic execution."""

    contrast: Contrast
    score: Score
    shape: ShapeFacts
    covariates: Covariates
    checks: Checks
    estimator: str
    estimand: Estimand
    spec: dict
    also_run: list[str] = Field(default_factory=list)
    inference: str
    vce: str
    cluster: str | None = None
    bandwidths: Bandwidths
    sharp_bandwidth_used: bool = False
    placebos: list[str]
    target_units: str
    frozen_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def render(self) -> str:
        s, f = self.score, self.shape
        rule = f"{'at or ' if s.cutoff_value_treated else ''}{s.treated_side} {s.cutoff:g}"
        lines = [
            "DESIGN",
            f"  score        {s.column} treated when {rule}"
            + (f"; take-up {s.takeup_column} = {s.takeup_level!r}" if s.takeup_column else "; treatment is the cutoff rule itself")
            + f" ({s.reason})",
            f"  shape        {f.kind}; {f.rows_primary} rows ({f.n_left} control side, {f.n_right} treated side); {f.distinct_scores} distinct scores; "
            f"duplicates {f.duplicate_share_left:.0%}/{f.duplicate_share_right:.0%}; {f.rows_at_cutoff} at the cutoff"
            + (f", effective cutoff moved by {f.cutoff_shift:g}" if f.cutoff_shift else ""),
        ]
        if f.takeup_left is not None:
            lines.append(f"  take-up      {f.takeup_left:.2f} on the control side, {f.takeup_right:.2f} on the treated side")
        lines += ["  " + ln for ln in self.covariates.render().splitlines()]
        for r in self.checks.results:
            lines.append(f"  check        {r.level:4} {r.address}  {r.detail}")
        lines.append(f"  estimator    {self.estimator} ({self.estimand}): {self.spec}" + (f"   also {', '.join(self.also_run)}" if self.also_run else ""))
        lines.append(f"  inference    {self.inference}  vce {self.vce}" + (f"  cluster {self.cluster}" if self.cluster else ""))
        bw = self.bandwidths
        lines.append(
            f"  bandwidths   {bw.rule}: h {bw.h:.4g}  b {bw.b:.4g}"
            + (f"  h_cer {bw.h_cer:.4g}" if bw.h_cer is not None else "")
            + f"  effective rows {bw.n_h_left}/{bw.n_h_right}"
            + ("  (sharp bandwidth used: take-up does not vary on one side)" if self.sharp_bandwidth_used else "")
        )
        lines.append(f"  placebos     {', '.join(self.placebos) or 'none'}")
        lines.append(f"  target       {self.target_units}")
        lines.append(f"  frozen at    {self.frozen_at}")
        return "\n".join(lines)


class RDInterpretation(Interpretation):
    """The interpretation, with the fields the gate compares against the Design and the primary estimate."""

    estimand: Estimand = Field(
        description="which quantity the estimate is: effect_at_cutoff (sharp), complier_effect_at_cutoff (fuzzy), or itt_at_cutoff (effect of crossing the cutoff, whatever was taken up)"
    )
    bandwidth_stated: float = Field(description="the estimation bandwidth, copied from the design")
    n_left_stated: int = Field(description="effective rows on the control side, copied from the estimate")
    n_right_stated: int = Field(description="effective rows on the treated side, copied from the estimate")
    ci_low_stated: float = Field(description="lower end of the robust interval, copied from the estimate")
    ci_high_stated: float = Field(description="upper end of the robust interval, copied from the estimate")
