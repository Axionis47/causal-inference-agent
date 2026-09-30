"""Discontinuity lane contracts. What each node writes, and the rungs of the ladder the design is climbed on; the records every
ladder shares (the threats, heterogeneity, what a rung would not guess) come from the harness.

Lane-invariant artifacts (Contrast, Checks, Estimate, Refutation, Interpretation, Feasibility) live in
causal_agent.common.contracts. These are the ones only this lane needs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from causal_agent.common.contracts import Checks, Cited, Contrast, Departure, Interpretation
from causal_agent.lane.ladder import Heterogeneity, LadderBase, Threats, Unsure

Estimand = Literal["effect_at_cutoff", "complier_effect_at_cutoff", "itt_at_cutoff"]


class Score(BaseModel):
    """Rung 0: the running variable, the cutoff, which side got the change, and who actually took it up."""

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
    by: Literal["pack", "judgement"] = "judgement"

    def lines(self) -> list[tuple[str, str]]:
        if not self.column or self.cutoff is None:
            return [("ladder:score.rule", "no cutoff rule on a numeric score")]
        rule = f"{'at or ' if self.cutoff_value_treated else ''}{self.treated_side} {self.cutoff:g}"
        return [
            ("ladder:score.rule", f"{self.column} treated when {rule} (set by the {self.by})"),
            ("ladder:score.takeup", f"{self.takeup_column} = {self.takeup_level!r}" if self.takeup_column else "none recorded: the rule is the change"),
        ]


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

    def lines(self) -> list[tuple[str, str]]:
        return [
            ("ladder:shape.sides", f"{self.n_left} rows on the control side, {self.n_right} on the treated side; {self.distinct_scores} distinct scores"),
            (
                "ladder:shape.kind",
                f"{self.kind}"
                + (f"; take-up {self.takeup_left:.2f} control side, {self.takeup_right:.2f} treated side" if self.takeup_left is not None else ""),
            ),
        ]


LineRisk = Literal["manipulation", "other_change_at_line", "score_set_after", "cutoff_known_in_advance"]


class Risk(Cited):
    """One thing that could break the comparison at the line, named from the story: units moving their own score (manipulation),
    something else switching at the same line (other_change_at_line), a score set after the change was decided (score_set_after),
    a cutoff units knew before their score was fixed (cutoff_known_in_advance)."""

    name: LineRisk


class Line(BaseModel):
    """Rung 2: whether the line is clean, argued from the story: what set the score, whether units could move it, what else
    switches there."""

    clean: bool = Field(
        description="whether the story supports that the score was set before the decision, could not be moved, and nothing else switches at the line"
    )
    why: str = Field(description="one or two sentences from the story and the facts")
    risks: list[Risk] = Field(default_factory=list, description="every risk the story raises, each cited; none when it raises none")
    cites: list[str] = Field(default_factory=list)
    unsure: list[Unsure] = Field(default_factory=list)
    by: Literal["judgement"] = "judgement"

    def lines(self) -> list[tuple[str, str]]:
        return [("ladder:line.clean", "yes" if self.clean else "no"), ("ladder:line.why", self.why)] + [
            (f"ladder:line.risk.{r.name}", r.reason) for r in self.risks
        ]


class CovariateRelation(BaseModel):
    """One column's standing at the cutoff, read with every other candidate in view."""

    column: str
    predetermined: bool = Field(
        description="the column's value was fixed before the score was set and the change decided, so it must be continuous at the cutoff"
    )
    affected_by_treatment: bool = Field(
        description="the column's value could have been changed by the treatment; adjusting for it would remove part of the effect"
    )
    is_outcome_measure: bool = Field(description="another measure of the outcome, or a later outcome, not a covariate")
    modifier_candidate: bool = Field(
        default=False, description="a predetermined characteristic the effect at the cutoff could plausibly differ by, per the story"
    )
    reasons: list[Cited] = Field(description="one entry per claim marked true, each citing the card that supports it")
    departures: list[Departure] = Field(default_factory=list)

    def word(self) -> str:
        base = (
            "another measure of the outcome"
            if self.is_outcome_measure
            else "changed by the treatment"
            if self.affected_by_treatment
            else "fixed before the line"
            if self.predetermined
            else "not fixed before the line"
        )
        return base + ("; a candidate modifier" if self.modifier_candidate else "")


class CovariateRoles(BaseModel):
    """Rung 3: every candidate covariate placed together."""

    items: list[CovariateRelation] = Field(description="one per column listed, all of them")
    unsure: list[Unsure] = Field(default_factory=list, description="what you would not guess, with why; the answer above still stands")

    def lines(self) -> list[tuple[str, str]]:
        return [(f"ladder:covariates.{r.column}", r.word()) for r in self.items]


class Bandwidth(BaseModel):
    """How wide the window around the line is, by code at the freeze."""

    h: float
    rule: str
    why: str

    def lines(self) -> list[tuple[str, str]]:
        return [("ladder:bandwidth.h", f"{self.h:.4g} in the score's units ({self.rule})"), ("ladder:bandwidth.why", self.why)]


class Ladder(LadderBase):
    """The rungs climbed so far: the score and the line, the shape, whether the line is clean, the covariates, where the effect
    could differ, the threats, the window. A rung reads the rungs below it; every line has an address."""

    ORDER: ClassVar[tuple[str, ...]] = ("score", "shape", "line", "covariates", "heterogeneity", "threats", "bandwidth")

    score: Score | None = None
    shape: ShapeFacts | None = None
    line: Line | None = None
    covariates: CovariateRoles | None = None
    heterogeneity: Heterogeneity | None = None
    threats: Threats | None = None
    bandwidth: Bandwidth | None = None


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
    modifiers: list[str] = Field(
        default_factory=list, description="the predetermined characteristics the effect at the cutoff is also estimated within, level by level"
    )
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
        lines.append(f"  modifiers    {', '.join(self.modifiers) or 'none'}")
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
