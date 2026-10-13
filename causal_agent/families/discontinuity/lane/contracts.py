"""RD evidence, reasoning and immutable execution contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from causal_agent.common.contracts import Checks, Cited, Contrast, Departure, FrozenDesign
from causal_agent.lane.ladder import LadderBase, Unsure

Estimand = Literal["effect_at_cutoff", "complier_effect_at_cutoff", "itt_at_cutoff", "effect_in_window", "complier_effect_in_window", "itt_in_window"]
LineRisk = Literal["manipulation", "other_change_at_line", "score_set_after", "cutoff_known_in_advance"]


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
    clusters: int | None = Field(default=None, description="distinct values of the cluster column among the primary rows; null when none is declared")

    def lines(self) -> list[tuple[str, str]]:
        return [
            ("ladder:shape.sides", f"{self.n_left} rows on the control side, {self.n_right} on the treated side; {self.distinct_scores} distinct scores"),
            (
                "ladder:shape.kind",
                f"{self.kind}"
                + (f"; take-up {self.takeup_left:.2f} control side, {self.takeup_right:.2f} treated side" if self.takeup_left is not None else ""),
            ),
        ]


class BinomialWindow(BaseModel):
    """One window either side of the line and how the rows split across it: under no manipulation the split is a coin toss."""

    width: float
    n_left: int
    n_right: int
    p: float


class DensityFacts(BaseModel):
    """Rung 1, the evidence for the line, by code before the line is judged: the density test at the line, the binomial split
    in nested windows, the histogram either side, the mass points, and whether the rows were drawn by side (which silences the
    test). Every line is addressed so the line rung must read it and its gate can require citing it."""

    status: Literal["tested", "uninformative", "not_computable"]
    reason: str = ""
    p: float | None = None
    t: float | None = None
    hat_left: float | None = None
    hat_right: float | None = None
    h_left: float | None = None
    h_right: float | None = None
    n_eff_left: int = 0
    n_eff_right: int = 0
    windows: list[BinomialWindow] = Field(default_factory=list)
    histogram: list[tuple[float, float, int]] = Field(default_factory=list, description="(low edge, high edge, rows), the line as an edge")
    mass_share_left: float = 0.0
    mass_share_right: float = 0.0
    sampled_by_side: bool = False
    flagged: bool = Field(default=False, description="the test or the smallest window says the rows bunch on one side, by the declared thresholds")

    def lines(self) -> list[tuple[str, str]]:
        if self.status == "uninformative":
            test = "uninformative: the rows were drawn by side of the line, so their density says nothing about manipulation"
        elif self.status == "not_computable":
            test = f"not computable: {self.reason}"
        else:
            test = (
                f"density {self.hat_left:.3g} just below the line, {self.hat_right:.3g} just above; test p = {self.p:.3g} on "
                f"{self.n_eff_left}/{self.n_eff_right} effective rows" + ("; the rows bunch on one side" if self.flagged else "; no sign of bunching")
            )
        out = [("ladder:density.test", test)]
        if self.windows:
            out.append(
                (
                    "ladder:density.windows",
                    "rows below | above the line within nested windows, and the coin-toss p: "
                    + "; ".join(f"±{w.width:.3g}: {w.n_left} | {w.n_right} (p = {w.p:.2g})" for w in self.windows)
                    + " (only the narrowest window speaks to bunching; a wider window's split follows the slope of the density)",
                )
            )
        if self.histogram:
            below = [n for lo, hi, n in self.histogram if hi <= 0]
            above = [n for lo, hi, n in self.histogram if lo >= 0]
            out.append(
                ("ladder:density.histogram", f"rows per bin, control side then treated side: {', '.join(map(str, below))} | {', '.join(map(str, above))}")
            )
        out.append(
            (
                "ladder:density.mass_points",
                f"duplicated scores: {self.mass_share_left:.0%} on the control side, {self.mass_share_right:.0%} on the treated side",
            )
        )
        return out


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

    def lines(self) -> list[tuple[str, str]]:
        return [("ladder:line.clean", "yes" if self.clean else "no"), ("ladder:line.why", self.why)] + [
            (f"ladder:line.risk.{r.name}", r.reason) for r in self.risks
        ]


class BalanceItem(BaseModel):
    """One candidate's standing at the line before it is placed: for a number, the jump at the line from a sharp local linear fit
    at the coverage-error width with its robust p; for a category, the difference in the share of its commonest level between the
    two sides within the density's window."""

    column: str
    how: Literal["jump", "share", "untested"]
    jump: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p: float | None = None
    n_left: int = 0
    n_right: int = 0
    width: float | None = None
    level: str | None = None
    error: str | None = None

    def flagged(self, threshold: float) -> bool:
        return self.p is not None and self.p < threshold

    def line(self, threshold: float) -> str:
        if self.how == "untested":
            return f"could not be tested ({self.error})"
        verdict = "; differs at the line" if self.flagged(threshold) else "; no detected difference at the line"
        if self.how == "share":
            return f"share of {self.level!r} differs by {self.jump:+.2f} between the sides within {self.width:.3g} of the line (p = {self.p:.2g}, {self.n_left}/{self.n_right} rows){verdict}"
        return f"jump {self.jump:.3g} at the line (robust p = {self.p:.3g}, {self.n_left}/{self.n_right} rows within {self.width:.3g}){verdict}"


class BalanceFacts(BaseModel):
    """Rung 3's evidence, by code before the covariates are placed: every candidate's standing at the line. A column fixed before
    the line should not differ across it; a detected discontinuity contests continuity, not known measurement timing."""

    items: list[BalanceItem] = Field(default_factory=list)
    threshold: float = 0.05

    def item(self, column: str) -> BalanceItem | None:
        return next((i for i in self.items if i.column == column), None)

    def lines(self) -> list[tuple[str, str]]:
        return [(f"ladder:balance.{i.column}", i.line(self.threshold)) for i in self.items] or [
            ("ladder:balance.none", "no candidate column to test at the line")
        ]


class CovariateRelation(BaseModel):
    """One column's standing at the cutoff, read with every other candidate in view."""

    column: str
    predetermined: bool = Field(
        description="the column's value was fixed before the score was set and the change decided, so continuity is an identifying assumption to assess"
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


class Ladder(LadderBase):
    """The rungs climbed so far: the score and the line, the shape, the density at the line, whether the line is clean, every
    candidate's standing at the line, the covariates, where the effect could differ, the threats, the window. A rung reads the
    rungs below it; every line has an address."""

    ORDER: ClassVar[tuple[str, ...]] = ("score", "shape", "density", "line", "balance", "covariates")

    score: Score | None = None
    shape: ShapeFacts | None = None
    density: DensityFacts | None = None
    line: Line | None = None
    balance: BalanceFacts | None = None
    covariates: CovariateRoles | None = None


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


class Bandwidths(BaseModel):
    """The window the design fits in, on each side of the line, as the window rung set it."""

    selector: str = Field(description="the selector the window rung chose, or the rule's own name")
    rule: Literal["mse", "cer", "support_points", "local_randomisation"] = "mse"
    h_left: float = Field(description="the estimation bandwidth on the control side")
    h_right: float = Field(description="the estimation bandwidth on the treated side")
    b_left: float = Field(description="the bias bandwidth on the control side")
    b_right: float = Field(description="the bias bandwidth on the treated side")
    h_cer_left: float | None = Field(
        default=None, description="the coverage-error-optimal width on the control side, for the grid; null when the rule has none"
    )
    h_cer_right: float | None = None
    n_h_left: int = 0
    n_h_right: int = 0

    def two_sided(self) -> bool:
        return abs(self.h_left - self.h_right) > 1e-12

    @property
    def h(self) -> list[float]:
        return [self.h_left, self.h_right]

    @property
    def b(self) -> list[float]:
        return [self.b_left, self.b_right]


class AssumptionDecision(BaseModel):
    endpoint_id: str
    admissible: bool
    reason: str
    cites: list[str]
    assumptions: list[str]


class AssumptionDecisions(BaseModel):
    items: list[AssumptionDecision]


class Candidate(BaseModel):
    id: str
    endpoint_id: str
    status: Literal["proposed", "surviving", "blocked", "failed"] = "proposed"
    reason: str
    cites: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    adjusted: bool = False
    estimator: str
    check_results: list[dict] = Field(default_factory=list)
    terminal_kind: (
        Literal["context_deficiency", "unidentified_effect", "unsupported_method", "inadequate_support", "exhausted_reasoning", "execution_failure"] | None
    ) = None


class Design(FrozenDesign):
    """One immutable, standalone RD execution plan."""

    model_config = ConfigDict(frozen=True)
    id: str
    endpoint_id: str
    context_id: str
    ledger_revision: int
    target: dict[str, Any]
    contrast: Contrast
    score: Score
    shape: ShapeFacts
    covariates: Covariates
    checks: Checks
    estimator: str
    estimand: Estimand
    spec: dict[str, Any]
    inference: str
    vce: str
    cluster: str | None = None
    bandwidths: Bandwidths
    fuzzy: bool = False
    assumptions: list[str] = Field(default_factory=list)
    falsifications: list[str] = Field(default_factory=list)
    sensitivities: list[str] = Field(default_factory=list)
    modifiers: list[str] = Field(default_factory=list)
    rows: list[int] = Field(default_factory=list)
    table_fingerprint: str
    identification_certificate: dict[str, Any]
    support: dict[str, Any]
    weights_policy: dict[str, Any]
    selector_policy: dict[str, Any]
    frozen_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def render(self) -> str:
        return self.model_dump_json(indent=2)


class RDInterpretation(BaseModel):
    target_contrast: Literal["assignment", "receipt", "eligibility"] = Field(
        description="Copy the frozen target.contrast exactly; eligibility is ITT, receipt is a complier effect, assignment applies under sharp compliance."
    )
    design_id: str
    endpoint_id: str
    contrast: str
    answer: str
    effect_stated: float | None = Field(description="Copy the primary recorded estimate.value exactly, or null only when no primary point estimate exists.")
    caveats: list[str]
    cites: list[str]
