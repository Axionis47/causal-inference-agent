"""Diff-in-diff lane contracts. What each node writes, and the rungs of the ladder the design is climbed on; the records every
ladder shares (the threats, heterogeneity, what a rung would not guess) come from the harness.

Lane-invariant artifacts (Contrast, Checks, Estimate, Refutation, Interpretation, Feasibility) live in
causal_agent.common.contracts. These are the ones only this lane needs.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from causal_agent.common.contracts import Checks, Cited, Contrast, Departure
from causal_agent.lane.ladder import Heterogeneity, LadderBase, Threats, Unsure


class Groups(BaseModel):
    """Rung 0, who got the change: the column that marks it and the level that means treated."""

    column: str = Field(description="the column whose value says whether a unit got the change")
    treated_level: str = Field(description="the exact value meaning the unit got it")
    reason: str
    cites: list[str]
    by: Literal["pack", "judgement"] = "judgement"

    def lines(self) -> list[tuple[str, str]]:
        return [("ladder:groups.treated", f"{self.column} = {self.treated_level!r}; every other level is the comparison group (set by the {self.by})")]


class Periods(BaseModel):
    """Before and after. Either a time column with a first post period, or two columns holding the measure before and after."""

    kind: Literal["long", "wide"]
    time_column: str | None = Field(default=None, description="long only: the column that orders rows in time")
    first_post: str | None = Field(default=None, description="long only: the first time value at or after the change, as it appears in the data")
    window_start: str | None = Field(default=None, description="long only: first time value to use, or null for all")
    window_end: str | None = Field(default=None, description="long only: last time value to use, or null for all")
    before_column: str | None = Field(default=None, description="wide only: the column holding the outcome measured before the change")
    after_column: str | None = Field(default=None, description="wide only: the column holding the outcome measured after the change")
    reason: str
    cites: list[str]
    by: Literal["pack", "judgement"] = "judgement"

    def lines(self) -> list[tuple[str, str]]:
        when = (
            f"long: time {self.time_column}, first post {self.first_post}, window {self.window_start or 'start'} to {self.window_end or 'end'}"
            if self.kind == "long"
            else f"wide: before {self.before_column}, after {self.after_column}"
        )
        return [("ladder:periods.clock", f"{when} (set by the {self.by})")]


class ShapeFacts(BaseModel):
    """Rung 2, the panel's shape by code: who got the change and when, who never did, the periods either side of the earliest
    change, and how each unit's first treated period was read."""

    kind: Literal["long", "wide"]
    rows: int
    units_treated: int = Field(description="units that got the change at some point")
    units_control: int = Field(description="units that never got the change; under staggered adoption the not-yet-treated periods also serve as comparison")
    units_never_treated: int = 0
    never_treated_exists: bool = True
    periods_pre: int = Field(description="periods before the earliest first-treated period")
    periods_post: int = Field(description="periods at or after it")
    cohorts: int = Field(description="distinct first-treated periods among treated units; 1 means one-shot adoption")
    units_by_cohort: dict[str, int] = Field(default_factory=dict, description="first-treated period -> units")
    adoption: Literal["one_shot", "staggered"] = "one_shot"
    first_treated_source: Literal["label", "indicator", "cohort_column"] = "label"
    treatment_reversals: int = Field(default=0, description="units that left the treatment after getting it; the shape stops when any")
    clusters: int | None = Field(default=None, description="distinct values of the cluster column, or the units when none is declared")
    balanced: bool = Field(default=True, description="every unit is observed in every period")
    time_values: list[str] = Field(default_factory=list)

    def lines(self) -> list[tuple[str, str]]:
        cohorts = (
            f"one first-treated period ({next(iter(self.units_by_cohort), '')})"
            if self.cohorts <= 1
            else f"{self.cohorts} first-treated periods: " + ", ".join(f"{k} ({v} units)" for k, v in self.units_by_cohort.items())
        )
        return [
            ("ladder:shape.units", f"{self.units_treated} treated units, {self.units_control} never treated"),
            ("ladder:shape.periods", f"{self.periods_pre} before the earliest change, {self.periods_post} after"),
            (
                "ladder:shape.adoption",
                f"{self.adoption.replace('_', '-')}: {cohorts}; read from the {self.first_treated_source.replace('_', ' ')}"
                + ("" if self.never_treated_exists else "; no unit is never treated, so the comparison is units not yet treated")
                + ("" if self.balanced else "; the panel is not balanced"),
            ),
        ]


RiskName = Literal["anticipation", "spillover", "composition", "other_shock", "group_choice"]


class Risk(Cited):
    """One thing that could break the comparison, named from the story: units acting before the change (anticipation), treated
    units reaching the comparison units (spillover), who is in each group changing over time (composition), something else that
    hit one group at the same time (other_shock), the treated group chosen for where its outcome was heading (group_choice)."""

    name: RiskName


class Comparison(BaseModel):
    """Rung 3: whether the comparison group is a fair stand-in for the treated group without the change, and what could break it."""

    fair: bool = Field(description="whether the story supports that, apart from the change, the groups would have moved together")
    why: str = Field(description="one or two sentences from the story and the facts")
    risks: list[Risk] = Field(default_factory=list, description="every risk the story raises, each cited; none when it raises none")
    cites: list[str] = Field(default_factory=list)
    unsure: list[Unsure] = Field(default_factory=list)
    by: Literal["judgement"] = "judgement"

    def lines(self) -> list[tuple[str, str]]:
        return [("ladder:comparison.fair", "yes" if self.fair else "no"), ("ladder:comparison.why", self.why)] + [
            (f"ladder:comparison.risk.{r.name}", r.reason) for r in self.risks
        ]


class ControlRelation(BaseModel):
    """One column's fitness as a control in a before-after comparison, read with every other candidate in view."""

    column: str
    affected_by_treatment: bool = Field(
        description="the column's value could have been changed by the treatment, so adjusting for it removes part of the effect"
    )
    usable_as_control: bool = Field(
        description="the column moves over time within a unit, predates the outcome, and could drive the outcome differently across groups"
    )
    modifier_candidate: bool = Field(default=False, description="a trait of the unit, fixed over time, that the effect could plausibly differ by")
    reasons: list[Cited] = Field(description="one entry per claim marked true, each citing the card that supports it")
    departures: list[Departure] = Field(default_factory=list)

    def word(self) -> str:
        base = "changed by the treatment" if self.affected_by_treatment else "a control" if self.usable_as_control else "not a control"
        return base + ("; a candidate modifier" if self.modifier_candidate else "")


class ControlRoles(BaseModel):
    """Rung 4: every candidate control placed together."""

    items: list[ControlRelation] = Field(description="one per column listed, all of them")
    unsure: list[Unsure] = Field(default_factory=list, description="what you would not guess, with why; the answer above still stands")

    def lines(self) -> list[tuple[str, str]]:
        return [(f"ladder:controls.{r.column}", r.word()) for r in self.items]


class Cluster(BaseModel):
    """Where the errors cluster and how the p-value is computed, by code from the shape, the pack's word and the inference catalogue."""

    level: str
    why: str
    clusters: int | None = None
    inference: str = ""
    resample: str | None = Field(default=None, description="ritest or wildboottest when the p-value is resampled, else none")

    def lines(self) -> list[tuple[str, str]]:
        return [
            ("ladder:cluster.level", self.level + (f" ({self.clusters} clusters)" if self.clusters is not None else "")),
            ("ladder:cluster.inference", self.inference + (f"; the p-value by {self.resample}" if self.resample else "")),
            ("ladder:cluster.why", self.why),
        ]


class Ladder(LadderBase):
    """The rungs climbed so far: who got the change, the clock, the shape, the comparison, the controls, where the effect could
    differ, the threats, the clustering. A rung reads the rungs below it; every line has an address."""

    ORDER: ClassVar[tuple[str, ...]] = ("groups", "periods", "shape", "comparison", "controls", "heterogeneity", "threats", "cluster")

    groups: Groups | None = None
    periods: Periods | None = None
    shape: ShapeFacts | None = None
    comparison: Comparison | None = None
    controls: ControlRoles | None = None
    heterogeneity: Heterogeneity | None = None
    threats: Threats | None = None
    cluster: Cluster | None = None


class Excluded(BaseModel):
    column: str
    why: str


class Controls(BaseModel):
    included: list[str] = Field(default_factory=list)
    dropped_fixed: list[Excluded] = Field(default_factory=list, description="fixed within unit or within period; absorbed by the fixed effects")
    excluded: list[Excluded] = Field(default_factory=list, description="affected by the treatment, or ruled out by a revision")

    def render(self) -> str:
        lines = [f"controls: {', '.join(self.included) or 'none'}"]
        lines += [f"  absorbed {x.column}: {x.why}" for x in self.dropped_fixed]
        lines += [f"  excluded {x.column}: {x.why}" for x in self.excluded]
        return "\n".join(lines)


class Revision(BaseModel):
    column: str
    change: Literal["add_control", "remove_control"]
    reason: str
    cites: list[str] = Field(default_factory=list)


class DesignAssessment(BaseModel):
    action: Literal["proceed", "revise", "stop"]
    revisions: list[Revision] = Field(default_factory=list)
    reason: str = Field(description="one or two sentences citing the flag numbers")
    cites: list[str] = Field(default_factory=list)


class EstimatorPick(BaseModel):
    name: str = Field(description="one of the names offered")
    reason: str
    cites: list[str] = Field(default_factory=list)


class Design(BaseModel):
    """The frozen design. Everything after this is deterministic execution."""

    contrast: Contrast
    groups: Groups
    periods: Periods
    shape: ShapeFacts
    controls: Controls
    checks: Checks
    estimator: str
    engine: str = "feols"
    formula: str = Field(description="what runs: the feols formula, the two stages, or the surface's own words")
    also_run: str | None = None
    also_formula: str | None = None
    inference: str
    vcov: str | dict
    placebos: list[str]
    target_units: str
    modifiers: list[str] = Field(default_factory=list, description="the unit traits the effect is also estimated within, level by level")
    frozen_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def render(self) -> str:
        p, s = self.periods, self.shape
        when = (
            f"long: time {p.time_column}, first post {p.first_post}, window {p.window_start or 'start'} to {p.window_end or 'end'}"
            if p.kind == "long"
            else f"wide: before {p.before_column}, after {p.after_column}"
        )
        lines = [
            "DESIGN",
            f"  groups       {self.groups.column} = {self.groups.treated_level!r} treated, every other level control ({self.groups.reason})",
            f"  periods      {when} ({p.reason})",
            f"  shape        {s.rows} rows; {s.units_treated} treated units, {s.units_control} control; {s.periods_pre} pre, {s.periods_post} post; cohorts {s.cohorts}",
        ]
        lines += ["  " + ln for ln in self.controls.render().splitlines()]
        for r in self.checks.results:
            lines.append(f"  check        {r.level:4} {r.address}  {r.detail}")
        lines.append(f"  estimator    {self.estimator}: {self.formula}" + (f"   also {self.also_run}: {self.also_formula}" if self.also_run else ""))
        lines.append(f"  inference    {self.inference}  vcov {self.vcov}")
        lines.append(f"  placebos     {', '.join(self.placebos) or 'none'}")
        lines.append(f"  target       {self.target_units}")
        lines.append(f"  modifiers    {', '.join(self.modifiers) or 'none'}")
        lines.append(f"  frozen at    {self.frozen_at}")
        return "\n".join(lines)
