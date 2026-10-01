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


ChosenOn = Literal["levels", "trends", "neither", "unknown"]


class Mechanism(BaseModel):
    """Rung 3: how the treated group came to be chosen. Code fills the kind, the level and the drivers from the pack; a judgement
    fills what the pack leaves open: whether the group was picked for where its outcome stood (levels), for where it was heading
    (trends), or for neither; and whether the story states a lead during which units could act on the change before it came."""

    kind: str | None = None
    level_column: str | None = Field(default=None, description="the level the change was decided at, a column above the unit, when the pack names one")
    drivers: list[str] = Field(default_factory=list, description="the columns the choice looked at; only columns in play; none if the story names none")
    chosen_on: ChosenOn = Field(
        default="unknown",
        description="levels: the group was picked for where its outcome or its traits stood; trends: for where its outcome was heading; neither: the story gives no such reason; unknown: you cannot tell",
    )
    anticipation_periods: int | None = Field(
        default=None,
        description="the periods before the change during which units could act on it, only when the story states an announcement or a lead; null otherwise",
    )
    staggered: bool = False
    never_treated_exists: bool = True
    reason: str = ""
    cites: list[str] = Field(default_factory=list)
    unsure: list[Unsure] = Field(default_factory=list, description="what you would not guess, with why")
    by: Literal["pack", "judgement"] = "pack"

    def lines(self) -> list[tuple[str, str]]:
        words = {
            "levels": "the group was chosen for where its outcome or its traits stood",
            "trends": "the group was chosen for where its outcome was heading",
            "neither": "the story gives no sign the group was chosen for its outcome",
            "unknown": "how the group was chosen is not known",
        }
        return [
            ("ladder:mechanism.kind", f"{self.kind or 'not said'} (set by the {self.by})"),
            ("ladder:mechanism.level", self.level_column or "the unit itself"),
            ("ladder:mechanism.drivers", ", ".join(self.drivers) or "none named"),
            ("ladder:mechanism.chosen_on", words[self.chosen_on]),
            (
                "ladder:mechanism.anticipation",
                f"{self.anticipation_periods} period(s) in which units could act before the change" if self.anticipation_periods else "none stated",
            ),
            (
                "ladder:mechanism.adoption",
                ("staggered" if self.staggered else "one-shot") + ("" if self.never_treated_exists else "; no unit is never treated"),
            ),
        ] + ([("ladder:mechanism.reason", self.reason)] if self.reason else [])


class PathPoint(BaseModel):
    time: str
    treated_mean: float | None = None
    control_mean: float | None = None
    n_treated: int = 0
    n_control: int = 0


class Composition(BaseModel):
    """Who is in the panel when: units per group in each period, and how many entered after the first period or left before the last."""

    per_period: list[tuple[str, int, int]] = Field(default_factory=list, description="(period, treated units, comparison units)")
    entries: int = 0
    exits: int = 0
    balanced: bool = True


class TrendFacts(BaseModel):
    """Rung 4's evidence, by code before the comparison is judged: the mean outcome by group in every period before the earliest
    change, how the gap between the groups drifted, the joint test that the pre-period coefficients are zero with each lead, and
    the composition of the panel over time. The post-period coefficients never become lines: the effect is the run's to find."""

    pre_paths: list[PathPoint] = Field(default_factory=list)
    pre_slope_gap: float | None = Field(default=None, description="the drift of the treated-minus-comparison gap per period, before the change")
    leads: dict[int, tuple[float, float, float]] = Field(default_factory=dict, description="pre-period coefficient -> (estimate, low, high)")
    leads_stat: float | None = None
    leads_p: float | None = None
    leads_k: int = 0
    leads_level: Literal["pass", "soft", "hard", "untested"] = "untested"
    leads_how: str = ""
    composition: Composition = Field(default_factory=Composition)

    @property
    def reading(self) -> Literal["parallel", "diverging", "untested"]:
        return "untested" if self.leads_level == "untested" else "parallel" if self.leads_level == "pass" else "diverging"

    def lines(self) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        if self.pre_paths:
            paths = "; ".join(
                f"{pt.time}: treated {pt.treated_mean:.4g} (n {pt.n_treated}), comparison {pt.control_mean:.4g} (n {pt.n_control})"
                if pt.treated_mean is not None and pt.control_mean is not None
                else f"{pt.time}: one group absent"
                for pt in self.pre_paths
            )
            out.append(("ladder:trends.paths", f"mean outcome by group in each period before the change: {paths}"))
        if self.pre_slope_gap is not None:
            out.append(("ladder:trends.gap_slope", f"the treated-minus-comparison gap moved {self.pre_slope_gap:+.4g} per period before the change"))
        if self.leads_level == "untested":
            out.append(("ladder:trends.leads", f"untested: {self.leads_how or 'one period before the change'}"))
        else:
            verdict = {
                "pass": "no sign of differing pre-trends",
                "soft": "the groups may already have been moving apart",
                "hard": "the groups were already moving apart",
            }[self.leads_level]
            out.append(
                (
                    "ladder:trends.leads",
                    f"joint test that the {self.leads_k} pre-period coefficients are zero: p = {self.leads_p:.3g} ({self.leads_level}); {verdict}{self.leads_how}",
                )
            )
            for k in sorted(self.leads, reverse=True)[:8]:
                est, lo, hi = self.leads[k]
                out.append((f"ladder:trends.lead.{k}", f"{est:+.4g} [{lo:.4g}, {hi:.4g}]"))
        c = self.composition
        if c.per_period:
            treated = ", ".join(str(t) for _, t, _ in c.per_period[:12]) + ("…" if len(c.per_period) > 12 else "")
            control = ", ".join(str(n) for _, _, n in c.per_period[:12]) + ("…" if len(c.per_period) > 12 else "")
            out.append(
                (
                    "ladder:trends.composition",
                    f"units present per period: treated {treated}; comparison {control}; {c.entries} entered after the first period, {c.exits} left before the last"
                    + ("" if c.balanced else "; the panel is not balanced"),
                )
            )
        return out


RiskName = Literal["anticipation", "spillover", "composition", "other_shock", "group_choice"]


class Risk(Cited):
    """One thing that could break the comparison, named from the story: units acting before the change (anticipation), treated
    units reaching the comparison units (spillover), who is in each group changing over time (composition), something else that
    hit one group at the same time (other_shock), the treated group chosen for where its outcome was heading (group_choice)."""

    name: RiskName


class Comparison(BaseModel):
    """Rung 4: whether the comparison group is a fair stand-in for the treated group without the change, judged over the trends
    rung's evidence and the story, and what could break it."""

    fair: bool = Field(description="whether the story and the evidence support that, apart from the change, the groups would have moved together")
    why: str = Field(description="one or two sentences from the evidence and the story")
    leads_read: Literal["parallel", "diverging", "untested"] = Field(
        default="untested",
        description="your reading of the joint test on the pre-period coefficients under ladder:trends.leads: parallel, diverging, or untested",
    )
    why_despite: Cited | None = Field(
        default=None,
        description="when you judge the comparison fair although the test says the paths diverged: why, citing the pack line that says the groups would still have moved together",
    )
    composition_read: Cited | None = Field(
        default=None,
        description="when units enter or leave the panel and you do not name it as a risk: why it does not matter, citing ladder:trends.composition",
    )
    risks: list[Risk] = Field(default_factory=list, description="every risk the story or the evidence raises, each cited; none when none")
    cites: list[str] = Field(default_factory=list)
    unsure: list[Unsure] = Field(default_factory=list)
    by: Literal["judgement"] = "judgement"

    def lines(self) -> list[tuple[str, str]]:
        out = [("ladder:comparison.fair", "yes" if self.fair else "no"), ("ladder:comparison.leads_read", self.leads_read), ("ladder:comparison.why", self.why)]
        if self.why_despite is not None:
            out.append(("ladder:comparison.why_despite", self.why_despite.reason))
        if self.composition_read is not None:
            out.append(("ladder:comparison.composition_read", self.composition_read.reason))
        return out + [(f"ladder:comparison.risk.{r.name}", r.reason) for r in self.risks]


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
    """The rungs climbed so far: who got the change, the clock, the shape, how the group was chosen, the paths before the
    change, the comparison, the controls, where the effect could differ, the threats, the clustering. A rung reads the rungs
    below it; every line has an address."""

    ORDER: ClassVar[tuple[str, ...]] = ("groups", "periods", "shape", "mechanism", "trends", "comparison", "controls", "heterogeneity", "threats", "cluster")

    groups: Groups | None = None
    periods: Periods | None = None
    shape: ShapeFacts | None = None
    mechanism: Mechanism | None = None
    trends: TrendFacts | None = None
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
    placebos: list[str] = Field(description="every falsification and sensitivity in the catalogue whose applies_when holds on this design")
    placebo_outcomes: list[str] = Field(default_factory=list, description="columns the change could not have moved, for the placebo-outcome falsification")
    target_units: str
    modifiers: list[str] = Field(default_factory=list, description="the unit traits the effect is also estimated within, level by level")
    excluded_rel_times: list[int] = Field(
        default_factory=list, description="periods relative to the change left out of the estimate: the anticipation window the mechanism rung named"
    )
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
        if self.excluded_rel_times:
            lines.append(f"  left out     periods {', '.join(str(k) for k in self.excluded_rel_times)} relative to the change (anticipation)")
        lines.append(f"  frozen at    {self.frozen_at}")
        return "\n".join(lines)
