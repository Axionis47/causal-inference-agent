"""Diff-in-diff lane contracts. What each node writes.

Lane-invariant artifacts (Contrast, Checks, Estimate, Refutation, Interpretation, Feasibility) live in
causal_agent.common.contracts. These are the ones only this lane needs.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

from causal_agent.common.contracts import Checks, Cited, Contrast


class Groups(BaseModel):
    """Who got the change: the column that marks it and the level that means treated."""

    column: str = Field(description="the column whose value says whether a unit got the change")
    treated_level: str = Field(description="the exact value meaning the unit got it")
    reason: str
    cites: list[str]


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


class ShapeFacts(BaseModel):
    kind: Literal["long", "wide"]
    rows: int
    units_treated: int
    units_control: int
    periods_pre: int
    periods_post: int
    cohorts: int = Field(description="distinct first-treated periods among treated units; 1 means one-shot adoption")
    switching_units: int = Field(description="units whose group label changes over time; must be 0")
    time_values: list[str] = Field(default_factory=list)


class ControlRelation(BaseModel):
    """One column's fitness as a control in a before-after comparison. Two claims, each cited when true."""

    column: str
    affected_by_treatment: bool = Field(description="the column's value could have been changed by the treatment, so adjusting for it removes part of the effect")
    usable_as_control: bool = Field(description="the column moves over time within a unit, predates the outcome, and could drive the outcome differently across groups")
    reasons: list[Cited] = Field(description="one entry per claim marked true, each citing the card that supports it")


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
    formula: str
    also_run: str | None = None
    also_formula: str | None = None
    inference: str
    vcov: str | dict
    placebos: list[str]
    target_units: str
    frozen_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def render(self) -> str:
        p, s = self.periods, self.shape
        when = (f"long: time {p.time_column}, first post {p.first_post}, window {p.window_start or 'start'} to {p.window_end or 'end'}"
                if p.kind == "long" else f"wide: before {p.before_column}, after {p.after_column}")
        lines = [
            "DESIGN",
            f"  groups       {self.groups.column} = {self.groups.treated_level!r} treated, every other level control ({self.groups.reason})",
            f"  periods      {when} ({p.reason})",
            f"  shape        {s.rows} rows; {s.units_treated} treated units, {s.units_control} control; {s.periods_pre} pre, {s.periods_post} post; cohorts {s.cohorts}",
        ]
        lines += ["  " + l for l in self.controls.render().splitlines()]
        for r in self.checks.results:
            lines.append(f"  check        {r.level:4} {r.address}  {r.detail}")
        lines.append(f"  estimator    {self.estimator}: {self.formula}" + (f"   also {self.also_run}: {self.also_formula}" if self.also_run else ""))
        lines.append(f"  inference    {self.inference}  vcov {self.vcov}")
        lines.append(f"  placebos     {', '.join(self.placebos) or 'none'}")
        lines.append(f"  target       {self.target_units}")
        lines.append(f"  frozen at    {self.frozen_at}")
        return "\n".join(lines)
