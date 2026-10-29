"""DiD context readings, append-only assumptions and frozen execution contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from causal_agent.common.contracts import Checks, Contrast, FrozenDesign
from causal_agent.lane.ledger import Evidence


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
    before_column: str | None = Field(default=None, description="wide only: the column holding the outcome measured before the change")
    after_column: str | None = Field(default=None, description="wide only: the column holding the outcome measured after the change")
    reason: str
    cites: list[str]
    by: Literal["pack", "judgement"] = "judgement"

    def lines(self) -> list[tuple[str, str]]:
        when = (
            f"long: time {self.time_column}, first post {self.first_post}"
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


class LedgerOp(BaseModel):
    kind: Literal["selection", "parallel_trends", "anticipation", "spillover", "other_shock", "composition", "control", "modifier", "exchangeability"]
    subject: str
    value: Any = Field(
        default=None,
        description="selection: levels/trends/neither/unknown; anticipation: integer periods or null; control/modifier/exchangeability/spillover/other_shock/composition: boolean or null; parallel_trends: boolean/null or {requires_controls:[column keys]}",
    )
    status: Literal["proposed", "supported", "contested"] = "proposed"
    evidence: list[Evidence] = Field(default_factory=list)
    reason: str


class Reading(BaseModel):
    notes: str = Field(description="reason from the supplied evidence before proposing operations; no invented study facts")
    operations: list[LedgerOp] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


class MechanismOp(LedgerOp):
    kind: Literal["selection", "anticipation", "exchangeability"]


class MechanismReading(Reading):
    operations: list[MechanismOp] = Field(default_factory=list)


class ComparisonOp(LedgerOp):
    kind: Literal["parallel_trends", "spillover", "other_shock", "composition"]


class ComparisonReading(Reading):
    operations: list[ComparisonOp] = Field(default_factory=list)


class ControlOp(LedgerOp):
    kind: Literal["control", "modifier"]


class ControlReading(Reading):
    operations: list[ControlOp] = Field(default_factory=list)


class ProbeOp(LedgerOp):
    kind: Literal["selection", "parallel_trends", "anticipation", "spillover", "other_shock", "control"]


class ProbeReading(Reading):
    operations: list[ProbeOp] = Field(default_factory=list)
    diagnostic_cites: list[str] = Field(
        default_factory=list,
        description="exact probe addresses supporting diagnostic notes, independent of causal ledger operations; operations may be empty when uncertainty remains",
    )


class EstimatorChoice(BaseModel):
    candidate_id: str
    estimator: str
    reason: str
    cites: list[str] = Field(
        description="bare address strings such as candidate:<candidate_id> or estimator:<estimator>; copy only the address inside brackets, without brackets or the rendered sentence"
    )


class EstimatorPicks(BaseModel):
    notes: str
    choices: list[EstimatorChoice]


class Candidate(BaseModel):
    id: str
    endpoint_id: str
    estimator_options: list[str]
    controls: list[str] = Field(default_factory=list)
    support: str = "admitted_window"
    target: dict = Field(default_factory=dict)
    status: Literal["proposed", "ready", "excluded", "frozen", "failed"] = "proposed"
    reason: str = ""
    assumptions: list[str] = Field(default_factory=list)
    checks: list = Field(default_factory=list)


class Design(FrozenDesign):
    """One sealed specification. Execution never changes its rows, formula or target."""

    id: str
    endpoint_id: str
    context_id: str
    ledger_revision: int
    identification_certificate: dict
    target: dict
    contrast: Contrast
    groups: Groups
    periods: Periods
    shape: ShapeFacts
    controls: tuple[str, ...] = ()
    checks: Checks
    estimator: str
    engine: str
    formula: str
    first_stage: str = ""
    second_stage: str = ""
    inference: str
    cluster_column: str
    vcov: str | dict
    panel_path: str
    row_ids: tuple[int, ...]
    coefficients: tuple[str, ...] = ()
    weights: dict[str, float] = Field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    falsifications: tuple[str, ...] = ()
    sensitivities: tuple[str, ...] = ()
    frozen_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    def render(self) -> str:
        return f"DESIGN {self.id} ({self.endpoint_id})\n  target aggregation={self.target.get('aggregation')} horizons={self.target.get('horizons')} cohorts={self.target.get('cohorts')}; {len(self.target.get('included_units', []))} target units; weighting={self.target.get('weighting')}; support={self.target.get('support')}\n  estimator {self.estimator}: {self.formula}\n  controls {list(self.controls)}\n  inference {self.inference}, {self.cluster_column}\n  rows {len(self.row_ids)}; ledger revision {self.ledger_revision}"
