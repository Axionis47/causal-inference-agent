"""The diff-in-diff family's block: what the lane must not guess, as the desk filled it."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from causal_agent.common.contracts import Belief, Design, register_design


@register_design
class DidDesign(Design):
    """What the diff-in-diff lane must not guess."""

    kind: Literal["diff_in_diff"] = "diff_in_diff"
    unit: str | None = None
    time: str | None = None
    period_kind: Literal["date", "integer"] | None = None
    change_period: str | None = Field(default=None, description="the first period at or after the change, as it appears in the time column")
    treated_group: dict = Field(default_factory=dict, description="{column, level}, or {cohort_column, adoption_periods} when adoption is staggered")
    staggered: bool | None = None
    never_treated_exists: bool | None = None
    pre_periods: int | None = None
    post_periods: int | None = None
    controls_allowed: list[str] = Field(default_factory=list)
    cluster_level: str | None = None
    trend_belief: Belief | None = None
    spillover: Belief | None = None

    def columns(self) -> list[str]:
        return [n for n in [self.unit, self.time, (self.treated_group or {}).get("column"), self.cluster_level, *self.controls_allowed] if n]

    def time_column(self) -> str | None:
        return self.time or None

    def render(self) -> str:
        tg = self.treated_group
        return "\n".join(
            [
                f"  unit: {self.unit or 'not known'}; time: {self.time or 'not known'} ({self.period_kind or 'kind not known'})",
                f"  change period: {self.change_period or 'not known'}; treated group: "
                + (f"{tg.get('column')} = {tg.get('level')!r}" if tg.get("column") else "not known"),
                f"  staggered: {self.staggered}; never-treated units: {self.never_treated_exists}; pre periods: {self.pre_periods}; post periods: {self.post_periods}",
                f"  controls allowed: {', '.join(self.controls_allowed) or 'none named'}; cluster at: {self.cluster_level or 'not known'}",
            ]
            + [f"  {b.render()}" for b in (self.trend_belief, self.spillover) if b is not None]
        )
