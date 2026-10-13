"""The diff-in-diff family's block: what the lane must not guess, as the desk filled it."""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, field_validator

from causal_agent.common.contracts import Design, register_design


@register_design
class DidDesign(Design):
    """What the diff-in-diff lane must not guess."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["diff_in_diff"] = "diff_in_diff"
    unit: str | None = None
    time: str | None = None
    period_kind: Literal["date", "integer"] | None = None
    change_period: str | None = Field(default=None, description="the first period at or after the change, as it appears in the time column")
    treated_group: dict = Field(
        default_factory=dict,
        description="{column, level}, and cohort_column when the pack names the column holding the period each unit first got the change",
    )
    before_column: str | None = None
    after_column: str | None = None
    adoption_representation: Literal["label", "indicator", "cohort_column"] = "label"
    aggregation: Literal["pooled", "horizon", "horizon_profile", "cohort", "cohort_profile", "cohort_horizon"] = "pooled"
    target_weighting: (
        Literal["treated_cell_shares", "residualized_treatment_projection", "linear_response_horizon_projection", "clean_adopter_projection"] | None
    ) = None
    horizons: list[int] = Field(default_factory=list, description="distinct nonnegative post-adoption period offsets")
    cohorts: list[str] = Field(default_factory=list)
    prefix_scope_accepted: bool = False
    cluster_level: str | None = None

    @field_validator("horizons")
    @classmethod
    def valid_horizons(cls, horizons: list[int]) -> list[int]:
        if any(h < 0 for h in horizons) or len(set(horizons)) != len(horizons):
            raise ValueError("horizons must be distinct nonnegative period offsets")
        return sorted(horizons)

    def columns(self) -> list[str]:
        tg = self.treated_group or {}
        return [
            n
            for n in [
                self.unit,
                self.time,
                tg.get("column"),
                tg.get("cohort_column"),
                self.cluster_level,
                self.before_column,
                self.after_column,
            ]
            if n
        ]

    def time_column(self) -> str | None:
        return self.time or None

    def render(self) -> str:
        return f"unit={self.unit}; time={self.time}; paired={self.before_column}/{self.after_column}; assignment={self.treated_group}; adoption={self.adoption_representation}; change={self.change_period}; cluster={self.cluster_level}; aggregation={self.aggregation}; weighting={self.target_weighting}; horizons={self.horizons}; cohorts={self.cohorts}; prefix accepted={self.prefix_scope_accepted}"
