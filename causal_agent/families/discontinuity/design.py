"""The discontinuity family's block: what the lane must not guess, as the desk filled it."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from causal_agent.common.contracts import Belief, Design, register_design


@register_design
class RdDesign(Design):
    """What the discontinuity lane must not guess."""

    kind: Literal["discontinuity"] = "discontinuity"
    score: str | None = None
    cutoff: float | None = None
    treated_side: Literal["above", "below"] | None = None
    cutoff_value_treated: bool | None = None
    score_fixed_before: bool | None = None
    movable: bool | None = Field(default=None, description="could a unit change its score after seeing the rule")
    takeup: dict | None = Field(
        default=None, description="{column, level} when a column records who took the change (fuzzy); null when the rule is the change (sharp)"
    )
    covariates_allowed: list[str] = Field(default_factory=list)
    cluster: str | None = None
    sampled_by_side: bool = False
    cutoff_only: Belief | None = None

    def columns(self) -> list[str]:
        return [n for n in [self.score, (self.takeup or {}).get("column"), self.cluster, *self.covariates_allowed] if n]

    def render(self) -> str:
        rule = f"{self.score} {self.treated_side} {self.cutoff:g}" if self.score and self.cutoff is not None and self.treated_side else "not known"
        tk = self.takeup or {}
        return "\n".join(
            [
                f"  score and cutoff: {rule}"
                + (
                    f"; the cutoff value itself is {'treated' if self.cutoff_value_treated else 'not treated'}" if self.cutoff_value_treated is not None else ""
                ),
                f"  score fixed before the decision: {self.score_fixed_before}; a unit could move it: {self.movable}",
                "  take-up: " + (f"{tk.get('column')} = {tk.get('level')!r} (fuzzy)" if tk.get("column") else "none recorded (sharp)"),
                f"  covariates allowed: {', '.join(self.covariates_allowed) or 'none named'}; cluster: {self.cluster or 'none'}; rows drawn by side: {self.sampled_by_side}",
            ]
            + ([f"  {self.cutoff_only.render()}"] if self.cutoff_only is not None else [])
        )
