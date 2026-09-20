"""The adjustment family's block: what the lane must not guess, as the desk filled it."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from causal_agent.common.contracts import Design, register_design


@register_design
class AdjustmentDesign(Design):
    """What the adjustment lane must not guess. Every field is optional: an empty field means the desk could not say."""

    kind: Literal["adjustment"] = "adjustment"
    adjustment_candidates: list[str] = Field(
        default_factory=list, description="columns the offer depended on plus every before-column the change could not have moved"
    )
    forbidden: list[str] = Field(default_factory=list, description="columns set at or after the change; never a parent of the outcome in the graph")
    instrument: str | None = None
    mediator: str | None = None
    unobserved_confounding: bool | None = Field(default=None, description="true: the person says something hidden drove both; the caveat must say so")
    voluntary_uptake: bool | None = Field(default=None, description="true when units chose after an offer, so overlap is expected; false for a strict rule")
    target_units: str = "average"
    contrast: str = "switch"

    def columns(self) -> list[str]:
        return [n for n in [self.instrument, self.mediator, *self.adjustment_candidates] if n]

    def render(self) -> str:
        return "\n".join(
            [
                f"  candidates to adjust for: {', '.join(self.adjustment_candidates) or 'none named'}",
                f"  never adjust for: {', '.join(self.forbidden) or 'none named'}",
                f"  instrument: {self.instrument or 'none named'} · mediator: {self.mediator or 'none named'}",
                f"  hidden confounding: {'yes, per the person' if self.unobserved_confounding else 'no, per the person' if self.unobserved_confounding is False else 'not known'}",
                f"  uptake: {'voluntary after an offer' if self.voluntary_uptake else 'by a rule' if self.voluntary_uptake is False else 'not known'}",
                f"  target: {self.target_units}; contrast: {self.contrast}",
            ]
        )
