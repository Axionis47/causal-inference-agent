"""What every contract builds on: the intent, a cited reason, a candidate column, the scope, the model's thought, and the slug an
address is made of."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

Intent = Literal["effect_of_change", "driver_search", "root_cause", "not_causal"]


class Cited(BaseModel):
    reason: str = Field(description="one or two sentences")
    cites: list[str] = Field(description="addresses from the pack, e.g. col:lunch.note, change:1.note, dataset.profile.grain")


class Candidate(Cited):
    column: str


class Scope(BaseModel):
    population_filter: str | None = Field(default=None, description="a filter on rows the question implies, in words, or null")
    window: str | None = Field(default=None, description="time window the question implies, or null")
    contrast: Literal["switch", "dose", "level_vs_level", "none"] = "switch"
    target: Literal["average", "on_treated", "on_untreated", "conditional", "counterfactual"] = "average"


class Thought(BaseModel):
    """Debug only. Never gated, cited, or read by another node."""

    node: str
    text: str
    thinking_tokens: int | None = None
    output_tokens: int | None = None


def _slug(s: str) -> str:

    out = re.sub(r"[^0-9a-zA-Z]+", "_", str(s).strip()).strip("_").lower()
    return out or "x"
