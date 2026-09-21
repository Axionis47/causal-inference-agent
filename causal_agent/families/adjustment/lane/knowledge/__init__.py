"""The lane's knowledge files: the entry models, and the shared loader bound to this folder. Read once, rendered for the model, filtered by facts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from causal_agent.lane.knowledge import Knowledge

_HERE = Path(__file__).parent


class EstimatorEntry(BaseModel):
    name: str
    dowhy: str
    applies_when: dict[str, Any]
    in_words: str
    assumes: str
    weak_when: str
    prefer_over: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    rank: int = 99
    also_run: str | None = None

    @property
    def estimand(self) -> str:
        return str(self.applies_when.get("estimand"))

    def applies(self, *, estimand: str, treatment: str, outcome: str, adjustment_set: str, roads: list[str] | None = None) -> bool:
        """The entry runs on a design whose taken road, or any open road when `roads` is given, is the entry's."""
        w = self.applies_when
        on_road = w.get("estimand") in (roads or [estimand])
        return on_road and treatment in w.get("treatment", []) and outcome in w.get("outcome", []) and adjustment_set in w.get("adjustment_set", [])

    def render(self) -> str:
        return f"estimator: {self.name}\n  what it does: {self.in_words}\n  assumes: {self.assumes}\n  weak when: {self.weak_when}"


class RefuterEntry(BaseModel):
    name: str
    kind: Literal["falsification", "sensitivity"]
    in_words: str
    applies_when: dict[str, Any]
    params: dict[str, Any] = Field(default_factory=dict)
    pass_when: dict[str, Any] = Field(default_factory=dict)

    def applies(self, *, estimand: str, adjustment_set: str, hidden: bool = False) -> bool:
        w = self.applies_when
        needs_hidden = w.get("hidden")
        return (
            estimand in w.get("estimand", [])
            and adjustment_set in w.get("adjustment_set", ["empty", "nonempty"])
            and (needs_hidden is None or hidden in needs_hidden)
        )


# ------------------------------------------------------------------ the files, through the shared loader

K = Knowledge(_HERE, estimator=EstimatorEntry, refuter=RefuterEntry)
load_estimators, load_refuters, load_checks, load_beliefs = K.estimators, K.refuters, K.checks, K.beliefs
estimator, refuter, render_preferences = K.estimator, K.refuter, K.render_preferences
