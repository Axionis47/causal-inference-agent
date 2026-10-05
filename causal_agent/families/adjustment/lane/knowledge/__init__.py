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

    @property
    def targets(self) -> list[str]:
        """The units the entry can be asked the effect for: ate, att, atc."""
        return [str(t) for t in self.applies_when.get("target", [])]

    def applies(self, *, estimand: str, treatment: str, outcome: str, adjustment_set: str, target: str | None = None) -> bool:
        """The entry runs on a design whose road, treatment kind, outcome kind and adjustment set are the entry's, and, when a
        target is given, one the entry serves."""
        w = self.applies_when
        return (
            w.get("estimand") == estimand
            and treatment in w.get("treatment", [])
            and outcome in w.get("outcome", [])
            and adjustment_set in w.get("adjustment_set", [])
            and (target is None or target in self.targets)
        )

    def render(self) -> str:
        return f"estimator: {self.name}\n  what it does: {self.in_words}\n  assumes: {self.assumes}\n  weak when: {self.weak_when}"


class RefuterEntry(BaseModel):
    name: str
    kind: Literal["falsification", "sensitivity"]
    in_words: str
    applies_when: dict[str, Any]
    params: dict[str, Any] = Field(default_factory=dict)
    pass_when: dict[str, Any] = Field(default_factory=dict)

    def applies(self, *, estimand: str, adjustment_set: str, sensitivity_required: bool = False) -> bool:
        w = self.applies_when
        needs = w.get("sensitivity_required")
        return (
            estimand in w.get("estimand", [])
            and adjustment_set in w.get("adjustment_set", ["empty", "nonempty"])
            and (needs is None or sensitivity_required in needs)
        )


# ------------------------------------------------------------------ the files, through the shared loader

K = Knowledge(_HERE, estimator=EstimatorEntry, refuter=RefuterEntry)
load_estimators, load_refuters, load_checks, load_beliefs = K.estimators, K.refuters, K.checks, K.beliefs
estimator, refuter, render_preferences = K.estimator, K.refuter, K.render_preferences
