"""Loaders for the three knowledge files. Read once, rendered for the model, filtered by facts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

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
        return (
            f"estimator: {self.name}\n"
            f"  what it does: {self.in_words}\n"
            f"  assumes: {self.assumes}\n"
            f"  weak when: {self.weak_when}"
        )


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
        return (estimand in w.get("estimand", []) and adjustment_set in w.get("adjustment_set", ["empty", "nonempty"])
                and (needs_hidden is None or hidden in needs_hidden))


@lru_cache(maxsize=1)
def load_estimators() -> list[EstimatorEntry]:
    raw = yaml.safe_load((_HERE / "estimators.yaml").read_text())
    return sorted((EstimatorEntry(name=k, **v) for k, v in raw.items()), key=lambda e: e.rank)


@lru_cache(maxsize=1)
def load_refuters() -> list[RefuterEntry]:
    raw = yaml.safe_load((_HERE / "refuters.yaml").read_text())
    return [RefuterEntry(name=k, **v) for k, v in raw.items()]


@lru_cache(maxsize=1)
def load_checks() -> dict[str, Any]:
    return yaml.safe_load((_HERE / "checks.yaml").read_text())


def estimator(name: str) -> EstimatorEntry:
    for e in load_estimators():
        if e.name == name:
            return e
    raise KeyError(name)


def refuter(name: str) -> RefuterEntry:
    for r in load_refuters():
        if r.name == name:
            return r
    raise KeyError(name)


def render_preferences(entries: list[EstimatorEntry]) -> str:
    names = {e.name for e in entries}
    lines = [f"- prefer {e.name} over {o}: {why}" for e in entries for o, why in e.prefer_over.items() if o in names]
    return "\n".join(lines) or "(none recorded among these)"
