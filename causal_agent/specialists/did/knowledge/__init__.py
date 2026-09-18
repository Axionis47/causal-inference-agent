"""Loaders for the four knowledge files. Read once, rendered for the model, filtered by facts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

_HERE = Path(__file__).parent


class EstimatorEntry(BaseModel):
    name: str
    in_words: str
    formula: str
    controls_mode: str = "plain"  # plain | csw0 | none
    applies_when: dict[str, Any]
    assumes: str
    weak_when: str
    prefer_over: dict[str, str] = Field(default_factory=dict)
    rank: int = 99
    also_run: str | None = None

    def applies(self, *, cohorts: int, periods_pre: int, periods_post: int) -> bool:
        w = self.applies_when
        if "cohorts" in w and cohorts not in w["cohorts"]:
            return False
        if "cohorts_min" in w and cohorts < w["cohorts_min"]:
            return False
        return periods_pre >= w.get("periods_pre_min", 0) and periods_post >= w.get("periods_post_min", 0)

    def render(self) -> str:
        return f"estimator: {self.name}\n  what it does: {self.in_words}\n  assumes: {self.assumes}\n  weak when: {self.weak_when}"


class InferenceEntry(BaseModel):
    name: str
    in_words: str
    applies_when: dict[str, Any]
    vcov: Any
    resample: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    note: str = ""

    def applies(self, *, units_treated: int, kind: str) -> bool:
        w = self.applies_when
        if "kind" in w and kind not in w["kind"]:
            return False
        return w.get("units_treated_min", 0) <= units_treated <= w.get("units_treated_max", 10**9)


class PlaceboEntry(BaseModel):
    name: str
    in_words: str
    applies_when: dict[str, Any]
    params: dict[str, Any] = Field(default_factory=dict)
    pass_when: dict[str, Any] = Field(default_factory=dict)
    note: str = ""

    def applies(self, *, units: int, periods_pre: int) -> bool:
        w = self.applies_when
        return units >= w.get("units_min", 0) and periods_pre >= w.get("periods_pre_min", 0)


@lru_cache(maxsize=1)
def load_estimators() -> list[EstimatorEntry]:
    raw = yaml.safe_load((_HERE / "estimators.yaml").read_text())
    return sorted((EstimatorEntry(name=k, **v) for k, v in raw.items()), key=lambda e: e.rank)


@lru_cache(maxsize=1)
def load_inference() -> list[InferenceEntry]:
    raw = yaml.safe_load((_HERE / "inference.yaml").read_text())
    return [InferenceEntry(name=k, **v) for k, v in raw.items()]  # file order is precedence


@lru_cache(maxsize=1)
def load_placebos() -> list[PlaceboEntry]:
    raw = yaml.safe_load((_HERE / "placebos.yaml").read_text())
    return [PlaceboEntry(name=k, **v) for k, v in raw.items()]


@lru_cache(maxsize=1)
def load_checks() -> dict[str, Any]:
    return yaml.safe_load((_HERE / "checks.yaml").read_text())


def estimator(name: str) -> EstimatorEntry:
    for e in load_estimators():
        if e.name == name:
            return e
    raise KeyError(name)


def placebo(name: str) -> PlaceboEntry:
    for p in load_placebos():
        if p.name == name:
            return p
    raise KeyError(name)


def pick_inference(*, units_treated: int, kind: str) -> InferenceEntry:
    for entry in load_inference():
        if entry.applies(units_treated=units_treated, kind=kind):
            return entry
    raise LookupError("no inference entry applies")


def render_preferences(entries: list[EstimatorEntry]) -> str:
    names = {e.name for e in entries}
    lines = [f"- prefer {e.name} over {o}: {why}" for e in entries for o, why in e.prefer_over.items() if o in names]
    return "\n".join(lines) or "(none recorded among these)"
