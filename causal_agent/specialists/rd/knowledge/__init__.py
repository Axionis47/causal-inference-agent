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
    params: dict[str, Any] = Field(default_factory=dict)
    fuzzy: Any = False  # True | False | "inherit"
    covs: bool = False
    pickable: bool = True
    applies_when: dict[str, Any] = Field(default_factory=dict)
    runs_when: dict[str, Any] = Field(default_factory=dict)
    estimand: str
    assumes: str
    weak_when: str
    prefer_over: dict[str, str] = Field(default_factory=dict)
    rank: int = 99
    also_run: list[str] = Field(default_factory=list)

    def applies(self, *, kind: str, first_stage: str | None, distinct_scores: int) -> bool:
        w = self.applies_when
        if "kind" in w and kind not in w["kind"]:
            return False
        if "first_stage" in w and first_stage not in w["first_stage"]:
            return False
        if "distinct_scores_max" in w and distinct_scores > w["distinct_scores_max"]:
            return False
        return True

    def render(self) -> str:
        return f"estimator: {self.name}\n  what it does: {self.in_words}\n  assumes: {self.assumes}\n  weak when: {self.weak_when}"


class InferenceEntry(BaseModel):
    name: str
    in_words: str
    applies_when: dict[str, Any]
    vce: str
    cluster: str | None = None
    point_row: str = "Conventional"
    interval_row: str = "Robust"

    def applies(self, *, cluster_column: bool) -> bool:
        w = self.applies_when
        return "cluster_column" not in w or w["cluster_column"] == cluster_column


class PlaceboEntry(BaseModel):
    name: str
    in_words: str
    source: str = ""
    spec: str = "primary"
    placement: str | None = None
    grid: list[str] = Field(default_factory=list)
    hold_b: bool = False
    radii_share_of_h: list[float] = Field(default_factory=list)
    applies_when: dict[str, Any] = Field(default_factory=dict)
    pass_when: dict[str, Any] = Field(default_factory=dict)

    def applies(self) -> bool:
        return True


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


def pick_inference(*, cluster_column: bool) -> InferenceEntry:
    for entry in load_inference():
        if entry.applies(cluster_column=cluster_column):
            return entry
    raise LookupError("no inference entry applies")


def render_preferences(entries: list[EstimatorEntry]) -> str:
    names = {e.name for e in entries}
    lines = [f"- prefer {e.name} over {o}: {why}" for e in entries for o, why in e.prefer_over.items() if o in names]
    return "\n".join(lines) or "(none recorded among these)"
