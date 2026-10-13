"""The lane's knowledge files: the entry models, and the shared loader bound to this folder. Read once, rendered for the model, filtered by facts."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from causal_agent.common.contracts import load_endpoints
from causal_agent.lane.knowledge import Knowledge, matches
from causal_agent.lane.rungs import load_spec
from causal_agent.lane.space import load_space

_HERE = Path(__file__).parent
SPEC = load_spec(_HERE / "rungs.yaml")  # the ladder, declared once: the graph is built from it and held to it


class EstimatorEntry(BaseModel):
    name: str
    in_words: str
    engine: Literal["local_polynomial", "local_randomisation"] = "local_polynomial"
    params: dict[str, Any] = Field(default_factory=dict)
    fuzzy: Any = False  # True | False | "inherit"
    applies_when: dict[str, Any] = Field(default_factory=dict)
    estimand: str
    assumes: str
    weak_when: str
    prefer_over: dict[str, str] = Field(default_factory=dict)
    rank: int = 99

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
    source: str = ""
    applies_when: dict[str, Any]
    vce: str
    cluster: str | None = None

    def applies(self, **facts: Any) -> bool:
        return matches(self.applies_when, facts)


class DiagnosticEntry(BaseModel):
    name: str
    in_words: str
    source: str = ""
    spec: str = "primary"
    placement: str | None = None
    grid: list[str] = Field(default_factory=list)
    hold_b: bool = False
    radii_share_of_h: list[float] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    applies_when: dict[str, Any] = Field(default_factory=dict)
    pass_when: dict[str, Any] = Field(default_factory=dict)

    def applies(self, **facts: Any) -> bool:
        return matches(self.applies_when, facts)


# ------------------------------------------------------------------ the files, through the shared loader

K = Knowledge(_HERE, estimator=EstimatorEntry, inference=InferenceEntry)
load_estimators, load_inference, load_checks = K.estimators, K.inference, K.checks
estimator, pick_inference, render_preferences = K.estimator, K.pick_inference, K.render_preferences
CATALOGUE = load_endpoints(_HERE / "endpoints.yaml")
SPACE = load_space(_HERE / "space.yaml")


@lru_cache
def _diagnostics(file: str) -> tuple[DiagnosticEntry, ...]:
    entries = yaml.safe_load((_HERE / f"{file}.yaml").read_text()) or {}
    return tuple(DiagnosticEntry(name=n, **v) for n, v in entries.items())


def load_falsifications() -> list[DiagnosticEntry]:
    return list(_diagnostics("falsifications"))


def load_sensitivities() -> list[DiagnosticEntry]:
    return list(_diagnostics("sensitivity"))
