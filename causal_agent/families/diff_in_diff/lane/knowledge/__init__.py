"""Executable endpoint, estimator and robustness catalogues."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from causal_agent.common.contracts.endpoints import load_endpoints
from causal_agent.lane.knowledge import matches
from causal_agent.lane.rungs import load_spec
from causal_agent.lane.space import load_space

HERE = Path(__file__).parent
SPEC = load_spec(HERE / "rungs.yaml")
SPACE = load_space(HERE / "space.yaml")
ENDPOINTS = load_endpoints(HERE / "endpoints.yaml")


class EstimatorEntry(BaseModel):
    name: str
    engine: Literal["feols", "did2s", "saturated", "lpdid"]
    aggregation: list[str]
    endpoints: list[str]
    rank: int
    in_words: str

    @property
    def applies_when(self) -> dict:
        return {"endpoint": self.endpoints}

    def applies(self, **facts: Any) -> bool:
        return facts.get("endpoint") in self.endpoints and facts.get("aggregation", "pooled") in self.aggregation

    def render(self) -> str:
        return f"[estimator:{self.name}] {self.in_words}"


class RobustnessEntry(BaseModel):
    name: str
    in_words: str
    applies_when: dict
    params: dict = Field(default_factory=dict)

    def applies(self, **facts: Any) -> bool:
        return matches(self.applies_when, facts)

    def render(self) -> str:
        return f"{self.name}: {self.in_words}"


def load_estimators() -> list[EstimatorEntry]:
    raw = yaml.safe_load((HERE / "estimators.yaml").read_text())
    return [EstimatorEntry(name=n, **v) for n, v in raw.items()]


def estimator(name: str) -> EstimatorEntry:
    return next(e for e in load_estimators() if e.name == name)


def _robustness(name: str) -> list[RobustnessEntry]:
    return [RobustnessEntry(name=n, **v) for n, v in yaml.safe_load((HERE / name).read_text()).items()]


def load_falsifications() -> list[RobustnessEntry]:
    return _robustness("falsifications.yaml")


def load_sensitivities() -> list[RobustnessEntry]:
    return _robustness("sensitivity.yaml")


def load_checks() -> dict:
    return yaml.safe_load((HERE / "checks.yaml").read_text())
