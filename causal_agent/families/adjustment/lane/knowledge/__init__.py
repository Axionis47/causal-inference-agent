"""Executable method declarations owned by the adjustment family."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import networkx as nx
import yaml
from pydantic import BaseModel, Field, model_validator

from causal_agent.lane.rungs import load_spec
from causal_agent.lane.space import load_space

_HERE = Path(__file__).parent
SPEC = load_spec(_HERE / "rungs.yaml")
SPACE = load_space(_HERE / "space.yaml")


class EstimatorEntry(BaseModel):
    name: str
    method: str
    endpoints: list[str]
    targets: list[str]
    outcomes: list[str]
    in_words: str
    assumes: str
    params: dict[str, Any] = Field(default_factory=dict)
    rank: int = 99

    def render(self) -> str:
        return f"[estimator:{self.name}] {self.in_words}; assumes {self.assumes}"


class FalsificationEntry(BaseModel):
    name: str
    endpoints: list[str]
    in_words: str
    params: dict = Field(default_factory=dict)
    pass_when: dict = Field(default_factory=dict)


class SensitivityEntry(BaseModel):
    name: str
    endpoints: list[str]
    in_words: str
    params: dict = Field(default_factory=dict)


def _entries(filename: str, cls):
    return [cls(name=name, **raw) for name, raw in yaml.safe_load((_HERE / filename).read_text()).items()]


def load_estimators() -> list[EstimatorEntry]:
    return _entries("estimators.yaml", EstimatorEntry)


def load_falsifications() -> list[FalsificationEntry]:
    return _entries("falsifications.yaml", FalsificationEntry)


def load_sensitivities() -> list[SensitivityEntry]:
    return _entries("sensitivity.yaml", SensitivityEntry)


def load_checks() -> dict:
    return yaml.safe_load((_HERE / "checks.yaml").read_text())


def load_endpoint_catalogue():
    from causal_agent.common.contracts.endpoints import load_endpoints

    return load_endpoints(_HERE / "endpoints.yaml")


class MotifCertificate(BaseModel):
    road: str
    variables: list[str] | None = None
    target: str = "ate"


class GraphMotif(BaseModel):
    id: str
    title: str
    category: Literal["base", "variant", "blocked"] = "base"
    account: str
    treatment: str = "t"
    outcome: str = "y"
    observed: list[str]
    latent: list[str] = Field(default_factory=list)
    edges: list[tuple[str, str]]
    evidence: list[str]
    assumptions: list[str]
    expected: list[MotifCertificate] = Field(default_factory=list)
    forbidden_adjustments: list[list[str]] = Field(default_factory=list)
    mediator: str | None = None
    modifiers: list[str] = Field(default_factory=list)
    stop: str | None = None

    @model_validator(mode="after")
    def valid_graph(self):
        nodes = self.observed + self.latent
        if len(nodes) != len(set(nodes)) or self.treatment not in self.observed or self.outcome not in self.observed:
            raise ValueError("motif nodes must be unique, with observed treatment and outcome")
        if any(src not in nodes or dst not in nodes for src, dst in self.edges):
            raise ValueError("motif edge references an undeclared node")
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(self.edges)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("motif must be a DAG")
        if self.mediator and self.mediator not in self.observed:
            raise ValueError("motif mediator must be observed")
        if any(modifier not in self.observed for modifier in self.modifiers):
            raise ValueError("motif modifier must be observed")
        if any(column not in nodes for group in self.forbidden_adjustments for column in group):
            raise ValueError("forbidden adjustment references an undeclared node")
        return self


def load_motifs() -> list[GraphMotif]:
    """Illustrative DAGs, never additional context or endpoint eligibility rules."""
    rows = [GraphMotif.model_validate(row) for row in yaml.safe_load((_HERE / "motifs.yaml").read_text())["motifs"]]
    if len({row.id for row in rows}) != len(rows) or sum(row.category == "base" for row in rows) != 10:
        raise ValueError("motifs require unique IDs and the ten illustrative base examples")
    return rows
