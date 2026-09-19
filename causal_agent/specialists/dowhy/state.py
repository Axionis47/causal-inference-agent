"""Specialist state. Shares handoff, dataset, specialist_result, debug with the router; the rest is its own.

The table never sits in state: `table_path` points at the run directory. DoWhy objects never sit in
state either; the per-contrast worker rebuilds the model from the Design.
"""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict

from causal_agent.common.contracts import Contrast, Estimate, Feasibility, Handoff, Interpretation, Refutation, Thought
from causal_agent.specialists.dowhy.contracts import Design, DesignAssessment, Estimand, EstimatorPick, Graph, Relation, Revision


class SpecialistState(TypedDict, total=False):
    # shared with the parent graph
    question: str
    handoff: Handoff | None
    dataset: str
    specialist_result: dict | None
    debug: Annotated[list[Thought], operator.add]

    # this lane
    run_dir: str
    table_path: str
    columns: dict[str, str]  # key -> raw name
    outcome_kind: str
    target_units: str
    treatment_levels: list[str]
    contrasts: list[Contrast]
    relations: Annotated[list[Relation], operator.add]
    relate_errors: dict[str, list[str]]  # column key -> errors from verify_graph, reruns only these
    relate_attempts: int
    graph: Graph | None
    estimand: Estimand | None
    checks: list  # list[CheckResult]
    assessment: DesignAssessment | None
    revisions: int
    applied_revisions: list[Revision]
    estimator: str | None
    estimator_pick: EstimatorPick | None
    excluded_estimators: list[str]
    pick_attempts: int
    design: Design | None
    estimates: Annotated[list[Estimate], operator.add]
    refutations: Annotated[list[Refutation], operator.add]
    hidden_dropped: bool
    ask: dict | None
    interpretations: Annotated[list[Interpretation], operator.add]
    interpret_errors: dict[str, list[str]]
    interpret_attempts: int
    feasibility: Feasibility | None
    report: str


class RelateTask(TypedDict):
    """Input to one relate worker. Three cards and the question; never the parent state."""

    question: str
    frame: str
    treatment_card: str
    outcome_card: str
    column: str
    card: str
    errors: str


class ContrastTask(TypedDict):
    """Input to one analyse worker: the frozen design and where the table is."""

    contrast: str  # contrast key
    design: dict
    table_path: str


class InterpretTask(TypedDict):
    question: str
    contrast: str
    material: str
    addresses: str
    errors: str
    primary_value: float | None
    tolerance: float
