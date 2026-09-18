"""Diff-in-diff lane state. Shares question, handoff, dataset, specialist_result, debug with the router.

The panel never sits in state: `panel_path` points at the run directory. pyfixest objects never sit in
state either; the estimate node refits from the Design.
"""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict

from causal_agent.common.contracts import Contrast, Estimate, Feasibility, Handoff, Interpretation, Refutation, Thought
from causal_agent.specialists.did.contracts import (
    ControlRelation,
    Controls,
    Design,
    DesignAssessment,
    EstimatorPick,
    Groups,
    Periods,
    Revision,
    ShapeFacts,
)


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
    panel_path: str
    columns: dict[str, str]  # key -> raw name
    unit_column: str | None
    target_units: str
    group_levels: list[str]
    groups: Groups | None
    periods: Periods | None
    contrast: Contrast | None
    shape: ShapeFacts | None
    relations: Annotated[list[ControlRelation], operator.add]
    relate_errors: dict[str, list[str]]
    relate_attempts: int
    controls: Controls | None
    applied_revisions: list[Revision]
    revisions: int
    checks: list  # list[CheckResult]
    assessment: DesignAssessment | None
    estimator: str | None
    estimator_pick: EstimatorPick | None
    excluded_estimators: list[str]
    pick_attempts: int
    design: Design | None
    estimates: Annotated[list[Estimate], operator.add]
    refutations: Annotated[list[Refutation], operator.add]
    dynamic: dict  # rel_time -> (value, lo, hi) when the dynamic model ran
    interpretations: Annotated[list[Interpretation], operator.add]
    interpret_errors: dict[str, list[str]]
    feasibility: Feasibility | None
    report: str


class RelateTask(TypedDict):
    question: str
    frame: str
    column: str
    card: str
    errors: str


class PlaceboTask(TypedDict):
    name: str
    design: dict
    panel_path: str
    observed: float


class InterpretTask(TypedDict):
    question: str
    contrast: str
    material: str
    addresses: str
    errors: str
    primary_value: float | None
    tolerance: float
