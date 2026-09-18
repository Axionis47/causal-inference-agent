"""Discontinuity lane state. Shares question, handoff, dataset, specialist_result, debug with the router.

Tables never sit in state: `table_path`, `canon_path`, and `xall_path` point at the run directory. Library
objects never sit in state either; every node that needs a fit refits from the Design.
"""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict

from causal_agent.common.contracts import Contrast, Estimate, Feasibility, Handoff, Refutation, Thought
from causal_agent.specialists.rd.contracts import (
    CovariateRelation,
    Covariates,
    Design,
    DesignAssessment,
    EstimatorPick,
    RDInterpretation,
    Score,
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
    canon_path: str
    xall_path: str
    columns: dict[str, str]  # key -> raw name
    cluster_column: str | None
    sampled_by_side: bool
    target_units: str
    score: Score | None
    shape: ShapeFacts | None
    contrast: Contrast | None
    candidates: list[str]
    relations: Annotated[list[CovariateRelation], operator.add]
    relate_errors: dict[str, list[str]]
    relate_attempts: int
    covariates: Covariates | None
    checks: list  # list[CheckResult]
    check_facts: dict
    assessment: DesignAssessment | None
    estimator: str | None
    estimator_pick: EstimatorPick | None
    excluded_estimators: list[str]
    pick_attempts: int
    design: Design | None
    primary: dict
    estimates: Annotated[list[Estimate], operator.add]
    refutations: Annotated[list[Refutation], operator.add]
    interpretations: Annotated[list[RDInterpretation], operator.add]
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
    canon_path: str
    primary: dict
