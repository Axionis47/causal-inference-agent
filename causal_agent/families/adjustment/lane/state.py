"""Specialist state. Shares handoff, dataset, specialist_result, debug with the desk; the harness keys with every lane.

The table never sits in state: `table_path` points at the run directory. DoWhy objects never sit in
state either; the per-contrast worker rebuilds the model from the Design.
"""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict

from causal_agent.common.contracts import Contrast, Estimate, Interpretation, Refutation
from causal_agent.families.adjustment.lane.contracts import Design, DesignAssessment, Estimand, EstimatorPick, Graph, Relation, Revision
from causal_agent.lane.state import InterpretTask as InterpretTask
from causal_agent.lane.state import LaneState, by_key, merge_dicts
from causal_agent.lane.state import RelateTask as BaseRelateTask


class SpecialistState(LaneState, total=False):
    outcome_kind: str
    target_units: str
    treatment_levels: list[str]
    contrasts: list[Contrast]
    relations: Annotated[list[Relation], operator.add]
    relate_errors: dict[str, list[str]]  # column key -> errors from verify_graph, reruns only these
    relate_attempts: int
    graph: Graph | None
    estimand: Estimand | None
    assessment: DesignAssessment | None
    revisions: int
    applied_revisions: list[Revision]
    estimator: str | None
    estimator_pick: EstimatorPick | None
    excluded_estimators: list[str]
    pick_attempts: int
    design: Design | None
    estimates: Annotated[list[Estimate], by_key(lambda e: (e.contrast, e.method))]  # a re-pick replaces, never duplicates
    refutations: Annotated[list[Refutation], by_key(lambda r: (r.contrast, r.refuter))]
    hidden_dropped: bool
    interpretations: Annotated[list[Interpretation], operator.add]
    interpret_errors: Annotated[dict[str, list[str]], merge_dicts]
    interpret_attempts: int


class RelateTask(BaseRelateTask):
    """The adjustment lane shows the treatment's and the outcome's cards beside the column's."""

    treatment_card: str
    outcome_card: str


class ContrastTask(TypedDict):
    """Input to one analyse worker: the frozen design and where the table is."""

    contrast: str  # contrast key
    design: dict
    table_path: str
