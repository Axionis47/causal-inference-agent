"""Diff-in-diff lane state. Shares the harness keys with every lane; the rest is its own.

The panel never sits in state: `panel_path` points at the run directory. pyfixest objects never sit in
state either; the estimate node refits from the Design.
"""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict

from causal_agent.common.contracts import Contrast, Estimate, Interpretation, Refutation
from causal_agent.families.diff_in_diff.lane.contracts import (
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
from causal_agent.lane.state import LaneState, by_key, merge_dicts


class SpecialistState(LaneState, total=False):
    panel_path: str
    unit_column: str | None
    cluster_column: str | None
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
    assessment: DesignAssessment | None
    estimator: str | None
    estimator_pick: EstimatorPick | None
    excluded_estimators: list[str]
    pick_attempts: int
    design: Design | None
    estimates: Annotated[list[Estimate], by_key(lambda e: (e.contrast, e.method))]
    refutations: Annotated[list[Refutation], by_key(lambda r: (r.contrast, r.refuter))]
    dynamic: dict  # rel_time -> (value, lo, hi) when the dynamic model ran
    placebo_draws: Annotated[dict, merge_dicts]  # placebo name -> every placebo effect, so the spread can be drawn
    interpretations: Annotated[list[Interpretation], operator.add]
    interpret_errors: Annotated[dict[str, list[str]], merge_dicts]


class RelateTask(TypedDict):
    question: str
    frame: str
    column: str
    card: str
    settled: str
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
