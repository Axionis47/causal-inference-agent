"""Discontinuity lane state. Shares the harness keys with every lane; the rest is its own.

Tables never sit in state: `table_path`, `canon_path`, and `xall_path` point at the run directory. Library
objects never sit in state either; every node that needs a fit refits from the Design.
"""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict

from causal_agent.common.contracts import Contrast, Estimate, Refutation
from causal_agent.families.discontinuity.lane.contracts import (
    Covariates,
    Design,
    DesignAssessment,
    EstimatorPick,
    Ladder,
    RDInterpretation,
    Score,
    ShapeFacts,
)
from causal_agent.lane.state import LaneState, by_key, merge_dicts


class SpecialistState(LaneState, total=False):
    canon_path: str
    xall_path: str
    cluster_column: str | None
    sampled_by_side: bool
    density_facts: dict  # the density test as the library returned it, computed once by the density rung
    window_table: dict  # every selector's widths and rows, as the window rung built it
    target_units: str
    score: Score | None
    shape: ShapeFacts | None
    contrast: Contrast | None
    candidates: list[str]
    ladder: Ladder | None  # the rungs climbed so far
    covariates: Covariates | None
    assessment: DesignAssessment | None
    estimator: str | None
    estimator_pick: EstimatorPick | None
    excluded_estimators: list[str]
    pick_attempts: int
    design: Design | None
    primary: dict
    estimates: Annotated[list[Estimate], by_key(lambda e: (e.contrast, e.method, e.modifier, e.level))]
    refutations: Annotated[list[Refutation], by_key(lambda r: (r.contrast, r.refuter))]
    placebo_points: Annotated[dict, merge_dicts]  # placebo name -> every refit, so the curve and the cutoffs can be drawn
    interpretations: Annotated[list[RDInterpretation], operator.add]
    interpret_errors: Annotated[dict[str, list[str]], merge_dicts]


class PlaceboTask(TypedDict):
    name: str
    design: dict
    canon_path: str
    primary: dict
