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
    CovariateRelation,
    Covariates,
    Design,
    DesignAssessment,
    EstimatorPick,
    RDInterpretation,
    Score,
    ShapeFacts,
)
from causal_agent.lane.state import LaneState, by_key, merge_dicts
from causal_agent.lane.state import RelateTask as RelateTask


class SpecialistState(LaneState, total=False):
    canon_path: str
    xall_path: str
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
    assessment: DesignAssessment | None
    estimator: str | None
    estimator_pick: EstimatorPick | None
    excluded_estimators: list[str]
    pick_attempts: int
    design: Design | None
    primary: dict
    estimates: Annotated[list[Estimate], by_key(lambda e: (e.contrast, e.method))]
    refutations: Annotated[list[Refutation], by_key(lambda r: (r.contrast, r.refuter))]
    placebo_points: Annotated[dict, merge_dicts]  # placebo name -> every refit, so the curve and the cutoffs can be drawn
    interpretations: Annotated[list[RDInterpretation], operator.add]
    interpret_errors: Annotated[dict[str, list[str]], merge_dicts]


class PlaceboTask(TypedDict):
    name: str
    design: dict
    canon_path: str
    primary: dict
