"""Specialist state. Shares handoff, dataset, specialist_result, debug with the desk; the harness keys with every lane.

The table never sits in state: `table_path` points at the run directory. DoWhy objects never sit in
state either; the per-contrast worker rebuilds the model from the Design.
"""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict

from causal_agent.common.contracts import Contrast, Estimate, Interpretation, Refutation
from causal_agent.families.adjustment.lane.contracts import Design, DesignAssessment, Estimand, EstimatorPick, Graph, Ladder, Revision
from causal_agent.lane.episode import EpisodeLog
from causal_agent.lane.state import InterpretTask as InterpretTask
from causal_agent.lane.state import LaneState, by_key, merge_dicts


class SpecialistState(LaneState, total=False):
    outcome_kind: str
    treatment_kind: str  # the treatment's place in the lane's space, observed at load
    target_units: str
    treatment_levels: list[str]
    contrasts: list[Contrast]
    ladder: Ladder | None  # the rungs climbed so far: the pair, the mechanism, time, the roles, the post-treatment roles
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
    estimates: Annotated[list[Estimate], by_key(lambda e: (e.contrast, e.method, e.modifier, e.level))]  # a re-pick replaces, never duplicates
    refutations: Annotated[list[Refutation], by_key(lambda r: (r.contrast, r.refuter))]
    hidden_dropped: bool
    mismatches: Annotated[dict[str, str], merge_dicts]  # contrast -> why the engine's identified set differed from the frozen design
    interpretations: Annotated[list[Interpretation], operator.add]
    interpret_errors: Annotated[dict[str, list[str]], merge_dicts]


class ContrastTask(TypedDict):
    """Input to one analyse worker: the frozen design and where the table is."""

    contrast: str  # contrast key
    design: dict
    table_path: str


# ------------------------------------------------------------------ reading the state: the reasoning, the execution and the artifacts share these


def ladder_of(state: SpecialistState) -> Ladder:
    lad = state.get("ladder")
    return lad if isinstance(lad, Ladder) else Ladder()


def episodes_of(state: SpecialistState) -> dict[str, EpisodeLog]:
    return {k: v for k, v in (state.get("episodes") or {}).items() if isinstance(v, EpisodeLog)}


def made_here(state: SpecialistState):
    """What this run itself made and a judgement may cite: the ladder's lines and the facts the episodes asked for."""
    lad, logs = ladder_of(state), list(episodes_of(state).values())
    return lambda a: lad.resolve(a) or any(log.resolve(a) for log in logs)


def latest_primary(state: SpecialistState) -> dict[str, Estimate]:
    """Per contrast, the primary estimate of the estimator picked last."""
    out: dict[str, Estimate] = {}
    for e in state.get("estimates") or []:
        if not e.secondary and e.modifier is None and e.method == state["estimator"]:
            out[e.contrast] = e
    return out
