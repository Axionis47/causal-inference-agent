"""Only frozen designs cross from adjustment reasoning into execution."""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict

from causal_agent.common.contracts import ContextDeficiency, Estimate, Falsification, Sensitivity
from causal_agent.families.adjustment.lane.contracts import Candidate, Design, Ladder
from causal_agent.lane.ledger import Ledger
from causal_agent.lane.state import LaneState, by_key, merge_dicts


class SpecialistState(LaneState, total=False):
    outcome_kind: str
    source_row_ids: list[int]
    contrasts: list
    ladder: Ladder
    ledger: Ledger
    candidates: list[Candidate]
    designs: dict[str, Design]
    lead_design_id: str
    context_deficiency: ContextDeficiency | None
    relate_retries: int
    reasoning_errors: list[str]
    diagnostic_candidates: list[dict]
    candidate_history: Annotated[list[dict], operator.add]
    reasoning_history: Annotated[list[dict], operator.add]
    execution_attempts: Annotated[list[dict], operator.add]
    comparisons: list[dict]
    estimates: Annotated[list[Estimate], by_key(lambda e: (e.design_id, e.contrast, e.method, e.modifier, e.level))]
    falsifications: Annotated[list[Falsification], by_key(lambda r: (r.design_id, r.contrast, r.refuter))]
    sensitivities: Annotated[list[Sensitivity], by_key(lambda r: (r.design_id, r.contrast, r.method))]
    interpretations: Annotated[list[dict], operator.add]
    execution_failures: Annotated[dict, merge_dicts]


class ContrastTask(TypedDict):
    contrast: str
    design: dict
    table_path: str


def ladder_of(state: SpecialistState) -> Ladder:
    return state.get("ladder") or Ladder()
