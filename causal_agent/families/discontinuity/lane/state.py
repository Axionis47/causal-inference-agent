"""RD workflow state. Frozen designs and independent worker records are keyed by design."""

from __future__ import annotations

import operator
from typing import Annotated

from typing_extensions import TypedDict

from causal_agent.common.contracts import ContextDeficiency, Contrast, Estimate, Falsification, Sensitivity
from causal_agent.families.discontinuity.lane.contracts import AssumptionDecisions, Candidate, Covariates, Design, Ladder, RDInterpretation, Score, ShapeFacts
from causal_agent.lane.ledger import Ledger
from causal_agent.lane.state import LaneState, by_key, merge_dicts


class SpecialistState(LaneState, total=False):
    canon_path: str
    xall_path: str
    cluster_column: str | None
    sampled_by_side: bool
    density_facts: dict
    score: Score
    shape: ShapeFacts
    contrast: Contrast
    column_candidates: list[str]
    covariates: Covariates
    ladder: Ladder
    ledger: Ledger
    assumption_decisions: AssumptionDecisions
    candidates: list[Candidate]
    candidate_history: Annotated[list[dict], operator.add]
    reasoning_history: Annotated[list[dict], operator.add]
    execution_attempts: Annotated[list[dict], operator.add]
    comparisons: list[dict]
    designs: dict[str, Design]
    lead_design_id: str | None
    context_deficiency: ContextDeficiency | None
    episode_visits: Annotated[list[dict], operator.add]
    estimates: Annotated[list[Estimate], by_key(lambda e: (e.design_id, e.contrast, e.method, e.modifier, e.level))]
    falsifications: Annotated[list[Falsification], by_key(lambda f: (f.design_id, f.contrast, f.refuter))]
    sensitivities: Annotated[list[Sensitivity], by_key(lambda s: (s.design_id, s.contrast, s.method))]
    interpretations: Annotated[list[RDInterpretation], operator.add]
    interpretation_errors: Annotated[dict[str, list[str]], merge_dicts]


class ExecuteTask(TypedDict):
    design: dict
    canon_path: str
