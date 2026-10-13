"""Reasoning state and immutable per-design execution artifacts."""

from __future__ import annotations

import operator
from typing import Annotated

from causal_agent.common.contracts import Estimate, Falsification, Interpretation, Sensitivity
from causal_agent.families.diff_in_diff.lane.contracts import Candidate, Design, Groups, Periods, ShapeFacts
from causal_agent.lane.ledger import Ledger
from causal_agent.lane.state import LaneState, by_key


class SpecialistState(LaneState, total=False):
    panel_path: str
    groups: Groups
    periods: Periods
    shape: ShapeFacts
    ledger: Ledger
    candidates: list[Candidate]
    designs: dict[str, Design]
    lead_design_id: str | None
    context_deficiency: dict | None
    chosen_estimators: dict[str, str]
    estimates: Annotated[list[Estimate], by_key(lambda e: (e.design_id, e.method, e.modifier, e.level))]
    falsifications: Annotated[list[Falsification], operator.add]
    sensitivities: Annotated[list[Sensitivity], operator.add]
    interpretations: Annotated[list[Interpretation], operator.add]
    reasoning_history: Annotated[list[dict], operator.add]
    candidate_history: Annotated[list[dict], operator.add]
