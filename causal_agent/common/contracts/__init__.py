"""Shared contracts, one package: what the desk writes for a lane (the pack), what the lanes write back (the artifacts), and
the question frame and decision in between. Every name is importable from here, as it always was."""

from __future__ import annotations

from causal_agent.common.contracts.artifacts import (
    CheckResult,
    Checks,
    Contrast,
    Decline,
    Estimate,
    Feasibility,
    Interpretation,
    LaneAsk,
    Refutation,
    RunRecord,
)
from causal_agent.common.contracts.base import (
    Candidate,
    Cited,
    Intent,
    Scope,
    Thought,
)
from causal_agent.common.contracts.frame import (
    FamilyDecision,
    FamilyVerdict,
    NeedCheck,
    PrefilterVote,
    QuestionFrame,
    Rejection,
)
from causal_agent.common.contracts.pack import (
    DESIGNS,
    Belief,
    ColumnBrief,
    ColumnFacts,
    Design,
    Handoff,
    Probe,
    Provenance,
    Role,
    Said,
    When,
    parse_design,
    register_design,
)
from causal_agent.common.contracts.text import (
    render_change_text,
    render_dataset_text,
)

__all__ = [
    "DESIGNS",
    "Belief",
    "Candidate",
    "CheckResult",
    "Checks",
    "Cited",
    "ColumnBrief",
    "ColumnFacts",
    "Contrast",
    "Decline",
    "Design",
    "Estimate",
    "FamilyDecision",
    "FamilyVerdict",
    "Feasibility",
    "Handoff",
    "Intent",
    "Interpretation",
    "LaneAsk",
    "NeedCheck",
    "PrefilterVote",
    "Probe",
    "Provenance",
    "QuestionFrame",
    "Refutation",
    "Rejection",
    "Role",
    "RunRecord",
    "Said",
    "Scope",
    "Thought",
    "When",
    "parse_design",
    "register_design",
    "render_change_text",
    "render_dataset_text",
]
