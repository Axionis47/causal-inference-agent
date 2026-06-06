"""Typed output for the domain knowledge agent.

The agent extracts causal-inference context from dataset metadata: hypotheses
about treatment/outcome/confounder roles, temporal ordering, immutable variables
that cannot be caused by treatment, and uncertainties that limit confidence.

This is the typed slot stored at AnalysisState.domain_knowledge.
"""

from pydantic import BaseModel, ConfigDict, Field


class Hypothesis(BaseModel):
    """A causal-role hypothesis the agent recorded during investigation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    claim: str
    confidence: str = "medium"
    evidence: str = ""
    revised: bool = False
    superseded_by: str | None = None


class Uncertainty(BaseModel):
    """A flagged limitation in the metadata or causal interpretation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    issue: str
    impact: str


class DomainKnowledge(BaseModel):
    """Output of the DomainKnowledgeAgent.

    `hypotheses` is the live list (revisions filtered out); `all_hypotheses`
    keeps every claim including superseded ones so downstream agents and the
    notebook can show the evolution.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    hypotheses: list[Hypothesis] = Field(default_factory=list)
    all_hypotheses: list[Hypothesis] = Field(default_factory=list)
    uncertainties: list[Uncertainty] = Field(default_factory=list)
    temporal_understanding: str | None = None
    immutable_vars: list[str] = Field(default_factory=list)
    investigation_complete: bool = False
