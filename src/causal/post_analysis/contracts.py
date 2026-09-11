"""Small public contracts. Authority stays in referenced upstream artifacts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from causal.post_analysis.visualization.contracts import CausalDiagram, DataTable
from causal.shared.contracts import ArtifactRef


class Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class InputIssue(Model):
    code: str
    owner: Literal["analysis", "design", "post_analysis", "storage"]
    source: ArtifactRef | None = None
    path: str
    expected: Any
    received: Any
    required_action: str


class InputError(ValueError):
    def __init__(self, *issues: InputIssue) -> None:
        self.issues = issues
        super().__init__("; ".join(issue.code for issue in issues))


@dataclass(frozen=True)
class EvidencePacket:
    """Derived, read-only view; never an alternative scientific record."""
    analysis_id: str
    sources: dict[str, ArtifactRef]
    evidence: dict[str, dict[str, Any]]
    tables: dict[str, DataTable]
    required_evidence: tuple[str, ...]
    diagram: CausalDiagram | None = None
    limitations: tuple[str, ...] = ()


class Citation(Model):
    evidence_id: str
    selector: str = ""


class Statement(Model):
    text: Annotated[str, Field(min_length=1, max_length=4000)]
    citations: Annotated[tuple[Citation, ...], Field(min_length=1, max_length=30)]


class Section(Model):
    title: Annotated[str, Field(min_length=1, max_length=200)]
    statements: Annotated[tuple[Statement, ...], Field(min_length=1, max_length=30)]
    visual_ids: tuple[str, ...] = ()


class ReportDraft(Model):
    title: Annotated[str, Field(min_length=1, max_length=200)]
    sections: Annotated[tuple[Section, ...], Field(min_length=1, max_length=12)]
    # Every expected result must be accounted for, including unavailable/failed results.
    coverage: dict[str, Annotated[str, Field(min_length=1, max_length=1000)]]


class Action(Model):
    tool: Literal["read_evidence", "render_visual", "render_dag", "write_report", "submit", "stop"]
    arguments: dict[str, Any] = {}
    decision_summary: Annotated[str, Field(min_length=1, max_length=2000)]


class Review(Model):
    verdict: Literal["pass", "revise"]
    issues: tuple[str, ...]
    decision_summary: Annotated[str, Field(min_length=1, max_length=3000)]


@dataclass(frozen=True)
class PostAnalysisResult:
    status: Literal["complete", "blocked", "incomplete"]
    analysis_id: str
    stage_run_id: str
    thread_id: str
    bundle: ArtifactRef | None = None
    issues: tuple[InputIssue, ...] = ()
    error_code: str | None = None
    counters: dict[str, int] = field(default_factory=dict)
