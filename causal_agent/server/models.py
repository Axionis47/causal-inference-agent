"""Request and response shapes. Views are projections of the graph's own contracts; nothing here is judged."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from causal_agent.common.contracts import ConfidenceSet, ContextAdmission, ContextDeficiency, Falsification, Sensitivity
from causal_agent.memory.context.contracts import ColumnIdentity
from causal_agent.memory.context.conversation import GapQuestion, Reference
from causal_agent.memory.context.review import ColumnDisposition, ContextEdit, KnowledgeStatement
from causal_agent.viz.spec import FigureSpec

# ------------------------------------------------------------------ requests


class IngestedDataset(BaseModel):
    dataset_id: str
    version_id: str
    title: str
    filename: str
    size_bytes: int
    sha256: str
    initial_context: str | None
    created_at: datetime
    status: Literal["saved"] = "saved"


class ConversationView(BaseModel):
    id: str
    dataset_id: str
    version_id: str
    title: str
    created_at: datetime


class ReviewStart(BaseModel):
    question: str = Field(min_length=1, max_length=10000)


class ReviewSave(BaseModel):
    generation: int = Field(ge=1)
    edits: list[ContextEdit] = Field(default_factory=list, max_length=1000)
    dispositions: list[ColumnDisposition] = Field(default_factory=list, max_length=200)


class ReviewFinish(BaseModel):
    generation: int = Field(ge=1)


class ReviewView(BaseModel):
    id: str
    dataset_id: str
    version_id: str
    question: str
    starting_revision_id: str
    completed_revision_id: str | None
    generation: int
    columns: list[ColumnIdentity]
    edits: list[ContextEdit]
    dispositions: list[ColumnDisposition]


class ReviewStatements(BaseModel):
    statements: list[KnowledgeStatement]
    next_offset: int | None


class ConversationStart(BaseModel):
    review_id: str
    generation: int = Field(ge=1)


class AnalysisReview(BaseModel):
    id: str
    revision_id: str
    question: str
    family: str
    assumptions: str


class ReadinessSummary(BaseModel):
    status: Literal["unprepared", "needs_context", "ready", "blocked"]
    missing: int


class EvidenceView(BaseModel):
    title: str
    text: str
    quotes: list[str] = Field(default_factory=list)


class ChatOverview(BaseModel):
    name: str
    dataset_id: str
    title: str
    question: str
    version_id: str
    revision_id: str
    status: Literal["idle", "running", "interrupted"]
    prompt: str
    focus: Reference | None
    focus_label: str | None = None
    fast_track: bool
    questions: list[GapQuestion]
    analysis_review: AnalysisReview | None
    message_count: int
    error: str | None
    readiness: ReadinessSummary


class ChatMessage(BaseModel):
    id: str
    role: str
    text: str
    references: list[Reference]
    created_at: datetime


class MessagePage(BaseModel):
    items: list[ChatMessage]
    before: str | None


class ChatContextSummary(BaseModel):
    revision_id: str
    revision: int
    columns: list[ColumnIdentity]
    dispositions: list[ColumnDisposition]
    latest_revision_id: str


class ContextChange(BaseModel):
    before: KnowledgeStatement | None
    after: KnowledgeStatement | None


class ContextChanges(BaseModel):
    from_revision_id: str
    to_revision_id: str
    items: list[ContextChange]
    next_offset: int | None


class ChatStatements(BaseModel):
    revision_id: str
    statements: list[KnowledgeStatement]
    next_offset: int | None


# ------------------------------------------------------------------ datasets


class DatasetListItem(BaseModel):
    dataset_id: str
    version_id: str
    title: str
    filename: str
    size_bytes: int
    created_at: datetime
    status: Literal["saved"] = "saved"
    has_initial_context: bool


class DatasetList(BaseModel):
    items: list[DatasetListItem]
    next_cursor: str | None


# ------------------------------------------------------------------ sessions


class CheckView(BaseModel):
    design_id: str = ""
    contrast: str | None = None
    name: str
    level: str | None = None
    value: Any = None
    threshold: Any = None
    detail: str | None = None


class InterpretationView(BaseModel):
    design_id: str = ""
    contrast: str | None = None
    answer: str
    caveats: list[str] = Field(default_factory=list)
    cites: list[str] = Field(default_factory=list)


class EstimateView(BaseModel):
    design_id: str = ""
    endpoint_id: str = ""
    target: dict[str, Any] = Field(default_factory=dict)
    confidence_set: ConfidenceSet | None = None
    modifier: str | None = None
    level: str | None = None
    contrast: str | None = None
    method: str | None = None
    value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n: int | None = None
    n_treated: int | None = None
    n_control: int | None = None
    secondary: bool = False
    error: str | None = None
    p_value: float | None = None
    p_value_source: str | None = None


class DeclineView(BaseModel):
    """Where the lane did not take the pack as given."""

    address: str
    stage: str
    kind: str
    about: str
    pack_value: str | None = None
    took: str | None = None
    reason: str
    check: str


class DecisionView(BaseModel):
    """The routing's decision as the page shows it."""

    chosen: str | None = None
    chosen_assumption: str | None = None
    why: str | None = None
    over: dict[str, str] = Field(default_factory=dict, description="rejected family -> reason")


class FeasibilityView(BaseModel):
    """Where and why a lane stopped."""

    terminal_kind: Literal["context_deficiency", "unidentified_effect", "unsupported_method", "inadequate_support", "exhausted_reasoning", "execution_failure"]
    stage: str
    reason: str
    facts: list[str] = Field(default_factory=list)
    what_would_fix: str = ""


class RunView(BaseModel):
    index: int
    question: str
    family: str | None = None
    specialist: str | None = None
    status: str
    run_id: str | None = None
    effect: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    estimator: str | None = None
    decision: DecisionView = Field(default_factory=DecisionView)
    decision_record: str = ""
    flags: list[CheckView] = Field(default_factory=list)
    checks: list[CheckView] = Field(default_factory=list)
    lead_design_id: str | None = None
    lead_selection: dict[str, Any] = Field(default_factory=dict)
    context_id: str = ""
    dataset_identity: str = ""
    context_admission: ContextAdmission | None = None
    context_deficiency: ContextDeficiency | None = None
    terminal: dict[str, Any] = Field(default_factory=dict)
    candidate_history: list[dict[str, Any]] = Field(default_factory=list)
    reasoning_history: list[dict[str, Any]] = Field(default_factory=list)
    context_facts: dict[str, Any] = Field(default_factory=dict)
    designs: dict[str, dict[str, Any]] = Field(default_factory=dict)
    ledger: dict[str, Any] = Field(default_factory=dict)
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    episode_visits: dict[str, Any] = Field(default_factory=dict)
    comparisons: list[dict[str, Any]] = Field(default_factory=list)
    execution_attempts: list[dict[str, Any]] = Field(default_factory=list)
    falsifications: list[Falsification] = Field(default_factory=list)
    sensitivities: list[Sensitivity] = Field(default_factory=list)
    interpretations: list[InterpretationView] = Field(default_factory=list)
    estimates: list[EstimateView] = Field(default_factory=list)
    feasibility: FeasibilityView | None = None
    files: list[str] = Field(default_factory=list)
    what_if: dict[str, str] = Field(default_factory=dict)
    differs: list[str] = Field(default_factory=list)
    figures: list[FigureSpec] = Field(default_factory=list, description="what the run left, in the order the chat shows them")
    declines: list[DeclineView] = Field(default_factory=list)


class ArtifactSummary(BaseModel):
    id: str
    ask: str
    origin: Literal["exploration", "analysis"]
    format: Literal["figure", "table", "report", "result"]
    summary: str
    conversation_id: str
    dataset_version_id: str
    design: int | None
    made_at: str
    url: str | None


class AnalysisSummary(BaseModel):
    id: str
    index: int
    question: str
    family: str | None
    status: str
    lead_design_id: str | None


class FileEntry(BaseModel):
    name: str
    size: int


class RunFiles(BaseModel):
    run_id: str
    files: list[FileEntry]
