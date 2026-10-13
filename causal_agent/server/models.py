"""Request and response shapes. Views are projections of the graph's own contracts; nothing here is judged."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from causal_agent.common.contracts import ConfidenceSet, ContextAdmission, ContextDeficiency, Falsification, Sensitivity
from causal_agent.viz.spec import FigureSpec

NAME_RE = r"^[a-z][a-z0-9_]{1,39}$"


# ------------------------------------------------------------------ requests


class DatasetCreate(BaseModel):
    """CSV only. The conversation starts with the causal question; nothing about the data is typed into a form."""

    name: str = Field(pattern=NAME_RE)
    title: str = Field(min_length=1, max_length=120)
    upload_id: str = Field(pattern=r"^[0-9a-f]{8}$")


class MessageIn(BaseModel):
    text: str = Field(min_length=1)


# ------------------------------------------------------------------ datasets


class NumericShape(BaseModel):
    min: float
    p25: float
    p50: float
    p75: float
    max: float
    mean: float


class TopValue(BaseModel):
    value: str
    count: int
    share: float


class DatetimeShape(BaseModel):
    first: str
    last: str
    frequency: str | None = None


class Sentinel(BaseModel):
    value: str
    count: int
    reason: str


class ColumnSummary(BaseModel):
    """One column as the profiler read it. Facts only: shape, missingness, and anything odd."""

    name: str
    key: str
    kind: str
    nulls: int
    null_rate: float = 0.0
    distinct: int
    constant: bool = False
    examples: list[str]
    numeric: NumericShape | None = None
    top_values: list[TopValue] = Field(default_factory=list)
    datetime: DatetimeShape | None = None
    sentinels: list[Sentinel] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class ProfileOut(BaseModel):
    upload_id: str
    filename: str
    rows: int
    columns: list[ColumnSummary]
    head: list[list[str]] = Field(default_factory=list)
    duplicate_rows: int = 0
    candidate_keys: list[list[str]] = Field(default_factory=list)
    grain: list[str] | None = None
    co_missing: list[list[str]] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class SessionBrief(BaseModel):
    stage: str
    phase: str
    runs: int


class DatasetSummary(BaseModel):
    name: str
    title: str
    csv: str
    rows: int | None = None
    columns: int | None = None
    created_at: str | None = None
    shipped: bool
    has_claims: bool
    question: str | None = None
    session: SessionBrief | None = None


class DatasetList(BaseModel):
    datasets: list[DatasetSummary]


# ------------------------------------------------------------------ sessions


class AskItemView(BaseModel):
    """One line of a readback the page can answer on its own: picking another option or typing a value sends `address = value`."""

    address: str
    label: str
    value: str
    options: list[str] = Field(default_factory=list)
    recommended: str | None = None


class QuestionView(BaseModel):
    """The one question the desk asks this turn, as the page shows it: the field addresses it settles, its form, its chips, the
    option the desk recommends first, and for a readback each draft as an item."""

    keys: list[str]
    kind: str
    text: str
    options: list[str] = Field(default_factory=list)
    recommended: str | None = None
    items: list[AskItemView] = Field(default_factory=list)
    evidence_cites: list[str] = Field(default_factory=list)
    because: list[str] = Field(default_factory=list, description="the families that need these fields")


class ClaimView(BaseModel):
    key: str
    kind: str
    status: str
    fields: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    evidence: list[str] = Field(default_factory=list)
    check_detail: str | None = None
    asked: int = 0
    refutations: int = 0


class StatusView(BaseModel):
    endpoint_admissions: dict[str, ContextAdmission] = Field(default_factory=dict)
    selected_family: str | None = None
    table: dict[str, dict[str, str]]
    surviving: list[str]
    struck: dict[str, str]
    required: list[str]
    settled: list[str]
    open: list[str]
    ready: bool
    contradictions: list[str]


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
    artifacts: list[ArtifactView] = Field(default_factory=list, description="the artifacts about this run, oldest first")
    declines: list[DeclineView] = Field(default_factory=list)


class ArtifactView(BaseModel):
    """An artifact a tool made from the file on request: what it is, what it shows, the numbers it holds, and where its files
    are served from."""

    id: str
    address: str = Field(description="artifact:<id>, as the chat cites it")
    kind: Literal["exploration", "picture"]
    design: int | None = Field(default=None, description="the design run it is about; null before any run")
    ask: str = Field(description="what was asked for, in the person's words")
    summary: str = Field(description="what it shows, in words; a picture's caption")
    facts: dict[str, float] = Field(default_factory=dict, description="every number it holds, by name; each is citable as artifact:<id>.<name>")
    figure: bool = False
    table: str | None = Field(default=None, description="the table file's name, when it wrote one")
    made_at: str
    url: str | None = Field(default=None, description="where the page fetches figure.png, when there is one")
    table_url: str | None = Field(default=None, description="where the page fetches the table, when there is one")


class Turn(BaseModel):
    role: Literal["user", "assistant", "system"]
    text: str
    phase: str = "before"
    at: str
    kind: str | None = None
    figure: FigureSpec | None = Field(default=None, description="shown under the text")
    artifact: ArtifactView | None = Field(default=None, description="a picture drawn on request, shown under the text")


class StepView(BaseModel):
    """One step of the conversation's journal, with the run it names resolved for the page."""

    n: int
    address: str
    kind: str
    by: str
    at: str
    memory_version: int
    design: int | None = Field(default=None, description="the design run the step belongs to; null for the steps leading to the first design")
    read: list[str] = Field(default_factory=list)
    left: list[str] = Field(default_factory=list)
    note: str = ""
    run: int | None = Field(default=None, description="the design index a left entry names, when one does")
    run_id: str | None = Field(default=None, description="the lane run dir's name, when a left entry is one")


class VersionView(BaseModel):
    """One table version the data agent wrote: what was asked, what changed, and where it came from."""

    v: int
    parent: int
    address: str
    ask: str
    summary: str
    rows: int
    added: list[str] = Field(default_factory=list)
    dropped: list[str] = Field(default_factory=list)
    made_at: str


class HandoffField(BaseModel):
    """One field the design rests on, as the memory holds it: the value beside how it was settled."""

    address: str
    value: Any = None
    status: str
    source: str | None = None
    said: str | None = None
    reason: str | None = None


class GapView(BaseModel):
    address: str
    reason: str
    cites: list[str] = Field(default_factory=list)


class JudgeView(BaseModel):
    """A second reading of the context before the run: advice beside the fields, never a block."""

    verdict: Literal["ready", "risky", "not_ready"]
    summary: str
    gaps: list[GapView] = Field(default_factory=list)


class HandoffView(BaseModel):
    context_id: str = ""
    dataset_identity: str = ""
    data_version: int = 0
    admission: ContextAdmission | None = None
    """The pack the ready moment read back, shown before the word: every field the design rests on with its status and the
    sentence it rests on. A correction is sent as a message of `address = value` lines; the desk writes each as the person's
    word and shows the ready moment again."""

    family: str
    specialist: str
    outcome: str
    treatment: str | None = None
    bets_on: str = ""
    memory_version: int = 0
    fields: list[HandoffField] = Field(default_factory=list, description="the claim fields first, then every carried column's fields")
    judge: JudgeView | None = Field(default=None, description="the judge's reading of these fields, when it gave one")


class Prompt(BaseModel):
    text: str = ""
    status: str = ""
    ready: bool = False
    open: list[str] = Field(default_factory=list)
    runs: int = 0
    phase: str = "before"
    kind: str | None = Field(
        default=None,
        description="question: the causal question is being asked; chat: a turn with nothing asked; ask: a question rides with the reply; "
        "ready: the design the desk would run, waiting for the word; after: the chat after a run",
    )


class Activity(BaseModel):
    node: str
    since: str


class SessionView(BaseModel):
    name: str
    title: str
    question: str | None = None
    stage: Literal["busy", "waiting", "ended", "stale", "error", "new"]
    phase: str = "before"
    activity: Activity | None = None
    ready: bool = False
    prompt: Prompt | None = None
    questions: list[QuestionView] = Field(default_factory=list)
    claims: list[ClaimView] = Field(default_factory=list)
    status: StatusView | None = None
    runs: list[RunView] = Field(default_factory=list)
    brief: str = ""
    transcript: list[Turn] = Field(default_factory=list)
    error: str | None = None
    analysis: str | None = Field(default=None, description="a<k>: this conversation's journal")
    journal: list[StepView] = Field(default_factory=list, description="the conversation's steps, in order")
    artifacts: list[ArtifactView] = Field(default_factory=list, description="the artifacts about the data before any run, oldest first")
    handoff: HandoffView | None = Field(default=None, description="at the ready moment: the context the run would rest on, field by field")
    data_version: int = Field(default=0, description="the table version in play; 0 is the uploaded file")
    versions: list[VersionView] = Field(default_factory=list, description="every table version the data agent wrote, oldest first")


# ------------------------------------------------------------------ runs


class FileEntry(BaseModel):
    name: str
    size: int


class RunFiles(BaseModel):
    run_id: str
    files: list[FileEntry]
