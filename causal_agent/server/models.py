"""Request and response shapes. Views are projections of the graph's own contracts; nothing here is judged."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

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


class QuestionView(BaseModel):
    """The one question the desk asks this turn, as the page shows it: the field addresses it settles, its form, its chips."""

    keys: list[str]
    field: str | None = None
    kind: str
    text: str
    options: list[str] = Field(default_factory=list)
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
    table: dict[str, dict[str, str]]
    surviving: list[str]
    struck: dict[str, str]
    required: list[str]
    settled: list[str]
    open: list[str]
    ready: bool
    contradictions: list[str]


class CheckView(BaseModel):
    contrast: str | None = None
    name: str
    level: str | None = None
    value: Any = None
    threshold: Any = None
    detail: str | None = None


class RefutationView(BaseModel):
    contrast: str | None = None
    refuter: str
    kind: str | None = None
    passed: bool | None = None
    p_value: float | None = None
    new_effect: float | None = None
    detail: str | None = None


class InterpretationView(BaseModel):
    contrast: str | None = None
    answer: str
    caveats: list[str] = Field(default_factory=list)
    cites: list[str] = Field(default_factory=list)


class EstimateView(BaseModel):
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
    decision: dict[str, Any] = Field(default_factory=dict)
    decision_record: str = ""
    flags: list[CheckView] = Field(default_factory=list)
    checks: list[CheckView] = Field(default_factory=list)
    refutations: list[RefutationView] = Field(default_factory=list)
    interpretations: list[InterpretationView] = Field(default_factory=list)
    estimates: list[EstimateView] = Field(default_factory=list)
    feasibility: dict[str, Any] | None = None
    files: list[str] = Field(default_factory=list)
    what_if: dict[str, str] = Field(default_factory=dict)
    differs: list[str] = Field(default_factory=list)
    figures: list[dict] = Field(default_factory=list)


class Turn(BaseModel):
    role: Literal["user", "assistant", "system"]
    text: str
    phase: str = "before"
    at: str
    kind: str | None = None
    figure: dict | None = Field(default=None, description="a FigureSpec (causal_agent/viz/spec.py) shown under the text")


class Prompt(BaseModel):
    text: str = ""
    status: str = ""
    ready: bool = False
    open: list[str] = Field(default_factory=list)
    runs: int = 0
    phase: str = "before"
    kind: str | None = Field(default=None, description="question: the causal question is being asked; ask: a field; after: the chat after a run")


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
    written: dict[str, Any] | None = None
    error: str | None = None


# ------------------------------------------------------------------ runs


class FileEntry(BaseModel):
    name: str
    size: int


class RunFiles(BaseModel):
    run_id: str
    files: list[FileEntry]
