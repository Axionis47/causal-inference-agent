"""Job-related Pydantic schemas for API."""

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.analysis.agents.base import JobStatus

# Kaggle URL pattern: https://www.kaggle.com/datasets/owner/dataset-name
KAGGLE_URL_PATTERN = re.compile(
    r"^https?://(?:www\.)?kaggle\.com/datasets/[\w-]+/[\w-]+/?$",
    re.IGNORECASE
)

# Maximum lengths for string fields
MAX_KAGGLE_URL_LENGTH = 500
MAX_VARIABLE_NAME_LENGTH = 100


class CreateJobRequest(BaseModel):
    """Request to create a new analysis job."""

    kaggle_url: str = Field(
        ...,
        description="Kaggle dataset URL",
        min_length=1,
        max_length=MAX_KAGGLE_URL_LENGTH,
    )
    treatment_variable: str | None = Field(
        None,
        description="Optional treatment variable hint",
        max_length=MAX_VARIABLE_NAME_LENGTH,
    )
    outcome_variable: str | None = Field(
        None,
        description="Optional outcome variable hint",
        max_length=MAX_VARIABLE_NAME_LENGTH,
    )
    orchestrator_mode: Literal["standard", "react"] | None = Field(
        None,
        description=(
            "Which orchestrator to run for this job. 'standard' uses the "
            "fixed 13-agent pipeline; 'react' lets the orchestrator decide "
            "which agent runs next (experimental). Omit to use the "
            "server-side default."
        ),
    )
    user_context: str | None = Field(
        None,
        description=(
            "Optional prose context the analyst provides about the dataset: "
            "the study it came from, what the variables mean, known caveats. "
            "Flows to domain_knowledge so it has something to investigate "
            "even when Kaggle's authored metadata is empty. Plain text, "
            "no markdown processing."
        ),
        max_length=2000,
    )

    @field_validator("kaggle_url")
    @classmethod
    def validate_kaggle_url(cls, v: str) -> str:
        """Validate and normalize the Kaggle URL."""
        v = v.strip()
        if not v:
            raise ValueError("Kaggle URL cannot be empty")
        if not KAGGLE_URL_PATTERN.match(v):
            raise ValueError(
                "Invalid Kaggle URL format. Expected: https://www.kaggle.com/datasets/owner/dataset-name"
            )
        return v

    @field_validator("treatment_variable", "outcome_variable")
    @classmethod
    def validate_variable_name(cls, v: str | None) -> str | None:
        """Validate optional variable names."""
        if v is None:
            return None
        v = v.strip()
        if not v:
            return None  # Treat empty/whitespace-only as None
        # Basic validation: alphanumeric, underscores, hyphens
        if not re.match(r"^[\w\-]+$", v):
            raise ValueError(
                "Variable name must contain only alphanumeric characters, underscores, and hyphens"
            )
        return v


class JobResponse(BaseModel):
    """Basic job response.

    Includes a small digest (dataset_name, T/Y, iteration_count) that
    the JobsListPage renders directly so a user can scan a stack of
    jobs and see what each one did without opening it. These fields
    are already on the persisted record; populating them is free.
    """

    id: str
    kaggle_url: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime

    # Optional digest fields populated by /jobs list. Absent on
    # just-created jobs; that's fine — UI falls back gracefully.
    dataset_name: str | None = None
    treatment_variable: str | None = None
    outcome_variable: str | None = None
    iteration_count: int = 0

    model_config = ConfigDict(from_attributes=True)


class JobDetailResponse(JobResponse):
    """Detailed job response."""

    dataset_name: str | None = None
    current_agent: str | None = None
    iteration_count: int = 0
    error_message: str | None = None
    progress_percentage: int = 0
    treatment_variable: str | None = None
    outcome_variable: str | None = None


class JobStatusResponse(BaseModel):
    """Lightweight job status response."""

    id: str
    status: JobStatus
    progress_percentage: int
    current_agent: str | None = None


class JobListResponse(BaseModel):
    """Paginated list of jobs."""

    jobs: list[JobResponse]
    total: int
    limit: int
    offset: int


class TreatmentEffectResponse(BaseModel):
    """Treatment effect result."""

    method: str
    estimand: str
    estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float | None = None
    assumptions_tested: list[str] = []


class CausalGraphResponse(BaseModel):
    """Causal graph response."""

    nodes: list[str]
    edges: list[dict[str, Any]]
    discovery_method: str
    interpretation: str | None = None


class SensitivityResponse(BaseModel):
    """Sensitivity analysis result."""

    method: str
    robustness_value: float
    interpretation: str


class MethodConsensusResponse(BaseModel):
    """Consensus across multiple estimation methods."""

    n_methods: int = Field(..., description="Number of methods used")
    direction_agreement: float = Field(..., description="Proportion agreeing on effect direction (0-1)")
    all_significant: bool = Field(..., description="Whether all methods show statistical significance")
    estimate_range: tuple[float, float] = Field(..., description="Range of estimates (min, max)")
    median_estimate: float = Field(..., description="Median estimate across methods")
    consensus_strength: str = Field(..., description="'strong', 'moderate', or 'weak'")


class DataContextResponse(BaseModel):
    """Data context and quality information."""

    n_samples: int = Field(..., description="Total sample size")
    n_features: int = Field(..., description="Number of features")
    n_treated: int | None = Field(None, description="Number in treatment group")
    n_control: int | None = Field(None, description="Number in control group")
    missing_data_pct: float = Field(0.0, description="Overall missing data percentage")
    data_quality_issues: list[str] = Field(default_factory=list, description="Data quality warnings")


class ExecutiveSummaryResponse(BaseModel):
    """Executive summary of analysis findings."""

    headline: str = Field(..., description="One-line key finding")
    effect_direction: str = Field(..., description="'positive', 'negative', 'null', or 'mixed'")
    confidence_level: str = Field(..., description="'high', 'medium', or 'low'")
    key_findings: list[str] = Field(default_factory=list, description="Bullet points of key findings")


class AnalysisResultsResponse(BaseModel):
    """Complete analysis results."""

    job_id: str
    treatment_variable: str | None
    outcome_variable: str | None

    # New: Executive summary for quick understanding
    executive_summary: ExecutiveSummaryResponse | None = None

    # New: Method consensus information
    method_consensus: MethodConsensusResponse | None = None

    # New: Data context
    data_context: DataContextResponse | None = None

    # New: Plain-English narrative summary
    narrative_summary: str | None = None

    # Existing fields
    causal_graph: CausalGraphResponse | None = None
    treatment_effects: list[TreatmentEffectResponse] = []
    sensitivity_analysis: list[SensitivityResponse] = []
    recommendations: list[str] = []
    notebook_url: str | None = None

    # Decision audit trail
    decision_log: list[dict] = Field(default_factory=list)


class AgentTraceResponse(BaseModel):
    """Agent trace for observability."""

    agent_name: str
    timestamp: datetime
    action: str
    # Internal AgentTrace.reasoning is `str | None`; mirroring it here so a
    # single `reasoning=None` entry doesn't 500 the entire traces endpoint.
    reasoning: str | None = ""
    duration_ms: int
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    tools_called: list[str] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)


class AgentTracesResponse(BaseModel):
    """List of agent traces."""

    job_id: str
    traces: list[AgentTraceResponse]
    total_traces: int = 0
    total_duration_ms: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


class FileEntryResponse(BaseModel):
    """One file from a downloaded dataset archive."""

    name: str
    size_bytes: int
    format: str
    used: bool


class DownloadBlock(BaseModel):
    """Download lifecycle block of the dataset view."""

    status: str  # "pending" | "downloading" | "downloaded" | "failed"
    url: str | None = None
    files: list[FileEntryResponse] = Field(default_factory=list)
    error: str | None = None


class KaggleMetaData(BaseModel):
    """Kaggle-supplied metadata about the dataset."""

    description: str | None = None
    column_descriptions: dict[str, str] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    domain: str | None = None
    metadata_quality: str = "unknown"


class KaggleMetaBlock(BaseModel):
    """Kaggle metadata block.

    status='unavailable' means the source did not provide it (sparse
    dataset on Kaggle); 'error' means the fetch itself failed.
    """

    status: str  # "pending" | "loaded" | "unavailable" | "error"
    data: KaggleMetaData | None = None
    error: str | None = None


class ProfileBlock(BaseModel):
    """Computed data profile block."""

    status: str  # "pending" | "loaded" | "error"
    data: dict[str, Any] | None = None
    error: str | None = None


class SampleRowsBlock(BaseModel):
    """A small sample of the actual downloaded rows.

    Lets the analyst see real cell values, not just summary stats.
    status='unavailable' means the file type can't be previewed;
    'error' means the read itself failed.
    """

    status: str  # "pending" | "loaded" | "unavailable" | "error"
    columns: list[str] = Field(default_factory=list)
    rows: list[dict[str, Any]] = Field(default_factory=list)
    total_rows: int | None = None
    error: str | None = None


class DatasetViewResponse(BaseModel):
    """Live + persisted view of a job's dataset for the Data panel.

    Each block carries its own status so the UI can render exactly what
    is known so far without collapsing distinct missing-states into one.
    """

    download: DownloadBlock
    kaggle_meta: KaggleMetaBlock
    profile: ProfileBlock
    sample: SampleRowsBlock = Field(default_factory=lambda: SampleRowsBlock(status="pending"))


class CancelJobResponse(BaseModel):
    """Response for job cancellation."""

    job_id: str
    was_running: bool
    cancelled: bool
    status: str | None = None


class DeleteJobResponse(BaseModel):
    """Response for job deletion."""

    job_id: str
    found: bool
    cancelled: bool
    firestore_deleted: bool
    local_artifacts_deleted: dict[str, bool] = {}


# ── Human-approval gate ────────────────────────────────────────────────────


class DagEditPayload(BaseModel):
    """Optional DAG edits the human applies at the approval gate.

    Mirrors `src.domain.approval.DagEdit`; redeclared here so the API
    schema layer stays free of domain imports and OpenAPI shows only
    JSON-safe types.
    """

    adjustment_set: list[str] | None = None
    forbidden_edges: list[dict[str, str]] | None = None
    variable_roles: dict[str, str] | None = None


class ApprovalRequest(BaseModel):
    """Request body for POST /jobs/{job_id}/approval.

    `granted_at` is stamped server-side. `reason` is required when
    `decision == "rejected"` (validated by the domain model).
    """

    decision: Literal["approved", "rejected"]
    granted_by: str | None = Field(default=None, max_length=200)
    dag_edits: DagEditPayload | None = None
    appended_context: str | None = Field(default=None, max_length=4000)
    reason: str | None = Field(default=None, max_length=500)


class ApprovalSnapshotResponse(BaseModel):
    """Snapshot the UI renders at the gate (same shape as the SSE event).

    Fields are intentionally loosely typed (dict / list) — the snapshot
    is built from many state slots and is purely display.
    """

    treatment_variable: str | None = None
    outcome_variable: str | None = None
    eda_summary: dict[str, Any] | None = None
    proposed_dag: dict[str, Any] | None = None
    brief_flags: dict[str, dict[str, Any]] = Field(default_factory=dict)


class ApprovalResultResponse(BaseModel):
    """Result of POST /jobs/{job_id}/approval."""

    job_id: str
    resumed: bool
    status: str
