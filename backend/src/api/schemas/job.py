"""Job-related Pydantic schemas for API."""

import re
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.agents.base import JobStatus

# Kaggle URL pattern: https://www.kaggle.com/datasets/owner/dataset-name
KAGGLE_URL_PATTERN = re.compile(
    r"^https?://(?:www\.)?kaggle\.com/datasets/[\w-]+/[\w-]+/?$",
    re.IGNORECASE
)

# Maximum lengths for string fields
MAX_KAGGLE_URL_LENGTH = 500
MAX_VARIABLE_NAME_LENGTH = 256


class OrchestratorMode(str, Enum):
    """Orchestrator mode selection."""
    STANDARD = "standard"  # Original orchestrator with fixed workflow
    REACT = "react"  # Fully autonomous ReAct orchestrator


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
    analysis_preferences: dict[str, Any] | None = Field(None, description="Optional analysis preferences")
    orchestrator_mode: OrchestratorMode = Field(
        default=OrchestratorMode.STANDARD,
        description="Orchestrator mode: 'standard' for fixed workflow, 'react' for autonomous (experimental)"
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
    """Basic job response."""

    id: str
    kaggle_url: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


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

    # Existing fields
    causal_graph: CausalGraphResponse | None = None
    treatment_effects: list[TreatmentEffectResponse] = []
    sensitivity_analysis: list[SensitivityResponse] = []
    recommendations: list[str] = []
    notebook_url: str | None = None


class AgentTraceResponse(BaseModel):
    """Agent trace for observability."""

    agent_name: str
    timestamp: datetime
    action: str
    reasoning: str
    duration_ms: int


class AgentTracesResponse(BaseModel):
    """List of agent traces."""

    job_id: str
    traces: list[AgentTraceResponse]


class DatasetValidationRequest(BaseModel):
    """Request to validate a Kaggle URL."""

    kaggle_url: str


class DatasetValidationResponse(BaseModel):
    """Dataset validation result."""

    valid: bool
    dataset_name: str | None = None
    error: str | None = None


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
