"""Job-related Pydantic schemas for API."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from src.agents.base import JobStatus


class OrchestratorMode(str, Enum):
    """Orchestrator mode selection."""
    STANDARD = "standard"  # Original orchestrator with fixed workflow
    REACT = "react"  # Fully autonomous ReAct orchestrator


class CreateJobRequest(BaseModel):
    """Request to create a new analysis job."""

    kaggle_url: str = Field(..., description="Kaggle dataset URL")
    treatment_variable: str | None = Field(None, description="Optional treatment variable hint")
    outcome_variable: str | None = Field(None, description="Optional outcome variable hint")
    analysis_preferences: dict[str, Any] | None = Field(None, description="Optional analysis preferences")
    orchestrator_mode: OrchestratorMode = Field(
        default=OrchestratorMode.STANDARD,
        description="Orchestrator mode: 'standard' for fixed workflow, 'react' for autonomous (experimental)"
    )


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


class SensitivityResponse(BaseModel):
    """Sensitivity analysis result."""

    method: str
    robustness_value: float
    interpretation: str


class AnalysisResultsResponse(BaseModel):
    """Complete analysis results."""

    job_id: str
    treatment_variable: str | None
    outcome_variable: str | None
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
