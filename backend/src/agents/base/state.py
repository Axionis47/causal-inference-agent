"""Shared state management for the agentic system."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of an analysis job."""

    PENDING = "pending"
    FETCHING_DATA = "fetching_data"
    PROFILING = "profiling"
    EXPLORATORY_ANALYSIS = "exploratory_analysis"
    DISCOVERING_CAUSAL = "discovering_causal"
    ESTIMATING_EFFECTS = "estimating_effects"
    SENSITIVITY_ANALYSIS = "sensitivity_analysis"
    CRITIQUE_REVIEW = "critique_review"
    ITERATING = "iterating"
    GENERATING_NOTEBOOK = "generating_notebook"
    COMPLETED = "completed"
    FAILED = "failed"


class DatasetInfo(BaseModel):
    """Information about a dataset."""

    url: str
    name: str | None = None
    local_path: str | None = None
    n_samples: int | None = None
    n_features: int | None = None


class DataProfile(BaseModel):
    """Profile of a dataset from the data profiler agent."""

    n_samples: int
    n_features: int
    feature_names: list[str]
    feature_types: dict[str, str]  # feature_name -> type (numeric, categorical, etc.)
    missing_values: dict[str, int]  # feature_name -> count of missing
    numeric_stats: dict[str, dict[str, float]]  # feature_name -> {mean, std, min, max}
    categorical_stats: dict[str, dict[str, int]]  # feature_name -> {value: count}

    # Causal-specific profiling
    treatment_candidates: list[str] = Field(default_factory=list)
    outcome_candidates: list[str] = Field(default_factory=list)
    potential_confounders: list[str] = Field(default_factory=list)
    potential_instruments: list[str] = Field(default_factory=list)
    has_time_dimension: bool = False
    time_column: str | None = None
    discontinuity_candidates: list[str] = Field(default_factory=list)


class CausalEdge(BaseModel):
    """An edge in a causal graph."""

    source: str
    target: str
    edge_type: str = "directed"  # directed, bidirected, undirected
    confidence: float = 1.0


class CausalDAG(BaseModel):
    """A causal directed acyclic graph."""

    nodes: list[str]
    edges: list[CausalEdge]
    discovery_method: str
    treatment_variable: str | None = None
    outcome_variable: str | None = None


class TreatmentEffectResult(BaseModel):
    """Result of a treatment effect estimation."""

    method: str
    estimand: str  # ATE, ATT, CATE
    estimate: float
    std_error: float
    ci_lower: float
    ci_upper: float
    p_value: float | None = None
    assumptions_tested: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


class SensitivityResult(BaseModel):
    """Result of a sensitivity analysis."""

    method: str
    robustness_value: float
    interpretation: str
    details: dict[str, Any] = Field(default_factory=dict)


class EDAResult(BaseModel):
    """Result of exploratory data analysis."""

    # Correlation analysis
    correlation_matrix: dict[str, dict[str, float]] = Field(default_factory=dict)
    high_correlations: list[dict[str, Any]] = Field(default_factory=list)  # pairs with |r| > 0.7

    # Distribution analysis
    distribution_stats: dict[str, dict[str, Any]] = Field(default_factory=dict)  # skewness, kurtosis, normality

    # Outlier detection
    outliers: dict[str, dict[str, Any]] = Field(default_factory=dict)  # column -> {count, indices, method}

    # Multicollinearity
    vif_scores: dict[str, float] = Field(default_factory=dict)  # Variance Inflation Factor
    multicollinearity_warnings: list[str] = Field(default_factory=list)

    # Covariate balance (for treatment/control)
    covariate_balance: dict[str, dict[str, float]] = Field(default_factory=dict)  # SMD, p-values
    balance_summary: str = ""

    # Data quality
    data_quality_score: float = 0.0
    data_quality_issues: list[str] = Field(default_factory=list)

    # Visualizations (paths to generated plots)
    plot_paths: dict[str, str] = Field(default_factory=dict)

    # Summary statistics
    summary_table: dict[str, dict[str, Any]] = Field(default_factory=dict)


class CritiqueDecision(str, Enum):
    """Decision from the critique agent."""

    APPROVE = "APPROVE"
    ITERATE = "ITERATE"
    REJECT = "REJECT"


class CritiqueFeedback(BaseModel):
    """Feedback from the critique agent."""

    decision: CritiqueDecision
    iteration: int
    scores: dict[str, int]  # dimension -> score (1-5)
    issues: list[str]
    improvements: list[str]
    reasoning: str


class AgentTrace(BaseModel):
    """Trace of an agent action for observability."""

    agent_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    action: str
    reasoning: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    tools_called: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)


class AnalysisState(BaseModel):
    """Shared state for the entire analysis pipeline.

    This state is passed between agents and tracks the progress of the analysis.
    """

    job_id: str
    dataset_info: DatasetInfo
    status: JobStatus = JobStatus.PENDING

    # Set by user or inferred
    treatment_variable: str | None = None
    outcome_variable: str | None = None

    # Populated by Data Profiler
    data_profile: DataProfile | None = None
    dataframe_path: str | None = None  # Path to pickled DataFrame

    # Populated by EDA Agent
    eda_result: EDAResult | None = None

    # Populated by Causal Discovery
    proposed_dag: CausalDAG | None = None

    # Populated by Effect Estimator
    treatment_effects: list[TreatmentEffectResult] = Field(default_factory=list)

    # Populated by Sensitivity Analyst
    sensitivity_results: list[SensitivityResult] = Field(default_factory=list)

    # Populated by ConfounderDiscoveryAgent
    confounder_discovery: dict[str, Any] | None = None

    # Populated by PSDiagnosticsAgent
    ps_diagnostics: dict[str, Any] | None = None

    # Populated by DataRepairAgent
    data_repairs: list[dict[str, Any]] = Field(default_factory=list)

    # Populated by CritiqueAgent (multi-perspective debate)
    debate_history: list[dict[str, Any]] = Field(default_factory=list)

    # Critique feedback history
    critique_history: list[CritiqueFeedback] = Field(default_factory=list)

    # Iteration control
    iteration_count: int = 0
    max_iterations: int = 3

    # Final outputs
    notebook_path: str | None = None
    recommendations: list[str] = Field(default_factory=list)

    # Execution traces
    agent_traces: list[AgentTrace] = Field(default_factory=list)

    # Error tracking
    error_message: str | None = None
    error_agent: str | None = None

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    def add_trace(self, trace: AgentTrace) -> None:
        """Add an agent trace to the history."""
        self.agent_traces.append(trace)
        self.updated_at = datetime.utcnow()

    def get_latest_critique(self) -> CritiqueFeedback | None:
        """Get the most recent critique feedback."""
        if self.critique_history:
            return self.critique_history[-1]
        return None

    def should_iterate(self) -> bool:
        """Check if we should iterate based on critique feedback."""
        latest = self.get_latest_critique()
        if latest is None:
            return False
        return (
            latest.decision == CritiqueDecision.ITERATE
            and self.iteration_count < self.max_iterations
        )

    def is_approved(self) -> bool:
        """Check if the analysis has been approved."""
        latest = self.get_latest_critique()
        return latest is not None and latest.decision == CritiqueDecision.APPROVE

    def mark_failed(self, error: str, agent: str) -> None:
        """Mark the analysis as failed."""
        self.status = JobStatus.FAILED
        self.error_message = error
        self.error_agent = agent
        self.updated_at = datetime.utcnow()

    def mark_completed(self) -> None:
        """Mark the analysis as completed."""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
