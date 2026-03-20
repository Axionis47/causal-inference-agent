"""Shared state management for the agentic system."""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(StrEnum):
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
    CANCELLING = "cancelling"  # Graceful shutdown in progress
    CANCELLED = "cancelled"    # User-initiated cancellation


class DatasetInfo(BaseModel):
    """Information about a dataset."""

    url: str
    name: str | None = None
    local_path: str | None = None
    n_samples: int | None = None
    n_features: int | None = None

    # Kaggle semantic metadata for variable understanding
    kaggle_description: str | None = None
    kaggle_column_descriptions: dict[str, str] = Field(default_factory=dict)
    kaggle_tags: list[str] = Field(default_factory=list)
    kaggle_domain: str | None = None  # Inferred domain (healthcare, economics, etc.)
    metadata_quality: str = "unknown"  # high, medium, low, unknown


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


class TreatmentEncoding(BaseModel):
    """Profiler-determined encoding for categorical treatments.

    When the treatment variable is a multi-level categorical (string),
    the data profiler LLM decides how to encode it. This encoding is stored
    in state and applied deterministically by all downstream methods.
    """

    original_type: str  # "binary", "multi_categorical", "continuous"
    strategy: str  # "none", "label_encode", "collapse_to_binary"
    control_value: str | None = None  # e.g., "No E-Mail"
    value_mapping: dict[str, int] | None = None  # e.g., {"No E-Mail": 0, "Mens E-Mail": 1}


class CausalPair(BaseModel):
    """A treatment-outcome pair that was analyzed."""

    treatment: str
    outcome: str
    rationale: str = ""
    priority: int = 1  # 1 = primary, 2 = secondary, 3 = exploratory


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
    interpretation: str = ""  # LLM-generated interpretation of the graph
    # dag_expert outputs (pullable by downstream agents via context tools)
    forbidden_edges: list[dict[str, str]] | None = None  # [{"source": ..., "target": ..., "reason": ...}]
    variable_roles: dict[str, str] | None = None  # {"age": "confounder", "treat": "treatment", ...}
    adjustment_set: list[str] | None = None  # ["age", "education", "race"]


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

    # Which variables were analyzed (for multi-pair analysis)
    treatment_variable: str | None = None
    outcome_variable: str | None = None


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

    # Summary statistics
    summary_table: dict[str, dict[str, Any]] = Field(default_factory=dict)


class CritiqueDecision(StrEnum):
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
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
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

    # Trace management constants
    MAX_TRACES: int = 100  # Keep last 100 detailed traces (12-agent pipeline needs room)
    MAX_TRACE_OUTPUT_LEN: int = 1000  # Truncate large outputs (method diagnostics need ~800 chars)
    MAX_TRACE_REASONING_LEN: int = 1000  # Truncate long reasoning

    job_id: str
    dataset_info: DatasetInfo
    status: JobStatus = JobStatus.PENDING

    # Set by user or inferred
    treatment_variable: str | None = None
    outcome_variable: str | None = None

    # Raw metadata from Kaggle (populated by metadata extractor)
    raw_metadata: dict[str, Any] | None = None

    # Domain knowledge (populated by DomainKnowledgeAgent)
    # Contains: hypotheses, uncertainties, temporal_understanding, immutable_vars
    domain_knowledge: dict[str, Any] | None = None

    # Populated by Data Profiler
    data_profile: DataProfile | None = None
    dataframe_path: str | None = None  # Path to parquet DataFrame

    # Populated by EDA Agent
    eda_result: EDAResult | None = None

    # Populated by Causal Discovery
    proposed_dag: CausalDAG | None = None

    # Populated by Effect Estimator
    treatment_effects: list[TreatmentEffectResult] = Field(default_factory=list)
    analyzed_pairs: list[CausalPair] = Field(default_factory=list)  # Pairs that were analyzed
    treatment_binarization_threshold: float | None = None  # MED3: shared median-split threshold
    treatment_encoding: TreatmentEncoding | None = None  # Profiler-guided categorical encoding

    # Populated by Sensitivity Analyst
    sensitivity_results: list[SensitivityResult] = Field(default_factory=list)

    # Populated by ConfounderDiscoveryAgent
    confounder_discovery: dict[str, Any] | None = None

    # Populated by PSDiagnosticsAgent
    ps_diagnostics: dict[str, Any] | None = None

    # Populated by DataRepairAgent
    data_repairs: list[dict[str, Any]] = Field(default_factory=list)

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

    # SSE event stream for real-time agent-level updates
    sse_events: list[dict] = Field(default_factory=list)

    # Progress tracking
    progress_percentage: int = 0

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    def add_trace(self, trace: AgentTrace) -> None:
        """Add an agent trace with automatic truncation and compression.

        Traces are truncated to prevent large outputs from consuming memory,
        and old traces are compressed into summaries when the list grows too large.
        """
        # Truncate large fields in the trace
        trace = self._truncate_trace(trace)

        # Add to main list
        self.agent_traces.append(trace)
        self.updated_at = datetime.now(timezone.utc)

        # Only compress when significantly over limit (prevents running on every add)
        if len(self.agent_traces) > 2 * self.MAX_TRACES:
            self._compress_traces()

    def _truncate_trace(self, trace: AgentTrace) -> AgentTrace:
        """Truncate large fields in a trace to save tokens."""
        # Truncate outputs
        if trace.outputs:
            truncated_outputs = {}
            for k, v in trace.outputs.items():
                str_v = str(v)
                if len(str_v) > self.MAX_TRACE_OUTPUT_LEN:
                    truncated_outputs[k] = str_v[:self.MAX_TRACE_OUTPUT_LEN] + "...[truncated]"
                else:
                    truncated_outputs[k] = v
            trace.outputs = truncated_outputs

        # Truncate reasoning
        if trace.reasoning and len(trace.reasoning) > self.MAX_TRACE_REASONING_LEN:
            trace.reasoning = trace.reasoning[:self.MAX_TRACE_REASONING_LEN] + "...[truncated]"

        return trace

    def _compress_traces(self) -> None:
        """Compress old traces into a summary to prevent unbounded growth."""
        if len(self.agent_traces) <= self.MAX_TRACES:
            return

        # Keep last half of max traces
        keep_count = self.MAX_TRACES // 2
        old_traces = self.agent_traces[:-keep_count]
        recent_traces = self.agent_traces[-keep_count:]

        # Create summary of old traces
        summary = self._create_trace_summary(old_traces)

        # Replace with summary + recent
        self.agent_traces = [summary] + recent_traces

    def _create_trace_summary(self, traces: list[AgentTrace]) -> AgentTrace:
        """Create a rich summary trace from multiple traces.

        Preserves per-agent tools used, last reasoning preview,
        and duration breakdown so compressed traces remain useful.
        """
        # Group by agent with richer detail
        agent_info: dict[str, dict] = {}
        total_duration = 0
        total_tokens: dict[str, int] = {"input": 0, "output": 0}

        for trace in traces:
            agent = trace.agent_name
            if agent not in agent_info:
                agent_info[agent] = {
                    "actions": [],
                    "duration_ms": 0,
                    "tools": set(),
                    "last_reasoning": "",
                }
            info = agent_info[agent]
            info["actions"].append(trace.action)
            info["duration_ms"] += trace.duration_ms
            if trace.tools_called:
                info["tools"].update(trace.tools_called)
            if trace.reasoning:
                info["last_reasoning"] = trace.reasoning[:200]

            total_duration += trace.duration_ms
            total_tokens["input"] += trace.token_usage.get("input_tokens", trace.token_usage.get("input", 0))
            total_tokens["output"] += trace.token_usage.get("output_tokens", trace.token_usage.get("output", 0))

        # Build summary text
        summary_text = f"Summary of {len(traces)} compressed traces:\n"
        for agent, info in agent_info.items():
            tools_str = f", tools={list(info['tools'])}" if info["tools"] else ""
            summary_text += (
                f"  {agent}: {len(info['actions'])} actions, "
                f"{info['duration_ms']}ms{tools_str}\n"
            )

        return AgentTrace(
            agent_name="trace_summary",
            action="compressed_history",
            reasoning=summary_text,
            outputs={
                "trace_count": len(traces),
                "agents": {
                    a: {
                        "n_actions": len(info["actions"]),
                        "duration_ms": info["duration_ms"],
                        "tools_used": sorted(info["tools"]),
                        "last_reasoning_preview": info["last_reasoning"],
                    }
                    for a, info in agent_info.items()
                },
                "total_duration_ms": total_duration,
                "compressed_at": datetime.now(timezone.utc).isoformat(),
            },
            duration_ms=0,
            token_usage=total_tokens,
        )

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
        self.updated_at = datetime.now(timezone.utc)

    def mark_completed(self) -> None:
        """Mark the analysis as completed."""
        self.status = JobStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def mark_cancelled(self, reason: str = "Cancelled by user") -> None:
        """Mark the analysis as cancelled."""
        self.status = JobStatus.CANCELLED
        self.error_message = reason
        self.updated_at = datetime.now(timezone.utc)

    def push_sse_event(self, event_type: str, data: dict) -> None:
        """Push an SSE event for real-time streaming to clients.

        Maintains a FIFO buffer of max 100 events.

        Args:
            event_type: Type of event (e.g. "agent_started", "agent_completed")
            data: Event payload dict
        """
        event = {
            "event_type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.sse_events.append(event)
        # FIFO: keep only the last 100 events
        if len(self.sse_events) > 100:
            self.sse_events = self.sse_events[-100:]

    def get_primary_pair(self) -> tuple[str | None, str | None]:
        """Get the primary treatment-outcome pair for analysis.

        Priority order:
        1. User-specified variables (treatment_variable, outcome_variable)
        2. First analyzed pair (from effect estimation)
        3. First treatment effect's variables
        4. None, None

        Returns:
            Tuple of (treatment_variable, outcome_variable)
        """
        # Priority 1: User specified
        if self.treatment_variable and self.outcome_variable:
            return self.treatment_variable, self.outcome_variable

        # Priority 2: First analyzed pair
        if self.analyzed_pairs:
            pair = self.analyzed_pairs[0]
            return pair.treatment, pair.outcome

        # Priority 3: From treatment effects
        if self.treatment_effects:
            effect = self.treatment_effects[0]
            if effect.treatment_variable and effect.outcome_variable:
                return effect.treatment_variable, effect.outcome_variable

        return None, None

    def get_effects_for_pair(
        self, treatment: str, outcome: str
    ) -> list[TreatmentEffectResult]:
        """Get treatment effects for a specific pair.

        Args:
            treatment: Treatment variable name
            outcome: Outcome variable name

        Returns:
            List of effects for this pair
        """
        return [
            e for e in self.treatment_effects
            if e.treatment_variable == treatment and e.outcome_variable == outcome
        ]
