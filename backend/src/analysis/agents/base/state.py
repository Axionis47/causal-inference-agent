"""Shared state management for the agentic system."""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, model_validator

from src.analysis.agents.causal_discovery import CausalDAG, CausalEdge, CausalPair
from src.analysis.agents.critique import CritiqueDecision, CritiqueFeedback
from src.analysis.agents.data_profiler import DataProfile, TreatmentEncoding
from src.analysis.agents.domain_knowledge import DomainKnowledge
from src.analysis.agents.eda import EDAResult
from src.analysis.agents.effect_estimator import TreatmentEffectResult
from src.analysis.agents.sensitivity_analyst import SensitivityResult


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


class FileEntry(BaseModel):
    """One file inside a downloaded dataset archive.

    Captured before the temp dir is wiped so the UI can show what was
    actually delivered, not just the file we ended up loading.
    """

    name: str
    size_bytes: int
    format: str  # e.g. "csv", "parquet", "json", "txt", "other"
    used: bool = False  # True for the file we loaded into the DataFrame


class DatasetInfo(BaseModel):
    """Information about a dataset."""

    url: str
    name: str | None = None
    local_path: str | None = None
    n_samples: int | None = None
    n_features: int | None = None
    files: list[FileEntry] = Field(default_factory=list)

    # Kaggle semantic metadata for variable understanding
    kaggle_description: str | None = None
    kaggle_column_descriptions: dict[str, str] = Field(default_factory=dict)
    kaggle_tags: list[str] = Field(default_factory=list)
    kaggle_domain: str | None = None  # Inferred domain (healthcare, economics, etc.)
    metadata_quality: str = "unknown"  # high, medium, low, unknown


class AnalysisDecision(BaseModel):
    """A recorded decision made by an agent during analysis."""

    agent: str
    decision_type: str  # method_selected, method_rejected, confounder_selected, dag_edge_added, quality_gate, iteration_decision, agent_dispatched
    choice: str
    reason: str
    alternatives: list[dict[str, str]] = Field(default_factory=list)  # [{"option": "IPW", "reason": "poor overlap"}]
    timestamp: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class AgentTrace(BaseModel):
    """Trace of an agent action for observability."""

    agent_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    action: str
    reasoning: str | None = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    tools_called: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    token_usage: dict[str, int] = Field(default_factory=dict)


class AnalysisState(BaseModel):
    """Shared state for the entire analysis pipeline.

    This state is passed between agents and tracks the progress of the analysis.
    """

    # Trace management constants (defaults match settings; overridden at init)
    MAX_TRACES: int = 100  # Keep last 100 detailed traces (12-agent pipeline needs room)
    MAX_TRACE_OUTPUT_LEN: int = 1000  # Truncate large outputs (method diagnostics need ~800 chars)
    MAX_TRACE_REASONING_LEN: int = 1000  # Truncate long reasoning

    def model_post_init(self, __context: Any) -> None:
        """Load trace limits from settings (if available) so they stay configurable."""
        try:
            from src.config.settings import get_settings
            settings = get_settings()
            self.MAX_TRACES = settings.max_agent_traces
            self.MAX_TRACE_OUTPUT_LEN = settings.max_trace_output_len
        except Exception:
            pass  # Keep defaults during testing / import bootstrapping

    @model_validator(mode="before")
    @classmethod
    def _backfill_proposed_dag(cls, data: Any) -> Any:
        """Migrate older state docs that only carry the legacy proposed_dag.

        Before this commit, both causal_discovery and dag_expert wrote to a
        single proposed_dag field. State documents persisted under that
        schema are loaded as if dag_expert produced the value (the typical
        last writer), so the new refined_dag slot picks them up.
        """
        if not isinstance(data, dict):
            return data
        legacy = data.pop("proposed_dag", None)
        if legacy is not None:
            data.setdefault("refined_dag", legacy)
        return data

    job_id: str
    dataset_info: DatasetInfo
    download_id: str | None = None
    status: JobStatus = JobStatus.PENDING

    # Which orchestrator handles this job. Recorded on the state so a
    # single-process JobManager can run different modes per job, and so
    # the choice round-trips through persistence for audit. Default
    # "standard" keeps deserialised older states valid.
    orchestrator_mode: Literal["standard", "react"] = "standard"

    # Set by user or inferred
    treatment_variable: str | None = None
    outcome_variable: str | None = None

    # Raw metadata from Kaggle (populated by metadata extractor)
    raw_metadata: dict[str, Any] | None = None

    # Domain knowledge (populated by DomainKnowledgeAgent)
    domain_knowledge: DomainKnowledge | None = None

    # Populated by Data Profiler
    data_profile: DataProfile | None = None
    dataframe_path: str | None = None  # Path to parquet DataFrame

    # Populated by EDA Agent
    eda_result: EDAResult | None = None

    # DAG storage. Two stages, two slots, so neither agent's output is
    # silently overwritten and the notebook can show the evolution.
    #   - discovered_dag: data-driven, written by causal_discovery
    #     using PC / FCI / GES / NOTEARS / LiNGAM ensemble.
    #   - refined_dag: domain-aware, written by dag_expert after fusing
    #     discovery's edges with metadata-derived constraints.
    # Read via the proposed_dag computed property below, which prefers
    # refined_dag when available and falls back to discovered_dag.
    discovered_dag: CausalDAG | None = None
    refined_dag: CausalDAG | None = None

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

    # Decision audit trail
    decisions: list[AnalysisDecision] = Field(default_factory=list)

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

    @computed_field  # type: ignore[prop-decorator]
    @property
    def proposed_dag(self) -> CausalDAG | None:
        """The "best" DAG available for downstream readers.

        Returns refined_dag if dag_expert has run, otherwise the raw
        discovery output. Writes are not allowed; agents must target
        discovered_dag or refined_dag explicitly so each stage's output
        is preserved.
        """
        return self.refined_dag if self.refined_dag is not None else self.discovered_dag

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

    def is_rejected(self) -> bool:
        """Check if the analysis has been rejected by critique.

        REJECT means the analysis is unrecoverable through iteration; the
        job should be marked FAILED rather than retried or approved. This
        is symmetric with is_approved() and should_iterate(), so all three
        decisions have an explicit consumer.
        """
        latest = self.get_latest_critique()
        return latest is not None and latest.decision == CritiqueDecision.REJECT

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
        # FIFO: keep only the last N events (configured via settings)
        try:
            from src.config.settings import get_settings
            max_events = get_settings().max_sse_events
        except Exception:
            max_events = 100
        if len(self.sse_events) > max_events:
            self.sse_events = self.sse_events[-max_events:]

    def push_decision(
        self,
        agent: str,
        decision_type: str,
        choice: str,
        reason: str,
        alternatives: list[dict[str, str]] | None = None,
    ) -> None:
        """Record a decision for the audit trail."""
        self.decisions.append(
            AnalysisDecision(
                agent=agent,
                decision_type=decision_type,
                choice=choice,
                reason=reason,
                alternatives=alternatives or [],
            )
        )
        # Cap decisions (configured via settings)
        try:
            from src.config.settings import get_settings
            max_dec = get_settings().max_decisions
        except Exception:
            max_dec = 100
        if len(self.decisions) > max_dec:
            self.decisions = self.decisions[-max_dec:]

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
