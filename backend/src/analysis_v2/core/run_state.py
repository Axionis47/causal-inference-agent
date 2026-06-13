"""AnalysisRunState: the single source of truth for one analysis run.

Agents never mutate this directly. They return typed outputs; the
orchestrator validates and commits them here through the methods below,
which keep the slots, the artifact registry, the agent runs, and the
immutable event log consistent. Persisted as its own record kind
(analysis_runs), separate from the input slice's parked AnalysisState.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field, ValidationError

from src.analysis_v2.spec import (
    CausalDAG,
    CausalSpec,
    ClaimCritique,
    DatasetDossier,
    DesignCandidate,
    DiagnosticsResult,
    EDAPlan,
    EDASummary,
    EstimateResult,
    MethodPlan,
    NotebookBuildResult,
    NotebookVerificationResult,
    PlanCritique,
    ProfileSummary,
    ReadinessResult,
    SensitivityResult,
    ToolEligibility,
)

from .agent_run import AgentRun, TokenUsage
from .artifacts import Artifact, ArtifactRegistry
from .events import StateEvent
from .gates import GateResult
from .stages import AnalysisStage

RUN_SCHEMA_VERSION = 1


class StaleRunState(ValueError):
    """A persisted run state could not be loaded under the current schema."""


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING_FOR_USER = "waiting_for_user"
    FAILED = "failed"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AnalysisRunState(BaseModel):
    schema_version: int = RUN_SCHEMA_VERSION
    job_id: str = Field(min_length=1)

    # Inputs snapshotted at pickup from the confirmed AnalysisState.
    causal_question: str = Field(min_length=1)
    user_context: str | None = None
    dataset_name: str | None = None
    ignored_columns: list[str] = Field(default_factory=list)

    # Typed slots, one producer each, filled as the spine advances.
    causal_spec: CausalSpec | None = None
    dataset_profile: ProfileSummary | None = None
    dataset_dossier: DatasetDossier | None = None
    causal_dag: CausalDAG | None = None
    design_candidates: list[DesignCandidate] = Field(default_factory=list)
    selected_design: DesignCandidate | None = None
    tool_eligibility: ToolEligibility | None = None
    eda_plan: EDAPlan | None = None
    eda_summary: EDASummary | None = None
    plan_critique: PlanCritique | None = None
    method_plan: MethodPlan | None = None
    readiness: ReadinessResult | None = None
    estimate_result: EstimateResult | None = None
    diagnostics_result: DiagnosticsResult | None = None
    sensitivity_result: SensitivityResult | None = None
    claim_critique: ClaimCritique | None = None
    notebook_build: NotebookBuildResult | None = None
    notebook_verification: NotebookVerificationResult | None = None

    # Bookkeeping.
    artifact_registry: ArtifactRegistry = Field(default_factory=ArtifactRegistry)
    agent_runs: list[AgentRun] = Field(default_factory=list)
    state_events: list[StateEvent] = Field(default_factory=list)
    current_state: AnalysisStage = AnalysisStage.S0_DATASET_SAVED
    state_version: int = 0
    status: RunStatus = RunStatus.PENDING
    error_message: str | None = None
    total_tokens: TokenUsage = Field(default_factory=TokenUsage)
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    # -- commit discipline -------------------------------------------------

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)

    def record_transition(
        self,
        *,
        to_state: AnalysisStage,
        agent_name: str | None,
        gate_result: GateResult | None = None,
        input_artifacts: list[str] | None = None,
        output_artifacts: list[str] | None = None,
        warnings: list[str] | None = None,
        tokens: TokenUsage | None = None,
        cost_usd: float = 0.0,
    ) -> StateEvent:
        """Append one immutable transition event and advance the stage."""
        if gate_result is not None:
            gate_result.validate_back_target(self.current_state)
        event = StateEvent(
            sequence=len(self.state_events),
            from_state=self.current_state,
            to_state=to_state,
            agent_name=agent_name,
            gate_result=gate_result,
            input_artifacts=input_artifacts or [],
            output_artifacts=output_artifacts or [],
            warnings=warnings or [],
            tokens=tokens or TokenUsage(),
            cost_usd=cost_usd,
        )
        self.state_events.append(event)
        self.current_state = to_state
        self.state_version += 1
        self.add_usage(event.tokens, event.cost_usd)
        self.touch()
        return event

    def register_artifact(self, artifact: Artifact) -> Artifact:
        self.artifact_registry.register(artifact)
        self.touch()
        return artifact

    def add_agent_run(self, run: AgentRun) -> AgentRun:
        self.agent_runs.append(run)
        self.touch()
        return run

    def latest_run_for(self, agent: str) -> AgentRun | None:
        for run in reversed(self.agent_runs):
            if run.agent == agent:
                return run
        return None

    def add_usage(self, tokens: TokenUsage, cost_usd: float) -> None:
        self.total_tokens = self.total_tokens.add(tokens)
        self.total_cost_usd += cost_usd

    def mark_failed(self, message: str) -> None:
        self.status = RunStatus.FAILED
        self.error_message = message
        self.completed_at = datetime.now(timezone.utc)
        self.touch()

    def mark_completed(self) -> None:
        self.status = RunStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.touch()

    # -- persistence guard ---------------------------------------------------

    @classmethod
    def load(cls, payload: dict) -> "AnalysisRunState":
        stored = payload.get("schema_version", RUN_SCHEMA_VERSION)
        if stored != RUN_SCHEMA_VERSION:
            raise StaleRunState(
                f"analysis run schema v{stored} is incompatible with v{RUN_SCHEMA_VERSION}"
            )
        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            raise StaleRunState(
                f"analysis run could not be loaded under schema v{RUN_SCHEMA_VERSION}"
            ) from exc
