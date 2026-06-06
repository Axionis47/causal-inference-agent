"""Typed contracts for orchestrator-worker handoffs.

AgentBrief is what a specialist worker returns from each dispatch.
AgentCapability is a worker's declaration of what it answers, what
state it needs, what it delivers, and what tests must pass. Flag is
the closed enum of issue kinds workers raise.

The orchestrator reads briefs to decide the next move; full worker
output stays in AnalysisState (the blackboard) and is fetched lazily
by other workers via ContextTools. The brief is the compressed signal,
the only thing the orchestrator's prompt grows by per dispatch.

See docs/adr-002-orchestrator-worker-pattern.md for the design
rationale and docs/agent-contracts.md for per-agent contracts.
"""
from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Status = Literal["done", "failed", "refused"]


class Flag(StrEnum):
    """Closed vocabulary of issue kinds a worker can raise on its brief.

    Grouped by category for readability. Adding a kind is a deliberate
    code change: it must be picked from this enum, not invented per-worker.
    """

    # Refusal: orchestrator handed input the agent cannot accept
    NEEDS_NOT_MET = "needs_not_met"
    PRECONDITION_FAILED = "precondition_failed"
    STATE_INCONSISTENT = "state_inconsistent"
    CAPABILITY_MISMATCH = "capability_mismatch"

    # Data shape (profiler, repair)
    ENCODING_REQUIRED = "encoding_required"
    HIGH_MISSINGNESS = "high_missingness"
    TYPE_MISMATCH = "type_mismatch"
    REPAIR_FAILED = "repair_failed"
    DATA_LOST = "data_lost"

    # Distributional (eda)
    OUTCOME_SKEWED = "outcome_skewed"
    TC_IMBALANCE = "tc_imbalance"
    SUSPECT_CORRELATION = "suspect_correlation"

    # Causal structure (discovery, dag_expert, confounder)
    DAG_CONFLICT = "dag_conflict"
    LOW_STABILITY = "low_stability"
    CYCLE_DETECTED = "cycle_detected"
    NO_ADJUSTMENT_SET = "no_adjustment_set"
    NO_IV_AVAILABLE = "no_iv_available"
    WEAK_CONFOUNDER_EVIDENCE = "weak_confounder_evidence"
    COLLIDER_SUSPECTED = "collider_suspected"

    # Identification (ps_diagnostics)
    POOR_OVERLAP = "poor_overlap"
    SMD_HIGH = "smd_high"
    CALIBRATION_OFF = "calibration_off"

    # Estimation (effect_estimator)
    METHOD_UNSTABLE = "method_unstable"
    ATE_OUTLIER = "ate_outlier"
    WEAK_INSTRUMENT = "weak_instrument"

    # Robustness (sensitivity_analyst)
    NOT_ROBUST = "not_robust"
    WEAK_TO_UNOBSERVED = "weak_to_unobserved"


class Criterion(BaseModel):
    """One testable success condition for an agent.

    Tests are expected to assert each criterion by id, so a criterion
    declared on a capability without a matching test surfaces as a
    coverage gap rather than silent drift.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    raises_flag: Flag | None = None


class AgentCapability(BaseModel):
    """What an agent advertises to the orchestrator.

    Declared as a class attribute on each specialist; never built at
    runtime. The orchestrator reads `name` and `answers` to assemble
    its capability menu, reads `needs` to compute which workers are
    unblocked, and reads `success_criteria` for routing and test audit.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    answers: str = Field(min_length=1)
    needs: tuple[str, ...] = ()
    delivers: tuple[str, ...] = ()
    refuses_when: tuple[str, ...] = ()
    success_criteria: tuple[Criterion, ...] = ()

    def one_line(self) -> str:
        """Compact menu entry the orchestrator joins into its prompt."""
        return f"{self.name}: {self.answers}"


class AgentBrief(BaseModel):
    """Typed return from one agent dispatch.

    The brief is the only thing the orchestrator's prompt grows by
    per dispatch. Full analysis output lives in AnalysisState; the
    brief carries headline + flags + reroute hint + artifact pointers.
    """

    model_config = ConfigDict(extra="forbid")

    agent: str = Field(min_length=1)
    status: Status
    headline: str = Field(min_length=1, max_length=200)
    flags: list[Flag] = Field(default_factory=list)
    raised_issues: list[str] = Field(default_factory=list)
    reroute_to: str | None = None
    artifact_keys: list[str] = Field(default_factory=list)

    @field_validator("flags")
    @classmethod
    def _no_duplicate_flags(cls, v: list[Flag]) -> list[Flag]:
        if len(set(v)) != len(v):
            raise ValueError("flags must not contain duplicates")
        return v

    @field_validator("raised_issues")
    @classmethod
    def _issues_are_short(cls, v: list[str]) -> list[str]:
        for s in v:
            if not s:
                raise ValueError("raised_issues entries must be non-empty")
            if len(s) > 280:
                raise ValueError("raised_issues entries must be <= 280 chars")
        return v


def refusal_brief(
    agent: str,
    flag: Flag,
    headline: str,
    issues: list[str] | None = None,
) -> AgentBrief:
    """Construct a refusal brief in one call.

    Used by worker pre-flight checks when declared `needs` are absent
    or `refuses_when` conditions are triggered. The orchestrator sees
    status=refused, reads the flag, and routes to a worker that can
    repair the missing precondition.
    """
    return AgentBrief(
        agent=agent,
        status="refused",
        headline=headline,
        flags=[flag],
        raised_issues=issues or [],
        artifact_keys=[],
    )
