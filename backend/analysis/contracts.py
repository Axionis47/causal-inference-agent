"""Typed scientific decisions and their common, inspectable checkpoint receipt."""

from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import Field

from backend.context.contracts import AnalysisBrief, ColumnIdentity
from backend.responses import Reply
from backend.validation import Contract, ReplyNumber


class ClaimScope(Contract):
    variable_ids: list[str] = Field(default_factory=list, max_length=100)
    effect_ids: list[str] = Field(default_factory=list, max_length=20)
    treatment_ids: list[str] = Field(default_factory=list, max_length=20)
    assignment_event_key: str | None = None


class SourceExcerpt(Contract):
    reference: str
    statement: str
    qualifications: str = ""
    conditions: str = ""
    timing: str = ""
    basis: str
    status: str


class ClaimAdmission(Contract):
    status: Literal["admitted", "candidate", "rejected"]
    authority: Literal["source", "deduction", "assumption", "none"]
    rule: str
    predicate: str
    source_refs: list[str] = Field(default_factory=list)
    sources: list[SourceExcerpt] = Field(default_factory=list)
    scope: ClaimScope
    original_proposal: dict[str, Any]
    reason: str
    ledger_id: str | None = None


class Assumption(Contract):
    id: str = Field(
        min_length=1,
        max_length=100,
        description="Bare ledger identity, e.g. no_interference. References use assumption:no_interference.",
    )
    statement: str = Field(min_length=1, max_length=10000)
    rationale: str = Field(min_length=1, max_length=3000)
    depends_on: list[str] = Field(default_factory=list, max_length=20)
    affects: list[str] = Field(min_length=1, max_length=30)
    status: Literal["candidate", "relied_upon", "rejected"] | None = Field(default=None, exclude_if=lambda value: value is None)
    required_obligation: str | None = Field(default=None, exclude_if=lambda value: value is None)
    scope: ClaimScope | None = Field(default=None, exclude_if=lambda value: value is None)
    original_proposal: dict[str, Any] | None = Field(default=None, exclude_if=lambda value: value is None)
    admission_reason: str | None = Field(default=None, exclude_if=lambda value: value is None)


class Claim(Contract):
    statement: str = Field(min_length=1, max_length=10000)
    basis: Literal["supplied", "deduced", "assumption", "unknown"]
    references: list[str] = Field(default_factory=list, max_length=20)
    prerequisites: list[str] = Field(default_factory=list, max_length=20, description="Accepted decision references used as context or dependencies. These never supply factual or causal authority; references must still contain the required direct evidence or declared assumption.")
    admission: ClaimAdmission | None = Field(default=None, exclude_if=lambda value: value is None)


class Decision(Contract):
    status: Literal["ready", "needs_input", "unidentified", "unsupported"]
    summary: str = Field(min_length=1, max_length=4000)
    citations: list[str] = Field(default_factory=list, max_length=60)
    assumptions: list[Assumption] = Field(default_factory=list, max_length=2000)
    unresolved: list[str] = Field(default_factory=list, max_length=20)
    changes: str = Field(min_length=1, max_length=3000)
    permits: str = Field(min_length=1, max_length=3000)
    requested_keys: list[str] = Field(default_factory=list, max_length=6)
    widening_reason: str = Field(default="", max_length=1000)


class Variable(Contract):
    id: str = Field(min_length=1, max_length=100)
    label: str = Field(min_length=1, max_length=200)
    column_ids: list[str] = Field(min_length=1, max_length=30)
    meaning: Claim
    derivation: str = Field(
        default="",
        max_length=3000,
        description="Empty for a directly observed column. Otherwise the exact transformation, not a description such as 'directly observed'.",
    )


class RequestedEffect(Contract):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    question: Claim
    priority: Literal["primary", "secondary"]


class Target(Decision):
    outcome: Variable | None = None
    measurement: Claim | None = None
    population: Claim | None = None
    unit: Claim | None = None
    timing: Claim | None = None
    effect_question: str = Field(default="", max_length=3000)
    requested_effects: list[RequestedEffect] = Field(
        default_factory=list,
        max_length=20,
        description="The effects requested by the supplied question, each with its stable identity and priority. Freeze these before binding treatment variables. Auxiliary computations such as a ratio's numerator or denominator are not additional effects unless separately requested as effects.",
    )


class Treatment(Contract):
    variable: Variable
    change: Claim
    timing: Claim
    assignment: Claim | None = Field(default=None, description="Historical per-factor account. New analyses describe assignment once in assignments[].")
    properties: list[Claim] = Field(default_factory=list, max_length=20)


class TypedValue(Contract):
    kind: Literal["text", "number", "boolean"]
    literal: str = Field(max_length=1000, description="Exact source representation. Text '1' differs from number '1'. Numbers must be finite decimal literals; booleans use true or false. Never discover levels from target outcomes.")


class TreatmentValue(Contract):
    id: str = Field(min_length=1, max_length=100, description="Proposal-local value identity. The harness binds it canonically to variable_id and the typed literal.")
    variable_id: str = Field(min_length=1, max_length=100, description="Exact selected factor or offered cohort/eligibility variable. Metadata variables remain separate from intervention factors.")
    value: TypedValue
    meaning: Claim


class ContrastTerm(Contract):
    weight: float
    value_ids: list[str] = Field(min_length=1, max_length=20, description="One exact value for every factor in this joint state. The list and signed weight are frozen, not empirical probabilities.")


class StaticContrast(Contract):
    terms: list[ContrastTerm] = Field(min_length=2, max_length=100, description="Ordered signed linear contrast over explicit complete states. Weights sum to zero. Averaging over another factor requires its explicit states and declared weights, never inferred independence.")


class GroupTimeContrast(Contract):
    cohort_value_id: str
    comparison_value_ids: list[str] = Field(min_length=1, max_length=30, description="Exact comparison cohorts on the same cohort variable, including an explicitly coded never-treated cohort where requested. No inferred not-yet-treated selection.")
    time_variable_id: str
    adoption_period: TypedValue = Field(description="Explicit adoption time on time_variable_id's numeric scale. Never infer it from the cohort label or numeric cohort code.")
    period: TypedValue
    baseline: TypedValue


class DiscontinuityContrast(Contract):
    running_variable_id: str
    cutoff: TypedValue
    cutoff_in: Literal["above", "below"] = Field(description="Which side contains equality: above means >= cutoff; below means <= cutoff.")
    direction: Literal["above_minus_below", "below_minus_above"]
    estimand: Literal["outcome_jump", "local_wald"] = Field(description="local_wald is the outcome jump divided by the receipt jump in the same direction. These computational components are not additional requested effects.")
    treatment_value_ids: list[str] = Field(min_length=2, max_length=2, description="Ordered [control, treated] levels on the selected factor. For fuzzy receipt these levels do not assert deterministic receipt at the cutoff.")
    assignment_value_ids: list[str] = Field(default_factory=list, max_length=2, description="Optional ordered [ineligible, eligible] levels on a distinct observed eligibility variable. Empty means eligibility is defined by the explicit threshold rule, not an observed assignment column.")


class EffectWeight(Contract):
    effect_id: str
    weight: float


class AggregateContrast(Contract):
    terms: list[EffectWeight] = Field(min_length=1, max_length=20, description="An explicitly requested weighted average of named component effects. Nonnegative fixed weights sum to one; no empirical reweighting.")


class ContrastBinding(Contract):
    kind: Literal["static", "group_time", "discontinuity", "aggregate", "unsupported"]
    evidence: Claim
    static: StaticContrast | None = None
    group_time: GroupTimeContrast | None = None
    discontinuity: DiscontinuityContrast | None = None
    aggregate: AggregateContrast | None = None
    unsupported: Claim | None = Field(default=None, description="Faithful description of an unimplemented history, adaptive regime or nonlinear contrast. Preserve it and return unsupported, never replace it with a static contrast.")


class EffectRequest(Contract):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    treatment_ids: list[str] = Field(min_length=1, max_length=20)
    contrast: Claim
    estimand: str = Field(min_length=1, max_length=3000)
    priority: Literal["primary", "secondary"]
    comparison: ContrastBinding | None = Field(default=None, description="Historical records may omit this. Fresh analyses freeze a typed comparison separately from the factor and assignment process; exactly the payload named by kind is present.")


class DiagnosticWindow(Contract):
    unit_variable_id: str
    time_variable_id: str
    cutoff: float
    assignment_encoding: Literal["indicator", "group"]
    assignment_variable_id: str | None = Field(default=None, description="Exact observed variable carrying treated/control codes, distinct from a selected cohort regime when appropriate. Required for fresh diagnostic windows, never inferred from a name or selected factor.")
    treated_values: list[str] = Field(min_length=1, max_length=30)
    control_values: list[str] = Field(min_length=1, max_length=30)
    adoption_variable_id: str | None = None
    never_adoption_values: list[str] = Field(default_factory=list, max_length=30)
    evidence: Claim
    history_evidence: Claim


class AssignmentProcess(Contract):
    id: str = Field(min_length=1, max_length=100)
    treatment_ids: list[str] = Field(min_length=1, max_length=20)
    kind: Literal["point", "joint", "absorbing_adoption", "switching", "dynamic_regime", "unknown"] = Field(
        description="Temporal/factor structure, not identification or randomization. point means one factor at one occasion, including threshold assignment and fuzzy/noncompliant receipt. joint means a declared process assigning several factors together. Neither implies randomization or perfect compliance. absorbing_adoption means treatment starts and remains on; switching or dynamic_regime require unsupported regime capabilities. Use unknown only when this structure is unresolved.",
    )
    unit: Claim
    timing: Claim
    mechanism: Claim
    history_variables: list[str] = Field(default_factory=list, max_length=100)
    diagnostic_window: DiagnosticWindow | None = None


class Treatments(Decision):
    treatments: list[Treatment] = Field(
        default_factory=list,
        max_length=20,
        description="Intervention variables/factors, NOT treatment arms or levels. One binary or categorical column is one Treatment. Its values belong in values[] and the requested comparisons in effects[].comparison.",
    )
    effects: list[EffectRequest] = Field(default_factory=list, max_length=20)
    assignments: list[AssignmentProcess] = Field(default_factory=list, max_length=20)
    values: list[TreatmentValue] = Field(default_factory=list, max_length=200, description="Source-qualified typed value registry shared across comparisons. Contains no inferred or data-discovered levels. Historical records may omit it.")


class RoleAssignment(Contract):
    variable: Variable
    roles: list[
        Literal[
            "confounder",
            "mediator",
            "collider",
            "effect_modifier",
            "identifier",
            "time",
            "instrument",
            "running_variable",
            "selection",
            "pre_outcome",
            "irrelevant",
            "unknown",
        ]
    ] = Field(min_length=1, description="Scientific roles relative to the requested effects. irrelevant means no applicable scientific role, not merely excluded from the adjustment set. A pre_outcome, confounder, modifier or other scientific role still requires explicit scope even when also marked irrelevant or not adjusted for.")
    timing: Literal["before_treatment", "at_treatment", "after_treatment", "unknown"]
    account: Claim
    timing_evidence: Claim
    outcome_proxy: bool | None = Field(default=None, description="Harness-owned lineage classification. Return null; the harness derives it from sourced lineage and accepted timing.")
    proxy_evidence: Claim | None = None
    reason: str = Field(min_length=1, max_length=3000)
    effect_ids: list[str] = Field(default_factory=list, max_length=20, description="Exact requested effects to which this role applies. Required for every scientific role or operational use, including relevant variables intentionally excluded from adjustment.")
    treatment_ids: list[str] = Field(default_factory=list, max_length=20, description="Exact accepted treatment identities whose events define this role and timing. Required for every scientific role or operational use; irrelevant does not cancel another role's scope.")
    time_scope: str = Field(default="", max_length=3000, description="Explicit measurement/reference window relative to the selected treatment_ids. Required for scientific roles and operational uses. A timing label alone does not identify the measurement window; irrelevant does not mean merely unadjusted.")
    operational_use: list[Literal["unit_key", "time_key", "cluster_key", "assignment_key"]] = Field(default_factory=list, max_length=4)
    assignment_event_key: str | None = Field(default=None, description="Harness-owned canonical assignment-event identity for this role receipt. Models do not author this field; historical records may omit it.")
    proposed_roles: list[str] | None = Field(default=None, exclude_if=lambda value: value is None)
    coverage_status: Literal["assessed", "unresolved"] | None = Field(default=None, exclude_if=lambda value: value is None)


class Roles(Decision):
    assignments: list[RoleAssignment] = Field(default_factory=list, max_length=2000)


class CausalNode(Contract):
    id: str
    label: str = ""
    column_ids: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)


class CausalEdge(Contract):
    source: str
    target: str
    basis: Literal["supplied", "deduced", "assumption", "unknown"]
    references: list[str] = Field(max_length=20)
    prerequisites: list[str] = Field(default_factory=list, max_length=20, description="Accepted decisions used as dependencies, separate from direct evidence or assumption authority.")
    explanation: str = Field(min_length=1, max_length=3000)
    admission: ClaimAdmission | None = Field(default=None, exclude_if=lambda value: value is None)


class IdentificationObligation(Contract):
    name: str = Field(min_length=1, max_length=100)
    claim: Claim


class IdentificationClaim(Contract):
    effect_id: str
    strategy: Literal["backdoor", "structural_zero", "source_assignment", "did", "discontinuity", "aggregate", "other"] = Field(description="aggregate is deterministic closure of the exact frozen weighted component effects, permitted only for a typed aggregate comparison. It adds no adjustment or new identification premises.")
    adjustment: list[str] = Field(default_factory=list, max_length=100)
    evidence: Claim
    obligations: list[IdentificationObligation] = Field(default_factory=list, max_length=30, description="Named identification obligations: did requires parallel_trends, no_anticipation, comparison_support; discontinuity requires cutoff_rule and continuity or local_randomization. Graph checks are conditional on the accepted causal account.")


class RelationshipDisposition(Contract):
    reference: str
    effect_ids: list[str] = Field(min_length=1, max_length=20)
    disposition: Literal["applicable", "not_applicable", "unresolved"]
    source_id: str = ""
    target_id: str = ""
    scope: str = Field(default="", max_length=3000)
    rationale: Claim


class CausalAccount(Decision):
    nodes: list[CausalNode] = Field(default_factory=list, max_length=100)
    edges: list[CausalEdge] = Field(default_factory=list, max_length=300)
    identification: str = Field(min_length=1, max_length=5000)
    threats: list[str] = Field(default_factory=list, max_length=30)
    identifications: list[IdentificationClaim] = Field(default_factory=list, max_length=20)
    relationships: list[RelationshipDisposition] = Field(default_factory=list, max_length=300)


class Frame(Contract):
    target: Target
    treatments: Treatments
    roles: Roles
    causal: CausalAccount
    graph_hash: str


class DiagnosticAction(Contract):
    action: Literal["run_code", "finish"]
    question: str = Field(min_length=1, max_length=3000)
    consequence: str = Field(min_length=1, max_length=3000)
    code: str = Field(default="", max_length=30000)
    feasible: bool = True
    limitations: list[str] = Field(default_factory=list, max_length=30)


class EffectPlan(Contract):
    effect_id: str
    estimator: str = Field(min_length=1, max_length=3000)
    adjustment: list[str] = Field(
        default_factory=list,
        max_length=100,
        description="Identifying causal adjustment only. Copy exactly the accepted identification's adjustment set. Estimator regressors such as an RD running score, and operational unit/time keys, belong in the estimator specification and are not additional identifying adjustment variables.",
    )
    adjustment_reason: str = Field(min_length=1, max_length=3000)
    inference: str = Field(min_length=1, max_length=3000)
    result_kind: Literal["estimate", "structural_zero", "confidence_set", "estimate_without_inference"] = Field(
        default="estimate",
        description=(
            "Use estimate for an ordinary point estimate with its confidence interval, including t/Welch, "
            "normal, robust or bootstrap intervals. Use confidence_set only for a prespecified nonstandard "
            "set-valued procedure that can produce an empty, disconnected or unbounded region, such as "
            "test inversion under weak identification; an ordinary confidence interval does not require it. "
            "Use estimate_without_inference for a prespecified justified point-only result, and structural_zero "
            "only for an exactly zero effect established by the accepted causal graph."
        ),
    )
    result_reason: str = Field(default="", max_length=3000)


class Plan(Decision):
    effects: list[EffectPlan] = Field(default_factory=list, max_length=20)
    population: str = Field(min_length=1, max_length=3000)
    missingness: str = Field(min_length=1, max_length=3000)
    exclusions: str = Field(min_length=1, max_length=3000)
    multiplicity: str = Field(min_length=1, max_length=3000)
    reporting: str = Field(min_length=1, max_length=3000)
    diagnostics: list[str] = Field(default_factory=list, max_length=20)
    limitations: list[str] = Field(min_length=1, max_length=30)
    seed: int = Field(default=1729, ge=0, le=2147483647)


class Program(Contract):
    code: str = Field(min_length=1, max_length=60000)
    summary: str = Field(min_length=1, max_length=3000)


class NumericalApiSite(Contract):
    line: int
    api: str
    coverage: float | None
    expected: float | None


class NumericalApiVerification(Contract):
    sites: list[NumericalApiSite]
    target_data_access: Literal[False]
    limitations: str


class ProgramVerification(Contract):
    numerical_api: NumericalApiVerification | None = None
    analyzer: str
    version: str
    source_hash: str
    rules: list[str]
    passed: bool
    target_data_access: bool
    limitations: str
    facts_writers_checked: int | None = None
    json_writers_unknown: int | None = None
    other_json_writers: int | None = None


class CodeReview(Contract):
    accepted: bool
    summary: str = Field(min_length=1, max_length=4000)
    issues: list[str] = Field(default_factory=list, max_length=30)


class AnalysisReport(Contract):
    text: str = Field(min_length=1, max_length=20000)
    numbers: list[ReplyNumber] = Field(default_factory=list, max_length=200)


class Conclusion(Decision):
    report: AnalysisReport


class InputProjection(Contract):
    policy: Literal["canonical_columns_v1"] = "canonical_columns_v1"
    columns: list[ColumnIdentity] = Field(min_length=1, max_length=2000)


class Snapshot(Contract):
    version_id: UUID
    context_revision_id: UUID | None
    source_sha256: str
    context_hash: str
    question: str
    frame: Frame
    plan: Plan
    diagnostic_refs: list[str]
    prior_exposure: str
    input_projection: InputProjection | None = None


class GateCheck(Contract):
    rule: str
    kind: Literal["deterministic", "semantic"]
    passed: bool
    detail: str


class CheckpointDefinition(Contract):
    id: str
    title: str
    question: str
    phase: Literal["semantic", "computation", "execution", "interpretation"]
    prerequisites: list[str]
    tools: list[str]


class CheckpointReceipt(Contract):
    complete: bool = True
    id: str
    checkpoint: str
    attempt: int
    status: Literal["accepted", "blocked", "rejected", "failed", "exhausted"]
    input_hash: str
    output_hash: str
    input_refs: list[str]
    prerequisite_ids: list[str]
    offered: list[str]
    retrieved: list[str]
    cited: list[str]
    checks: list[GateCheck]
    output: dict[str, Any]
    assumptions: list[Assumption]
    started_at: datetime
    completed_at: datetime
    budgets: dict[str, int]
    evidence_scope: dict[str, Any] | None = None
    trace_id: str | None = None
    trace_url: str | None = None


class CheckpointView(CheckpointDefinition):
    status: Literal["pending", "running", "accepted", "blocked", "rejected", "failed", "exhausted"]
    attempts: list[CheckpointReceipt]
    blocked_by: list[str]


class AnalysisTaskView(Contract):
    id: str
    key: str
    checkpoint: str
    kind: str
    reason: str
    status: Literal["pending", "running", "accepted", "rejected", "blocked", "failed", "exhausted", "expanded"]
    budget_kind: str
    column_ids: list[str]
    obligation_keys: list[str]
    scope: dict[str, Any]
    dependency_receipt_ids: list[str]
    input_hash: str
    packet: dict[str, Any]
    allowance: dict[str, int]
    attempts: list[CheckpointReceipt]
    attempt_details: list[dict[str, Any]]


class SharedAttemptView(Contract):
    id: UUID
    key: str
    logical_key: str
    obligation_key: str | None
    number: int
    status: Literal["reserved", "accepted", "rejected", "failed", "uncertain", "blocked", "exhausted", "expanded"] = Field(
        description="Provider invocation state. Accepted means a typed proposal was received, not that a scientific gate accepted it."
    )
    input_hash: str
    request_hash: str | None
    started_at: datetime
    completed_at: datetime | None
    metrics: dict[str, Any]
    has_request: bool
    has_raw_output: bool
    has_proposal: bool


class SharedAttemptDetail(SharedAttemptView):
    request: dict[str, Any] | None
    proposal: dict[str, Any] | None
    raw_output: str | None


class ProgramView(Contract):
    revision: int
    hash: str
    science_hash: str
    runtime_hash: str
    program: Program
    review: CodeReview
    verification: ProgramVerification | None = None
    diff: str
    approved_at: datetime | None = None
    rejected_reason: str | None = None
    execution: dict[str, Any] | None = None


class PrepareAnalysis(Contract):
    submission_id: UUID
    question: str | None = Field(default=None, min_length=1, max_length=8000)
    expected_context_revision_id: UUID | None
    expected_readiness_fingerprint: str = Field(min_length=64, max_length=64)


class FrozenContextView(Contract):
    dataset_version_id: UUID
    source_sha256: str
    context_revision_id: UUID | None
    context_hash: str
    question: str
    brief: AnalysisBrief
    readiness_fingerprint: str
    readiness: dict[str, Any]
    policy_version: str


class AdmissionView(Contract):
    receipt_id: str
    checkpoint: str
    receipt_status: str
    path: str
    statement: str | None = None
    admission: ClaimAdmission


class AssumptionView(Contract):
    receipt_id: str
    checkpoint: str
    receipt_status: str
    entry: Assumption


class AuthorityView(Contract):
    claims: list[AdmissionView] = Field(default_factory=list)
    ledger: list[AssumptionView] = Field(default_factory=list)


class ApproveAnalysis(Contract):
    snapshot_hash: str = Field(min_length=64, max_length=64)
    program_hash: str = Field(min_length=64, max_length=64)


class RejectAnalysis(ApproveAnalysis):
    reason: str = Field(min_length=1, max_length=4000)


class EventView(Contract):
    sequence: int
    stage: str
    kind: str
    summary: str
    detail: dict[str, Any]
    created_at: datetime


class Result(Contract):
    report: Reply
    facts: dict[str, float]
    filenames: list[str]
    execution_ok: bool
    stdout: str
    stderr: str


class AgentSummary(Contract):
    id: UUID | None
    agent: Literal["study", "design", "computation", "report"]
    title: str
    status: str
    summary: str
    model_calls: int
    model_limit: int
    submissions: int
    handoff_id: str | None
    handoff_hash: str | None
    dependencies: list[str]
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float


class AgentDetail(Contract):
    id: UUID
    agent: str
    packet: dict[str, Any]
    draft: dict[str, Any]
    handoff: dict[str, Any] | None
    handoff_hash: str | None
    sessions: list[dict[str, Any]]


class AnalysisView(Contract):
    read_only: bool = False
    contract: str | None = None
    frozen_context: FrozenContextView | None = Field(default=None, exclude_if=lambda value: value is None)
    authority: AuthorityView | None = Field(default=None, exclude_if=lambda value: value is None)
    id: UUID
    conversation_id: UUID
    version_id: UUID
    context_revision_id: UUID | None
    question: str
    status: Literal["queued", "running", "awaiting_approval", "succeeded", "blocked", "failed"]
    stage: str
    calls: int
    budgets: dict[str, int]
    frame: Frame | None
    snapshot: Snapshot | None
    snapshot_hash: str | None
    programs: list[ProgramView]
    checkpoints: list[CheckpointView]
    tasks: list[AnalysisTaskView] = Field(default_factory=list)
    agents: list[AgentSummary] = Field(default_factory=list)
    outcome: Literal["unsupported", "unidentified"] | None = None
    recovery_actions: list[Literal["retry", "approve", "revise_program"]] = Field(default_factory=list)
    shared_attempts: list[SharedAttemptView] = Field(default_factory=list)
    context: dict[str, Any]
    diagnostics: list[dict[str, Any]]
    legacy_record: dict[str, Any] | None
    result: Result | None
    error: str | None
    retryable: bool
    events: list[EventView]
    trace_id: UUID
    created_at: datetime


class AnalysisSummary(Contract):
    read_only: bool = False
    id: UUID
    question: str
    status: str
    stage: str
    created_at: datetime
