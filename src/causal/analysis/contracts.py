"""Versioned immutable contracts for the standalone analysis boundary.

Historical estimation payloads live in integration.contracts. They are not
reinterpreted as these contracts. Approval is a caller's attestation, not an
authentication service; the runner verifies what was attested.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Annotated, Any, Literal, cast

from pydantic import Field, model_validator

from causal.analysis.common.candidate import (
    Boundary,
    CandidateAssertion,
    CandidateDraft,
    CandidateEvaluation,
    CandidateOutcome,
    CapabilityExploration,
    CapabilityGraph,
    CapabilityRequestError,
    CheckObligation,
    Eligibility,
    GraphCoverage,
    GraphEdge,
    GraphNode,
    OptionState,
    Relation,
    RequirementResult,
    RequirementStatus,
    ResolutionKind,
    ResolutionTarget,
    RoleSlot,
    Selection,
    VariableBinding,
)
from causal.analysis.common.definitions import ColumnRequirement
from causal.analysis.common.models import Applicability, Issue, Model, Scalar
from causal.analysis.methods.aipw.specification import Specification as AIPWSpecification
from causal.analysis.methods.did.specification import Specification as DiDSpecification
from causal.analysis.methods.randomized.specification import (
    Specification as RandomizedSpecification,
)
from causal.analysis.methods.rdd.specification import Specification as RDDSpecification

__all__ = ['AnalysisEvidence', 'AnalysisSpecification', 'ApprovedPlan', 'Assessment', 'Boundary', 'BoundaryError', 'CandidateAssertion', 'CandidateDraft', 'CandidateEvaluation', 'CandidateOutcome', 'CapabilityExploration', 'CapabilityGraph', 'CapabilityRequestError', 'CheckObligation', 'CompiledPlan', 'Computation', 'DatasetIdentity', 'DiagnosticPlan', 'Eligibility', 'Estimate', 'Exclusion', 'FixedCandidate', 'FixedDesign', 'GraphCoverage', 'GraphEdge', 'GraphNode', 'Guidance', 'Hash', 'Limitation', 'Measurement', 'Method', 'MethodSpecification', 'MethodSummary', 'OptionState', 'Outcome', 'PlotData', 'PlotPlan', 'PreflightResult', 'Provenance', 'Relation', 'RequirementResult', 'RequirementStatus', 'ResolutionKind', 'ResolutionTarget', 'RoleSlot', 'Selection', 'SensitivityPlan', 'StudyFact', 'Topic', 'VariableBinding', 'fingerprint']

Method = Literal["randomized", "aipw", "did", "rdd"]
Topic = Literal["overview", "configuration", "diagnostics", "diagnostic_details"]
Hash = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
MethodSpecification = Annotated[
    RandomizedSpecification | AIPWSpecification | DiDSpecification | RDDSpecification,
    Field(discriminator="method"),
]


def fingerprint(value: Model) -> str:
    encoded = json.dumps(value.model_dump(mode="json"), sort_keys=True,
                         separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


class StudyFact(Model):
    name: str
    value: Scalar
    evidence: tuple[str, ...] = ()
    original_answer: str | None = None


class Outcome(Model):
    column: Annotated[str, Field(min_length=1)]
    kind: Literal["continuous", "binary"]
    units: Annotated[str, Field(min_length=1)]


class FixedDesign(Model):
    """Historical scientific-frame reader; insufficient for a new accepted run."""

    reference: Annotated[str, Field(min_length=1)]
    content_hash: Hash
    method: Method
    population: Annotated[str, Field(min_length=1)]
    outcome: Outcome
    facts: tuple[StudyFact, ...] = ()

    @model_validator(mode="after")
    def unique_facts(self) -> FixedDesign:
        if len({fact.name for fact in self.facts}) != len(self.facts):
            raise ValueError("each study fact must have exactly one recorded value")
        return self

    def evidenced_facts(self) -> dict[str, Scalar]:
        return {f.name: f.value for f in self.facts
                if f.value is not None and f.evidence and all(s.strip() for s in f.evidence)}


class FixedCandidate(Model):
    """Complete caller-referenced snapshot. Its content hash grants no approval."""

    schema_version: Literal["analysis-fixed-design.v2"] = "analysis-fixed-design.v2"
    reference: Annotated[str, Field(min_length=1)]
    content_hash: Hash
    capability_version: Annotated[str, Field(min_length=1)]
    candidate: CandidateDraft

    @staticmethod
    def hash_payload(candidate: CandidateDraft, reference: str, capability_version: str) -> str:
        payload = {"schema_version": "analysis-fixed-design.v2", "reference": reference,
                   "capability_version": capability_version,
                   "candidate": candidate.model_dump(mode="json")}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(encoded.encode()).hexdigest()

    @model_validator(mode="after")
    def hash_matches_snapshot(self) -> FixedCandidate:
        if self.content_hash != self.hash_payload(
                self.candidate, self.reference, self.capability_version):
            raise ValueError("The fixed-design content hash does not match its complete candidate.")
        return self

    @classmethod
    def from_candidate(cls, candidate: CandidateDraft | Mapping[str, Any],
                       reference: str) -> FixedCandidate:
        """Freeze a design-ready proposal under a real caller-supplied reference."""
        from causal.analysis.common.evaluation import evaluate_candidate

        draft = CandidateDraft.model_validate(candidate.model_dump() if isinstance(
            candidate, CandidateDraft) else candidate)
        evaluation = evaluate_candidate(draft)
        if evaluation.status != "design_ready" or evaluation.capability_version is None:
            raise BoundaryError("Only a design-ready candidate can be fixed.", evaluation.issues)
        version = evaluation.capability_version
        return cls(reference=reference, capability_version=version, candidate=draft,
                   content_hash=cls.hash_payload(draft, reference, version))

    @property
    def method(self) -> Method:
        return cast(Method, self.candidate.method)

    @property
    def population(self) -> str:
        return cast(str, self.candidate.population)

    @property
    def outcome(self) -> Outcome:
        outcome = self.candidate.outcome
        if outcome is None:
            raise ValueError("The fixed candidate has no outcome measurement.")
        return Outcome(column=cast(str, outcome.column), kind=cast(Any, outcome.kind),
                       units=cast(str, outcome.units))

    @property
    def facts(self) -> tuple[StudyFact, ...]:
        return tuple(StudyFact(name=f.name, value=f.value, evidence=f.evidence,
                               original_answer=f.original_answer) for f in self.candidate.facts)

    def evidenced_facts(self) -> dict[str, Scalar]:
        return {f.name: f.value for f in self.candidate.facts
                if f.value is not None and f.evidence and all(s.strip() for s in f.evidence)}


class DatasetIdentity(Model):
    name: Annotated[str, Field(min_length=1)]
    content_hash: Hash


class AnalysisSpecification(Model):
    schema_version: Literal["analysis-specification.v1", "analysis-specification.v2"] = "analysis-specification.v2"
    design: FixedCandidate | FixedDesign
    dataset: DatasetIdentity
    configuration: MethodSpecification
    diagnostics: tuple[str, ...] = ()
    sensitivities: tuple[str, ...] = ()
    seed: Annotated[int, Field(ge=0, le=2**32 - 2)] = 0

    @model_validator(mode="before")
    @classmethod
    def derive_executable_projection(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        raw = dict(value)
        if raw.get("schema_version", "analysis-specification.v2") == "analysis-specification.v1":
            return raw
        design = raw.get("design")
        if isinstance(design, FixedCandidate):
            fixed = FixedCandidate.model_validate(design.model_dump())
        elif isinstance(design, Mapping) and design.get("schema_version") == "analysis-fixed-design.v2":
            fixed = FixedCandidate.model_validate(design)
        else:
            raise ValueError("A current specification requires a complete FixedCandidate snapshot.")
        from causal.analysis.common.catalog import method_module
        from causal.analysis.common.evaluation import evaluate_candidate

        evaluation = evaluate_candidate(fixed.candidate)
        if evaluation.status != "design_ready" or evaluation.configuration is None:
            raise ValueError("The fixed candidate must satisfy current design requirements.")
        if evaluation.capability_version != fixed.capability_version:
            raise ValueError("The fixed candidate capability version is no longer current.")
        configuration_type = method_module(fixed.method).DEFINITION.specification
        expected_configuration = configuration_type.model_validate(evaluation.configuration)
        expected = {"configuration": expected_configuration,
                    "diagnostics": fixed.candidate.diagnostics,
                    "sensitivities": fixed.candidate.sensitivities,
                    "seed": fixed.candidate.seed if fixed.candidate.seed is not None else 0}
        for name, projected in expected.items():
            if name in raw:
                actual = raw[name]
                if name == "configuration":
                    actual = configuration_type.model_validate(actual)
                elif name in ("diagnostics", "sensitivities"):
                    if not isinstance(actual, tuple | list):
                        raise ValueError(f"The executable {name} must be a list of selected computations.")
                    actual = tuple(actual)
                elif name == "seed" and (isinstance(actual, bool) or not isinstance(actual, int)):
                    raise ValueError("The executable seed must be an integer, not a coerced value.")
                if actual != projected:
                    raise ValueError(f"The executable {name} differs from the complete fixed candidate.")
            raw[name] = projected
        raw["design"] = fixed
        return raw

    @model_validator(mode="after")
    def versioned_boundary(self) -> AnalysisSpecification:
        if self.schema_version == "analysis-specification.v1" and isinstance(self.design, FixedCandidate):
            raise ValueError("Historical v1 specifications cannot represent the complete fixed boundary.")
        return self


class Assessment(Model):
    schema_version: Literal["analysis-assessment.v1", "analysis-assessment.v2"] = "analysis-assessment.v2"
    status: Literal["ready", "needs_information", "rejected"]
    capability_version: str | None
    specification_hash: str | None
    specification: AnalysisSpecification | None
    issues: tuple[Issue, ...]
    evaluation: CandidateEvaluation | None = None

    @model_validator(mode="before")
    @classmethod
    def historical_response_reader(cls, value: Any) -> Any:
        if isinstance(value, Mapping) and "schema_version" not in value:
            specification = value.get("specification")
            version = (specification.schema_version if isinstance(specification, AnalysisSpecification)
                       else specification.get("schema_version") if isinstance(specification, Mapping) else None)
            if version == "analysis-specification.v1":
                return dict(value) | {"schema_version": "analysis-assessment.v1"}
        return value

    @model_validator(mode="after")
    def executable_acceptance_only(self) -> Assessment:
        if self.schema_version == "analysis-assessment.v1":
            if self.specification is not None and self.specification.schema_version != "analysis-specification.v1":
                raise ValueError("Historical assessments cannot certify the complete current specification boundary.")
            return self
        if self.status == "ready":
            if self.specification is None or self.specification_hash is None or self.issues:
                raise ValueError("Ready assessment requires an executable specification without unresolved issues.")
            if self.specification.schema_version != "analysis-specification.v2":
                raise ValueError("Ready assessment requires the complete current specification boundary.")
            if self.specification_hash != fingerprint(self.specification):
                raise ValueError("The assessment hash must identify its exact executable specification.")
        elif self.specification is not None or self.specification_hash is not None:
            raise ValueError("Only ready assessment may expose an executable specification or its hash.")
        return self


class Measurement(Model):
    name: str
    value: Scalar


class DiagnosticPlan(Model):
    diagnostic_id: str
    title: str
    obligation: Literal["required", "optional"]
    applicability: Applicability
    explanation: str
    selected: bool
    severity: str
    limitation_category: Literal["population", "causal_interpretation", "measurement", "confidence"] = "confidence"
    thresholds: tuple[Measurement, ...] = ()
    trigger: str
    evaluation_stage: Literal["specification", "preflight", "execution"]
    applicability_boundary: Literal["design"] = "design"
    measurement_boundary: Literal["execution"] = "execution"


class SensitivityPlan(Model):
    sensitivity_id: str
    title: str
    purpose: str
    delta: tuple[Measurement, ...]


class PreflightResult(Model):
    schema_version: Literal["analysis-preflight.v1", "analysis-preflight.v2"] = "analysis-preflight.v2"
    specification_hash: Hash
    capability_version: str
    implementation_hash: Hash
    data: DatasetIdentity
    ready: bool
    issues: tuple[Issue, ...]
    observations: tuple[Measurement, ...] = ()
    resolution_targets: tuple[ResolutionTarget, ...] = ()


class PlotPlan(Model):
    """Deprecated historical v1 reader; new analysis plans contain no visual prescriptions."""
    plot_id: str
    dependency: str
    applicability: Applicability
    explanation: str


class CompiledPlan(Model):
    schema_version: Literal["analysis-plan.v1", "analysis-plan.v2", "analysis-plan.v3"] = "analysis-plan.v3"
    specification: AnalysisSpecification
    capability_version: str
    implementation_hash: Hash
    preflight_hash: Hash
    diagnostics: tuple[DiagnosticPlan, ...]
    sensitivities: tuple[SensitivityPlan, ...]
    plots: tuple[PlotPlan, ...] = ()

    @model_validator(mode="after")
    def numerical_plan_only(self) -> CompiledPlan:
        if self.schema_version != "analysis-plan.v1" and self.plots:
            raise ValueError("visual prescriptions are historical v1 fields, not analysis computations")
        if self.schema_version == "analysis-plan.v3" and self.specification.schema_version != "analysis-specification.v2":
            raise ValueError("Current plans require the complete v2 fixed-design boundary.")
        return self

    @property
    def plan_hash(self) -> str:
        return fingerprint(self)


class ApprovedPlan(Model):
    plan: CompiledPlan
    approved_hash: Hash
    approved_by: Annotated[str, Field(min_length=1)]
    approval_reference: Annotated[str, Field(min_length=1)]


class MethodSummary(Model):
    method: str
    title: str
    summary: str
    capability_version: str
    graph_entry: str


class Exclusion(Model):
    option: str
    reason: str


class Guidance(Model):
    schema_version: Literal["analysis-guidance.v2"] = "analysis-guidance.v2"
    method: str
    topic: Topic
    capability_version: str
    explanation: str
    input_schema: dict[str, Any]
    applicable_options: tuple[str, ...]
    required_facts: tuple[str, ...] = ()
    required_columns: tuple[ColumnRequirement, ...] = ()
    measurement_units: tuple[Measurement, ...] = ()
    derived_quantities: tuple[str, ...] = ()
    sensitivities: tuple[SensitivityPlan, ...] = ()
    unresolved_prerequisites: tuple[Issue, ...]
    exclusions: tuple[Exclusion, ...]
    diagnostics: tuple[DiagnosticPlan, ...]
    follow_up_topics: tuple[Topic, ...]
    graph_entry: str
    evaluation: CandidateEvaluation
    options: tuple[OptionState, ...]
    role_slots: tuple[RoleSlot, ...]


class Estimate(Model):
    contrast: str
    estimand: str
    label: str
    estimate: float
    units: str
    standard_error: float
    confidence_level: float
    interval_lower: float
    interval_upper: float
    p_value: float | None
    uncertainty_method: str
    convergence: Literal["converged", "not_converged", "not_applicable"]
    population: tuple[Measurement, ...]
    method_quantities: tuple[Measurement, ...] = ()


class Computation(Model):
    computation_id: str
    status: Literal["computed", "failed", "blocked", "unavailable", "inapplicable", "not_selected"]
    explanation: str
    estimates: tuple[Estimate, ...] = ()
    measurements: tuple[Measurement, ...] = ()
    policy: Literal["acceptable", "warning", "invalidating", "descriptive"] = "descriptive"


class Limitation(Model):
    category: Literal["population", "causal_interpretation", "measurement", "confidence"]
    source: str
    explanation: str


class PlotData(Model):
    """Deprecated historical v1 reader; post-analysis owns new display coordinates."""
    plot_id: str
    columns: tuple[str, ...]
    rows: tuple[tuple[Scalar, ...], ...]


class Provenance(Model):
    plan_hash: Hash
    data: DatasetIdentity
    capability_version: str
    seed: int
    approved_by: str
    approval_reference: str
    environment: tuple[Measurement, ...]


class AnalysisEvidence(Model):
    schema_version: Literal["analysis-evidence.v1", "analysis-evidence.v2"] = "analysis-evidence.v2"
    status: Literal["completed", "completed_with_limitations", "incomplete", "failed"]
    population: str
    primary: Computation
    diagnostics: tuple[Computation, ...]
    sensitivities: tuple[Computation, ...]
    limitations: tuple[Limitation, ...]
    supporting_data: tuple[Computation, ...] = ()
    plotting_data: tuple[PlotData, ...] = ()
    plots: tuple[Computation, ...] = ()
    provenance: Provenance

    @model_validator(mode="after")
    def numerical_evidence_only(self) -> AnalysisEvidence:
        if self.schema_version == "analysis-evidence.v2" and (self.plotting_data or self.plots):
            raise ValueError("display coordinates are historical v1 fields, not numerical evidence")
        return self


class BoundaryError(ValueError):
    def __init__(self, message: str, issues: tuple[Issue, ...] = ()) -> None:
        super().__init__(message)
        self.issues = issues
