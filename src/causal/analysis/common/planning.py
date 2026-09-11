"""Freeze computations from shared evaluation and current exact-data readiness."""
from __future__ import annotations

from causal.analysis.common.candidate import CandidateDraft, CandidateEvaluation
from causal.analysis.common.catalog import method_module
from causal.analysis.common.evaluation import evaluate_candidate
from causal.analysis.common.identity import implementation_hash
from causal.analysis.contracts import (
    AnalysisSpecification,
    BoundaryError,
    CompiledPlan,
    DiagnosticPlan,
    FixedCandidate,
    Measurement,
    PreflightResult,
    SensitivityPlan,
    fingerprint,
)


def diagnostic_plans(draft: CandidateDraft,
                     evaluation: CandidateEvaluation | None = None) -> tuple[DiagnosticPlan, ...]:
    if draft.method is None:
        return ()
    current = evaluate_candidate(draft) if evaluation is None else evaluation
    if current.candidate_fingerprint != fingerprint(draft):
        raise BoundaryError("Diagnostic planning requires evaluation of this exact candidate.")
    definition = method_module(draft.method).DEFINITION
    if current.capability_version != definition.version:
        raise BoundaryError("Diagnostic planning requires the current capability evaluation.")
    options = {option.node_id: option for option in current.options}
    rows = []
    for diagnostic in definition.diagnostics:
        option = options[f"{draft.method}:diagnostic:{diagnostic.id}"]
        rows.append(DiagnosticPlan(
            diagnostic_id=diagnostic.id, title=diagnostic.title,
            obligation=diagnostic.obligation, applicability=option.applicability,
            explanation=option.explanation,
            selected=diagnostic.obligation == "required" or diagnostic.id in draft.diagnostics,
            severity=diagnostic.severity, limitation_category=diagnostic.limitation_category,
            trigger=diagnostic.trigger, evaluation_stage="execution",
            applicability_boundary="design", measurement_boundary="execution",
            thresholds=tuple(Measurement(name=k, value=v) for k, v in diagnostic.thresholds)))
    return tuple(rows)


def compile_checked(spec: AnalysisSpecification, readiness: PreflightResult) -> CompiledPlan:
    from causal.analysis.common.assessment import assess

    assessment = assess(spec.design, spec)
    if assessment.status != "ready" or assessment.specification is None or assessment.evaluation is None:
        raise BoundaryError("The specification is not ready for compilation.", assessment.issues)
    if (readiness.schema_version != "analysis-preflight.v2" or not readiness.ready or readiness.issues
            or readiness.specification_hash != fingerprint(spec)
            or readiness.data != spec.dataset
            or readiness.capability_version != assessment.capability_version
            or readiness.implementation_hash != implementation_hash(spec.configuration.method)):
        raise BoundaryError("Preflight is failed or stale; check the exact specification and data again.")
    if not isinstance(spec.design, FixedCandidate):
        raise BoundaryError("Compilation requires the complete current fixed candidate.")
    definition = method_module(spec.configuration.method).DEFINITION
    diagnostics = diagnostic_plans(spec.design.candidate, assessment.evaluation)
    return CompiledPlan(
        specification=assessment.specification, capability_version=definition.version,
        implementation_hash=readiness.implementation_hash,
        preflight_hash=fingerprint(readiness), diagnostics=diagnostics,
        sensitivities=tuple(SensitivityPlan(sensitivity_id=s.id, title=s.title, purpose=s.purpose,
                                            delta=tuple(Measurement(name=k, value=v) for k, v in s.delta))
                            for s in definition.sensitivities if s.id in spec.sensitivities))
