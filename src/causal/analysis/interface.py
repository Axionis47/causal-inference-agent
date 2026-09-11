"""Agent-independent capability exploration, acceptance, and exact execution."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast, get_args

from pydantic import ValidationError

from causal.analysis.common.assessment import assess
from causal.analysis.common.candidate import (
    CandidateDraft,
    CapabilityRequestError,
    ResolutionTarget,
    RoleSlot,
)
from causal.analysis.common.catalog import method_module
from causal.analysis.common.definitions import ColumnRequirement
from causal.analysis.common.evaluation import coerce_candidate, evaluate_candidate
from causal.analysis.common.graph import explore_capabilities
from causal.analysis.common.identity import implementation_hash
from causal.analysis.common.models import Issue
from causal.analysis.common.planning import compile_checked, diagnostic_plans
from causal.analysis.common.preflight import identify_data, inspect_data
from causal.analysis.contracts import (
    AnalysisEvidence,
    AnalysisSpecification,
    ApprovedPlan,
    Assessment,
    CompiledPlan,
    Exclusion,
    FixedCandidate,
    FixedDesign,
    Guidance,
    Measurement,
    MethodSummary,
    PreflightResult,
    SensitivityPlan,
    Topic,
    fingerprint,
)

__all__ = [
    "assess_specification", "compile_plan", "evaluate_candidate", "execute",
    "explore_capabilities", "identify_data", "list_methods", "preflight", "retrieve_guidance",
]


def list_methods() -> tuple[MethodSummary, ...]:
    from causal.analysis.common.catalog import METHODS

    return tuple(MethodSummary(method=d.method, title=d.title, summary=d.summary,
                               capability_version=d.version, graph_entry=f"method:{d.method}")
                 for method in METHODS for d in (method_module(method).DEFINITION,))


def retrieve_guidance(method: str, topic: Topic,
                      draft: CandidateDraft | AnalysisSpecification | Mapping[str, Any] | None = None,
                      diagnostic_id: str | None = None) -> Guidance:
    """Explanatory compatibility view; evaluation owns every eligibility decision."""
    if topic not in get_args(Topic):
        raise CapabilityRequestError("topic", f"Unknown guidance topic {topic!r}.")
    module = method_module(method)
    definition = module.DEFINITION
    try:
        # Legacy bare configuration requests use their explicitly requested family.
        # Structured candidates always retain the caller's actual nomination.
        if draft is None:
            candidate = CandidateDraft(method=method)
        elif isinstance(draft, CandidateDraft | AnalysisSpecification):
            candidate = coerce_candidate(draft)
        else:
            raw = dict(draft)
            if not set(raw) & {"design", "configuration", "population", "outcome", "facts", "schema_version"}:
                candidate = CandidateDraft(method=raw.get("method", method), configuration=raw)
            else:
                candidate = coerce_candidate(raw)
                if candidate.method is None and "schema_version" not in raw:
                    candidate = candidate.model_copy(update={"method": method})
        evaluation = evaluate_candidate(candidate)
    except ValidationError:
        evaluation = evaluate_candidate(draft)
        candidate = CandidateDraft()
    same_method = candidate.method == method
    options = evaluation.options if same_method else ()
    diagnostics = diagnostic_plans(candidate, evaluation) if same_method and evaluation.configuration is not None else ()
    if topic == "diagnostic_details":
        if diagnostic_id not in {d.id for d in definition.diagnostics}:
            raise CapabilityRequestError("diagnostic_id", "Select a diagnostic returned by this method's graph.")
        diagnostics = tuple(d for d in diagnostics if d.diagnostic_id == diagnostic_id)
    exclusions = tuple(Exclusion(option=option.node_id.split(":", 2)[-1], reason=option.explanation)
                       for option in options if option.state in ("blocked", "unsupported", "conditional"))
    if topic in ("diagnostics", "diagnostic_details"):
        applicable = tuple(option.node_id.split(":", 2)[-1] for option in options
                           if option.node_id.startswith(f"{method}:diagnostic:") and option.state == "available")
    else:
        applicable = tuple(option.node_id.split(":", 2)[-1] for option in options
                           if (f"{method}:decision:" in option.node_id or f"{method}:option:" in option.node_id)
                           and option.state == "available")
    available = {option.node_id for option in options if option.state == "available"}
    sensitivities = tuple(SensitivityPlan(sensitivity_id=branch.id, title=branch.title, purpose=branch.purpose,
                                         delta=tuple(Measurement(name=key, value=value) for key, value in branch.delta))
                          for branch in definition.sensitivities if f"{method}:sensitivity:{branch.id}" in available)
    role_slots = evaluation.role_slots if same_method else tuple(
        RoleSlot(node_id=f"{method}:role:{role.id}", role=role.id, field=role.field,
                 kind=role.kind, minimum=role.minimum, maximum=role.maximum,
                 description=role.description) for role in definition.roles)
    columns = tuple(ColumnRequirement(name, slot.role, cast(Any, slot.kind)) for slot in role_slots
                    for name in slot.bound_columns)
    units: tuple[Measurement, ...] = () if candidate.outcome is None or candidate.outcome.units is None else (
        Measurement(name=candidate.outcome.column or "outcome", value=candidate.outcome.units),)
    if method == "rdd" and candidate.configuration.get("running_units"):
        units += (Measurement(name=candidate.configuration.get("running_column") or "running_variable",
                              value=candidate.configuration["running_units"]),)
    guidance_dir = Path(str(module.__file__)).parent / "guidance"
    requested = guidance_dir / f"{topic}.md"
    source = requested if requested.exists() else guidance_dir / "identification.md"
    explanation = source.read_text(encoding="utf-8")
    if topic == "diagnostic_details":
        explanation += "\n" + next(d.purpose for d in definition.diagnostics if d.id == diagnostic_id)
    return Guidance(method=method, topic=topic, capability_version=definition.version,
                    explanation=explanation, input_schema=definition.specification.model_json_schema(),
                    applicable_options=applicable, required_facts=definition.required_facts,
                    required_columns=columns, measurement_units=units,
                    derived_quantities=definition.derived_quantities, sensitivities=sensitivities,
                    unresolved_prerequisites=evaluation.issues, exclusions=exclusions,
                    diagnostics=diagnostics if topic in ("diagnostics", "diagnostic_details") else (),
                    follow_up_topics=tuple(t for t in get_args(Topic) if t != topic),
                    graph_entry=f"method:{method}", evaluation=evaluation, options=options,
                    role_slots=role_slots)


def assess_specification(fixed_design: FixedCandidate | FixedDesign | Mapping[str, Any],
                         draft: AnalysisSpecification | Mapping[str, Any]) -> Assessment:
    return assess(fixed_design, draft)


def preflight(specification: AnalysisSpecification, data: object) -> PreflightResult:
    result = assess_specification(specification.design, specification)
    identity = identify_data(data, specification.dataset.name)
    issues = result.issues
    if identity != specification.dataset:
        issues += (Issue(category="incompatible_data", field="dataset.content_hash",
                         finding="The supplied data differs from the declared dataset.",
                         requirement="Bind the intended exact dataset before approval.",
                         explanation="Changed values, schema or row order invalidate readiness."),)
    observations: tuple[Measurement, ...] = ()
    if result.specification is not None:
        data_issues, observations = inspect_data(result.specification, data)
        issues += data_issues
    targets = tuple(target for row in result.evaluation.requirements for target in row.resolution_targets
                    ) if result.evaluation is not None else ()
    assigned = {target.field for target in targets}
    targets += tuple(ResolutionTarget(
        kind="data_preparation" if i.category in ("missing_data", "incompatible_data") else "configuration",
        field=i.field, requirement_id=f"{specification.configuration.method}:check:data"
        if i.category in ("missing_data", "incompatible_data") else "analysis:requirement:input")
        for i in issues if i.field not in assigned)
    return PreflightResult(specification_hash=fingerprint(specification),
                           implementation_hash=implementation_hash(specification.configuration.method),
                           capability_version=method_module(specification.configuration.method).DEFINITION.version,
                           data=identity, ready=not issues, issues=issues, observations=observations,
                           resolution_targets=targets)


def compile_plan(specification: AnalysisSpecification, preflight: PreflightResult) -> CompiledPlan:
    return compile_checked(specification, preflight)


def execute(approved_plan: ApprovedPlan, data: object) -> AnalysisEvidence:
    from causal.analysis.runner import execute_approved

    return execute_approved(approved_plan, data)
