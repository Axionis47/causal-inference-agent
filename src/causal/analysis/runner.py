"""Execute exactly a compiled, approved plan. No repair or effect-driven selection."""
from __future__ import annotations

import math
from importlib import metadata
from typing import Any, Literal, cast

from pydantic import ValidationError

from causal.analysis.contracts import (
    AnalysisEvidence,
    ApprovedPlan,
    BoundaryError,
    Computation,
    DiagnosticPlan,
    Estimate,
    Limitation,
    Measurement,
    Provenance,
)


def _measurements(values: dict[str, Any]) -> tuple[Measurement, ...]:
    return tuple(Measurement(name=name, value=value) for name, value in sorted(values.items()))


def _estimates(items: Any, ctx: Any) -> tuple[Estimate, ...]:
    return tuple(Estimate(
        contrast=i.contrast_id, estimand=i.estimand_id, label=i.estimand_label,
        estimate=i.estimate, units=i.estimate_units, standard_error=i.standard_error,
        confidence_level=i.confidence_level, interval_lower=i.interval_lower,
        interval_upper=i.interval_upper, p_value=i.p_value, uncertainty_method=i.uncertainty_method, convergence=i.convergence,
        population=_measurements(ctx.population_counts(i)),
        method_quantities=_measurements(ctx.restore_measurements({"derived": i.method_quantities})["derived"])) for i in items)


def _policy(row: DiagnosticPlan, values: dict[str, Any]) -> Literal["acceptable", "warning", "invalidating", "descriptive"]:
    # These are shared threshold comparisons, not scientific applicability decisions.
    from causal.analysis.common.legacy_diagnostics import _bound, severity_result

    thresholds = {v.name: v.value for v in row.thresholds}
    for key in thresholds:
        name, _ = _bound(key)
        value = values.get(name)
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise ValueError(f"The diagnostic did not produce a finite numerical measurement for {name!r}.")
    policy = severity_result(cast(Any, row.severity), values, thresholds)
    return "invalidating" if row.severity == "required_blocking" and policy == "warning" else policy


def _threshold_explanation(row: DiagnosticPlan, values: dict[str, Any]) -> str:
    from causal.analysis.common.legacy_diagnostics import _bound

    findings = []
    for threshold in row.thresholds:
        name, lower = _bound(threshold.name)
        value, limit = values.get(name), threshold.value
        if (isinstance(value, int | float) and isinstance(limit, int | float)
                and ((value < limit) if lower else (value > limit))):
            findings.append(f"{name.replace('_', ' ')} was {value:.4g}; the planned "
                                f"{'minimum' if lower else 'maximum'} is {limit:.4g}")
    return "; ".join(findings) + f". This limits {row.limitation_category.replace('_', ' ')}."


def _diagnostic(row: DiagnosticPlan, harvest: dict[str, Any], errors: dict[str, str],
                primary_ok: bool) -> Computation:
    if row.applicability == "inapplicable":
        return Computation(computation_id=row.diagnostic_id, status="inapplicable",
                           explanation=row.explanation)
    if not row.selected:
        return Computation(computation_id=row.diagnostic_id, status="not_selected",
                           explanation="This optional check was not selected in the approved plan.")
    if not primary_ok:
        return Computation(computation_id=row.diagnostic_id, status="blocked",
                           explanation="The primary numerical fit failed; this check could not run.")
    if row.diagnostic_id in errors:
        return Computation(computation_id=row.diagnostic_id, status="failed",
                           explanation=errors[row.diagnostic_id], policy="warning")
    values = harvest.get(row.diagnostic_id)
    if not values:
        return Computation(computation_id=row.diagnostic_id, status="unavailable",
                           explanation="The implementation returned no diagnostic measurements.", policy="warning")
    try:
        policy = _policy(row, values)
        explanation = row.title
        if policy in ("warning", "invalidating"):
            explanation += ": " + _threshold_explanation(row, values)
        return Computation(computation_id=row.diagnostic_id, status="computed",
                           explanation=explanation, measurements=_measurements(values), policy=policy)
    except (ValueError, TypeError) as error:
        return Computation(computation_id=row.diagnostic_id, status="unavailable",
                           explanation=str(error), policy="warning")


def _supporting_data(harvest: dict[str, Any], errors: dict[str, str]) -> tuple[Computation, ...]:
    """Expose complete harvested quantities, without inventing display coordinates or fits."""
    rows = []
    for name in sorted(harvest.keys() | errors.keys()):
        values = harvest.get(name, {})
        unavailable = tuple(key for key, value in values.items()
                            if isinstance(value, float) and not math.isfinite(value))
        clean = {key: None if key in unavailable else value for key, value in values.items()}
        explanation = errors.get(name, "Frozen supporting numerical measurements from the primary analysis.")
        if unavailable:
            explanation += " Non-finite source values are unavailable: " + ", ".join(unavailable)
        rows.append(Computation(computation_id=name,
            status="failed" if name in errors else "unavailable" if unavailable or not values else "computed",
            explanation=explanation, measurements=_measurements(clean)))
    return tuple(rows)


def _limitations(plan: Any, diagnostics: tuple[Computation, ...],
                 sensitivities: tuple[Computation, ...], supporting_data: tuple[Computation, ...]
                  ) -> tuple[Limitation, ...]:
    rows = {row.diagnostic_id: row for row in plan.diagnostics}
    limits = [Limitation(category="causal_interpretation", source="fixed_design",
                         explanation="The causal interpretation relies on the recorded study assumptions; numerical checks cannot establish them.")]
    for computation in (*diagnostics, *sensitivities,
                        *(row for row in supporting_data if row.computation_id not in rows)):
        if computation.status in ("failed", "blocked", "unavailable") or computation.policy in (
                "warning", "invalidating"):
            row = rows.get(computation.computation_id)
            limits.append(Limitation(category=row.limitation_category if row else "confidence",
                                     source=computation.computation_id, explanation=computation.explanation))
    return tuple(limits)


def _environment() -> tuple[Measurement, ...]:
    libraries = ("polars", "numpy", "scipy", "pyfixest", "scikit-learn", "rdrobust", "rddensity")
    versions = []
    for name in libraries:
        try:
            value = metadata.version(name)
        except metadata.PackageNotFoundError:
            value = "not installed"
        versions.append(Measurement(name=name, value=value))
    return tuple(versions)


def execute_approved(approved: ApprovedPlan, data: object) -> AnalysisEvidence:
    from causal.analysis.interface import compile_plan, preflight

    try:
        approval = ApprovedPlan.model_validate(approved.model_dump())
    except ValidationError as error:
        raise BoundaryError("The approved plan is not a valid immutable boundary artifact.") from error
    plan = approval.plan
    if plan.schema_version != "analysis-plan.v3" or plan.specification.schema_version != "analysis-specification.v2":
        raise BoundaryError("Historical plans remain readable but cannot authorize a new execution.")
    if approval.approved_hash != plan.plan_hash:
        raise BoundaryError("Approval does not match the compiled plan hash.")
    readiness = preflight(plan.specification, data)
    current = compile_plan(plan.specification, readiness)
    if current != plan:
        raise BoundaryError("The compiled plan or capability definition changed; obtain new approval.")
    from causal.analysis.integration.numerical import prepare

    ctx, result = None, None
    harvest: dict[str, Any] = {}
    errors: dict[str, str] = {}
    try:
        ctx = prepare(plan, cast(Any, data))
        result = ctx.fit()
        estimates = _estimates(result.items, ctx)
        if not estimates or any(not math.isfinite(e.estimate) for e in estimates):
            raise ValueError("The estimator returned no finite primary contrast.")
        converged = all(e.convergence != "not_converged" for e in estimates)
        primary = Computation(computation_id="primary", status="computed" if converged else "failed",
                              estimates=estimates, explanation="The approved primary numerical fit completed." if converged else
                              "The estimator reported nonconvergence. Its estimates are retained for audit.")
    except Exception as error:  # noqa: BLE001 -- every planned computation returns a visible failure
        primary = Computation(computation_id="primary", status="failed", explanation=f"{type(error).__name__}: " + (ctx.restore_names(str(error)) if ctx is not None else str(error)))
    if ctx is not None and result is not None:
        try:
            harvest, errors = ctx.collect(result)
        except Exception as error:  # noqa: BLE001 -- every planned computation returns a visible failure
            errors = {r.diagnostic_id: f"Diagnostic collection failed: {error}" for r in plan.diagnostics}
    public_measurements = ctx.restore_measurements(harvest) if ctx is not None else harvest
    diagnostics = tuple(_diagnostic(r, public_measurements, errors, result is not None) for r in plan.diagnostics)
    sensitivities = _sensitivities(ctx, plan, primary.status == "computed")
    supporting_data = _supporting_data(public_measurements, errors)
    limitations = _limitations(plan, diagnostics, sensitivities, supporting_data)
    incomplete = any(c.status in ("failed", "blocked", "unavailable") for c in (*diagnostics, *sensitivities))
    blocking = {r.diagnostic_id for r in plan.diagnostics if r.severity == "required_blocking"}
    incomplete = incomplete or any(c.computation_id in blocking and c.policy == "invalidating" for c in diagnostics)
    status: Literal["completed", "completed_with_limitations", "incomplete", "failed"] = ("failed" if primary.status == "failed" else "incomplete" if incomplete else
              "completed_with_limitations" if len(limitations) > 1 else "completed")
    return AnalysisEvidence(status=status, population=plan.specification.design.population,
                            primary=primary, diagnostics=diagnostics, sensitivities=sensitivities,
                            limitations=limitations, supporting_data=supporting_data,
                            provenance=Provenance(plan_hash=plan.plan_hash, data=plan.specification.dataset,
                                                  capability_version=plan.capability_version, seed=plan.specification.seed,
                                                  approved_by=approval.approved_by, approval_reference=approval.approval_reference,
                                                  environment=_environment()))


def _sensitivities(ctx: Any, plan: Any, primary_ok: bool) -> tuple[Computation, ...]:
    rows = []
    for branch in plan.sensitivities:
        if not primary_ok:
            rows.append(Computation(computation_id=branch.sensitivity_id, status="blocked",
                                    explanation="The primary numerical fit failed."))
            continue
        try:
            result = ctx.fit({m.name: m.value for m in branch.delta})
            estimates = _estimates(result.items, ctx)
            if not estimates:
                raise ValueError("The sensitivity returned no estimates.")
            converged = all(e.convergence != "not_converged" for e in estimates)
            rows.append(Computation(computation_id=branch.sensitivity_id, status="computed" if converged else "failed",
                                    explanation=branch.purpose if converged else "The sensitivity reported nonconvergence; estimates retained for audit.", estimates=estimates))
        except Exception as error:  # noqa: BLE001 -- every planned computation returns a visible failure
            rows.append(Computation(computation_id=branch.sensitivity_id, status="failed",
                                    explanation=f"{type(error).__name__}: {ctx.restore_names(str(error))}", policy="warning"))
    return tuple(rows)
