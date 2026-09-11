"""Apply the planned diagnostic policy to measurements from each analysis method.

Methods collect measurements from their own fitted models and prepared inputs. This
module assigns execution status, applies registered thresholds and severity, checks
applicability, and preserves diagnostic policy results for interpretation downstream. It neither chooses a
new analysis from observed effects nor fits a model to repair an adverse diagnostic.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from causal.analysis.integration import contracts as ec
from causal.analysis.integration.packs import EstimationPackV1

# Preserve existing artifact provenance: this extraction changes ownership, not policy.
IMPLEMENTATION_VERSION: Final = "estimation-engine.v1"
DIAGNOSTIC_NOT_COMPUTED: Final = "diagnostic_not_computed"


def _as_float(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


def _bound(key: str) -> tuple[str, bool]:
    # One registered threshold key as (measured value name, whether it is a lower bound).
    for prefix, lower in (("max_", False), ("min_", True), ("minimum_", True)):
        if key.startswith(prefix):
            return key[len(prefix):], lower
    return (key[:-6], True) if key.endswith("_floor") else (key, False)


def severity_result(severity: ec.DiagnosticSeverity, values: ec.ValueMap,
                    thresholds: ec.ValueMap) -> ec.PolicyResult:
    # §14.2: the prespecified severity and the pack's threshold row decide together. Nothing
    # here may promote, demote, or reinterpret a severity after seeing the value.
    if severity == "descriptive":
        return "descriptive"
    breached = False
    for key, limit in thresholds.items():
        name, lower = _bound(key)
        found, bound = _as_float(values.get(name)), _as_float(limit)
        breached = breached or (found is not None and bound is not None
                                and (found < bound if lower else found > bound))
    if not breached:
        return "acceptable"
    return "invalidating" if severity == "invalidation_guard" else "warning"


def run_diagnostics(pack: EstimationPackV1, harvest: Mapping[str, ec.ValueMap],
                    base: Mapping[str, Any]) -> tuple[ec.DiagnosticResultV1, ...]:
    # §26.1: the registered order, one at a time. Every required row reaches a visible terminal
    # result; an unharvested row is `not_computable` and an empty harvest is `failed`.
    results: list[ec.DiagnosticResultV1] = []
    for row in pack.required_diagnostics:
        values = harvest.get(row.diagnostic_id)
        status: ec.ExecutionStatus = ("not_computable" if values is None
                                      else "computed" if values else "failed")
        policy: ec.PolicyResult = severity_result(
            row.severity, values, row.threshold_params) if values else (
            "descriptive" if row.severity == "descriptive" else "warning")
        results.append(ec.DiagnosticResultV1(
            **base, diagnostic_id=row.diagnostic_id, diagnostic_version="v1",
            severity=row.severity, threshold_context=dict(row.threshold_params),
            execution_status=status, policy_result=policy, values=dict(values or {}),
            warnings=() if status == "computed" else (DIAGNOSTIC_NOT_COMPUTED,),
            interpreting_rule_id=row.diagnostic_id, implementation_version=IMPLEMENTATION_VERSION))
    return tuple(results)


def qualified_inapplicable_diagnostic(plan: ec.EstimationPlanV1,
                                     row: ec.DiagnosticResultV1) -> bool:
    # Covariate continuity has no scientific target when the approved sharp RDD contains no
    # predetermined covariates. This one inapplicable row remains visible and qualifying.
    # A missing row, a failed computation, or any approved covariate never receives this waiver.
    covariate_roles = {"predetermined_covariate", "precision_covariate", "confounder_candidate"}
    return (plan.method_id == "sharp_rdd" and row.diagnostic_id == "covariate_continuity"
            and plan.required_diagnostics.get(row.diagnostic_id) == "qualification_guard"
            and row.severity == "qualification_guard" and row.execution_status == "not_computable"
            and row.policy_result == "warning" and not row.values
            and not any(role.partition("__")[0] in covariate_roles for role in plan.role_columns))




