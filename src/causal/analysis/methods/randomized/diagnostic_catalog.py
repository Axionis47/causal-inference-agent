"""Implemented checks and fixed sensitivity deltas for randomized experiments."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from causal.analysis.common.definitions import DiagnosticDefinition, SensitivityDefinition
from causal.analysis.common.models import Applicability, Scalar

if TYPE_CHECKING:
    from causal.analysis.methods.randomized.specification import Specification

from causal.analysis.common.definitions import DiagnosticDefinition as D
from causal.analysis.common.definitions import SensitivityDefinition as S

CORE_REQUIREMENTS = ("contrast", "estimator", "assignment_mechanism", "assignment_unit", "precision_covariate")

DIAGNOSTICS: tuple[DiagnosticDefinition, ...] = (
    D("randomization_unit_reconciliation", "Assignment and analysis units", "required",
      "Retain the randomized population when counting outcome availability.", "required_blocking", limitation_category="population"),
    D("arm_cluster_stratum_contribution_counts", "Contributing groups", "required",
      "Make arm, cluster and stratum support visible.", "required_blocking", limitation_category="population"),
    D("baseline_balance", "Baseline balance", "optional",
      "Describe imbalance without treating a balance test as evidence for randomization.",
      "descriptive", trigger="precision_covariate", limitation_category="measurement"),
    D("outcome_attrition_by_arm", "Missing outcomes", "required",
      "Show overall and differential attrition from the assigned population.", "qualification_guard",
      (("max_overall_attrition", 0.2), ("max_differential_attrition", 0.1)), limitation_category="population"),
    D("covariance_cluster_adequacy", "Uncertainty support", "required",
      "Check the number of independent assignment units supporting uncertainty.", "invalidation_guard",
      (("minimum_clusters", 8),)),
    D("influential_cluster_leverage", "Concentration in assignment units", "required",
      "Flag concentration of observations in one assignment cluster.", "qualification_guard",
      (("max_single_cluster_leverage_share", 0.25),)),
    D("model_convergence_integrity", "Numerical convergence", "required",
      "Verify that the declared primary contrast was computed.", "required_blocking",
      (("min_converged_contrasts", 1),)),
    D("multiplicity_handling", "Contrast family disclosure", "required",
      "Record the number of confirmatory contrasts.", "descriptive"),
)

DIAGNOSTICS = tuple(replace(check, dependencies=CORE_REQUIREMENTS + (
    ("precision_timing",) if check.trigger == "precision_covariate" else ()))
    for check in DIAGNOSTICS)

SENSITIVITIES: tuple[SensitivityDefinition, ...] = (
    S("unadjusted_itt", "Unadjusted contrast", "Compare the approved ANCOVA with no precision adjustment.",
      (("specification", "difference_in_means"),), ("precision_covariate",)),
    S("sensitivity_covariance_profile", "Alternative robust covariance",
      "Compare HC3 or CRV3 uncertainty with the primary HC2 or CRV1 calculation.",
      (("cluster_covariance", "cluster_robust_cr3"),)),
)

SENSITIVITIES = tuple(replace(branch, dependencies=CORE_REQUIREMENTS + (
    ("precision_timing",) if branch.id == "unadjusted_itt" else ()))
    for branch in SENSITIVITIES)


def diagnostic_applicability(
    definition: DiagnosticDefinition, config: Specification, facts: Mapping[str, Scalar],
) -> tuple[Applicability, str]:
    if definition.trigger == "precision_covariate" and config.precision_covariate_column is None:
        return "inapplicable", "No baseline precision covariate is selected."
    if definition.trigger == "precision_covariate":
        return "applicable", "A baseline precision covariate is selected."
    return "applicable", "This check applies to the declared randomized contrast."


def sensitivity_applicability(
    definition: SensitivityDefinition, config: Specification, facts: Mapping[str, Scalar],
) -> tuple[Applicability, str]:
    if definition.id == "unadjusted_itt":
        if config.estimator is None:
            return "unresolved", "The primary estimation specification is not selected."
        if config.estimator != "ancova":
            return "inapplicable", "The primary contrast is already unadjusted."
    return "applicable", "The fixed sensitivity changes an implemented part of this specification."


applicability = diagnostic_applicability
