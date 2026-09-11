"""Sharp-RDD checks and justified fixed comparisons; no arbitrary donut or placebo."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from causal.analysis.common.definitions import DiagnosticDefinition, SensitivityDefinition
from causal.analysis.common.models import Applicability, Scalar

if TYPE_CHECKING:
    from causal.analysis.methods.rdd.specification import Specification

from causal.analysis.common.definitions import DiagnosticDefinition as D
from causal.analysis.common.definitions import SensitivityDefinition as S

CORE_REQUIREMENTS = ("contrast", "running_measurement", "sharp_assignment", "assignment_direction", "cutoff_rule")

DIAGNOSTICS: tuple[DiagnosticDefinition, ...] = (
    D("cutoff_side_support", "Local cutoff support", "required",
      "Check effective observations on both sides within the selected bandwidth.", "required_blocking",
      (("min_effective_observations_per_side", 20),), limitation_category="population"),
    D("selected_bandwidth_report", "Selected bandwidth", "required",
      "Report the mechanically selected bandwidth on each side.", "required_blocking"),
    D("mass_points_and_heaping", "Running-variable heaping", "required",
      "Flag repeated running-variable values affecting local estimation.", "qualification_guard",
      (("max_repeated_value_share", 0.2),), limitation_category="measurement"),
    D("density_manipulation_test", "Density near the cutoff", "required",
      "Measure a discontinuity in density; failure to detect one does not prove no sorting.", "invalidation_guard",
      (("density_p_value_floor", 0.05),), limitation_category="causal_interpretation"),
    D("covariate_continuity", "Predetermined covariate continuity", "required",
      "Assess a discontinuity in the approved predetermined covariate.", "qualification_guard",
      (("max_covariate_jump_z", 2.0),), trigger="predetermined_covariate", limitation_category="measurement"),
    D("bandwidth_polynomial_sensitivity", "Local specification sensitivity", "required",
      "Run half-bandwidth, double-bandwidth and quadratic probes against the primary local linear fit.",
      "qualification_guard", (("max_relative_estimate_change", 0.5),)),
    D("influence_leverage_near_cutoff", "Near-cutoff concentration", "required",
      "Describe concentration of observations near the cutoff.", "qualification_guard",
      (("max_single_observation_leverage_share", 0.25),)),
    D("sorting_and_other_policy_qualification", "Other cutoff mechanisms", "required",
      "Retain limitations from sorting and other policies sharing the cutoff.", "descriptive", limitation_category="causal_interpretation"),
    D("robust_bias_correction_integrity", "Bias-correction integrity", "required",
      "Verify that robust bias-corrected quantities were computed.", "required_blocking"),
)

DIAGNOSTICS = tuple(replace(check, dependencies=CORE_REQUIREMENTS + (
    ("covariate_timing",) if check.trigger == "predetermined_covariate" else ()))
    for check in DIAGNOSTICS)

SENSITIVITIES: tuple[SensitivityDefinition, ...] = (
    S("half_bandwidth", "Half bandwidth", "Use half each selected primary bandwidth.",
      (("bandwidth_multiplier", 0.5),)),
    S("double_bandwidth", "Double bandwidth", "Use twice each selected primary bandwidth.",
      (("bandwidth_multiplier", 2.0),)),
    S("local_quadratic", "Local quadratic", "Compare a quadratic fit with the primary local linear fit.",
      (("polynomial_order", 2),)),
    S("alternative_kernel", "Uniform kernel", "Compare a uniform kernel with the primary triangular kernel.",
      (("kernel", "uniform"),)),
    S("covariate_adjusted", "Predetermined covariate adjustment",
      "Adjust for the single approved predetermined covariate.", (("covariate_adjustment", True),),
      ("predetermined_covariate",)),
)

SENSITIVITIES = tuple(replace(branch, dependencies=CORE_REQUIREMENTS + (
    ("covariate_timing",) if branch.id == "covariate_adjusted" else ()))
    for branch in SENSITIVITIES)


def diagnostic_applicability(
    definition: DiagnosticDefinition, config: Specification, facts: Mapping[str, Scalar],
) -> tuple[Applicability, str]:
    if definition.trigger == "predetermined_covariate" and config.covariate_column is None:
        return "inapplicable", "No predetermined covariate is selected."
    return "applicable", "This check applies at the declared sharp cutoff."


def sensitivity_applicability(
    definition: SensitivityDefinition, config: Specification, facts: Mapping[str, Scalar],
) -> tuple[Applicability, str]:
    if definition.id == "covariate_adjusted" and config.covariate_column is None:
        return "inapplicable", "Adjustment requires an approved predetermined covariate."
    return "applicable", "The fixed comparison is evaluated against the declared primary local fit."


applicability = diagnostic_applicability
