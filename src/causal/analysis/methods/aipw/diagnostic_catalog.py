"""Implemented independent-unit AIPW diagnostics; no unsupported confounding bound."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from causal.analysis.common.definitions import DiagnosticDefinition, SensitivityDefinition
from causal.analysis.common.models import Applicability, Scalar

if TYPE_CHECKING:
    from causal.analysis.methods.aipw.specification import Specification

from causal.analysis.common.definitions import DiagnosticDefinition as D
from causal.analysis.common.definitions import SensitivityDefinition as S

CORE_REQUIREMENTS = ("contrast", "estimand", "adjustment_set", "adjustment_timing")

DIAGNOSTICS: tuple[DiagnosticDefinition, ...] = (
    D("fold_assignment_treatment_support", "Cross-fitting support", "required",
      "Check treatment support and convergence in the fixed folds.", "required_blocking",
      (("max_folds_without_both_states", 0), ("min_converged_folds", 5))),
    D("nuisance_calibration_performance", "Nuisance model diagnostics", "required",
      "Describe predictive calibration; this cannot establish causal exchangeability.", "qualification_guard",
      (("max_calibration_slope_deviation", 0.25),)),
    D("propensity_common_support", "Treatment overlap", "required",
      "Identify target-population regions with poor treatment support.", "invalidation_guard",
      (("min_propensity", 0.01), ("max_out_of_support_share", 0.05)), limitation_category="causal_interpretation"),
    D("weight_tail_effective_sample", "Weight concentration", "required",
      "Show whether a few observations dominate the estimate.", "qualification_guard",
      (("min_effective_sample_fraction", 0.3), ("max_single_weight_share", 0.1))),
    D("weighted_covariate_balance", "Weighted baseline balance", "required",
      "Describe observed covariate balance after weighting.", "qualification_guard",
      (("max_standardized_difference", 0.1),), limitation_category="measurement"),
    D("influence_score_distribution", "Influence concentration", "required",
      "Flag strong dependence on individual observations.", "qualification_guard",
      (("max_single_unit_influence_share", 0.05),)),
    D("cross_fit_score_integrity", "Cross-fitting integrity", "required",
      "Verify fold isolation and score computation.", "required_blocking",
      (("max_validation_rows_fitted", 0),)),
    D("preprocessing_missingness_usage", "Preprocessing disclosure", "required",
      "Record the preprocessing used by nuisance fits.", "descriptive", limitation_category="measurement"),
    D("unmeasured_confounding_qualification", "Unmeasured confounding", "required",
      "Retain the limitation that observed adjustment cannot rule out unmeasured confounding.", "descriptive", limitation_category="causal_interpretation"),
)

DIAGNOSTICS = tuple(replace(check, dependencies=CORE_REQUIREMENTS) for check in DIAGNOSTICS)

SENSITIVITIES: tuple[SensitivityDefinition, ...] = (
    S("alternative_cross_fit_seed", "Alternative cross-fitting assignment",
      "Refit with the next seed and ten folds, retaining the same target population and estimand.",
      (("seed_offset", 1), ("fold_count", 10))),
    S("propensity_bound_sensitivity", "Alternative numerical propensity bound",
      "Compare the fixed numerical bound without deleting rows or redefining the population.",
      (("propensity_bound", 0.02),)),
)

SENSITIVITIES = tuple(replace(branch, dependencies=CORE_REQUIREMENTS) for branch in SENSITIVITIES)


def diagnostic_applicability(
    definition: DiagnosticDefinition, config: Specification, facts: Mapping[str, Scalar],
) -> tuple[Applicability, str]:
    return "applicable", "This check applies to the declared independent-unit AIPW specification."


def sensitivity_applicability(
    definition: SensitivityDefinition, config: Specification, facts: Mapping[str, Scalar],
) -> tuple[Applicability, str]:
    return "applicable", "The fixed sensitivity retains the chosen estimand and target population."


applicability = diagnostic_applicability
