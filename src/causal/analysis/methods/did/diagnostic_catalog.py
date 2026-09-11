"""Panel diagnostics and sensitivities whose parameter changes are implemented."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from causal.analysis.common.definitions import DiagnosticDefinition, SensitivityDefinition
from causal.analysis.common.models import Applicability, Scalar

if TYPE_CHECKING:
    from causal.analysis.methods.did.specification import Specification

from causal.analysis.common.definitions import DiagnosticDefinition as D
from causal.analysis.common.definitions import SensitivityDefinition as S

CORE_REQUIREMENTS = ("panel", "profile", "adoption_schedule", "parallel_trends", "no_anticipation")

DIAGNOSTICS: tuple[DiagnosticDefinition, ...] = (
    D("unit_period_schema_reconciliation", "Panel structure", "required",
      "Check persistent units and unit-period support.", "required_blocking", limitation_category="population"),
    D("group_time_cohort_support", "Cohort support", "required",
      "Check support in the cohort-period comparisons.", "invalidation_guard", (("min_cell_units", 5),), limitation_category="population"),
    D("pre_and_post_period_placement", "Pre/post period support", "required",
      "Check that the adopted design has usable pre-treatment and post-treatment periods.", "required_blocking",
      (("min_pre_periods", 2), ("min_post_periods", 1)), limitation_category="population"),
    D("treatment_timing_anticipation", "Treatment timing", "required",
      "Disclose the approved anticipation convention.", "qualification_guard", (("max_anticipation_periods", 1),), limitation_category="causal_interpretation"),
    D("event_study_pre_period_test", "Pre-treatment event-time evidence", "required",
      "Describe departures before adoption without interpreting a nonsignificant test as proof of parallel trends.",
      "qualification_guard", (("joint_pre_period_p_value_floor", 0.05),), limitation_category="causal_interpretation"),
    D("attrition_composition_change", "Panel composition", "required",
      "Flag changing unit composition across periods.", "qualification_guard", (("max_composition_change_share", 0.1),), limitation_category="population"),
    D("cluster_covariance_adequacy", "Cluster uncertainty support", "required",
      "Check the independent clusters supporting uncertainty.", "invalidation_guard", (("minimum_clusters", 10),)),
    D("aggregate_weight_sensitivity", "Aggregation weights", "required",
      "Disclose which cohort and event-time cells contribute to the aggregate.", "descriptive"),
    D("concurrent_event_qualification", "Concurrent events", "required",
      "Retain the possibility of concurrent differential shocks.", "descriptive", limitation_category="causal_interpretation"),
    D("reference_period_numerical_integrity", "Reference period", "required",
      "Verify that the fixed pre-treatment reference remains supported.", "required_blocking"),
)

DIAGNOSTICS = tuple(replace(check, dependencies=CORE_REQUIREMENTS) for check in DIAGNOSTICS)

SENSITIVITIES: tuple[SensitivityDefinition, ...] = (
    S("alternative_anticipation_window", "One-period anticipation",
      "Compare a one-period anticipation window with the approved zero-anticipation primary.",
      (("anticipation_periods", 1),)),
    S("alternative_event_time_aggregation", "Equal event-time aggregation",
      "Compare weighting across event times within the staggered-adoption specification.",
      (("aggregation", "equal_weighted_event_time"),)),
    S("balanced_panel_contribution", "Balanced-panel contribution",
      "Assess the restriction to units represented in every observed period, reporting the changed population.",
      (("mask_rule_id", "balanced_panel_cell"),)),
)

SENSITIVITIES = tuple(replace(branch, dependencies=CORE_REQUIREMENTS) for branch in SENSITIVITIES)


def diagnostic_applicability(
    definition: DiagnosticDefinition, config: Specification, facts: Mapping[str, Scalar],
) -> tuple[Applicability, str]:
    return "applicable", "This check applies to the declared panel and event-time specification."


def sensitivity_applicability(
    definition: SensitivityDefinition, config: Specification, facts: Mapping[str, Scalar],
) -> tuple[Applicability, str]:
    if config.profile is None:
        return "unresolved", "The adoption profile has not been selected."
    if definition.id == "alternative_event_time_aggregation" and config.profile != "staggered":
        return "inapplicable", "The simultaneous estimator does not aggregate staggered event-time cells."
    return "applicable", "The fixed sensitivity changes an implemented part of this panel specification."


applicability = diagnostic_applicability
