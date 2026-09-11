"""Sharp RDD at an evidenced cutoff, with the above-cutoff state treated."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, cast

from pydantic import Field

from causal.analysis.common.definitions import (
    ColumnRequirement,
    FixedPolicyDefinition,
    MethodDefinition,
    assess_contrast,
    check_contrast,
    check_independent_units,
    fact_contradiction,
    missing_fields,
)
from causal.analysis.common.definitions import (
    RequirementDefinition as R,
)
from causal.analysis.common.definitions import (
    RoleDefinition as Role,
)
from causal.analysis.common.models import Issue, Model, Scalar
from causal.analysis.methods.rdd.diagnostic_catalog import (
    CORE_REQUIREMENTS,
    DIAGNOSTICS,
    SENSITIVITIES,
)

Column = Annotated[str, Field(min_length=1)]


class Specification(Model):
    method: Literal["rdd"] = "rdd"
    estimand: Literal["late_at_cutoff"] = "late_at_cutoff"
    treatment_column: Column | None = None
    unit_column: Column | None = None
    treated_value: Column | None = None
    comparator_value: Column | None = None
    running_column: Column | None = None
    cutoff: Annotated[float, Field(strict=True)] | None = None
    running_units: Column | None = None
    assignment_direction: Literal["above"] = "above"
    covariate_column: Column | None = None
    cluster_column: Column | None = None
    polynomial_order: Literal[1] = 1
    kernel: Literal["triangular"] = "triangular"
    bandwidth_selector: Literal["mserd"] = "mserd"
    confidence_level: Annotated[float, Field(ge=0.95, le=0.95)] = 0.95


def columns(config: Specification) -> tuple[ColumnRequirement, ...]:
    return tuple(ColumnRequirement(name, role, cast(Any, kind)) for name, role, kind in (
        (config.treatment_column, "treatment", "categorical"),
        (config.unit_column, "unit_identifier", "identifier"),
        (config.running_column, "running_variable", "numeric"),
        (config.covariate_column, "predetermined_covariate", "numeric"),
        (config.cluster_column, "cluster", "identifier")) if name is not None)


def _fact(facts: Mapping[str, Scalar], name: str, expected: Scalar) -> tuple[Issue, ...]:
    return tuple(issue.model_copy(update={"field": f"facts.{name}"})
                 for issue in fact_contradiction(facts, name, expected))


REQUIREMENTS = (
    R("contrast", ("configuration.treatment_column", "configuration.unit_column",
                   "configuration.treated_value", "configuration.comparator_value"), "choice",
      "Bind one analysis unit and two distinct treatment states.",
      "Sharp RDD compares the declared treated and comparator states at the policy threshold.",
      lambda c, f: assess_contrast(c)),
    R("running_measurement", ("configuration.running_column", "configuration.cutoff",
                              "configuration.running_units"), "role",
      "Identify a numeric running measurement, its units and the policy cutoff.",
      "The policy cutoff is a scientific input; bandwidth selection cannot choose or change it.",
      lambda c, f: missing_fields(c, "running_column", "cutoff", "running_units")),
    R("sharp_assignment", ("facts.sharp_assignment",), "fact",
      "Source evidence establishes sharp assignment at the cutoff.",
      "A selected threshold does not establish sharp assignment. Fuzzy compliance is unsupported.",
      lambda c, f: _fact(f, "sharp_assignment", True), fact_name="sharp_assignment"),
    R("assignment_direction", ("facts.assignment_direction",), "fact",
      "The state at or above the cutoff is treated.",
      "The current estimator reports the above-cutoff treatment effect and cannot reverse its meaning.",
      lambda c, f: _fact(f, "assignment_direction", "above"), fact_name="assignment_direction"),
    R("cutoff_rule", ("facts.cutoff", "configuration.cutoff"), "fact",
      "The proposed cutoff equals the source-supported assignment threshold.",
      "The analysis cannot move the threshold to satisfy observed data or improve a fitted result.",
      lambda c, f: _fact(f, "cutoff", c.cutoff) if c.cutoff is not None else (),
      dependencies=("running_measurement",), fact_name="cutoff"),
    R("covariate_timing", ("facts.covariate_pre_treatment", "configuration.covariate_column"), "fact",
      "The selected numeric covariate was determined before treatment assignment.",
      "Binding a column does not establish that it is predetermined; timing evidence must name this covariate.",
      lambda c, f: _fact(f, "covariate_pre_treatment", True),
      fact_name="covariate_pre_treatment", active_when=lambda c: c.covariate_column is not None,
      scope_fields=("configuration.covariate_column",)),
)

SCIENTIFIC_REQUIREMENTS = tuple(rule.id for rule in REQUIREMENTS if rule.fact_name is not None)


def check_data(config: Specification, frame: Any) -> tuple[Issue, ...]:
    issues = check_independent_units(config, frame) + check_contrast(config, frame)
    if (config.running_column is None or config.running_column not in frame.columns
            or config.treatment_column is None or config.treatment_column not in frame.columns
            or config.cutoff is None or config.treated_value is None or config.comparator_value is None):
        return issues
    import polars as pl

    expected = pl.when(pl.col(config.running_column) >= config.cutoff).then(
        pl.lit(config.treated_value)).otherwise(pl.lit(config.comparator_value))
    if frame.filter(pl.col(config.treatment_column).cast(pl.String) != expected).height:
        issues += (Issue(category="incompatible_data", field=f"dataset.{config.treatment_column}",
                         finding="Treatment does not follow the approved sharp cutoff rule.",
                         requirement="Every unit at or above the cutoff is treated; all others are controls.",
                         explanation="Assignment contradictions cannot be repaired by moving the cutoff."),)
    for name, predicate in (("below", pl.col(config.running_column) < config.cutoff),
                            ("above", pl.col(config.running_column) >= config.cutoff)):
        side = frame.filter(predicate)
        if side.height < 20 or side[config.running_column].n_unique() < 4:
            issues += (Issue(category="incompatible_data", field=f"dataset.{config.running_column}",
                             finding=f"The {name}-cutoff side has insufficient observations or distinct values.",
                             requirement="At least 20 observations and four distinct running values per side.",
                             explanation="The primary fit and mandatory quadratic probe need local support; final bandwidth support is checked during execution."),)
    return issues


DEFINITION = MethodDefinition(
    "rdd", "rdd.v2", "Sharp regression discontinuity",
    "A local discontinuity with above-cutoff treatment, local linear fitting and robust bias correction.",
    Specification, tuple(rule.fact_name for rule in REQUIREMENTS
                         if rule.fact_name is not None and rule.active_when is None),
    (("below_cutoff_treated", "The current numerical adapter does not reverse its reported effect sign."),
     ("fuzzy_rdd", "No compliance-ratio estimator is implemented."),
     ("multiple_cutoffs", "The estimator consumes exactly one approved cutoff."),
     ("donut_or_placebo", "Arbitrary radii and offsets lack a study-specific justification."),
     ("multiple_covariates", "This adapter accepts one numeric predetermined covariate.")),
    DIAGNOSTICS, SENSITIVITIES,
    requirements=REQUIREMENTS,
    fixed_policies=(
        FixedPolicyDefinition('primary_quantity', 'robust_bias_corrected',
            'Report the robust bias-corrected primary discontinuity and uncertainty.'),
        FixedPolicyDefinition('mass_point_handling', 'checked',
            'Apply the implemented running-variable mass-point support checks.'),
        FixedPolicyDefinition('mask_rule_id', 'selected_bandwidth',
            'Restrict each local fit to its prespecified bandwidth procedure.'),
        FixedPolicyDefinition('bandwidth_multiplier', 1.0,
            'Use the selected bandwidth unchanged for the primary fit; multipliers are sensitivity comparisons.'),
        FixedPolicyDefinition('donut_radius', 0.0,
            'Retain observations next to the scientific cutoff in the primary fit.'),
        FixedPolicyDefinition('placebo_cutoff_offset', 0.0,
            'Use the source-supported assignment cutoff without a placebo offset.'),
        FixedPolicyDefinition('covariate_adjustment', False,
            'Keep the primary discontinuity unadjusted; a declared covariate supports continuity checks and the separate adjusted sensitivity.'),
    ),
    roles=(
        Role("treatment", "configuration.treatment_column", "categorical", minimum=1,
             description="The two states governed by the sharp assignment rule.", dependencies=("contrast",)),
        Role("unit_identifier", "configuration.unit_column", "identifier", minimum=1,
             description="One row per independently identified analysis unit.", dependencies=("contrast",)),
        Role("running_variable", "configuration.running_column", "numeric", minimum=1,
             description="The source-defined measurement used by the assignment threshold.", dependencies=("running_measurement",)),
        Role("predetermined_covariate", "configuration.covariate_column", "numeric",
             description="At most one numeric covariate determined before assignment.", dependencies=("covariate_timing",)),
        Role("cluster", "configuration.cluster_column", "identifier",
             description="An optional declared cluster for uncertainty estimation."),
    ),
    option_dependencies=tuple((field, SCIENTIFIC_REQUIREMENTS)
                              for field in Specification.model_fields if field != "covariate_column") + (
        ("covariate_column", CORE_REQUIREMENTS + ("covariate_timing",)),
    ),
    derived_quantities=("estimate", "standard_error", "confidence_interval", "selected_bandwidths",
                        "effective_observations_per_side", "bias_corrected_estimate"))
