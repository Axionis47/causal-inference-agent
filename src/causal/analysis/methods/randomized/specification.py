"""One prespecified two-arm ITT contrast, with optional precision adjustment."""
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
from causal.analysis.methods.randomized.diagnostic_catalog import DIAGNOSTICS, SENSITIVITIES

Column = Annotated[str, Field(min_length=1)]


class Specification(Model):
    method: Literal["randomized"] = "randomized"
    estimand: Literal["itt"] = "itt"
    treatment_column: Column | None = None
    unit_column: Column | None = None
    treated_value: Column | None = None
    comparator_value: Column | None = None
    estimator: Literal["difference_in_means", "ancova"] | None = None
    precision_covariate_column: Column | None = None
    stratum_column: Column | None = None
    cluster_column: Column | None = None
    confidence_level: Annotated[float, Field(ge=0.95, le=0.95)] = 0.95


def columns(config: Specification) -> tuple[ColumnRequirement, ...]:
    return tuple(ColumnRequirement(name, role, cast(Any, kind)) for name, role, kind in (
        (config.treatment_column, "treatment", "categorical"),
        (config.unit_column, "unit_identifier", "identifier"),
        (config.precision_covariate_column, "precision_covariate", "any"),
        (config.stratum_column, "stratum", "categorical"),
        (config.cluster_column, "cluster", "identifier")) if name is not None)


def _precision(config: Specification, facts: Mapping[str, Scalar]) -> tuple[Issue, ...]:
    issues: tuple[Issue, ...] = ()
    if config.estimator == "ancova":
        issues += missing_fields(config, "precision_covariate_column")
        if config.precision_covariate_column is not None and config.precision_covariate_column in {
            config.stratum_column, config.treatment_column, config.unit_column,
        }:
            issues += (Issue(category="contradictory_configuration", field="configuration.precision_covariate_column",
                             finding="The adjustment column duplicates a treatment, identity or absorbed stratum role.",
                             requirement="A distinct baseline precision covariate for ANCOVA.",
                             explanation="Duplicating this role adds no identifiable precision adjustment."),)
    return issues


def _assignment(config: Specification, facts: Mapping[str, Scalar]) -> tuple[Issue, ...]:
    mechanism = facts.get("assignment_mechanism")
    if mechanism is None or mechanism in ("cluster_randomized", "individual_randomized"):
        return ()
    return (Issue(category="contradictory_configuration", field="facts.assignment_mechanism",
                  finding="The stated assignment is not a supported randomized design.",
                  requirement="Individual randomization or cluster randomization.",
                  explanation="An ITT analysis requires evidence of randomized assignment."),)


def _assignment_unit(config: Specification, facts: Mapping[str, Scalar]) -> tuple[Issue, ...]:
    issues: tuple[Issue, ...] = ()
    mechanism = facts.get("assignment_mechanism")
    if mechanism == "cluster_randomized":
        issues += missing_fields(config, "cluster_column")
    if mechanism == "individual_randomized" and config.cluster_column is not None:
        issues += (Issue(category="contradictory_configuration", field="configuration.cluster_column",
                         finding="An assignment cluster is selected for individual randomization.",
                         requirement="Cluster columns describe the randomized assignment unit.",
                         explanation="General correlated-outcome designs require a separate specification."),)
    return issues


def _precision_timing(config: Specification, facts: Mapping[str, Scalar]) -> tuple[Issue, ...]:
    return tuple(issue.model_copy(update={"field": "facts.precision_covariate_pre_treatment"})
                 for issue in fact_contradiction(facts, "precision_covariate_pre_treatment", True))


REQUIREMENTS = (
    R("contrast", ("configuration.treatment_column", "configuration.unit_column",
                   "configuration.treated_value", "configuration.comparator_value"), "choice",
      "Bind one analysis unit and two distinct assigned treatment states.",
      "The implemented randomized analysis estimates one prespecified two-arm intention-to-treat contrast.",
      lambda c, f: assess_contrast(c)),
    R("estimator", ("configuration.estimator",), "choice",
      "Choose an unadjusted difference in means or ANCOVA.",
      "Precision adjustment is a prespecified scientific choice; it is not selected from fitted results.",
      lambda c, f: missing_fields(c, "estimator")),
    R("assignment_mechanism", ("facts.assignment_mechanism",), "fact",
      "Source evidence establishes individual or cluster randomization.",
      "Observed baseline balance does not establish that treatment was randomized.",
      _assignment, fact_name="assignment_mechanism"),
    R("assignment_unit", ("configuration.cluster_column", "facts.assignment_mechanism"), "consistency",
      "Bind an assignment cluster exactly when the design is cluster randomized.",
      "The cluster identifies the randomized assignment unit; arbitrary outcome correlation needs a different specification.",
      _assignment_unit, dependencies=("assignment_mechanism",)),
    R("precision_covariate", ("configuration.estimator", "configuration.precision_covariate_column"), "role",
      "ANCOVA requires one baseline covariate distinct from treatment, identity and absorbed strata.",
      "Repeating an absorbed role cannot provide an identifiable precision adjustment.",
      _precision),
    R("precision_timing", ("facts.precision_covariate_pre_treatment", "configuration.precision_covariate_column"), "fact",
      "The selected precision covariate was measured before treatment assignment.",
      "A baseline balance check or precision adjustment requires timing evidence for the selected covariate.",
      _precision_timing, fact_name="precision_covariate_pre_treatment",
      active_when=lambda c: c.precision_covariate_column is not None,
      scope_fields=("configuration.precision_covariate_column",)),
)

SCIENTIFIC_REQUIREMENTS = tuple(rule.id for rule in REQUIREMENTS if rule.fact_name is not None)


def check_data(config: Specification, frame: Any) -> tuple[Issue, ...]:
    issues = check_independent_units(config, frame) + check_contrast(config, frame)
    if config.treatment_column is None or config.treatment_column not in frame.columns:
        return issues
    counts = frame.group_by(config.treatment_column).len()["len"]
    if counts.len() and int(counts.min() or 0) < 2:
        issues += (Issue(category="incompatible_data", field=f"dataset.{config.treatment_column}",
                         finding="A treatment arm has fewer than two observed outcomes.",
                         requirement="At least two observed outcomes per treatment arm.",
                         explanation="The contrast needs within-arm information for uncertainty."),)
    covariate = config.precision_covariate_column
    if (config.estimator == "ancova" and covariate is not None and covariate in frame.columns
            and frame[covariate].n_unique() < 2):
        issues += (Issue(category="incompatible_data", field=f"dataset.{covariate}",
                         finding="The selected precision covariate is constant.",
                         requirement="A varying baseline covariate for the requested ANCOVA.",
                         explanation="A constant column cannot provide the selected adjustment."),)
    if config.cluster_column is None or config.cluster_column not in frame.columns:
        return issues
    import polars as pl

    counts = frame.group_by(config.cluster_column).agg(
        pl.col(config.treatment_column).n_unique().alias("__states"))
    if counts.filter(pl.col("__states") > 1).height:
        issues += (Issue(category="incompatible_data", field=f"dataset.{config.cluster_column}",
                         finding="A declared assignment cluster contains different treatments.",
                         requirement="Treatment must be constant within each randomized cluster.",
                         explanation="The data contradict the selected assignment unit."),)
    return issues


DEFINITION = MethodDefinition(
    "randomized", "randomized.v2", "Randomized experiment",
    "One two-arm intention-to-treat contrast for independent units or randomized clusters.",
    Specification, tuple(rule.fact_name for rule in REQUIREMENTS
                         if rule.fact_name is not None and rule.active_when is None),
    (("subgroups", "No implemented subgroup filtering or subgroup estimand."),
     ("attrition_bounds", "Observed extrema do not define justified outcome bounds."),
     ("multiple_covariates", "This adapter supports one precision covariate."),
     ("repeated_measurements", "One row per analysis unit is required."),
     ("cr2", "The legacy CR2 option actually computed CRV1.")),
    DIAGNOSTICS, SENSITIVITIES,
    requirements=REQUIREMENTS,
    fixed_policies=(
        FixedPolicyDefinition('outcome_scale', 'difference',
            'Report the prespecified arm contrast on the outcome difference scale.'),
        FixedPolicyDefinition('stratum_handling', 'fixed_effects',
            'Absorb a declared randomization stratum using fixed effects.'),
        FixedPolicyDefinition('cluster_covariance', 'cluster_robust_cr1',
            'Use the primary CRV1 profile for assignment clusters; independent-unit fits use HC2.'),
        FixedPolicyDefinition('minimum_clusters', 8,
            'Flag primary uncertainty supported by fewer than eight assignment clusters.'),
        FixedPolicyDefinition('mask_rule_id', 'outcome_observed',
            'Estimate the primary contrast among observed outcomes while retaining assigned-population attrition evidence.'),
        FixedPolicyDefinition('multiplicity_policy_id', 'holm_step_down.v1',
            'Use the implemented Holm step-down policy for the prespecified contrast family.'),
    ),
    roles=(
        Role("treatment", "configuration.treatment_column", "categorical", minimum=1,
             description="The two prespecified randomized arms.", dependencies=("contrast",)),
        Role("unit_identifier", "configuration.unit_column", "identifier", minimum=1,
             description="One row per analysis unit, including units with missing outcomes.", dependencies=("contrast",)),
        Role("precision_covariate", "configuration.precision_covariate_column", "any",
             description="At most one source-supported baseline precision covariate.", dependencies=("precision_covariate", "precision_timing")),
        Role("stratum", "configuration.stratum_column", "categorical",
             description="An optional declared randomization stratum."),
        Role("cluster", "configuration.cluster_column", "identifier",
             description="Required for cluster randomization and absent for individual randomization.", dependencies=("assignment_unit",)),
    ),
    option_dependencies=tuple((field, SCIENTIFIC_REQUIREMENTS)
                              for field in Specification.model_fields
                              if field not in ("precision_covariate_column", "cluster_column")) + (
        ("estimator=difference_in_means", SCIENTIFIC_REQUIREMENTS + ("assignment_unit", "contrast")),
        ("estimator=ancova", SCIENTIFIC_REQUIREMENTS + ("assignment_unit", "contrast", "precision_covariate")),
        ("precision_covariate_column", SCIENTIFIC_REQUIREMENTS + ("precision_covariate",)),
        ("cluster_column", SCIENTIFIC_REQUIREMENTS + ("assignment_unit",)),
    ),
    derived_quantities=("estimate", "standard_error", "confidence_interval", "assigned_and_observed_counts"))
