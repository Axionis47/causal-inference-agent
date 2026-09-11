"""ATE or ATT for independent units with a declared baseline adjustment set."""
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
from causal.analysis.methods.aipw.diagnostic_catalog import DIAGNOSTICS, SENSITIVITIES

Column = Annotated[str, Field(min_length=1)]


class Specification(Model):
    method: Literal["aipw"] = "aipw"
    estimand: Literal["ate", "att"] | None = None
    treatment_column: Column | None = None
    unit_column: Column | None = None
    treated_value: Column | None = None
    comparator_value: Column | None = None
    covariate_columns: tuple[Column, ...] = ()
    nuisance_profile: Literal["regularized_glm"] = "regularized_glm"
    fold_count: Literal[5] = 5
    propensity_bound: Annotated[float, Field(ge=0.01, le=0.01)] = 0.01
    confidence_level: Annotated[float, Field(ge=0.95, le=0.95)] = 0.95


def columns(config: Specification) -> tuple[ColumnRequirement, ...]:
    core = tuple(ColumnRequirement(name, role, cast(Any, kind)) for name, role, kind in (
        (config.treatment_column, "treatment", "categorical"),
        (config.unit_column, "unit_identifier", "identifier")) if name is not None)
    return core + tuple(ColumnRequirement(name, f"adjustment_covariate__{index}", "numeric")
                        for index, name in enumerate(config.covariate_columns))


def _adjustment(config: Specification, facts: Mapping[str, Scalar]) -> tuple[Issue, ...]:
    issues = missing_fields(config, "covariate_columns")
    if (len(set(config.covariate_columns)) != len(config.covariate_columns)
            or set(config.covariate_columns) & {config.treatment_column, config.unit_column}):
        issues += (Issue(category="contradictory_configuration", field="configuration.covariate_columns",
                         finding="Adjustment columns repeat or include treatment or unit identifiers.",
                         requirement="Distinct baseline covariates.",
                         explanation="Treatment and unit identity cannot be adjustment covariates."),)
    return issues


def _adjustment_timing(config: Specification, facts: Mapping[str, Scalar]) -> tuple[Issue, ...]:
    return tuple(issue.model_copy(update={"field": "facts.adjustment_set_pre_treatment"})
                 for issue in fact_contradiction(facts, "adjustment_set_pre_treatment", True))


REQUIREMENTS = (
    R("contrast", ("configuration.treatment_column", "configuration.unit_column",
                   "configuration.treated_value", "configuration.comparator_value"), "choice",
      "Bind independent analysis units and two distinct treatment states.",
      "The supported cross-fitting and uncertainty calculation require independent units and a two-arm contrast.",
      lambda c, f: assess_contrast(c)),
    R("estimand", ("configuration.estimand",), "choice",
      "Select either the ATE or ATT target population.",
      "ATE and ATT are separate scientific targets and require separately labelled specifications.",
      lambda c, f: missing_fields(c, "estimand")),
    R("adjustment_set", ("configuration.covariate_columns",), "role",
      "Bind at least one distinct numeric baseline covariate, excluding treatment and unit identity.",
      "The declared adjustment set fixes the measured baseline information used by both nuisance models.",
      _adjustment),
    R("adjustment_timing", ("facts.adjustment_set_pre_treatment", "configuration.covariate_columns"), "fact",
      "Source evidence establishes pre-treatment timing for every selected adjustment covariate.",
      "A declaration about a previous adjustment set cannot establish the timing of newly selected columns.",
      _adjustment_timing, fact_name="adjustment_set_pre_treatment",
      scope_fields=("configuration.covariate_columns",)),
)

SCIENTIFIC_REQUIREMENTS = tuple(rule.id for rule in REQUIREMENTS if rule.fact_name is not None)


def check_data(config: Specification, frame: Any) -> tuple[Issue, ...]:
    issues = check_independent_units(config, frame) + check_contrast(config, frame)
    if config.treatment_column is None or config.treatment_column not in frame.columns:
        return issues
    counts = frame.group_by(config.treatment_column).len()["len"]
    if counts.len() and int(counts.min() or 0) < config.fold_count:
        issues += (Issue(category="incompatible_data", field=f"dataset.{config.treatment_column}",
                         finding="A treatment arm has fewer units than the fixed fold count.",
                         requirement="At least five independent units in each treatment arm.",
                         explanation="Each cross-fitting fold needs both treatment states."),)
    return issues


def check_sensitivity_data(config: Specification, frame: Any,
                           selected_ids: tuple[str, ...]) -> tuple[Issue, ...]:
    if ("alternative_cross_fit_seed" not in selected_ids or config.treatment_column is None
            or config.treatment_column not in frame.columns):
        return ()
    counts = frame.group_by(config.treatment_column).len()["len"]
    if counts.len() and int(counts.min() or 0) >= 10:
        return ()
    return (Issue(category="incompatible_data", field="sensitivities.alternative_cross_fit_seed",
                  finding="The selected ten-fold sensitivity lacks ten units in each treatment arm.",
                  requirement="At least ten independent units in each arm for this selected sensitivity.",
                  explanation="A five-fold primary may be supported while the prespecified ten-fold comparison is not.",
                  resolutions=("Supply sufficient data or omit this sensitivity before approval.",)),)


DEFINITION = MethodDefinition(
    "aipw", "aipw.v2", "Augmented inverse-probability weighting",
    "An ATE or ATT using five-fold cross-fitting and regularized generalized linear nuisance models.",
    Specification, tuple(rule.fact_name for rule in REQUIREMENTS
                         if rule.fact_name is not None and rule.active_when is None),
    (("clustered_data", "Fold assignment and uncertainty require independent units."),
     ("categorical_covariates", "The old encoder assigned arbitrary ordinal codes; numeric inputs only."),
     ("missing_values", "Automatic zero filling is not an approved missing-data strategy."),
     ("e_value", "No implemented E-value calculation."),
     ("combined_ate_att", "ATE and ATT require separate clearly labelled specifications."),
     ("alternative_learners", "This version exposes the verified regularized GLM profile only.")),
    DIAGNOSTICS, SENSITIVITIES,
    requirements=REQUIREMENTS,
    fixed_policies=(
        FixedPolicyDefinition('aggregation', 'cross_fit_influence_mean',
            'Aggregate cross-fitted influence contributions for the selected estimand.'),
        FixedPolicyDefinition('seed_offset', 0,
            'Use the accepted primary seed without an additional offset.'),
        FixedPolicyDefinition('mask_rule_id', 'cross_fit_predicted',
            'Require cross-fitted predictions for the contributing primary observations.'),
        FixedPolicyDefinition('fold_assignment_rule_id', 'stratified_by_treatment_and_unit',
            'Assign cross-fitting folds using treatment strata and independent unit identities.'),
        FixedPolicyDefinition('nuisance.propensity_learner', 'sklearn_logistic_regression_l2',
            'Fit the propensity nuisance with the implemented L2 logistic regression learner.'),
        FixedPolicyDefinition('nuisance.outcome_learner', 'sklearn_ridge_regression_l2',
            'Fit the outcome nuisance with the implemented ridge regression learner.'),
        FixedPolicyDefinition('nuisance.regularization_c', 1.0,
            'Fix nuisance regularization strength at 1.0.'),
        FixedPolicyDefinition('nuisance.max_iter', 1000,
            'Allow at most 1,000 optimization iterations for the nuisance learner.'),
        FixedPolicyDefinition('nuisance.standardize', True,
            'Apply the implemented numeric standardization recipe to nuisance inputs.'),
        FixedPolicyDefinition('nuisance.random_state', 0,
            'Fix the learner-level random state at zero; fold randomness still uses the accepted analysis seed.'),
    ),
    roles=(
        Role("treatment", "configuration.treatment_column", "categorical", minimum=1,
             description="The declared treatment and comparator states.", dependencies=("contrast",)),
        Role("unit_identifier", "configuration.unit_column", "identifier", minimum=1,
             description="One row per independent analysis unit.", dependencies=("contrast",)),
        Role("adjustment_covariates", "configuration.covariate_columns", "numeric", minimum=1, maximum=None,
             description="One or more distinct numeric covariates with pre-treatment measurement evidence.",
             dependencies=("adjustment_set", "adjustment_timing")),
    ),
    option_dependencies=tuple((field, SCIENTIFIC_REQUIREMENTS)
                              for field in Specification.model_fields if field != "covariate_columns") + (
        ("estimand=ate", ("contrast", "adjustment_set", "adjustment_timing")),
        ("estimand=att", ("contrast", "adjustment_set", "adjustment_timing")),
        ("covariate_columns", ("adjustment_set", "adjustment_timing")),
    ),
    derived_quantities=("estimate", "standard_error", "confidence_interval", "learned_propensities",
                        "cross_fitted_outcome_predictions", "effective_sample_size"))
