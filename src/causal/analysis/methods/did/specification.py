"""Panel ATT with an explicit adoption schedule and never-treated comparison."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, cast

from pydantic import Field

from causal.analysis.common.definitions import (
    ColumnRequirement,
    FixedPolicyDefinition,
    MethodDefinition,
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
from causal.analysis.methods.did.diagnostic_catalog import DIAGNOSTICS, SENSITIVITIES

Column = Annotated[str, Field(min_length=1)]


class Specification(Model):
    method: Literal["did"] = "did"
    estimand: Literal["att_group_time_aggregate"] = "att_group_time_aggregate"
    profile: Literal["simultaneous", "staggered"] | None = None
    treatment_column: Column | None = None
    unit_column: Column | None = None
    time_column: Column | None = None
    adoption_column: Column | None = None
    adoption_time: Annotated[float, Field(gt=0, strict=True)] | None = None
    group_column: Column | None = None
    cluster_column: Column | None = None
    comparison_cohort: Literal["never_treated"] = "never_treated"
    anticipation_periods: Literal[0] = 0
    reference_period: Literal[-1] = -1
    event_window_lags: Literal[3] = 3
    confidence_level: Annotated[float, Field(ge=0.95, le=0.95)] = 0.95


def columns(config: Specification) -> tuple[ColumnRequirement, ...]:
    return tuple(ColumnRequirement(name, role, cast(Any, kind)) for name, role, kind in (
        (config.treatment_column, "treatment", "binary"),
        (config.unit_column, "unit_identifier", "identifier"),
        (config.time_column, "time", "numeric"),
        (config.adoption_column, "adoption_time", "numeric"),
        (config.group_column, "group", "categorical"),
        (config.cluster_column, "cluster", "identifier")) if name is not None)


def _schedule(config: Specification, facts: Mapping[str, Scalar]) -> tuple[Issue, ...]:
    issues: tuple[Issue, ...] = ()
    if config.adoption_column is None and config.adoption_time is None:
        issues += (Issue(category="missing_context", field="configuration.adoption_time",
                         finding="The adoption schedule is unspecified.",
                         requirement="An adoption column or a common positive adoption time.",
                         explanation="The analysis cannot infer a scientific adoption rule silently."),)
    elif config.adoption_column is not None and config.adoption_time is not None:
        issues += (Issue(category="contradictory_configuration", field="configuration.adoption_time",
                         finding="Both a common adoption time and an adoption column were supplied.",
                         requirement="One authoritative adoption schedule.",
                         explanation="Two schedules can imply different event times."),)
    if config.profile == "staggered":
        issues += missing_fields(config, "adoption_column")
    return issues


def _assumption(facts: Mapping[str, Scalar], name: str) -> tuple[Issue, ...]:
    return tuple(issue.model_copy(update={"field": f"facts.{name}"})
                 for issue in fact_contradiction(facts, name, True))


REQUIREMENTS = (
    R("panel", ("configuration.treatment_column", "configuration.unit_column", "configuration.time_column"), "role",
      "Bind a binary exposure history, persistent unit identifier and numeric observation period.",
      "This method requires repeated observations of units; preflight verifies the actual panel and irreversible treatment history.",
      lambda c, f: missing_fields(c, "treatment_column", "unit_column", "time_column")),
    R("profile", ("configuration.profile",), "choice",
      "Choose simultaneous or staggered adoption.",
      "The adoption profile determines which event-time comparisons and sensitivity aggregations are implemented.",
      lambda c, f: missing_fields(c, "profile")),
    R("adoption_schedule", ("configuration.adoption_column", "configuration.adoption_time", "configuration.profile"), "consistency",
      "Declare exactly one schedule: an adoption column or common positive time; staggered adoption requires the column.",
      "The analysis cannot infer adoption silently or reconcile competing schedules by choosing one.",
      _schedule),
    R("parallel_trends", ("facts.parallel_trends_assumption",), "assumption",
      "Explicitly affirm parallel untreated outcome trends for the proposed comparison.",
      "Pre-treatment diagnostics can qualify this assumption; a nonsignificant test cannot establish it.",
      lambda c, f: _assumption(f, "parallel_trends_assumption"),
      fact_name="parallel_trends_assumption", permits_assumption=True),
    R("no_anticipation", ("facts.no_anticipation_assumption",), "assumption",
      "Explicitly affirm no treatment anticipation for the primary analysis.",
      "The primary zero-anticipation convention is fixed; the one-period sensitivity does not change the accepted scientific schedule.",
      lambda c, f: _assumption(f, "no_anticipation_assumption"),
      fact_name="no_anticipation_assumption", permits_assumption=True),
)

SCIENTIFIC_REQUIREMENTS = tuple(rule.id for rule in REQUIREMENTS if rule.fact_name is not None)


def check_data(config: Specification, frame: Any) -> tuple[Issue, ...]:
    unit, time, treatment = config.unit_column, config.time_column, config.treatment_column
    if unit is None or time is None or treatment is None:
        return ()
    if any(name not in frame.columns for name in (unit, time, treatment)):
        return ()
    import polars as pl

    issues: list[Issue] = []

    def issue(field: str, finding: str, requirement: str) -> None:
        issues.append(Issue(category="incompatible_data", field=f"dataset.{field}",
                            finding=finding, requirement=requirement,
                            explanation="The declared panel specification requires this structure."))

    if frame.select(unit, time).unique().height != frame.height:
        issue(str(unit), "Unit-period pairs repeat.", "One row per unit and period.")
    if int(frame.group_by(unit).len()["len"].min() or 0) < 2:
        issue(str(unit), "A unit is observed only once.", "Repeated observations of each unit.")
    states = set(frame[treatment].drop_nulls().to_list())
    if states != {0, 1}:
        issue(str(treatment), "Treatment is not a supported binary exposure history.", "Numeric 0 and 1.")
        return tuple(issues)
    by_unit = frame.sort(time).group_by(unit).agg(
        pl.col(treatment).max().alias("__ever"), pl.col(treatment).diff().min().alias("__change"))
    if by_unit.filter(pl.col("__ever") == 0).height == 0:
        issue(str(treatment), "There is no never-treated comparison unit.", "A never-treated comparison group.")
    if by_unit.filter(pl.col("__change") < 0).height:
        issue(str(treatment), "Treatment switches off after adoption.", "Irreversible treatment adoption.")
    return tuple(issues) + _check_schedule(config, frame, unit, time, treatment)


def _check_schedule(config: Specification, frame: Any, unit: str, time: str,
                    treatment: str) -> tuple[Issue, ...]:
    import polars as pl

    if config.adoption_column is not None and config.adoption_column in frame.columns:
        schedule = pl.col(config.adoption_column)
    elif config.adoption_time is not None:
        schedule = pl.when(pl.col(treatment).max().over(unit) > 0).then(
            pl.lit(config.adoption_time)).otherwise(pl.lit(0.0))
    else:
        return ()
    timed = frame.with_columns(schedule.alias("__adoption"))
    issues: list[Issue] = []

    def issue(finding: str, requirement: str) -> None:
        issues.append(Issue(category="incompatible_data", field="configuration.adoption_time",
                            finding=finding, requirement=requirement,
                            explanation="The declared schedule defines the fixed event-time comparisons."))

    if timed.group_by(unit).agg(pl.col("__adoption").n_unique())["__adoption"].max() != 1:
        issue("The adoption time changes within a unit.", "One adoption time per unit.")
    if timed.filter(pl.col("__adoption") < 0).height:
        issue("The schedule contains negative adoption codes.", "Positive adoption periods or zero for never-treated units.")
    expected = ((pl.col("__adoption") > 0) & (pl.col(time) >= pl.col("__adoption"))).cast(pl.Int8)
    if timed.filter(pl.col(treatment) != expected).height:
        issue("The adoption schedule contradicts the observed treatment history.", "Treatment follows the declared schedule in every observed period.")
    treated = timed.filter(pl.col("__adoption") > 0).with_columns(
        (pl.col(time) - pl.col("__adoption")).alias("__relative"))
    cohorts = treated["__adoption"].n_unique()
    if config.profile == "simultaneous" and cohorts != 1:
        issue("Adoption is not simultaneous.", "One positive adoption time.")
    if config.profile == "staggered" and cohorts < 2:
        issue("There are fewer than two adoption cohorts.", "At least two positive adoption times.")
    support = treated.group_by("__adoption").agg(
        pl.col("__relative").filter(pl.col("__relative") < 0).n_unique().alias("__pre"),
        pl.col("__relative").filter(pl.col("__relative") >= 0).n_unique().alias("__post"),
        (pl.col("__relative") == -1).any().alias("__reference"))
    if not support.height or support.filter((pl.col("__pre") < 2) | (pl.col("__post") < 1)
                                            | ~pl.col("__reference")).height:
        issue("A treated cohort lacks the fixed pre/post diagnostic support.", "Each cohort needs event time -1, at least two pre-treatment periods and one post-treatment period.")
    return tuple(issues)


def check_sensitivity_data(config: Specification, frame: Any,
                           selected_ids: tuple[str, ...]) -> tuple[Issue, ...]:
    unit, time, treatment = config.unit_column, config.time_column, config.treatment_column
    if (unit is None or time is None or treatment is None
            or any(name not in frame.columns for name in (unit, time, treatment))):
        return ()
    import polars as pl

    issues: list[Issue] = []
    if "balanced_panel_contribution" in selected_ids:
        balanced = frame.filter(pl.col(time).n_unique().over(unit) == frame[time].n_unique())
        for found in check_data(config, balanced):
            issues.append(found.model_copy(update={
                "field": "sensitivities.balanced_panel_contribution",
                "explanation": "The selected balanced-panel contribution does not retain the required comparison structure. " + found.explanation}))
    if "alternative_anticipation_window" not in selected_ids:
        return tuple(issues)
    if config.adoption_column is not None and config.adoption_column in frame.columns:
        schedule = pl.col(config.adoption_column)
    elif config.adoption_time is not None:
        schedule = pl.when(pl.col(treatment).max().over(unit) > 0).then(
            pl.lit(config.adoption_time)).otherwise(pl.lit(0.0))
    else:
        return tuple(issues)
    timed = frame.with_columns(schedule.alias("__adoption")).filter(pl.col("__adoption") > 0)
    # The fixed anticipation delta is one period, so the new -1 reference is
    # event time -2 under the original scientific adoption schedule.
    support = timed.group_by("__adoption").agg(
        (pl.col(time) - pl.col("__adoption") == -2).any().alias("__reference"))
    if not support.height or support.filter(~pl.col("__reference")).height:
        issues.append(Issue(category="incompatible_data", field="sensitivities.alternative_anticipation_window",
                            finding="A cohort lacks the reference period for the one-period anticipation sensitivity.",
                            requirement="Each treated cohort needs an observation two periods before actual adoption.",
                            explanation="The shifted comparison must retain its own fixed pre-treatment reference.",
                            resolutions=("Supply the required period or omit this sensitivity before approval.",)))
    return tuple(issues)


DEFINITION = MethodDefinition(
    "did", "did.v2", "Difference in differences",
    "A panel ATT with simultaneous or staggered adoption and a never-treated comparison group.",
    Specification, tuple(rule.fact_name for rule in REQUIREMENTS
                         if rule.fact_name is not None and rule.active_when is None),
    (("repeated_cross_section", "The supported estimators absorb persistent unit fixed effects."),
     ("treatment_reversal", "The event-time construction assumes irreversible adoption."),
     ("no_never_treated_group", "This version fixes the primary comparison to never-treated units."),
     ("arbitrary_placebo", "A placebo schedule requires a separately justified scientific plan.")),
    DIAGNOSTICS, SENSITIVITIES,
    requirements=REQUIREMENTS,
    fixed_policies=(
        FixedPolicyDefinition('event_window_leads', 3,
            'Inspect the fixed three-period pre-adoption event window.'),
        FixedPolicyDefinition('fixed_effects', 'unit_and_time',
            'Absorb persistent unit and common period effects.'),
        FixedPolicyDefinition('weighting', 'cell_size',
            'Weight supported group-time cells using their contributing sample sizes.'),
        FixedPolicyDefinition('aggregation', 'cohort_weighted_overall',
            'Aggregate group-time effects using the implemented cohort weights.'),
        FixedPolicyDefinition('mask_rule_id', 'group_time_cell',
            'Use supported group-time cells under the fixed adoption schedule.'),
        FixedPolicyDefinition('cluster_level', 'unit',
            'Use the primary unit-level clustering profile.'),
        FixedPolicyDefinition('placebo_shift_periods', 0,
            'Keep the actual scientific adoption schedule unchanged for the primary analysis.'),
        FixedPolicyDefinition('influence_summary', 'none',
            'Do not add an unimplemented influence-summary computation to the primary fit.'),
    ),
    roles=(
        Role("treatment", "configuration.treatment_column", "binary", minimum=1,
             description="A numeric 0/1 history with irreversible adoption.", dependencies=("panel",)),
        Role("unit_identifier", "configuration.unit_column", "identifier", minimum=1,
             description="The persistent unit observed over multiple periods.", dependencies=("panel",)),
        Role("time", "configuration.time_column", "numeric", minimum=1,
             description="The observation period in the adoption schedule's units.", dependencies=("panel",)),
        Role("adoption_time", "configuration.adoption_column", "numeric",
             description="One positive adoption time per treated unit; zero identifies never-treated units.", dependencies=("adoption_schedule",)),
        Role("group", "configuration.group_column", "categorical",
             description="An optional declared grouping variable."),
        Role("cluster", "configuration.cluster_column", "identifier",
             description="An optional declared uncertainty cluster; unit clustering is otherwise mechanical."),
    ),
    option_dependencies=tuple((field, SCIENTIFIC_REQUIREMENTS)
                              for field in Specification.model_fields
                              if field not in ("adoption_column", "adoption_time")) + (
        ("profile=simultaneous", ("panel", "adoption_schedule", "parallel_trends", "no_anticipation")),
        ("profile=staggered", ("panel", "adoption_schedule", "parallel_trends", "no_anticipation")),
        ("adoption_column", SCIENTIFIC_REQUIREMENTS + ("profile", "adoption_schedule")),
        ("adoption_time", SCIENTIFIC_REQUIREMENTS + ("profile", "adoption_schedule")),
    ),
    derived_quantities=("estimate", "standard_error", "confidence_interval", "event_time_effects",
                        "cohort_weights", "contributing_units"))
