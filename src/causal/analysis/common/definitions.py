"""Static method descriptions and small shared checks; no estimator imports."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

from causal.analysis.common.models import Category, Issue, Model, Scalar


@dataclass(frozen=True)
class ColumnRequirement:
    name: str
    role: str
    kind: Literal["numeric", "categorical", "identifier", "binary", "any"]


@dataclass(frozen=True)
class RequirementDefinition:
    id: str
    fields: tuple[str, ...]
    kind: Literal["fact", "assumption", "choice", "role", "policy", "consistency"]
    expected: str
    explanation: str
    predicate: Callable[[Any, Mapping[str, Scalar]], tuple[Issue, ...]]
    dependencies: tuple[str, ...] = ()
    fact_name: str | None = None
    permits_assumption: bool = False
    active_when: Callable[[Any], bool] | None = None
    scope_fields: tuple[str, ...] = ()
    evaluation_boundary: Literal["design"] = "design"
    failure_category: Category = "contradictory_configuration"


@dataclass(frozen=True)
class RoleDefinition:
    id: str
    field: str
    kind: Literal["numeric", "categorical", "identifier", "binary", "any"]
    minimum: int = 0
    maximum: int | None = 1
    description: str = ""
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class DiagnosticDefinition:
    id: str
    title: str
    obligation: Literal["required", "optional"]
    purpose: str
    severity: str
    thresholds: tuple[tuple[str, Scalar], ...] = ()
    trigger: str = "always"
    evaluation_stage: Literal["specification", "preflight", "execution"] = "execution"
    limitation_category: Literal["population", "causal_interpretation", "measurement", "confidence"] = "confidence"
    dependencies: tuple[str, ...] = ()
    applicability_stage: Literal["design"] = "design"
    measurement_stage: Literal["execution"] = "execution"


@dataclass(frozen=True)
class SensitivityDefinition:
    id: str
    title: str
    purpose: str
    delta: tuple[tuple[str, Scalar], ...]
    required_roles: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class FixedPolicyDefinition:
    """An execution setting owned by the method, visible but never caller-selectable."""

    id: str
    value: Scalar
    description: str


@dataclass(frozen=True)
class MethodDefinition:
    method: str
    version: str
    title: str
    summary: str
    specification: type[Model]
    required_facts: tuple[str, ...]
    unsupported: tuple[tuple[str, str], ...]
    diagnostics: tuple[DiagnosticDefinition, ...]
    sensitivities: tuple[SensitivityDefinition, ...]
    derived_quantities: tuple[str, ...] = ()
    requirements: tuple[RequirementDefinition, ...] = ()
    roles: tuple[RoleDefinition, ...] = ()
    option_dependencies: tuple[tuple[str, tuple[str, ...]], ...] = ()
    fixed_policies: tuple[FixedPolicyDefinition, ...] = ()


def missing_fields(config: Model, *names: str) -> tuple[Issue, ...]:
    return tuple(Issue(category="missing_context", field=f"configuration.{name}",
                       finding=f"{name.replace('_', ' ')} is not specified.",
                       requirement="An explicit scientific selection is required.",
                       explanation="Analysis cannot choose this from the estimated result.")
                 for name in names if getattr(config, name) in (None, (), ""))


def assess_contrast(config: Any) -> tuple[Issue, ...]:
    issues = missing_fields(config, "treatment_column", "unit_column",
                            "treated_value", "comparator_value")
    if config.treated_value is not None and config.treated_value == config.comparator_value:
        issues += (Issue(category="contradictory_configuration", field="configuration.comparator_value",
                         finding="Treatment and comparator name the same state.",
                         requirement="Two distinct treatment states.",
                         explanation="A treatment contrast requires two different groups."),)
    return issues


def fact_contradiction(facts: Mapping[str, Scalar], name: str, expected: Scalar) -> tuple[Issue, ...]:
    value = facts.get(name)
    same = value == expected and (isinstance(value, bool) == isinstance(expected, bool))
    if value is None or same:
        return ()
    return (Issue(category="contradictory_configuration", field=f"design.facts.{name}",
                  finding=f"The evidenced {name.replace('_', ' ')} conflicts with this method.",
                  requirement=f"{name} must be {expected!r}.",
                  explanation="An analysis option cannot override the fixed study design."),)


def check_independent_units(config: Any, frame: Any) -> tuple[Issue, ...]:
    column = config.unit_column
    if column is None or column not in frame.columns or frame[column].n_unique() == frame.height:
        return ()
    return (Issue(category="incompatible_data", field=f"dataset.{column}",
                  finding="The unit identifier repeats across rows.",
                  requirement="One row per analysis unit.",
                  explanation="This specification does not model repeated measurements."),)


def check_contrast(config: Any, frame: Any) -> tuple[Issue, ...]:
    if (not config.treatment_column or config.treatment_column not in frame.columns
            or config.treated_value is None or config.comparator_value is None):
        return ()
    import polars as pl

    observed = set(frame[config.treatment_column].drop_nulls().cast(pl.String).to_list())
    if observed == {config.treated_value, config.comparator_value}:
        return ()
    return (Issue(category="incompatible_data", field=f"dataset.{config.treatment_column}",
                  finding="Observed treatment states do not match the two declared states.",
                  requirement="Both declared states, with no additional treatment arms.",
                  explanation="The supported specification is one two-arm contrast."),)
