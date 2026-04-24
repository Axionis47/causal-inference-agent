"""Typed treatment operands and contrast structure, without choosing an estimator."""

import hashlib
import json
import math
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from graphlib import TopologicalSorter
from typing import Any

from backend.analysis.contracts import Claim, Treatments, TypedValue


def literal_key(value: TypedValue) -> str:
    """Preserve text exactly; normalize only explicitly numeric representations."""
    if value.kind == "text":
        return value.literal
    if value.kind == "boolean":
        if value.literal not in {"true", "false"}:
            raise ValueError("Boolean values require the exact literal true or false.")
        return value.literal
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?", value.literal):
        raise ValueError("Numeric values require a finite decimal literal, not a label or nonfinite value.")
    try:
        number = Decimal(value.literal)
    except InvalidOperation as error:
        raise ValueError("Numeric value is not a finite decimal literal.") from error
    if not number.is_finite():
        raise ValueError("Numeric values must be finite.")
    if number.is_zero():
        return "0"
    sign, digits_tuple, exponent = number.as_tuple()
    # Decimal.normalize() can round under the ambient precision. Canonicalize
    # the exact tuple instead, without expanding large powers of ten.
    digits = list(digits_tuple)
    assert isinstance(exponent, int)
    while digits[-1] == 0:
        digits.pop()
        exponent += 1
    return f"{sign}:{''.join(str(digit) for digit in digits)}:{exponent}"


def value_id(variable_id: str, value: TypedValue) -> str:
    encoded = json.dumps([variable_id, value.kind, literal_key(value)], separators=(",", ":"), ensure_ascii=False)
    return "value_" + hashlib.sha256(encoded.encode()).hexdigest()[:32]


def canonicalize(value: Treatments, variable_aliases: dict[str, str]) -> Treatments:
    """Resolve declared aliases only. Raw proposals remain separately retained."""
    result = value.model_copy(deep=True)
    aliases: dict[str, str] = {}
    identities: dict[str, tuple[str, str, str]] = {}
    for item in result.values:
        item.variable_id = variable_aliases.get(item.variable_id, item.variable_id)
        canonical = value_id(item.variable_id, item.value)
        key = (item.variable_id, item.value.kind, literal_key(item.value))
        if item.id in aliases:
            raise ValueError("Value proposal identities must be unique.")
        if canonical in identities:
            raise ValueError("Treatment values must be unique; duplicate typed values or an identity collision cannot be merged.")
        identities[canonical] = key
        aliases[item.id] = canonical
        item.id = canonical
    if any(key in identities and key != identity for key, identity in aliases.items()):
        raise ValueError("A proposed value alias collides with another canonical value identity.")

    def walk(obj: Any) -> Any:
        if isinstance(obj, list):
            return [walk(item) for item in obj]
        if isinstance(obj, dict):
            return {
                key: aliases.get(item, item) if key == "cohort_value_id"
                else [aliases.get(identity, identity) for identity in item]
                if key in {"value_ids", "comparison_value_ids", "treatment_value_ids", "assignment_value_ids"}
                else variable_aliases.get(item, item) if key in {"time_variable_id", "running_variable_id"}
                else walk(item)
                for key, item in obj.items()
            }
        return obj

    for effect in result.effects:
        if effect.comparison:
            effect.comparison = type(effect.comparison).model_validate(walk(effect.comparison.model_dump(mode="json")))
    for process in result.assignments:
        process.history_variables = [variable_aliases.get(identity, identity) for identity in process.history_variables]
        if process.diagnostic_window:
            window = process.diagnostic_window
            for name in ("unit_variable_id", "time_variable_id", "adoption_variable_id", "assignment_variable_id"):
                identity = getattr(window, name)
                if identity is not None:
                    setattr(window, name, variable_aliases.get(identity, identity))
    return result


def validate(value: Treatments, offered_columns: set[str], outcome_id: str | None) -> None:
    """Validate exact frozen operands and dependencies, never source entailment."""
    factors = {item.variable.id for item in value.treatments}
    history_variables = {identity for process in value.assignments for identity in process.history_variables}
    registry = {item.id: item for item in value.values}
    effects = {item.id: item for item in value.effects}
    if len(registry) != len(value.values):
        raise ValueError("Treatment value identities must be unique.")
    dependencies: dict[str, set[str]] = {identity: set() for identity in effects}
    adoption_dates: dict[tuple[tuple[str, ...], str, str, str], str] = {}

    def supported(claim: Claim, label: str) -> None:
        if claim.basis == "unknown" or not claim.references:
            raise ValueError(f"{label} requires supplied evidence or an explicit working assumption.")

    for item in value.values:
        if item.variable_id not in factors | offered_columns or item.variable_id == outcome_id:
            raise ValueError("A treatment value must bind an exact offered factor or assignment metadata variable, never the target outcome.")
        if item.variable_id not in factors and item.variable_id not in history_variables:
            raise ValueError("Every nonfactor value variable requires an explicit assignment history_variables obligation, including unused registry levels.")
        if item.id != value_id(item.variable_id, item.value):
            raise ValueError("Value identity does not match its exact variable and typed literal.")
        supported(item.meaning, "Treatment value meaning")

    def operands(identities: list[str]) -> list[str]:
        if not identities or len(identities) != len(set(identities)) or not set(identities) <= registry.keys():
            raise ValueError("Contrast operands require distinct exact declared value identities.")
        return [registry[identity].variable_id for identity in identities]

    def total(weights: list[float], expected: float, *, nonnegative: bool = False) -> None:
        if any(not math.isfinite(weight) or weight == 0 or (nonnegative and weight < 0) for weight in weights):
            raise ValueError("Contrast weights must be finite and nonzero; weighted-average weights must be positive. Omit zero terms.")
        exact = sum((Fraction(str(weight)) for weight in weights), Fraction(0))
        if abs(exact - Fraction(str(expected))) > Fraction("1e-12"):
            raise ValueError(f"Declared contrast weights must sum to {expected:g}; do not infer weights from assignment frequencies.")

    for effect in value.effects:
        binding = effect.comparison
        if binding is None:
            raise ValueError("Every fresh requested effect needs a typed comparison separate from its factor definition.")
        fields = ("static", "group_time", "discontinuity", "aggregate", "unsupported")
        if [name for name in fields if getattr(binding, name) is not None] != [binding.kind]:
            raise ValueError("A contrast must contain exactly the payload named by its kind.")
        supported(binding.evidence, "Contrast binding")
        selected = set(effect.treatment_ids)
        processes = [process for process in value.assignments if selected & set(process.treatment_ids)]
        histories = {identity for process in processes for identity in process.history_variables}

        def metadata(identity: str, history: set[str] = histories) -> None:
            if identity == outcome_id or identity not in offered_columns | factors:
                raise ValueError("Contrast metadata must name an exact offered observed variable, never target-outcome data.")
            if identity not in factors and identity not in history:
                raise ValueError("Every nonfactor contrast variable must be declared in its assignment process history_variables for later role-registry closure.")

        if binding.static:
            states: set[frozenset[str]] = set()
            state_factors: set[str] | None = None
            for term in binding.static.terms:
                variables = operands(term.value_ids)
                if len(set(variables)) != len(variables) or not set(variables) <= factors:
                    raise ValueError("A static joint state requires exactly one value per selected intervention factor, not metadata variables or duplicate levels.")
                if state_factors is None:
                    state_factors = set(variables)
                if set(variables) != state_factors or not selected <= state_factors:
                    raise ValueError("All static terms need the same complete factor state, including every treatment of the requested effect.")
                state = frozenset(term.value_ids)
                if state in states:
                    raise ValueError("Duplicate joint states must not acquire separate conflicting weights.")
                states.add(state)
            for identity in selected:
                levels = {operand for term in binding.static.terms for operand in term.value_ids if registry[operand].variable_id == identity}
                if len(levels) < 2:
                    raise ValueError("Each requested treatment must vary across the declared static states; a fixed conditioning level is not its treatment contrast.")
            total([term.weight for term in binding.static.terms], 0)
            if any(process.kind in {"absorbing_adoption", "switching", "dynamic_regime"} for process in processes):
                raise ValueError("A temporal assignment cannot be silently replaced by a static arm contrast.")
        elif binding.group_time:
            comparison = binding.group_time
            variables = operands([comparison.cohort_value_id, *comparison.comparison_value_ids])
            if len(set(variables)) != 1:
                raise ValueError("Group-time and comparison cohort values must bind the same exact cohort variable.")
            metadata(variables[0])
            metadata(comparison.time_variable_id)
            if variables[0] == comparison.time_variable_id or any(item.kind != "number" for item in (comparison.period, comparison.baseline, comparison.adoption_period)):
                raise ValueError("Implemented group-time contrasts require separate cohort/time variables and explicit numeric period, baseline and adoption selectors.")
            literal_key(comparison.period)
            literal_key(comparison.baseline)
            literal_key(comparison.adoption_period)
            if not Decimal(comparison.baseline.literal) < Decimal(comparison.adoption_period.literal) <= Decimal(comparison.period.literal):
                raise ValueError("A group-time contrast requires a strictly earlier baseline than explicit adoption_period, and adoption_period at or before the target period. Cohort labels never establish adoption time.")
            if len(selected) != 1 or len(processes) != 1 or processes[0].kind != "absorbing_adoption":
                raise ValueError("Group-time selectors require one explicitly linked absorbing-adoption factor; more complex histories remain unsupported.")
            event = (tuple(sorted(processes[0].treatment_ids)), variables[0], comparison.cohort_value_id, comparison.time_variable_id)
            date = literal_key(comparison.adoption_period)
            if event in adoption_dates and adoption_dates[event] != date:
                raise ValueError("The same assignment process, cohort and time variable cannot declare conflicting adoption periods across effects.")
            adoption_dates[event] = date
        elif binding.discontinuity:
            discontinuity = binding.discontinuity
            metadata(discontinuity.running_variable_id)
            variables = operands(discontinuity.treatment_value_ids)
            if len(set(variables)) != 1 or set(variables) != selected or discontinuity.running_variable_id in factors:
                raise ValueError("A discontinuity requires ordered control/treated levels on its one selected factor and a distinct bound running variable.")
            if discontinuity.cutoff.kind != "number":
                raise ValueError("A discontinuity cutoff requires an explicit finite numeric representation.")
            literal_key(discontinuity.cutoff)
            if discontinuity.assignment_value_ids:
                assignment = operands(discontinuity.assignment_value_ids)
                if len(assignment) != 2 or len(set(assignment)) != 1 or assignment[0] in {variables[0], discontinuity.running_variable_id}:
                    raise ValueError("Observed eligibility levels require a distinct single assignment variable, separate from receipt and running score.")
                metadata(assignment[0])
            if any(process.kind != "point" for process in processes):
                raise ValueError("This discontinuity binding does not implement a temporal or multi-factor threshold regime.")
        elif binding.aggregate:
            identities = [term.effect_id for term in binding.aggregate.terms]
            if len(set(identities)) != len(identities) or not set(identities) <= effects.keys() or effect.id in identities:
                raise ValueError("An aggregate requires distinct declared component effects and cannot include itself.")
            if {identity for component in identities for identity in effects[component].treatment_ids} != selected:
                raise ValueError("An aggregate must preserve exactly the treatment scope of its named component effects.")
            total([term.weight for term in binding.aggregate.terms], 1, nonnegative=True)
            dependencies[effect.id] = set(identities)
        else:
            raise ValueError("The declared nonlinear or history contrast is unsupported. Preserve its description and return unsupported instead of changing the requested effect.")
    tuple(TopologicalSorter(dependencies).static_order())
    for process in value.assignments:
        window = process.diagnostic_window
        if window is None:
            continue
        assignment_id = window.assignment_variable_id
        if assignment_id is None:
            raise ValueError("Fresh diagnostic windows require an explicit assignment_variable_id; the selected factor may instead be a cohort regime.")
        if window.assignment_encoding == "group" and window.adoption_variable_id is None:
            raise ValueError("A group-coded diagnostic window requires an explicit observed adoption_variable_id; cohort labels cannot establish safe pre-adoption outcome access.")
        if assignment_id is not None and (assignment_id not in offered_columns or assignment_id == outcome_id or (assignment_id not in process.treatment_ids and assignment_id not in process.history_variables)):
            raise ValueError("The diagnostic assignment variable needs an exact observed binding and an explicit assignment-history obligation.")
