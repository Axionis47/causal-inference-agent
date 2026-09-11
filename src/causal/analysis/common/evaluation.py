"""One deterministic evaluator for partial design, navigation, and final acceptance."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, cast

from pydantic import ValidationError

from causal.analysis.common.candidate import (
    CandidateDraft,
    CandidateEvaluation,
    CheckObligation,
    Eligibility,
    OptionState,
    RequirementResult,
    ResolutionKind,
    ResolutionTarget,
    RoleSlot,
    Selection,
)
from causal.analysis.common.catalog import METHODS, diagnostics_module, method_module
from causal.analysis.common.definitions import MethodDefinition, RequirementDefinition
from causal.analysis.common.models import Issue, Model, Scalar


def digest(value: Any) -> str:
    raw = value.model_dump(warnings=False) if isinstance(value, Model) else value

    def json_input(item: Any) -> bool:
        if item is None or type(item) in (bool, int, str):
            return True
        if type(item) is float:
            return math.isfinite(item)
        if isinstance(item, Mapping):
            return all(isinstance(key, str) and json_input(child) for key, child in item.items())
        if isinstance(item, (tuple, list)):
            return all(json_input(child) for child in item)
        return False

    def invalid_input(item: Any) -> Any:
        # Tag the entire invalid tree, rather than replacing NaN with null or a
        # user-reproducible marker inside an otherwise ordinary candidate.
        if isinstance(item, Mapping):
            entries = [(invalid_input(key), invalid_input(child)) for key, child in item.items()]
            return ["mapping", sorted(entries, key=lambda pair: json.dumps(pair[0], sort_keys=True))]
        if isinstance(item, (tuple, list)):
            return ["sequence", [invalid_input(child) for child in item]]
        if isinstance(item, float) and not math.isfinite(item):
            return ["nonfinite", "nan" if math.isnan(item) else "+infinity" if item > 0 else "-infinity"]
        if item is None or type(item) in (bool, int, float, str):
            return [type(item).__name__, item]
        if isinstance(item, bytes):
            return ["bytes", item.hex()]
        if isinstance(item, (set, frozenset)):
            return ["set", sorted((invalid_input(child) for child in item), key=lambda child: json.dumps(child, sort_keys=True))]
        return ["non_json_type", type(item).__module__ + "." + type(item).__qualname__]

    if not json_input(raw):
        encoded = json.dumps(invalid_input(raw), sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False)
        return hashlib.sha256(("analysis-invalid-input.v1:" + encoded).encode()).hexdigest()
    if isinstance(value, Model):
        value = value.model_dump(mode="json")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def coerce_candidate(draft: CandidateDraft | Mapping[str, Any] | Model | None) -> CandidateDraft:
    """Compatibility projection preserves unknowns; never makes a fixed snapshot."""
    if isinstance(draft, CandidateDraft):
        # Revalidate even model_copy input and detach nested caller-owned dictionaries.
        return CandidateDraft.model_validate(draft.model_dump(warnings=False))
    raw = draft.model_dump(warnings=False) if isinstance(draft, Model) else dict(draft or {})
    design = raw.get("design")
    if isinstance(design, Model):
        design = design.model_dump()
    if isinstance(design, Mapping):
        extras = {key: value for key, value in raw.items()
                  if key not in CandidateDraft.model_fields and key not in ("design", "dataset")}
        if "candidate" in design:
            candidate = dict(design["candidate"])
            candidate.update({key: value for key, value in raw.items()
                              if key in CandidateDraft.model_fields and key != "schema_version"})
            candidate.update(extras)
            return CandidateDraft.model_validate(candidate)
        candidate = {key: value for key, value in design.items()
                     if key in CandidateDraft.model_fields}
        candidate.update({key: value for key, value in raw.items()
                          if key in CandidateDraft.model_fields and key != "schema_version"})
        candidate.update(extras)
        return CandidateDraft.model_validate(candidate)
    if "dataset" in raw:
        raw.pop("dataset")
        raw.pop("schema_version", None)
    if not set(raw) & (set(CandidateDraft.model_fields) - {"method"}):
        return CandidateDraft(method=raw.get("method"), configuration=raw)
    return CandidateDraft.model_validate(raw)


def issue(field: str, finding: str, expected: str, *, missing: bool = False,
          unsupported: bool = False) -> Issue:
    return Issue(category="missing_context" if missing else "unsupported_capability" if unsupported
                 else "contradictory_configuration", field=field, finding=finding,
                 requirement=expected, explanation=expected)


def result(identifier: str, fields: tuple[str, ...], explanation: str,
           issues: tuple[Issue, ...] = (), *, kind: ResolutionKind = "configuration",
           dependencies: tuple[str, ...] = (), active: bool = True) -> RequirementResult:
    missing = tuple(dict.fromkeys(i.field for i in issues if i.category == "missing_context"))
    return RequirementResult(
        requirement_id=identifier, fields=fields, explanation=explanation,
        status="inapplicable" if not active else "violated" if any(
            i.category != "missing_context" for i in issues) else "unresolved" if issues else "satisfied",
        dependencies=dependencies, missing_inputs=missing, issues=issues,
        resolution_targets=tuple(ResolutionTarget(kind=kind, field=i.field,
                                                  requirement_id=identifier) for i in issues))


def schema_values(schema: dict[str, Any]) -> tuple[Any, ...]:
    variants = schema.get("anyOf", (schema,))
    values = [value for variant in variants for value in variant.get("enum", ())]
    values += [variant["const"] for variant in variants if "const" in variant]
    return tuple(value for value in values if value is not None)


def fixed_value(schema: dict[str, Any]) -> bool:
    values = schema_values(schema)
    return len(values) == 1 or (schema.get("minimum") is not None
                               and schema.get("minimum") == schema.get("maximum"))


def _frame_requirements(draft: CandidateDraft) -> list[RequirementResult]:
    rows = []
    for field in ("population", "unit_grain", "outcome.column", "outcome.kind", "outcome.units"):
        value = (getattr(draft.outcome, field.split(".")[1]) if draft.outcome is not None else None
                 ) if field.startswith("outcome.") else getattr(draft, field)
        errors: tuple[Issue, ...] = ()
        if value is None or not str(value).strip():
            errors = (issue(field, "The scientific frame is incomplete.",
                            "Supply the declared study population, unit/grain and outcome meaning.", missing=True),)
        elif field == "outcome.kind" and value not in ("continuous", "binary"):
            errors = (issue(field, "Unsupported outcome kind.", "Use continuous or binary.", unsupported=True),)
        rows.append(result(f"analysis:requirement:{field}", (field,),
                           "Explicit scientific frame; no scientific defaults.", errors, kind="scientific_frame"))
    errors = ()
    if draft.seed is not None and (isinstance(draft.seed, bool) or not 0 <= draft.seed <= 2**32 - 2):
        errors = (issue("seed", "The seed is outside the executable domain.", "Use an integer from 0 to 2**32 - 2."),)
    rows.append(result("analysis:requirement:seed", ("seed",), "A fixed reproducible seed.", errors))
    names = [f.name for f in draft.facts]
    errors = () if len(names) == len(set(names)) else (
        issue("facts", "An assertion name is duplicated.", "Each fact has exactly one assertion."),)
    rows.append(result("analysis:requirement:facts", ("facts",), "Unambiguous assertions.", errors, kind="study_fact"))
    return rows


def _configuration(draft: CandidateDraft, definition: MethodDefinition
                   ) -> tuple[Any, tuple[RequirementResult, ...], bool]:
    raw = dict(draft.configuration)
    if draft.estimand is not None and "estimand" not in raw:
        raw["estimand"] = draft.estimand
    invalid: dict[str, list[Issue]] = {}
    schema = definition.specification.model_json_schema()

    def matches_type(value: Any, field_schema: dict[str, Any]) -> bool:
        if "$ref" in field_schema:
            field_schema = schema.get("$defs", {}).get(field_schema["$ref"].split("/")[-1], field_schema)
        if "anyOf" in field_schema:
            return any(matches_type(value, variant) for variant in field_schema["anyOf"])
        kind = field_schema.get("type")
        if kind == "null":
            return value is None
        if kind == "boolean":
            return type(value) is bool
        if kind == "integer":
            return type(value) is int
        if kind == "number":
            return type(value) is int or (type(value) is float and math.isfinite(value))
        if kind == "string":
            return isinstance(value, str)
        if kind == "array":
            return isinstance(value, (list, tuple)) and all(
                matches_type(child, field_schema.get("items", {})) for child in value)
        if kind == "object":
            return isinstance(value, Mapping)
        return True

    for field, value in raw.items():
        field_schema = schema.get("properties", {}).get(field)
        if field_schema is not None and not matches_type(value, field_schema):
            invalid[field] = [issue(
                f"configuration.{field}", "The submitted value has an incompatible scalar or collection type.",
                "Use the declared schema types without coercing booleans, strings or nonfinite numbers.",
                unsupported=True)]
    try:
        config = definition.specification.model_validate(raw)
    except ValidationError as error:
        for item in error.errors():
            field = str(item["loc"][0]) if item["loc"] else "configuration"
            invalid.setdefault(field, []).append(issue(
                f"configuration.{field}", item["msg"], "Use this field's declared input schema.",
                missing=item["type"] == "missing", unsupported=True))
    if invalid:
        kept = {key: value for key, value in raw.items()
                if key in definition.specification.model_fields and key not in invalid}
        # Only valid field values are used to explain independent constraints. Every
        # original invalid value remains a violated requirement and explicit selection.
        config = definition.specification.model_validate(kept)
    rows = [result(f"{definition.method}:requirement:schema.{field}",
                   (f"configuration.{field}",), "The declared configuration schema.",
                   tuple(invalid.get(field, ()))) for field in definition.specification.model_fields]
    extra = tuple(error for field, errors in invalid.items()
                  if field not in definition.specification.model_fields for error in errors)
    rows.append(result(f"{definition.method}:requirement:schema", ("configuration",),
                       "Only implemented configuration fields are supported.", extra))
    return config, tuple(rows), not invalid


def _assertion_result(draft: CandidateDraft, config: Any, rule: RequirementDefinition,
                      method: str) -> RequirementResult:
    active = rule.active_when is None or rule.active_when(config)
    errors: tuple[Issue, ...] = ()
    facts: dict[str, Scalar] = {f.name: f.value for f in draft.facts if f.value is not None}
    if active and rule.fact_name:
        assertion = next((f for f in draft.facts if f.name == rule.fact_name), None)
        field = f"facts.{rule.fact_name}"
        if assertion is None or assertion.value is None:
            errors += (issue(field, "The study assertion is unknown.", rule.expected, missing=True),)
        elif assertion.support == "assumption" and not rule.permits_assumption:
            errors += (issue(field, "This factual requirement cannot be replaced by an assumption.", rule.expected),)
        elif not assertion.evidence or not all(ref.strip() for ref in assertion.evidence):
            errors += (issue(field, "The assertion lacks declared supporting references.",
                             rule.expected, missing=True),)
        if assertion is not None and assertion.value is not None and rule.scope_fields:
            columns: set[str] = set()
            for name in rule.scope_fields:
                value = getattr(config, name.removeprefix("configuration."), None)
                if isinstance(value, str):
                    columns.add(value)
                elif isinstance(value, tuple):
                    columns.update(value)
            if not columns.issubset(assertion.scope):
                errors += (issue(field + ".scope", "The assertion does not cover the selected variables.",
                                 "Declare support scoped to: " + ", ".join(sorted(columns)), missing=True),)
    if active:
        errors += rule.predicate(config, facts)
    kind: ResolutionKind = ("scientific_assumption" if rule.kind == "assumption" else
                            "study_fact" if rule.fact_name else
                            "role_binding" if rule.kind == "role" else "configuration")
    return result(f"{method}:requirement:{rule.id}", rule.fields, rule.explanation,
                  errors, kind=kind, dependencies=tuple(f"{method}:requirement:{d}" for d in rule.dependencies),
                  active=active)


def _roles(draft: CandidateDraft, config: Any, definition: MethodDefinition
           ) -> tuple[tuple[RoleSlot, ...], tuple[RequirementResult, ...]]:
    slots, requirements = [], []
    for role in definition.roles:
        value = getattr(config, role.field.removeprefix("configuration."), None)
        columns = (value,) if isinstance(value, str) else tuple(value or ())
        bindings = tuple(b for b in draft.bindings if b.role == role.id)
        slots.append(RoleSlot(node_id=f"{definition.method}:role:{role.id}", role=role.id,
                              field=role.field, kind=role.kind, minimum=role.minimum,
                              maximum=role.maximum, bound_columns=columns,
                              source_references=tuple(ref for b in bindings for ref in b.source_references),
                              description=role.description))
        errors: tuple[Issue, ...] = ()
        if len(columns) < role.minimum:
            errors = (issue(role.field, "A required variable role is unbound.", role.description, missing=True),)
        elif role.maximum is not None and len(columns) > role.maximum:
            errors = (issue(role.field, "Too many columns are bound to this role.", role.description),)
        requirements.append(result(f"{definition.method}:requirement:role.{role.id}",
                                   (role.field,), role.description, errors, kind="role_binding"))
    outcome = draft.outcome.column if draft.outcome else None
    slots.append(RoleSlot(node_id=f"{definition.method}:role:outcome", role="outcome", field="outcome.column",
                          kind="numeric", minimum=1, maximum=1,
                          bound_columns=(outcome,) if outcome else (), description="The declared outcome measurement."))
    binding_errors: list[Issue] = []
    if outcome:
        binding_errors.extend(issue(slot.field, "The outcome is also assigned to an input role.",
                            "Keep the outcome distinct from all input roles.")
                      for slot in slots if slot.role != "outcome" and outcome in slot.bound_columns)
    requirements.append(result("analysis:requirement:role_separation", ("configuration", "outcome.column"),
                               "The outcome cannot also be a predictor or assignment variable.",
                               tuple(binding_errors), kind="role_binding"))
    binding_errors = []
    by_role = {slot.role: set(slot.bound_columns) for slot in slots}
    seen: set[tuple[str, str]] = set()
    for binding in draft.bindings:
        target = binding.expected_alias or binding.column
        allowed_columns = by_role.get(binding.role)
        if allowed_columns is None or target not in allowed_columns:
            binding_errors.append(issue("bindings", "A source binding differs from its executable role.",
                                "Declare the selected column or its expected prepared alias for a supported role."))
        if (binding.role, target) in seen:
            binding_errors.append(issue("bindings", "A binding is duplicated.", "Declare each role binding once."))
        seen.add((binding.role, target))
        if not binding.source_references or not all(ref.strip() for ref in binding.source_references):
            binding_errors.append(issue("bindings", "A supplied binding lacks source references.",
                                "Preserve the notebook or source references for each supplied binding.", missing=True))
    requirements.append(result("analysis:requirement:bindings", ("bindings",),
                               "Source bindings and declared aliases must match executable variables.",
                               tuple(binding_errors), kind="role_binding"))
    return tuple(slots), tuple(requirements)


def dependency_state(dependencies: tuple[str, ...], rows: tuple[RequirementResult, ...]
                     ) -> tuple[Eligibility, tuple[str, ...], str]:
    by_id = {row.requirement_id: row for row in rows}
    pending = list(dependencies)
    closure: set[str] = set()
    while pending:
        identifier = pending.pop()
        if identifier in closure:
            continue
        closure.add(identifier)
        row = by_id.get(identifier)
        # Inapplicable conditional requirements have no live prerequisites.
        if row is not None and row.status != "inapplicable":
            pending.extend(row.dependencies)
    selected = tuple(row for row in rows if row.requirement_id in closure)
    absent = tuple(sorted(closure - by_id.keys()))
    missing = tuple(dict.fromkeys(field for row in selected for field in row.missing_inputs))
    failed = tuple(row for row in selected if row.status == "violated")
    unresolved = tuple(row for row in selected if row.status == "unresolved")
    state: Eligibility = "blocked" if failed else "conditional" if unresolved or absent else "available"
    reasons = tuple(i.finding for row in failed + unresolved for i in row.issues)
    if absent:
        reasons += ("Prerequisite evaluation is unavailable: " + ", ".join(absent),)
    return state, missing, " ".join(dict.fromkeys(reasons)) or "Known local prerequisites are satisfied."


def _options(draft: CandidateDraft, config: Any, definition: MethodDefinition,
             rows: tuple[RequirementResult, ...]) -> tuple[OptionState, ...]:
    method = definition.method
    options = []
    schema = definition.specification.model_json_schema()["properties"]
    dependencies = dict(definition.option_dependencies)
    for field, description in schema.items():
        values = schema_values(description)
        for value in values or (None,):
            key = field if value is None else f"{field}={value}"
            node = f"{method}:decision:{field}" if value is None else f"{method}:option:{key}"
            related = tuple(rule.id for rule in definition.requirements
                            if f"configuration.{field}" in rule.fields)
            names = tuple(dict.fromkeys(dependencies.get(key, dependencies.get(field, ())) + related))
            deps = tuple(f"{method}:requirement:{name}" for name in names)
            local_rows = rows
            if value is not None:
                # A supported sibling is assessed as a hypothetical revision. In
                # particular, ANCOVA and staggered adoption reveal requirements
                # that may not apply to the currently selected sibling.
                proposed = config.model_dump()
                proposed[field] = value
                hypothetical = definition.specification.model_validate(proposed)
                hypothetical_results = tuple(_assertion_result(draft, hypothetical, rule, method)
                                             for rule in definition.requirements)
                overrides = {row.requirement_id: row for row in hypothetical_results}
                local_rows = tuple(overrides.get(row.requirement_id, row) for row in rows)
            # Type/schema contradictions block the field itself; alternative supported
            # values remain inspectable and can repair the submitted contradiction.
            if value is None:
                deps += (f"{method}:requirement:schema.{field}",)
            state, missing, reason = dependency_state(deps, local_rows)
            current = draft.configuration.get(field)
            if field == "estimand" and current is None:
                current = draft.estimand
            selected = field in draft.configuration if value is None else (
                type(current) is type(value) and current == value)
            options.append(OptionState(node_id=node, state=state, selected=selected,
                                       explanation=reason, dependencies=deps, missing_inputs=missing,
                                       invalidated=selected and state in ("blocked", "conditional"),
                                       applicability="applicable" if state == "available" else "unresolved"))
        if values:
            # The decision/policy itself describes the actual submitted state;
            # its child options describe independent supported revisions.
            related = tuple(rule.id for rule in definition.requirements
                            if f"configuration.{field}" in rule.fields)
            names = tuple(dict.fromkeys(dependencies.get(field, ()) + related))
            deps = tuple(f"{method}:requirement:{name}" for name in names)
            deps += (f"{method}:requirement:schema.{field}",)
            state, missing, reason = dependency_state(deps, rows)
            selected = field in draft.configuration or (field == "estimand" and draft.estimand is not None)
            options.append(OptionState(node_id=f"{method}:decision:{field}", state=state,
                                       selected=selected, explanation=reason,
                                       dependencies=deps, missing_inputs=missing,
                                       invalidated=selected and state != "available",
                                       applicability="applicable" if state == "available" else "unresolved"))
    catalog = diagnostics_module(method)
    for family, branches in (("diagnostic", definition.diagnostics), ("sensitivity", definition.sensitivities)):
        selections = draft.diagnostics if family == "diagnostic" else draft.sensitivities
        for branch in branches:
            deps = tuple(f"{method}:requirement:{name}" for name in branch.dependencies)
            state, missing, reason = dependency_state(deps, rows)
            function = catalog.applicability if family == "diagnostic" else catalog.sensitivity_applicability
            applicability, local_reason = function(branch, config, {f.name: f.value for f in draft.facts})
            if applicability == "inapplicable":
                state, reason = "blocked", local_reason
            elif state == "available" and applicability == "unresolved":
                state, reason = "conditional", local_reason
            elif state == "available":
                reason = local_reason
            if state != "available" and applicability != "inapplicable":
                applicability = "unresolved"
            selected = branch.id in selections or (family == "diagnostic" and getattr(branch, "obligation", None) == "required")
            options.append(OptionState(node_id=f"{method}:{family}:{branch.id}", state=state,
                                       selected=selected, explanation=reason, dependencies=deps,
                                       missing_inputs=missing, invalidated=branch.id in selections and state != "available",
                                       applicability=applicability))
    options.extend(OptionState(node_id=f"{method}:exclusion:{name}", state="unsupported",
                               explanation=reason) for name, reason in definition.unsupported)
    return tuple(options)


def evaluate_candidate(draft: CandidateDraft | Mapping[str, Any] | Model | None = None
                       ) -> CandidateEvaluation:
    try:
        candidate = coerce_candidate(draft)
    except ValidationError as error:
        raw = draft.model_dump(warnings=False) if isinstance(draft, Model) else dict(draft or {})
        errors = tuple(issue(".".join(map(str, item["loc"])), item["msg"],
                             "Use the partial candidate schema; invalid submissions remain rejected.")
                       for item in error.errors())
        return CandidateEvaluation(candidate_fingerprint=digest(raw), capability_version=None,
                                   status="rejected", requirements=(result("analysis:requirement:input", (),
                                   "The partial candidate schema.", errors),), issues=errors)
    identity = digest(candidate)
    method = candidate.method
    if method not in METHODS:
        errors = (issue("method", "No supported method has been nominated." if method is None else
                        f"{method!r} is unsupported.", "Nominate one discovered method.",
                        missing=method is None, unsupported=method is not None),)
        row = result("analysis:requirement:method", ("method",), "A nominated supported method.", errors)
        return CandidateEvaluation(candidate_fingerprint=identity, capability_version=None,
                                   status="needs_information" if method is None else "rejected",
                                   requirements=(row,), issues=errors)
    definition = cast(MethodDefinition, method_module(method).DEFINITION)
    config, schema_rows, valid_config = _configuration(candidate, definition)
    rows = _frame_requirements(candidate)
    errors = () if config.method == method else (
        issue("configuration.method", "Configuration names a different method.", "Use the nominated method."),)
    rows.append(result("analysis:requirement:method", ("method", "configuration.method"),
                       "The nomination and configuration must agree.", errors))
    errors = () if candidate.estimand is None or candidate.estimand == getattr(config, "estimand", None) else (
        issue("estimand", "Desired estimand and executable estimand conflict.", "Use one declared estimand."),)
    rows.append(result("analysis:requirement:estimand", ("estimand", "configuration.estimand"),
                       "Scientific and executable estimands must agree.", errors))
    rows.extend(schema_rows)
    rows.extend(_assertion_result(candidate, config, rule, method) for rule in definition.requirements)
    slots, role_rows = _roles(candidate, config, definition)
    rows.extend(role_rows)
    options = _options(candidate, config, definition, tuple(rows))
    for family, selected, catalog_ids in (
        ("diagnostics", candidate.diagnostics, {d.id for d in definition.diagnostics}),
        ("sensitivities", candidate.sensitivities, {s.id for s in definition.sensitivities}),
    ):
        errors = tuple(issue(family, f"{name!r} is not implemented for this method.",
                             "Select a supported computation.", unsupported=True)
                       for name in selected if name not in catalog_ids)
        if len(selected) != len(set(selected)):
            errors += (issue(family, "A computation was selected more than once.", "Select each computation once."),)
        rows.append(result(f"{method}:requirement:{family}", (family,), "Registered, unique computation selections.", errors))
    obligations = [CheckObligation(node_id=f"{method}:check:data", boundary="data_preflight", selected=True,
                                   applicability="unresolved", explanation="Check the exact prepared frame without fitting effects.")]
    for option in options:
        parts = option.node_id.split(":", 2)
        if len(parts) < 3 or parts[1] not in ("diagnostic", "sensitivity"):
            continue
        family = "diagnostics" if parts[1] == "diagnostic" else "sensitivities"
        explicit = parts[2] in getattr(candidate, family)
        errors = ()
        if option.selected and option.state != "available" and (option.applicability != "inapplicable" or explicit):
            errors = (issue(f"{family}.{parts[2]}", option.explanation,
                            "Resolve prerequisites for required or selected computations.", missing=option.state == "conditional"),)
        rows.append(result(option.node_id, (f"{family}.{parts[2]}",), option.explanation,
                           errors, dependencies=option.dependencies,
                           active=option.applicability != "inapplicable" or explicit))
        obligations.append(CheckObligation(node_id=option.node_id, boundary="execution", selected=option.selected,
                                            applicability=option.applicability, explanation=option.explanation,
                                            dependencies=option.dependencies))
    selections = []
    schema = definition.specification.model_json_schema()["properties"]
    for field, value in config.model_dump(mode="json").items():
        explicit = field in candidate.configuration or (field == "estimand" and candidate.estimand is not None)
        if not explicit and (value is None or value == []):
            # An unbound scientific slot is an unknown, not a default selection.
            continue
        selections.append(Selection(field=f"configuration.{field}",
                                    value=candidate.configuration.get(field, value),
                                    origin="explicit" if explicit else "fixed_policy" if fixed_value(schema[field]) else "mechanical_default",
                                    node_id=f"{method}:decision:{field}"))
    for field, value in candidate.configuration.items():
        if field not in schema:
            selections.append(Selection(field=f"configuration.{field}", value=value, origin="explicit",
                                        node_id=f"{method}:requirement:schema"))
    selections.append(Selection(field="seed", value=candidate.seed if candidate.seed is not None else 0,
                                origin="explicit" if candidate.seed is not None else "mechanical_default",
                                node_id="analysis:requirement:seed"))
    selections.extend(Selection(field=f"policy.{policy.id}", value=policy.value, origin="fixed_policy",
                                node_id=f"{method}:policy:{policy.id}") for policy in definition.fixed_policies)
    for field in ("method", "population", "unit_grain", "estimand", "population_policy", "missingness_policy"):
        if (value := getattr(candidate, field)) is not None:
            selections.append(Selection(field=field, value=value, origin="explicit",
                                        node_id=f"analysis:requirement:{field}"))
    if candidate.outcome is not None:
        for field in ("column", "kind", "units", "meaning"):
            if (value := getattr(candidate.outcome, field)) is not None:
                selections.append(Selection(field=f"outcome.{field}", value=value, origin="explicit",
                                            node_id=f"analysis:requirement:outcome.{field if field != 'meaning' else 'units'}"))
    fact_rules = {rule.fact_name: rule.id for rule in definition.requirements if rule.fact_name}
    for assertion in candidate.facts:
        selections.append(Selection(field=f"facts.{assertion.name}", value=assertion.model_dump(mode="json"),
                                    origin="explicit", node_id=f"{method}:requirement:{fact_rules[assertion.name]}"
                                    if assertion.name in fact_rules else "analysis:requirement:facts"))
    for family, names in (("diagnostic", candidate.diagnostics), ("sensitivity", candidate.sensitivities)):
        known = {option.node_id for option in options}
        for name in names:
            node_id = f"{method}:{family}:{name}"
            selections.append(Selection(field=family, value=name, origin="explicit", node_id=node_id if node_id in known
                                        else f"{method}:requirement:{'diagnostics' if family == 'diagnostic' else 'sensitivities'}"))
    for binding in candidate.bindings:
        selections.append(Selection(field="bindings", value=binding.model_dump(mode="json"),
                                    origin="explicit", node_id="analysis:requirement:bindings"))
    errors = tuple(i for row in rows for i in row.issues)
    status = "rejected" if any(row.status == "violated" for row in rows) else "needs_information" if errors else "design_ready"
    return CandidateEvaluation(candidate_fingerprint=identity, capability_version=definition.version,
                               status=cast(Any, status), requirements=tuple(rows), options=options,
                               selections=tuple(selections), role_slots=slots, issues=errors,
                               obligations=tuple(obligations), configuration=config.model_dump(mode="json") if valid_config else None)
