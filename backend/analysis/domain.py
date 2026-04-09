"""Column-independent gates and the owner-controlled diagnostic data projection."""

import ast
import csv
import hashlib
import io
import math
import re
from collections import Counter
from graphlib import CycleError, TopologicalSorter
from typing import Any

from backend.analysis import authority, code_checks, contrasts, lineage
from backend.analysis.contracts import (
    Assumption,
    CausalAccount,
    CausalNode,
    Claim,
    Decision,
    Frame,
    GateCheck,
    InputProjection,
    Plan,
    Program,
    RelationshipDisposition,
    Roles,
    Target,
    Treatments,
    Variable,
)
from backend.context.contracts import ContextContent, statement_entries
from backend.context.service import fingerprint


def supplied(context: ContextContent) -> set[str]:
    return {
        key
        for key, value in statement_entries(context)
        if value.basis == "source_statement" and value.status not in {"withdrawn", "unresolved"}
    } | {"question"}


def resolve(
    value: Decision, mapping: dict[str, str], *, known_assumptions: set[str] | None = None,
) -> Decision:
    """Decode exact typed references, including only the prior ledger offered by the caller."""
    declared = Counter(item.id.removeprefix("assumption:") for item in value.assumptions)
    offered_assumptions = {identity.removeprefix("assumption:") for identity in known_assumptions or set()}
    bare_assumptions = {
        identity: "assumption:" + identity
        for identity in offered_assumptions | declared.keys()
        if declared[identity] <= 1
    }

    def reference(ref: str) -> str:
        if ref in bare_assumptions:
            if (
                ref == "question" or ref.startswith(("columns.", "dataset.", "relationships.", "concepts.", "decision:", "diagnostic:", "assumption:"))
                or (ref in mapping and mapping[ref] != bare_assumptions[ref])
                or ref in mapping.values()
            ):
                raise ValueError("A bare assumption identity collides with an evidence reference; use its explicit assumption: namespace.")
            return bare_assumptions[ref]
        return mapping.get(ref, ref)

    def walk(obj: Any) -> Any:
        if isinstance(obj, list):
            return [walk(item) for item in obj]
        if isinstance(obj, dict):
            resolved = {
                key: item if key in {"admission", "original_proposal"} else [reference(ref) for ref in item]
                if key in {"references", "citations", "depends_on", "prerequisites"}
                else mapping.get(item, item) if key == "reference" and isinstance(item, str)
                else walk(item)
                for key, item in obj.items()
            }
            # An explicit assumption dependency makes the entire claim conditional.
            # Only reduce authority, preserving the proposed statement and all references.
            # This is not evidence validation: the gate still requires a declared ledger
            # entry, offered references, consistent dependencies and semantic obligations.
            if resolved.get("basis") == "supplied" and any(
                isinstance(ref, str) and ref.startswith("assumption:")
                for ref in resolved.get("references", [])
            ):
                resolved["basis"] = "assumption"
            return resolved
        return obj

    resolved = type(value).model_validate(walk(value.model_dump(mode="json")))
    # Accept an explicitly namespaced ledger ID without adding the namespace twice.
    # References are still exact, and duplicate/conflicting identities fail provenance checks.
    for assumption in resolved.assumptions:
        assumption.id = assumption.id.removeprefix("assumption:")
    # A directly observed variable has one authoritative identity: the explicitly selected column.
    # Resolve only aliases declared in this proposal, never fuzzy labels or invented evidence.
    selected = (
        [resolved.outcome]
        if isinstance(resolved, Target) and resolved.outcome
        else [item.variable for item in resolved.treatments]
        if isinstance(resolved, Treatments)
        else [item.variable for item in resolved.assignments]
        if isinstance(resolved, Roles)
        else []
    )
    aliases: dict[str, str] = {}
    for variable in selected:
        if len(variable.column_ids) == 1 and not variable.derivation:
            canonical = variable.column_ids[0]
            if variable.id in aliases and aliases[variable.id] != canonical:
                raise ValueError("One proposed identity cannot refer to different source columns.")
            aliases[variable.id] = canonical
            variable.id = canonical
    if isinstance(resolved, Treatments):
        for effect in resolved.effects:
            effect.treatment_ids = [aliases.get(ref, ref) for ref in effect.treatment_ids]
        for assignment in resolved.assignments:
            assignment.treatment_ids = [aliases.get(ref, ref) for ref in assignment.treatment_ids]
        resolved = contrasts.canonicalize(resolved, aliases)
    return resolved


def references(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set().union(
            *(
                set(item)
                if key in {"references", "citations", "depends_on", "prerequisites"} and isinstance(item, list)
                else references(item)
                for key, item in value.items() if key not in {"admission", "original_proposal"}
            )
        )
    if isinstance(value, list):
        return set().union(*(references(item) for item in value))
    return set()


def normalize_role_scope(value: Roles, prior: dict[str, Any], role_scope: dict[str, Any] | None = None) -> Roles:
    """Bind operational metadata to a single accepted effect, without inferring timing.

    Causal roles and multiple-effect scopes need explicit semantic decisions. This
    projection preserves the raw proposal and never replaces a conflicting scope.
    """
    normalized = value.model_copy(deep=True)
    if role_scope is not None:
        event_key = fingerprint(role_scope["event"])
        for assignment in normalized.assignments:
            if assignment.assignment_event_key not in {None, event_key}:
                raise ValueError("A role proposal cannot substitute a different assignment event.")
            assignment.assignment_event_key = event_key
    if "treatments" not in prior:
        return normalized
    treatments = Treatments.model_validate(prior["treatments"])
    if role_scope is not None:
        treatments.effects = [effect for effect in treatments.effects if effect.id in role_scope["effect_treatments"]]
        for effect in treatments.effects:
            selected = set(role_scope["effect_treatments"][effect.id])
            if not selected <= set(effect.treatment_ids):
                return normalized
            effect.treatment_ids = sorted(selected)
    if len(treatments.effects) != 1:
        return normalized
    effect = treatments.effects[0]
    scope = set(effect.treatment_ids)
    if not scope or len(scope) != len(effect.treatment_ids) or not scope <= {item.variable.id for item in treatments.treatments}:
        return normalized
    for assignment in normalized.assignments:
        if (
            not assignment.operational_use
            or not set(assignment.roles) <= {"identifier", "time", "irrelevant"}
            or not set(assignment.roles) & {"identifier", "time"}
        ):
            continue
        if assignment.effect_ids not in ([], [effect.id]) or (assignment.treatment_ids and set(assignment.treatment_ids) != scope):
            continue
        if not assignment.effect_ids:
            assignment.effect_ids = [effect.id]
        if not assignment.treatment_ids:
            assignment.treatment_ids = list(effect.treatment_ids)
        if not assignment.time_scope.strip():
            assignment.time_scope = "Operational metadata for the bound effect"
    return normalized


def role_scopes(context: ContextContent, target: Target, treatments: Treatments) -> list[dict[str, Any]]:
    """Canonical role obligations from frozen factors and finite assignment events.

    Effects sharing an assignment event share a request, but retain independent
    coverage and attempt obligations. Names, prose and model process aliases do
    not establish new events or grant new allowances.
    """
    bound = variables(target, treatments)
    factors = {item.variable.id for item in treatments.treatments}
    assigned = [identity for process in treatments.assignments for identity in process.treatment_ids]
    if Counter(assigned) != Counter({identity: 1 for identity in factors}):
        raise ValueError("Role scopes require accepted assignment processes partitioning every treatment factor.")
    effects = {item.id: set(item.treatment_ids) for item in treatments.effects}
    if len(effects) != len(treatments.effects) or any(not selected or not selected <= factors for selected in effects.values()):
        raise ValueError("Role scopes require distinct accepted effects with exact treatment bindings.")
    effect_registry = {item.id: item for item in treatments.effects}
    value_registry = {item.id: item for item in treatments.values}
    if len(value_registry) != len(treatments.values):
        raise ValueError("Role events require distinct accepted treatment-value identities.")
    for process in treatments.assignments:
        if process.kind not in {"point", "joint", "absorbing_adoption"}:
            raise ValueError("Role scopes require a supported, resolved assignment event.")

    def event_bindings(identity: str, visiting: frozenset[str] = frozenset()) -> dict[str, tuple[dict[str, Any], set[str]]]:
        if identity not in effect_registry or identity in visiting:
            raise ValueError("Role scopes require an acyclic aggregate of exact declared component effects.")
        effect = effect_registry[identity]
        comparison = effect.comparison
        events: dict[str, tuple[dict[str, Any], set[str]]] = {}
        if comparison and comparison.aggregate:
            for component in comparison.aggregate.terms:
                for key, (inherited_event, selected) in event_bindings(component.effect_id, visiting | {identity}).items():
                    if key in events:
                        events[key][1].update(selected)
                    else:
                        events[key] = (inherited_event, set(selected))
            if set().union(*(selected for _, selected in events.values())) != effects[identity]:
                raise ValueError("Aggregate role scopes must preserve the exact factor scope of their components.")
            return events
        for process in treatments.assignments:
            selected = effects[identity] & set(process.treatment_ids)
            if not selected:
                continue
            event: dict[str, Any] = {"kind": process.kind, "treatment_ids": sorted(process.treatment_ids)}
            if comparison and comparison.group_time:
                group_time = comparison.group_time
                cohort = value_registry.get(group_time.cohort_value_id)
                if cohort is None or process.kind != "absorbing_adoption" or len(selected) != 1:
                    raise ValueError("Group-time role events require an exact cohort value and one absorbing-adoption factor.")
                adoption_literal = contrasts.literal_key(group_time.adoption_period)
                if group_time.adoption_period.kind == "number" and adoption_literal != "0":
                    sign, digits, exponent = adoption_literal.split(":")
                    adoption_literal = ("-" if sign == "1" else "") + digits + "e" + exponent
                event["group_time"] = {
                    "cohort_variable_id": cohort.variable_id,
                    "cohort_value_id": contrasts.value_id(cohort.variable_id, cohort.value),
                    "time_variable_id": group_time.time_variable_id,
                    "adoption_period": {"kind": group_time.adoption_period.kind,
                                        "literal": adoption_literal},
                }
            events[fingerprint(event)] = (event, selected)
        return events

    grouped: dict[str, tuple[dict[str, Any], dict[str, set[str]]]] = {}
    for identity in sorted(effects):
        for key, (event, selected) in event_bindings(identity).items():
            if key not in grouped:
                grouped[key] = (event, {})
            grouped[key][1][identity] = selected
    outcome_columns = set(target.outcome.column_ids) if target.outcome else set()
    result = []
    for event_key, (event, event_effects) in grouped.items():
        for column in sorted(set(context.columns) - outcome_columns):
            scoped = {
                effect: sorted(selected)
                for effect, selected in sorted(event_effects.items())
                if column not in {cid for identity in selected for cid in bound[identity].column_ids}
            }
            if scoped:
                result.append({
                    "key": f"role:{column}:{event_key}",
                    "column_id": column,
                    "event": event,
                    "effect_treatments": scoped,
                    "obligation_keys": [f"role:{column}:{effect}:{event_key}" for effect in scoped],
                })
    return sorted(result, key=lambda item: item["key"])


def validate_role_scope(value: Roles, scope: dict[str, Any], *, allow_legacy: bool = False) -> None:
    """Certify only the exact owned column/effect/event pairs, including exclusions."""
    expected = {(effect, treatment) for effect, identities in scope["effect_treatments"].items() for treatment in identities}
    covered: set[tuple[str, str]] = set()
    seen: set[tuple[str, str, str]] = set()
    event_key = fingerprint(scope["event"])
    for assignment in value.assignments:
        if assignment.assignment_event_key != event_key and not (allow_legacy and assignment.assignment_event_key is None):
            raise ValueError("Every role disposition must retain its exact harness-bound assignment event.")
        if scope["column_id"] not in assignment.variable.column_ids:
            raise ValueError("A role task may assess only its owned column measurement.")
        pairs = {(effect, treatment) for effect in assignment.effect_ids for treatment in assignment.treatment_ids}
        if not pairs or not pairs <= expected:
            raise ValueError("Every role disposition, including irrelevant, must explicitly name only its owned effect and assignment-event treatment scopes.")
        identities = {(assignment.variable.id, effect, treatment) for effect, treatment in pairs}
        if identities & seen:
            raise ValueError("Repeated role dispositions for the same measurement/effect/event cannot conflict or duplicate; use one explicit role assignment.")
        covered.update(pairs)
        seen.update(identities)
    if value.status == "ready" and covered != expected:
        missing = ", ".join(f"{effect}/{treatment}" for effect, treatment in sorted(expected - covered))
        raise ValueError("Role assessment omitted owned effect/event obligations: " + missing)


def validate_role_events(value: Roles, context: ContextContent, target: Target, treatments: Treatments) -> None:
    """Keep aggregate and repeated-factor event obligations distinct after assembly."""
    scopes = role_scopes(context, target, treatments)
    allowed = {(scope["column_id"], fingerprint(scope["event"])) for scope in scopes}
    for assignment in value.assignments:
        if not any((column, assignment.assignment_event_key) in allowed for column in assignment.variable.column_ids):
            raise ValueError("A role disposition has no accepted column and assignment-event obligation.")
    for scope in scopes:
        assignments = [item for item in value.assignments
                       if scope["column_id"] in item.variable.column_ids
                       and item.assignment_event_key == fingerprint(scope["event"])]
        validate_role_scope(value.model_copy(update={"assignments": assignments}), scope)


def adopt_relationships(value: CausalAccount, accepted: list[dict[str, Any]]) -> CausalAccount:
    """Assemble already accepted relevant interpretations without asking the model again.

    The caller selects the dependency-relevant accepted records. This function does
    not decide relevance, infer edges or reinterpret qualified source statements.
    Raw proposals are preserved; source/provenance gates still validate the result.
    """
    normalized = value.model_copy(deep=True)
    dispositions: dict[str, RelationshipDisposition] = {}
    ledger: dict[str, Assumption] = {}
    for record in accepted:
        if record.get("status") != "ready":
            raise ValueError("Only accepted ready relationship decisions can be assembled.")
        for raw in record.get("relationships", []):
            item = RelationshipDisposition.model_validate(raw)
            if item.reference in dispositions and dispositions[item.reference] != item:
                raise ValueError("Accepted relationship decisions conflict for the same source record.")
            dispositions[item.reference] = item
        for raw in record.get("assumptions", []):
            assumption = Assumption.model_validate(raw)
            if assumption.id in ledger and ledger[assumption.id] != assumption:
                raise ValueError("Accepted relationship decisions contain conflicting assumption definitions.")
            ledger[assumption.id] = assumption
    seen: set[str] = set()
    for item in normalized.relationships:
        if item.reference in seen or dispositions.get(item.reference) != item:
            raise ValueError("The causal proposal cannot add, duplicate or change an accepted relationship interpretation.")
        seen.add(item.reference)
    normalized.relationships = [dispositions[key] for key in sorted(dispositions)]
    if normalized.status == "ready" and any(item.disposition == "unresolved" for item in normalized.relationships):
        raise ValueError("A relevant unresolved relationship cannot support a ready causal account.")
    model_ids: set[str] = set()
    for assumption in normalized.assumptions:
        if assumption.id in model_ids or (assumption.id in ledger and ledger[assumption.id] != assumption):
            raise ValueError("The causal proposal cannot duplicate or redefine an accepted assumption.")
        model_ids.add(assumption.id)
        ledger[assumption.id] = assumption
    needed = references(normalized.model_dump())
    while True:
        expanded = needed | {
            ref for identity, assumption in ledger.items()
            if "assumption:" + identity in needed
            for ref in assumption.depends_on
        }
        if expanded == needed:
            break
        needed = expanded
    normalized.assumptions.extend(
        ledger[identity] for identity in sorted(ledger)
        if identity not in model_ids and "assumption:" + identity in needed
    )
    return CausalAccount.model_validate(normalized.model_dump(mode="json"))


def variables(
    target: Target, treatments: Treatments | None = None, roles: Roles | None = None
) -> dict[str, Variable]:
    items = (
        ([target.outcome] if target.outcome else [])
        + ([item.variable for item in treatments.treatments] if treatments else [])
        + ([item.variable for item in roles.assignments] if roles else [])
    )
    registry: dict[str, Variable] = {}
    for item in items:
        if item.id in registry and item != registry[item.id]:
            raise ValueError("A variable identity cannot acquire a different binding or meaning in another role.")
        registry[item.id] = item
    if target.outcome and treatments and target.outcome.id in {t.variable.id for t in treatments.treatments}:
        raise ValueError("The target measurement cannot also be its own treatment.")
    return registry


def canonical_graph(value: CausalAccount, target: Target, treatments: Treatments, roles: Roles) -> CausalAccount:
    """Project exact observed bindings, preserving the authored graph up to explicit aliases.

    A label never establishes identity. An unknown node ID is an alias only when its
    explicit ordered column binding identifies exactly one accepted variable. This
    does not infer edges, merge measurements or change identification choices.
    """
    value = value.model_copy(deep=True)
    bound = variables(target, treatments, roles)
    bindings: dict[tuple[str, ...], list[str]] = {}
    for identity, variable in bound.items():
        bindings.setdefault(tuple(variable.column_ids), []).append(identity)
    reserved = (
        {item.id for item in treatments.effects}
        | {item.id for item in treatments.assignments}
        | {item.id for item in value.assumptions}
        | {column for variable in bound.values() for column in variable.column_ids}
    )
    aliases: dict[str, str] = {}
    identities: set[str] = set()
    for node in value.nodes:
        if node.id in aliases:
            raise ValueError("Graph nodes must have unique declared identities.")
        identity = node.id
        if identity in bound:
            if node.column_ids and node.column_ids != bound[identity].column_ids:
                raise ValueError("An established graph identity cannot acquire a conflicting column binding.")
        elif node.column_ids:
            matches = bindings.get(tuple(node.column_ids), [])
            if len(matches) != 1:
                raise ValueError("A graph node alias requires an exact column binding to one accepted variable.")
            if identity in reserved:
                raise ValueError("A graph node alias collides with another established identity.")
            identity = matches[0]
        if identity in identities:
            raise ValueError("Distinct graph node aliases cannot duplicate one accepted variable.")
        aliases[node.id] = identity
        identities.add(identity)
    assigned: dict[str, set[str]] = {}
    for item in roles.assignments:
        assigned.setdefault(item.variable.id, set()).update(item.roles)
    for treatment in treatments.treatments:
        assigned.setdefault(treatment.variable.id, set()).add("treatment")
    if target.outcome:
        assigned.setdefault(target.outcome.id, set()).add("outcome")
    for index, node in enumerate(value.nodes):
        identity = aliases[node.id]
        if identity in bound:
            variable = bound[identity]
            value.nodes[index] = CausalNode(id=identity, label=variable.label,
                column_ids=list(variable.column_ids), roles=sorted(assigned[identity]))
    for edge in value.edges:
        edge.source = aliases.get(edge.source, edge.source)
        edge.target = aliases.get(edge.target, edge.target)
    for identification in value.identifications:
        identification.adjustment = [aliases.get(identity, identity) for identity in identification.adjustment]
    for assumption in value.assumptions:
        assumption.affects = [aliases.get(identity, identity) for identity in assumption.affects]
    return value


def _ancestors(parents: dict[str, set[str]], roots: set[str]) -> set[str]:
    found = set(roots)
    pending = list(roots)
    while pending:
        for parent in parents[pending.pop()]:
            if parent not in found:
                found.add(parent)
                pending.append(parent)
    return found


def backdoor_valid(nodes: set[str], edges: set[tuple[str, str]], treatments: set[str], outcome: str, adjustment: set[str]) -> bool:
    """The backdoor criterion, conditional on this explicit acyclic graph, including latent nodes.

    Moralization of the ancestral graph implements d-separation. No graph is inferred from data,
    and passing this criterion does not establish that the supplied causal graph is true.
    """
    if not treatments or not (treatments | {outcome} | adjustment) <= nodes or adjustment & (treatments | {outcome}):
        return False
    parents: dict[str, set[str]] = {node: set() for node in nodes}
    for source, destination in edges:
        parents[destination].add(source)
    tuple(TopologicalSorter(parents).static_order())
    if any(_ancestors(parents, {node}) & treatments for node in adjustment):
        return False
    cut = {node: predecessors - treatments for node, predecessors in parents.items()}
    relevant = _ancestors(cut, treatments | {outcome} | adjustment)
    moral: dict[str, set[str]] = {node: set() for node in relevant}
    for node in relevant:
        predecessors = cut[node] & relevant
        for parent in predecessors:
            moral[node].add(parent)
            moral[parent].add(node)
            moral[parent].update(predecessors - {parent})
    seen = set(treatments)
    pending = list(treatments)
    while pending:
        current = pending.pop()
        if current == outcome:
            return False
        for neighbor in moral[current] - adjustment - seen:
            seen.add(neighbor)
            pending.append(neighbor)
    return True


def validate_relationships(
    dispositions: list[RelationshipDisposition], context: ContextContent, effect_ids: set[str],
) -> None:
    """Validate source-preserving interpretations independently of a proposed graph."""
    if len({item.reference for item in dispositions}) != len(dispositions):
        raise ValueError("A source relationship must have one interpretation per accepted task.")
    for item in dispositions:
        reference = item.reference
        relation = context.relationships.get(reference.removeprefix("relationships.")) if reference.startswith("relationships.") else None
        if relation is None or relation.status == "withdrawn":
            raise ValueError("Relationship disposition names an absent or withdrawn source relationship.")
        if len(set(item.effect_ids)) != len(item.effect_ids) or not set(item.effect_ids) <= effect_ids:
            raise ValueError("Relationship scopes must use distinct requested effect identities.")
        if reference not in item.rationale.references:
            raise ValueError("A relationship disposition must cite its exact source relationship.")
        participants = {p.id for p in relation.participants}
        if any(endpoint and endpoint not in participants for endpoint in (item.source_id, item.target_id)):
            raise ValueError("Relationship endpoints must be explicit participants in the exact source record.")
        if item.disposition != "applicable":
            continue
        if item.rationale.basis == "unknown":
            raise ValueError("An applicable relationship requires an explicit interpretation or supplied account.")
        sources = {p.id for p in relation.participants if p.role == "from"}
        targets = {p.id for p in relation.participants if p.role == "to"}
        causal_direction = relation.kind in {"causal_influence", "no_causal_influence", "temporal_order"}
        non_directed = relation.kind in {None, "association", "membership", "unclassified"}
        if non_directed and (item.source_id or item.target_id):
            raise ValueError("Association or co-participation does not define directed graph endpoints.")
        if causal_direction or (not non_directed and (item.source_id or item.target_id)):
            if item.source_id not in sources or item.target_id not in targets or item.source_id == item.target_id:
                raise ValueError("Applicable directed relationships require exact selected from/to participants; co-participation never implies all pairwise edges.")
        qualified = bool(
            relation.conditions or relation.timing or relation.qualifications or len(participants) > 2
            or any(p.role in {"condition", "intermediate"} for p in relation.participants)
            or (causal_direction and (len(sources) != 1 or len(targets) != 1))
        )
        if qualified and not item.scope.strip():
            raise ValueError("A qualified or multi-party relationship needs an explicit accepted scope; do not flatten it into an unconditional pair.")
        if relation.kind == "no_causal_influence" and relation.absence_scope == "unspecified":
            raise ValueError("An unspecified absence cannot become a direct-edge or all-pathways constraint.")


def _relationship_checks(
    value: CausalAccount, context: ContextContent, bound: dict[str, Variable],
    effects: dict[str, Any], edges: set[tuple[str, str]], ancestors: dict[str, set[str]],
) -> None:
    """Check accepted relationship interpretations without flattening qualified source prose."""
    active_columns = {column for identity in ancestors if identity in bound for column in bound[identity].column_ids}
    relevant = {
        f"relationships.{rid}": relation
        for rid, relation in context.relationships.items()
        if relation.kind in {"causal_influence", "no_causal_influence", "temporal_order"}
        and relation.basis == "source_statement" and relation.status not in {"withdrawn", "unresolved"}
        and any(p.id in active_columns or p.id in ancestors for p in relation.participants)
    }
    dispositions = {item.reference: item for item in value.relationships}
    if len(dispositions) != len(value.relationships) or not relevant.keys() <= dispositions.keys():
        missing = sorted(relevant.keys() - dispositions.keys())
        raise ValueError("Every supplied causal/temporal relationship incident to the accepted graph needs one explicit scoped disposition. Missing: " + ", ".join(missing) + ". Use set_relationships to select these exact frozen records and state their scope.")
    validate_relationships(value.relationships, context, set(effects))
    for reference, item in dispositions.items():
        relation = context.relationships[reference.removeprefix("relationships.")]
        if item.disposition != "applicable" or relation.kind not in {"causal_influence", "no_causal_influence", "temporal_order"}:
            continue

        def binding(participant: str) -> str:
            candidates = {identity for identity, variable in bound.items() if participant in variable.column_ids and identity in ancestors}
            if participant in ancestors:
                candidates.add(participant)
            if len(candidates) != 1:
                raise ValueError("A relationship endpoint must resolve to one graph measurement. Preserve distinct temporal or derived measurements explicitly.")
            return next(iter(candidates))

        source, target = binding(item.source_id), binding(item.target_id)
        if relation.kind == "causal_influence":
            if relation.directness == "direct" and (source, target) not in edges:
                raise ValueError("The graph omits an applicable supplied direct causal influence.")
            if relation.directness != "direct" and source not in ancestors[target]:
                raise ValueError("The graph omits the pathway in an applicable supplied causal influence.")
        elif relation.kind == "no_causal_influence":
            if relation.absence_scope == "direct_effect" and (source, target) in edges:
                raise ValueError("The graph contradicts an applicable supplied absence of a direct effect.")
            if relation.absence_scope == "any_pathway" and source in ancestors[target]:
                raise ValueError("The graph contradicts an applicable supplied absence of every causal pathway.")
        elif relation.kind == "temporal_order" and target in ancestors[source]:
            raise ValueError("The graph contradicts an applicable supplied temporal order.")


def validate_variable(value: Variable, context: ContextContent) -> None:
    if (
        len(value.column_ids) != len(set(value.column_ids))
        or not set(value.column_ids) <= context.columns.keys()
    ):
        raise ValueError("Variable bindings must use unique exact dataset column identities.")
    if len(value.column_ids) == 1 and not value.derivation and value.id != value.column_ids[0]:
        raise ValueError("A directly observed variable uses its column identity as its variable identity.")
    if len(value.column_ids) > 1 and not value.derivation.strip():
        raise ValueError("A variable derived from multiple columns requires its explicit derivation.")


def checks(
    value: Decision,
    context: ContextContent,
    prior: dict[str, Any],
    computed: set[str] | None = None,
    *,
    role_columns: set[str] | None = None,
    scientific_version: int = 7,
) -> list[GateCheck]:
    outcomes: list[GateCheck] = []

    def check(rule: str, action: Any) -> None:
        try:
            detail = action()
            outcomes.append(GateCheck(rule=rule, kind="deterministic", passed=True, detail=detail if isinstance(detail, str) else "Validated."))
        except (ValueError, CycleError) as error:
            outcomes.append(GateCheck(rule=rule, kind="deterministic", passed=False, detail=str(error)))

    known_assumptions = {
        a["id"]: a for item in prior.values() if isinstance(item, dict) for a in item.get("assumptions", [])
        if scientific_version < 19 or a.get("status") == "relied_upon"
    }
    roots = supplied(context)
    offered = (
        {key for key, statement in statement_entries(context) if statement.status != "withdrawn"}
        | {"question"}
        | (computed or set())
    )
    decisions = {"decision:" + key for key in prior}

    def provenance() -> None:
        assumptions = dict(known_assumptions)
        for assumption in value.assumptions:
            if scientific_version >= 19 and assumption.status != "relied_upon":
                continue
            saved = assumption.model_dump(mode="json")
            if assumption.id in assumptions and assumptions[assumption.id] != saved:
                raise ValueError("An existing assumption cannot be silently changed.")
            assumptions[assumption.id] = saved
        if len({a.id for a in value.assumptions}) != len(value.assumptions):
            raise ValueError("Assumption identities must be unique.")
        assumption_refs = {"assumption:" + key for key in assumptions}
        active = value.model_dump()
        if scientific_version >= 19:
            active["assumptions"] = [item for item in active.get("assumptions", []) if item.get("status") == "relied_upon"]
        unknown = references(active) - (offered | decisions | assumption_refs)
        if unknown:
            raise ValueError(
                "References outside this task's evidence packet: "
                + ", ".join(sorted(unknown))
                + ". Use offered evidence handles or an explicitly declared assumption:<id>."
            )
        graph = {
            key: {
                ref.removeprefix("assumption:")
                for ref in entry["depends_on"]
                if ref.startswith("assumption:")
            }
            for key, entry in assumptions.items()
        }
        tuple(TopologicalSorter(graph).static_order())

        def walk(obj: Any, path: str = "decision") -> None:
            if isinstance(obj, list):
                for index, item in enumerate(obj):
                    walk(item, f"{path}[{index}]")
            if isinstance(obj, dict):
                if obj.get("basis") in {"supplied", "deduced", "assumption", "unknown"} and not set(obj.get("prerequisites", [])) <= decisions:
                    raise ValueError(f"{path}: prerequisite references must name accepted decisions offered to this task; they cannot confer source authority.")
                if obj.get("basis") in {"supplied", "assumption"} and "references" in obj:
                    refs = set(obj["references"])
                    valid = bool(refs) and (
                        refs <= roots if obj["basis"] == "supplied" else bool(refs & assumption_refs)
                    )
                    if not valid:
                        raise ValueError(
                            f"{path}: basis={obj['basis']} requires "
                            + (
                                "direct supplied source references"
                                if obj["basis"] == "supplied"
                                else "at least one assumption:<id> reference to a declared ledger entry"
                            )
                            + ". Use basis=unknown when unknown; do not promote an inference into a supplied fact."
                        )
                for key, item in obj.items():
                    if key not in {"admission", "original_proposal"}:
                        walk(item, f"{path}.{key}")

        walk(active)

    check("evidence_authority_and_dependencies", provenance)
    if scientific_version >= 19:
        phase = "roles" if isinstance(value, Roles) else "causal" if isinstance(value, CausalAccount) else "freeze" if isinstance(value, Plan) else "target" if isinstance(value, Target) else "treatments"
        check("claim_admission_and_ledger", lambda: authority.validate(value, prior, phase=phase, frozen="causal" in prior))

    def readiness() -> None:
        if value.status == "needs_input" and not value.unresolved:
            raise ValueError("A blocked information gate needs a focused missing-information question.")
        if value.status == "ready" and not references(value.model_dump()):
            raise ValueError("An accepted scientific decision needs cited evidence.")

    check("decision_is_actionable", readiness)

    def meaning(claim: Claim | None, name: str, essential: bool = False) -> None:
        if claim is None or (essential and claim.basis != "supplied"):
            raise ValueError(f"{name} requires a supplied meaning before proceeding.")

    def bindings() -> None:
        if isinstance(value, Target):
            if value.outcome:
                validate_variable(value.outcome, context)
            if value.status == "ready":
                if not value.outcome or not value.effect_question.strip():
                    raise ValueError("The target requires a bound outcome and an explicit effect question.")
                meaning(value.outcome.meaning, "Outcome", True)
                meaning(value.measurement, "Outcome measurement", True)
                for name in ("population", "unit", "timing"):
                    meaning(getattr(value, name), name)
                if scientific_version >= 9:
                    if (
                        not value.requested_effects
                        or len({item.id for item in value.requested_effects}) != len(value.requested_effects)
                        or sum(item.priority == "primary" for item in value.requested_effects) != 1
                    ):
                        raise ValueError("The target must freeze distinct requested effect identities with exactly one primary effect.")
                    for requested in value.requested_effects:
                        meaning(requested.question, "Requested effect question", True)
        if isinstance(value, Treatments):
            target = Target.model_validate(prior["target"])
            direct = [
                tuple(item.variable.column_ids) for item in value.treatments if not item.variable.derivation
            ]
            if len(direct) != len(set(direct)):
                raise ValueError(
                    "Treatments are intervention variables, not arms/levels. Represent one categorical column once, with its levels in effect.contrast."
                )
            variables(target, value)
            for item in value.treatments:
                validate_variable(item.variable, context)
                if value.status == "ready":
                    meaning(item.variable.meaning, "Treatment", True)
                    meaning(item.change, "Intervention change", True)
            if value.status == "ready":
                if not value.treatments or not value.effects:
                    raise ValueError("At least one treatment and one prespecified effect are required.")
                if (
                    len({item.id for item in value.effects}) != len(value.effects)
                    or sum(item.priority == "primary" for item in value.effects) != 1
                ):
                    raise ValueError(
                        "Effect identities must be unique and exactly one effect designated primary."
                    )
                if scientific_version >= 9 and (
                    not target.requested_effects
                    or {item.id: item.priority for item in value.effects}
                    != {item.id: item.priority for item in target.requested_effects}
                ):
                    raise ValueError("Preserve the accepted target's requested effect identities and priorities; auxiliary computations are not additional effects.")
                ids = {item.variable.id for item in value.treatments}
                covered: set[str] = set()
                for effect in value.effects:
                    if not set(effect.treatment_ids) <= ids or len(effect.treatment_ids) != len(
                        set(effect.treatment_ids)
                    ):
                        raise ValueError("Each effect must name its exact treatment bindings.")
                    meaning(effect.contrast, "Treatment contrast", True)
                    covered.update(effect.treatment_ids)
                if covered != ids:
                    raise ValueError("Every selected treatment must belong to a requested effect.")
                if scientific_version >= 8:
                    assigned: set[str] = set()
                    if len({item.id for item in value.assignments}) != len(value.assignments):
                        raise ValueError("Assignment process identities must be unique.")
                    for process in value.assignments:
                        factors = set(process.treatment_ids)
                        if len(factors) != len(process.treatment_ids) or not factors <= ids or factors & assigned:
                            raise ValueError("Assignment processes must partition the selected treatment factors exactly once.")
                        if process.kind in {"switching", "dynamic_regime"}:
                            raise ValueError("Switching and history-adaptive treatment regimes are not an implemented parity capability. Preserve the requested regime and return unsupported.")
                        if process.kind == "unknown" or any(c.basis == "unknown" for c in (process.unit, process.timing, process.mechanism)):
                            raise ValueError("A ready assignment account needs explicit unit, timing and mechanism, as supplied claims or declared assumptions.")
                        if process.kind == "point" and len(factors) != 1:
                            raise ValueError("Several factors assigned together require a joint assignment process.")
                        if len(set(process.history_variables)) != len(process.history_variables):
                            raise ValueError("Assignment history identities must be distinct.")
                        window = process.diagnostic_window
                        if window is not None and (
                            process.kind != "absorbing_adoption" or not math.isfinite(window.cutoff)
                            or window.evidence.basis != "supplied" or window.history_evidence.basis != "supplied"
                            or not window.evidence.references or not window.history_evidence.references
                            or set(window.treated_values) & set(window.control_values)
                            or len(set(window.treated_values)) != len(window.treated_values)
                            or len(set(window.control_values)) != len(window.control_values)
                        ):
                            raise ValueError("A diagnostic history window requires absorbing adoption, supplied timing/history evidence, a finite numeric cutoff and disjoint exact assignment codes.")
                        assigned.update(factors)
                    if assigned != ids:
                        raise ValueError("Every treatment factor needs its linked assignment process.")
                if scientific_version >= 15:
                    contrasts.validate(value, set(context.columns), target.outcome.id if target.outcome else None)
        if isinstance(value, Roles):
            target, treatments = (
                Target.model_validate(prior["target"]),
                Treatments.model_validate(prior["treatments"]),
            )
            all_vars = variables(target, treatments, value)
            for assignment in value.assignments:
                validate_variable(assignment.variable, context)
                if len(set(assignment.roles)) != len(assignment.roles):
                    raise ValueError("Do not duplicate role labels.")
                if scientific_version < 8 and assignment.proxy_evidence and assignment.outcome_proxy is True and assignment.timing == "before_treatment":
                    raise ValueError("Supplied target-outcome lineage contradicts pre-treatment timing. Resolve that evidence conflict before continuing.")
                if scientific_version >= 8:
                    effects = {effect.id: set(effect.treatment_ids) for effect in treatments.effects}
                    if len(set(assignment.effect_ids)) != len(assignment.effect_ids) or not set(assignment.effect_ids) <= effects.keys():
                        raise ValueError("Role scopes must name distinct requested effect identities.")
                    treatment_scope = set(assignment.treatment_ids)
                    if len(treatment_scope) != len(assignment.treatment_ids) or not treatment_scope <= {t.variable.id for t in treatments.treatments}:
                        raise ValueError("Role timing must name distinct selected treatment identities.")
                    relevant = not set(assignment.roles) <= {"irrelevant", "unknown"} or bool(assignment.operational_use)
                    missing_scope = [name for name, present in (
                        ("effect_ids", bool(assignment.effect_ids)), ("treatment_ids", bool(treatment_scope)),
                        ("time_scope", bool(assignment.time_scope.strip())),
                    ) if not present]
                    if relevant and missing_scope:
                        raise ValueError(f"Role variable {assignment.variable.id} is missing required scope fields: {', '.join(missing_scope)}. Supply exact accepted IDs and the measurement window; irrelevant does not cancel another scientific role or operational use.")
                    if any(not treatment_scope <= effects[identity] for identity in assignment.effect_ids):
                        raise ValueError("A role's treatment scope must belong to each named effect. Split roles whose effect or timing scopes differ.")
                    if len(set(assignment.operational_use)) != len(assignment.operational_use):
                        raise ValueError("Operational uses must be distinct.")
                    compatible_roles = {
                        "unit_key": {"identifier"},
                        "time_key": {"time"},
                        "cluster_key": {"identifier"},
                        "assignment_key": {"identifier", "time", "running_variable", "instrument"},
                    }
                    for use in assignment.operational_use:
                        if (not authority.established(assignment.account) if scientific_version >= 19 else assignment.account.basis != "supplied" or not assignment.account.references):
                            raise ValueError(f"Operational use {use} needs an explicitly supplied account with direct source references.")
                        if not set(assignment.roles) & compatible_roles[use]:
                            raise ValueError(f"Operational use {use} needs a compatible role: {', '.join(sorted(compatible_roles[use]))}. Do not invent an identifier role for an assignment score.")
            # Column-local tasks cannot close a registry that other role tasks are
            # still building. The combined Roles gate must resolve every history
            # variable before the scientific account can advance.
            if scientific_version >= 8 and role_columns is None:
                for process in treatments.assignments:
                    if not set(process.history_variables) <= all_vars.keys():
                        raise ValueError("An assignment process refers to an unbound history variable.")
            if scientific_version >= 15 and role_columns is None:
                validate_role_events(value, context, target, treatments)
            if value.status == "ready" and {col for item in all_vars.values() for col in item.column_ids} != (
                role_columns if role_columns is not None else set(context.columns)
            ):
                raise ValueError(
                    "Account for every column as selected, relevant, irrelevant or unresolved; do not silently omit columns."
                )

    if isinstance(value, (Target, Treatments, Roles)):
        check("semantic_bindings", bindings)

    if scientific_version < 8 and isinstance(value, Roles) and any(item.proxy_evidence for item in value.assignments):
        def lineage_agreement() -> None:
            expected = lineage.classify(value, context, Target.model_validate(prior["target"]))
            if any(
                (actual.outcome_proxy, actual.proxy_evidence) != (canonical.outcome_proxy, canonical.proxy_evidence)
                for actual, canonical in zip(value.assignments, expected.assignments, strict=True)
            ):
                raise ValueError("Outcome-lineage classifications must follow the deterministic source/timing policy.")

        check("derived_outcome_lineage", lineage_agreement)

    def causal() -> str | None:
        if not isinstance(value, CausalAccount):
            return None
        target, treatments, roles = (
            Target.model_validate(prior["target"]),
            Treatments.model_validate(prior["treatments"]),
            Roles.model_validate(prior["roles"]),
        )
        bound = variables(target, treatments, roles)
        nodes = {node.id: node for node in value.nodes}
        if len(nodes) != len(value.nodes):
            raise ValueError("Graph nodes must have unique identities.")
        expected_roles: dict[str, set[str]] = {}
        for item in roles.assignments:
            expected_roles.setdefault(item.variable.id, set()).update(item.roles)
        for treatment in treatments.treatments:
            expected_roles.setdefault(treatment.variable.id, set()).add("treatment")
        if target.outcome:
            expected_roles[target.outcome.id] = {"outcome"}
        for node in value.nodes:
            if node.column_ids:
                if (
                    node.id not in bound
                    or set(node.column_ids) != set(bound[node.id].column_ids)
                    or set(node.roles) != expected_roles[node.id]
                ):
                    raise ValueError(
                        f"Observed graph node {node.id} must preserve its established binding and roles. "
                        f"Expected columns: {bound[node.id].column_ids if node.id in bound else 'unbound'}. "
                        f"Expected roles: {sorted(expected_roles.get(node.id, set()))}; received: {node.roles}."
                    )
            elif node.roles != ["latent"]:
                raise ValueError("Unmeasured nodes must be explicitly latent.")
        required = {
            key
            for key, roleset in expected_roles.items()
            if not roleset <= {"irrelevant", "identifier", "time", "unknown"}
        }
        if value.status == "ready" and not required <= nodes.keys():
            raise ValueError(
                "The graph must include the target, all treatments and scientifically relevant variables."
            )
        edges: set[tuple[str, str]] = set()
        dependencies: dict[str, set[str]] = {key: set() for key in nodes}
        for edge in value.edges:
            pair = (edge.source, edge.target)
            if (
                edge.source not in nodes
                or edge.target not in nodes
                or edge.source == edge.target
                or pair in edges
            ):
                raise ValueError("Edges must uniquely connect distinct declared graph nodes.")
            edges.add(pair)
            dependencies[edge.target].add(edge.source)
        tuple(TopologicalSorter(dependencies).static_order())
        ancestors: dict[str, set[str]] = {node: set() for node in nodes}
        for identity in TopologicalSorter(dependencies).static_order():
            for parent in dependencies[identity]:
                ancestors[identity].update({parent, *ancestors[parent]})
        treatment_ids = {item.variable.id for item in treatments.treatments}
        outcome_id = target.outcome.id if target.outcome else ""
        for assignment in roles.assignments:
            identity = assignment.variable.id
            if identity not in nodes:
                continue
            assigned = set(assignment.roles)
            scoped_treatments = set(assignment.treatment_ids) if scientific_version >= 8 else treatment_ids
            if assignment.timing == "before_treatment" and scientific_version >= 8 and ancestors[identity] & scoped_treatments:
                raise ValueError("The graph makes a scoped pre-treatment measurement a descendant of that treatment.")
            if scientific_version >= 8:
                # Role labels explain a proposal; effect-specific identification below grants authority.
                # A sufficient adjustment variable need not itself be a common cause of T and Y.
                continue
            if "confounder" in assigned and not (
                identity in ancestors.get(outcome_id, set())
                and any(identity in ancestors[t] for t in scoped_treatments)
            ):
                raise ValueError(
                    "A confounder role requires a common-cause path to a selected treatment and the outcome."
                )
            if "mediator" in assigned and not (
                ancestors[identity] & scoped_treatments and identity in ancestors.get(outcome_id, set())
            ):
                raise ValueError("A mediator role requires a treatment-to-variable-to-outcome path.")
            if "collider" in assigned and len(dependencies[identity]) < 2:
                raise ValueError("A collider role requires converging causal arrows.")
            if "instrument" in assigned and not any(identity in ancestors[t] for t in scoped_treatments):
                raise ValueError("An instrument role requires a causal path to treatment.")
        if scientific_version >= 8 and value.status == "ready":
            effects = {effect.id: effect for effect in treatments.effects}
            if sorted(item.effect_id for item in value.identifications) != sorted(effects):
                raise ValueError("Each requested effect needs exactly one explicit identification claim.")
            identifications = {item.effect_id: item for item in value.identifications}
            component_dependencies: dict[str, set[str]] = {identity: set() for identity in effects}
            for identity, effect in effects.items():
                comparison = effect.comparison
                aggregate = comparison.aggregate if comparison and comparison.kind == "aggregate" else None
                if identifications[identity].strategy == "aggregate" and (scientific_version < 15 or aggregate is None):
                    raise ValueError("Aggregate identification requires the exact accepted typed aggregate comparison, never an ordinary effect or prose-derived component list.")
                if scientific_version >= 15 and comparison and comparison.kind == "aggregate":
                    if aggregate is None or identifications[identity].strategy != "aggregate":
                        raise ValueError("A frozen aggregate comparison requires aggregate identification through its exact component effects.")
                    components = [term.effect_id for term in aggregate.terms]
                    weights = [term.weight for term in aggregate.terms]
                    if len(components) != len(set(components)) or identity in components or not set(components) <= effects.keys():
                        raise ValueError("Aggregate identification requires distinct existing component effects and cannot depend on itself.")
                    if {tid for component in components for tid in effects[component].treatment_ids} != set(effect.treatment_ids):
                        raise ValueError("Aggregate identification must preserve exactly its component treatment scope.")
                    if any(not math.isfinite(weight) or not 0 < weight <= 1 for weight in weights) or abs(math.fsum(weights) - 1) > 1e-12:
                        raise ValueError("Aggregate identification requires the frozen positive weights to sum to one.")
                    component_dependencies[identity] = set(components)
            validated: set[str] = set()
            aggregate_details: list[str] = []
            for identity in TopologicalSorter(component_dependencies).static_order():
                identification = identifications[identity]
                selected = set(effects[identification.effect_id].treatment_ids)
                adjustment = set(identification.adjustment)
                if len(adjustment) != len(identification.adjustment) or not adjustment <= bound.keys():
                    raise ValueError("Identification adjustment uses distinct observed variable identities.")
                eligible = {r.variable.id for r in roles.assignments
                    if identification.effect_id in r.effect_ids and not set(r.roles) <= {"irrelevant", "unknown", "identifier"}}
                eligible.update(treatment_ids - selected)
                if not adjustment <= eligible:
                    raise ValueError("Identification adjustment must retain a reviewed role for this exact effect, or another established treatment factor.")
                obligations = {item.name: item.claim for item in identification.obligations}
                if len(obligations) != len(identification.obligations) or any(claim.basis == "unknown" for claim in obligations.values()):
                    raise ValueError("Identification obligations must be distinct and supported or explicitly assumed.")
                if identification.evidence.basis == "unknown":
                    raise ValueError("Identification needs supplied evidence or a declared conditional assumption.")
                if identification.strategy == "aggregate":
                    if adjustment or obligations:
                        raise ValueError("Aggregate identification inherits its validated components' conditional premises and cannot add adjustment or new obligations.")
                    if not component_dependencies[identity] or not component_dependencies[identity] <= validated:
                        raise ValueError("Every aggregate component must first pass its own identification criterion under this same graph.")
                    comparison = effects[identity].comparison
                    assert comparison is not None and comparison.aggregate is not None
                    terms = ", ".join(f"{term.effect_id} (weight {term.weight!r})" for term in comparison.aggregate.terms)
                    aggregate_details.append(f"{identity}: {terms}")
                if identification.strategy == "backdoor" and not backdoor_valid(set(nodes), edges, selected, outcome_id, adjustment):
                    raise ValueError(f"Effect {identification.effect_id}: the declared adjustment does not satisfy the backdoor criterion in the accepted graph.")
                if identification.strategy == "structural_zero" and ancestors.get(outcome_id, set()) & selected:
                    raise ValueError("A structural-zero claim conflicts with a directed treatment-to-outcome path.")
                if identification.strategy == "source_assignment":
                    processes = [process for process in treatments.assignments if selected & set(process.treatment_ids)]
                    covered = Counter(identity for process in processes for identity in process.treatment_ids if identity in selected)
                    if not selected or any(covered[identity] != 1 for identity in selected):
                        raise ValueError("Source-assignment identification requires exactly one accepted assignment process for every selected treatment factor.")
                    for process in processes:
                        source_refs = set(process.mechanism.references)
                        if (
                            process.kind not in {"point", "joint", "absorbing_adoption"}
                            or process.mechanism.basis != "supplied"
                            or not source_refs or not source_refs <= roots
                        ):
                            raise ValueError("Source-assignment identification requires a supported process with a directly supplied assignment mechanism for every selected factor.")
                        if not source_refs & set(identification.evidence.references):
                            raise ValueError("Source-assignment identification must directly cite supplied mechanism evidence for each selected assignment process.")
                    # Conditional evidence retains its ledger and authority. Provenance
                    # checks validate every dependency; this source linkage does not
                    # establish the truth of a natural-language randomization account.
                    if not backdoor_valid(set(nodes), edges, selected, outcome_id, adjustment):
                        raise ValueError("Source-assignment identification contradicts an open confounding path in the accepted graph.")
                if identification.strategy == "did":
                    if not {"parallel_trends", "no_anticipation", "comparison_support"} <= obligations.keys():
                        raise ValueError("DiD must record parallel_trends, no_anticipation and comparison_support obligations.")
                    if not any(p.kind == "absorbing_adoption" and selected <= set(p.treatment_ids) for p in treatments.assignments):
                        raise ValueError("The implemented DiD timing scope requires an absorbing-adoption assignment account.")
                    operational = {use for item in roles.assignments if identification.effect_id in item.effect_ids for use in item.operational_use}
                    if not {"unit_key", "time_key"} <= operational:
                        raise ValueError("DiD requires effect-scoped unit and time identities.")
                if identification.strategy == "discontinuity" and (
                    "cutoff_rule" not in obligations or not {"continuity", "local_randomization"} & obligations.keys()
                ):
                    raise ValueError("Discontinuity needs cutoff_rule and continuity or local_randomization obligations.")
                comparison = effects[identity].comparison
                if (scientific_version >= 15 and comparison and comparison.discontinuity
                        and comparison.discontinuity.estimand == "local_wald"):
                    required = {"cutoff_rule", "exclusion", "monotonicity", "relevance"}
                    if not required <= obligations.keys() or not {"continuity", "local_randomization"} & obligations.keys():
                        raise ValueError("A typed local_wald receipt effect requires cutoff_rule, local eligibility exclusion, monotonicity, relevance, and continuity or local_randomization obligations. Declare supported or conditional premises; a numerical ratio does not establish them.")
                if identification.strategy == "other" and not obligations:
                    raise ValueError("Other identification arguments require explicit conditional obligations; no generic graph certificate is implied.")
                validated.add(identity)
            _relationship_checks(value, context, bound, effects, edges, ancestors)
            if aggregate_details:
                return "Validated. Aggregate identification closes over components validated under this graph, retaining their conditional premises: " + "; ".join(aggregate_details) + ". This checks structure and declared obligations, not source entailment."
        return None

    if isinstance(value, CausalAccount):
        check("causal_graph_consistency", causal)

    def specification() -> None:
        if not isinstance(value, Plan) or value.status != "ready":
            return
        treatments = Treatments.model_validate(prior["treatments"])
        if sorted(item.effect_id for item in value.effects) != sorted(item.id for item in treatments.effects):
            raise ValueError("The plan must implement every frozen effect exactly once.")
        roles = Roles.model_validate(prior["roles"])
        permissible = {
            item.variable.id
            for item in roles.assignments
            if not set(item.roles) & {"irrelevant", "unknown", "identifier"}
        }
        if scientific_version >= 8:
            permissible.update(item.variable.id for item in treatments.treatments)
        for effect in value.effects:
            if (
                len(effect.adjustment) != len(set(effect.adjustment))
                or not set(effect.adjustment) <= permissible
            ):
                raise ValueError(
                    "Adjustment must name distinct, established relevant variables; justify its suitability for this estimand."
                )
            if scientific_version >= 8:
                causal_account = CausalAccount.model_validate(prior["causal"])
                identification = next((item for item in causal_account.identifications if item.effect_id == effect.effect_id), None)
                if identification is None or set(effect.adjustment) != set(identification.adjustment):
                    raise ValueError("The executable plan must retain its effect's frozen identifying adjustment, not reselect it after diagnostics.")
                if effect.result_kind != "estimate" and not effect.result_reason.strip():
                    raise ValueError("A nonstandard result shape needs its prespecified scientific reason.")
                if effect.result_kind == "structural_zero" and identification.strategy != "structural_zero":
                    raise ValueError("A structural-zero result requires the frozen no-directed-path identification.")

    if isinstance(value, Plan):
        check("fixed_scientific_scope", specification)
    outcomes.append(
        GateCheck(
            rule="semantic_sufficiency",
            kind="semantic",
            passed=value.status == "ready",
            detail=value.summary
            + (" Unresolved: " + "; ".join(value.unresolved) if value.unresolved else ""),
        )
    )
    return outcomes


def _preperiod_access(
    frame: Frame, context: ContextContent, rows: list[list[str]], selected_treatments: set[str],
    scientific_version: int = 7,
) -> tuple[list[bool] | None, dict[str, Any] | None]:
    """Validate an explicitly supplied numeric adoption map without inspecting outcome cells."""
    processes = [p for p in frame.treatments.assignments if set(p.treatment_ids) & selected_treatments]
    if not any(p.diagnostic_window is not None for p in processes):
        return None, None
    if any(p.kind != "absorbing_adoption" or p.diagnostic_window is None for p in processes):
        raise ValueError("Preperiod access needs a supplied adoption window for every selected assignment process.")
    registry = variables(frame.target, frame.treatments, frame.roles)
    positions = {cid: column.identity.position for cid, column in context.columns.items()}
    target_columns = set(frame.target.outcome.column_ids) if frame.target.outcome else set()
    unsafe = target_columns | {
        cid for role in frame.roles.assignments
        if role.outcome_proxy is True or (role.proxy_evidence is not None and role.proxy_evidence.basis == "unknown" and role.proxy_evidence.references)
        for cid in role.variable.column_ids
    }
    known_paths, possible_paths = lineage.outcome_paths(context, frame.target, scientific_version=scientific_version)
    unsafe.update(known_paths)
    unsafe.update(possible_paths)

    def position(identity: str, operational_use: str | None = None) -> int:
        variable = registry.get(identity)
        if variable is None or len(variable.column_ids) != 1 or variable.derivation or variable.column_ids[0] not in positions:
            raise ValueError("Adoption metadata must bind one directly observed source column.")
        if variable.column_ids[0] in unsafe:
            raise ValueError("Outcome measurements or unresolved outcome lineage cannot serve as adoption metadata.")
        selected_assignment = operational_use == "assignment_key" and identity in selected_treatments
        if operational_use and not selected_assignment and not any(
            r.variable.id == identity and operational_use in r.operational_use and (authority.established(r.account) if scientific_version >= 19 else r.account.basis == "supplied")
            for r in frame.roles.assignments
        ):
            raise ValueError(f"Adoption metadata needs its supplied {operational_use} role.")
        return positions[variable.column_ids[0]]

    def numeric(raw: str) -> float:
        try:
            value = float(raw)
        except ValueError as error:
            raise ValueError("Diagnostic adoption periods require exact finite numeric mapping; dates and missing periods are not inferred.") from error
        if not math.isfinite(value):
            raise ValueError("Diagnostic adoption periods require exact finite numeric mapping.")
        return value

    mask = [True] * len(rows)
    bindings: set[tuple[str, str]] = set()
    receipts = []
    for process in processes:
        window = process.diagnostic_window
        assert window is not None
        if not math.isfinite(window.cutoff) or any(c.basis != "supplied" or not c.references for c in (window.evidence, window.history_evidence)):
            raise ValueError("Preperiod outcome access requires supplied timing and historical-measurement evidence.")
        if window.assignment_encoding == "group" and window.adoption_variable_id is None:
            raise ValueError("A fixed treatment-group label cannot establish adoption timing. Group-encoded diagnostic access requires an explicit adoption_variable_id.")
        selected_factors = set(process.treatment_ids) & selected_treatments
        if window.assignment_variable_id is None:
            if scientific_version >= 15:
                raise ValueError("Preperiod access requires its explicitly bound assignment_variable_id; an indicator is never inferred from column names or cohort values.")
            assignment_ids = sorted(selected_factors)
        else:
            assignment_ids = [window.assignment_variable_id]
            if window.assignment_variable_id not in selected_factors:
                if (
                    len(process.treatment_ids) != 1
                    or selected_factors != {window.adoption_variable_id}
                    or window.assignment_encoding != "indicator"
                ):
                    raise ValueError("A separate adoption indicator is supported only for one selected adoption/cohort factor with an explicit indicator encoding.")
                if window.assignment_variable_id not in process.history_variables:
                    raise ValueError("The separate adoption indicator must belong to the accepted assignment history registry.")
            elif len(selected_factors) != 1:
                raise ValueError("One assignment-variable mapping cannot represent several selected treatment factors.")
        bindings.add((window.unit_variable_id, window.time_variable_id))
        if len(bindings) > 1:
            raise ValueError("This diagnostic projection requires one shared unit/time mapping across selected assignments.")
        unit_index = position(window.unit_variable_id, "unit_key")
        time_index = position(window.time_variable_id, "time_key")
        if unit_index == time_index:
            raise ValueError("Unit identity and measurement time require distinct source columns.")
        treated, controls = set(window.treated_values), set(window.control_values)
        if not treated or not controls or treated & controls:
            raise ValueError("Adoption assignment codes must be explicit and disjoint.")
        times = [numeric(row[time_index]) for row in rows]
        units = [row[unit_index] for row in rows]
        if any(not unit for unit in units) or len(set(zip(units, times, strict=True))) != len(rows):
            raise ValueError("Preperiod access needs nonmissing units and unique unit-period observations.")
        earliest: list[float] = []
        for treatment_id in assignment_ids:
            treatment_index = position(treatment_id)
            if treatment_index in {unit_index, time_index}:
                raise ValueError("The assignment indicator must be distinct from unit and measurement-time keys.")
            histories: dict[str, list[tuple[float, bool]]] = {}
            for row, unit, time in zip(rows, units, times, strict=True):
                code = row[treatment_index]
                if code not in treated | controls:
                    raise ValueError("An observed assignment value is outside the supplied adoption coding.")
                histories.setdefault(unit, []).append((time, code in treated))
            for history in histories.values():
                ordered = sorted(history)
                if window.assignment_encoding == "indicator":
                    on = False
                    for time, current in ordered:
                        if on and not current:
                            raise ValueError("Observed treatment switching violates the frozen absorbing-adoption account.")
                        if current:
                            earliest.append(time)
                            on = True
                elif len({value for _, value in ordered}) != 1:
                    raise ValueError("A supplied fixed treatment-group label changes within a unit.")
        if window.adoption_variable_id:
            adoption_index = position(window.adoption_variable_id, "assignment_key")
            by_unit: dict[str, set[str]] = {}
            for row, unit in zip(rows, units, strict=True):
                by_unit.setdefault(unit, set()).add(row[adoption_index])
            for unit, values in by_unit.items():
                if len(values) != 1:
                    raise ValueError("Adoption periods must agree within each unit.")
                raw = next(iter(values))
                adoption = None if raw in window.never_adoption_values else numeric(raw)
                if adoption is not None:
                    earliest.append(adoption)
                for treatment_id in assignment_ids:
                    treatment_index = position(treatment_id)
                    for row, row_unit, time in zip(rows, units, times, strict=True):
                        if row_unit != unit:
                            continue
                        on = row[treatment_index] in treated
                        if window.assignment_encoding == "indicator" and on != (adoption is not None and time >= adoption):
                            raise ValueError("The observed assignment indicator contradicts its supplied adoption-period mapping.")
                        if adoption is None and on:
                            raise ValueError("A never-adoption code contradicts observed treatment assignment.")
        if earliest and window.cutoff > min(earliest):
            raise ValueError("The diagnostic cutoff exceeds the earliest observed adoption; later outcomes must remain withheld for every unit.")
        mask = [allowed and time < window.cutoff for allowed, time in zip(mask, times, strict=True)]
        receipts.append({"assignment_id": process.id, "cutoff": window.cutoff,
            "treatment_ids": sorted(selected_factors), "assignment_variable_ids": assignment_ids,
            "unit_variable_id": window.unit_variable_id, "time_variable_id": window.time_variable_id,
            "assignment_encoding": window.assignment_encoding, "adoption_variable_id": window.adoption_variable_id,
            "references": sorted(set(window.evidence.references + window.history_evidence.references))})
    if not any(mask):
        raise ValueError("No source rows precede the verified earliest-adoption cutoff.")
    return mask, {"rule": "strictly_before_every_selected_adoption", "windows": receipts, "visible_rows": sum(mask), "masked_rows": len(rows) - sum(mask)}


def execution_projection(
    context: ContextContent, raw: bytes, manifest: InputProjection | dict[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    """Apply only the approved column mapping, preserving every row and cell string.

    The caller verifies Snapshot.source_sha256 before this post-approval operation.
    No values are interpreted, filtered, converted, imputed or summarized here.
    """
    manifest = InputProjection.model_validate(manifest)
    ordered_source = sorted((item.identity for item in context.columns.values()), key=lambda item: item.position)
    if (
        [item.position for item in ordered_source] != list(range(len(ordered_source)))
        or len({item.name for item in ordered_source}) != len(ordered_source)
        or any(key != item.identity.column_id for key, item in context.columns.items())
    ):
        raise ValueError("The source registry must contain exact unique column names, identities and contiguous positions.")
    if len({item.column_id for item in manifest.columns}) != len(manifest.columns):
        raise ValueError("The execution projection cannot repeat a selected column identity.")
    for column in manifest.columns:
        if column.column_id not in context.columns or context.columns[column.column_id].identity != column:
            raise ValueError("The frozen execution projection differs from the bound source column registry.")
    selected = sorted(manifest.columns, key=lambda item: item.column_id)
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow([column.column_id for column in selected])
    rows = 0
    try:
        reader = csv.reader(io.StringIO(raw.decode("utf-8-sig"), newline=""), strict=True)
        if next(reader, []) != [column.name for column in ordered_source]:
            raise ValueError("Source schema differs from the bound dataset version.")
        for row in reader:
            if len(row) != len(ordered_source):
                raise ValueError("Source rows differ from the bound dataset schema.")
            writer.writerow([row[column.position] for column in selected])
            rows += 1
    except (UnicodeError, csv.Error) as error:
        raise ValueError("The source is not a valid UTF-8 CSV.") from error
    projected = output.getvalue().encode("utf-8")
    return projected, {
        "policy": manifest.policy,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "projection_sha256": hashlib.sha256(projected).hexdigest(),
        "manifest_hash": fingerprint(manifest.model_dump(mode="json")),
        "rows": rows,
        "columns": [column.column_id for column in selected],
        "column_count": len(selected),
        "source_column_count": len(ordered_source),
        "row_policy": "preserve_all_rows",
        "value_policy": "preserve_decoded_csv_cell_strings",
    }


def _baseline_access(frame: Frame, context: ContextContent, effect_ids: set[str], scientific_version: int = 8) -> dict[str, dict[str, Any]]:
    """Combine supplied timing proofs only across exact frozen measurement identities."""
    registry = variables(frame.target, frame.treatments, frame.roles)
    required = {(effect.id, identity) for effect in frame.treatments.effects if effect.id in effect_ids for identity in effect.treatment_ids}
    event_requirements: dict[str, set[tuple[str, str, str]]] = {}
    if scientific_version >= 15:
        for scope in role_scopes(context, frame.target, frame.treatments):
            event_requirements.setdefault(scope["column_id"], set()).update(
                (effect, treatment, fingerprint(scope["event"]))
                for effect, identities in scope["effect_treatments"].items() if effect in effect_ids
                for treatment in identities
            )
    roots = supplied(context)
    access: dict[str, dict[str, Any]] = {}
    for identity, variable in registry.items():
        assignments = [item for item in frame.roles.assignments if item.variable.id == identity and set(item.effect_ids) & effect_ids]
        if not assignments or not any(not set(item.roles) <= {"irrelevant", "identifier", "unknown"} for item in assignments):
            continue
        if any(
            item.timing != "before_treatment" or (not authority.established(item.timing_evidence) if scientific_version >= 19 else item.timing_evidence.basis != "supplied")
            or not item.timing_evidence.references or not set(item.timing_evidence.references) <= roots
            or item.outcome_proxy is not False
            for item in assignments
        ):
            continue
        covered = {(effect, treatment) for item in assignments for effect in set(item.effect_ids) & effect_ids for treatment in item.treatment_ids}
        covered_events = {(effect, treatment, item.assignment_event_key)
                          for item in assignments if item.assignment_event_key is not None
                          for effect in set(item.effect_ids) & effect_ids for treatment in item.treatment_ids}
        for column in variable.column_ids:
            required_events = event_requirements.get(column, set())
            if scientific_version >= 15:
                valid = (bool(required_events) and required_events <= covered_events
                         and all(item.assignment_event_key is not None for item in assignments))
            else:
                valid = required <= covered
            if valid:
                access[column] = {
                    "rule": "supplied_pre_treatment_measurement",
                    "references": sorted({ref for item in assignments for ref in item.timing_evidence.references}),
                    "effect_treatment_pairs": [list(pair) for pair in sorted(required)],
                }
                if scientific_version >= 15:
                    access[column]["effect_treatment_events"] = [list(triple) for triple in sorted(required_events)]
    return access


def projection(
    frame: Frame, context: ContextContent, csv_bytes: bytes, *, purpose: str = "feasibility",
    effect_ids: set[str] | None = None,
    scientific_version: int = 7,
) -> tuple[bytes, dict[str, Any]]:
    """Construct a purpose-specific projection from accepted semantics, never model access votes.

    Target-period measurements remain withheld. A supplied, verified numeric adoption window
    may expose earlier cells in a longitudinal target column while retaining every metadata row.
    """
    if purpose not in {"feasibility", "unit_structure", "adoption_structure"}:
        raise ValueError("Unknown diagnostic purpose.")
    if scientific_version >= 8:
        # Historical proxy fields are a readable derived view, never an access grant.
        # Recompute source lineage for this immutable frame; purpose/window rules below own access.
        frame = frame.model_copy(deep=True)
        frame.roles = lineage.classify(frame.roles, context, frame.target)
    requested = {effect.id for effect in frame.treatments.effects}
    selected_effects = requested if effect_ids is None else effect_ids
    if not selected_effects or not selected_effects <= requested:
        raise ValueError("Diagnostic scope must name requested effects.")
    assert frame.target.outcome
    selected_treatments = {identity for effect in frame.treatments.effects if effect.id in selected_effects for identity in effect.treatment_ids}
    treatment_columns = {col for item in frame.treatments.treatments if item.variable.id in selected_treatments for col in item.variable.column_ids}
    protected = set(frame.target.outcome.column_ids)
    if scientific_version >= 8:
        known_lineage, possible_lineage = lineage.outcome_paths(context, frame.target, scientific_version=scientific_version)
        protected.update(treatment_columns & (known_lineage.keys() | possible_lineage.keys()))
    operational: set[str] = set()
    group_keys: set[str] = set()
    access: dict[str, dict[str, Any]] = {}
    for item in frame.roles.assignments:
        scoped = not item.effect_ids or bool(set(item.effect_ids) & selected_effects)
        if not scoped:
            continue
        unresolved_lineage = item.proxy_evidence is not None and item.proxy_evidence.basis == "unknown" and bool(item.proxy_evidence.references)
        if item.operational_use and (authority.established(item.account) if scientific_version >= 19 else item.account.basis == "supplied" and item.account.references) and item.outcome_proxy is not True and not unresolved_lineage:
            operational.update(item.variable.column_ids)
            if set(item.operational_use) <= {"unit_key", "cluster_key"}:
                group_keys.update(item.variable.column_ids)
            for column in item.variable.column_ids:
                access[column] = {"rule": "supplied_operational_metadata", "uses": item.operational_use, "references": item.account.references}
        elif item.outcome_proxy is not False or item.timing in {"after_treatment", "unknown"}:
            # A selected factor remains accessible for its own assignment checks.
            # Its role relative to another event does not override that authority;
            # actual or qualified target lineage still withholds it.
            exempt = treatment_columns if scientific_version >= 8 and item.outcome_proxy is not True and not unresolved_lineage else set()
            protected.update(set(item.variable.column_ids) - exempt)
    allowed = set(treatment_columns) if purpose != "unit_structure" else set()
    for column in allowed:
        access[column] = {"rule": "selected_treatment", "references": []}
    allowed.update(operational)
    if purpose == "feasibility" and scientific_version >= 8:
        baseline = _baseline_access(frame, context, selected_effects, scientific_version)
        allowed.update(baseline)
        access.update(baseline)
    elif purpose == "feasibility":
        for item in frame.roles.assignments:
            if (
                (not item.effect_ids or bool(set(item.effect_ids) & selected_effects))
                and (not item.treatment_ids or selected_treatments <= set(item.treatment_ids))
                and item.timing == "before_treatment" and item.timing_evidence.basis == "supplied"
                and item.timing_evidence.references and item.outcome_proxy is False
                and not set(item.roles) <= {"irrelevant", "identifier"}
            ):
                allowed.update(item.variable.column_ids)
                for column in item.variable.column_ids:
                    access[column] = {"rule": "supplied_pre_treatment_measurement", "references": item.timing_evidence.references}
    longitudinal = purpose != "unit_structure" and any(p.diagnostic_window is not None for p in frame.treatments.assignments if set(p.treatment_ids) & selected_treatments)
    historical_columns = set(frame.target.outcome.column_ids) if longitudinal else set()
    selected = sorted((allowed - protected) | historical_columns)
    if not selected:
        raise ValueError(
            "No safe diagnostic input projection can be established from supplied timing and outcome lineage."
        )
    # Column identities encode source positions in ContextContent, but code only sees canonical IDs.
    reader = csv.reader(io.StringIO(csv_bytes.decode("utf-8-sig")))
    header = next(reader)
    ordered = sorted(context.columns.values(), key=lambda item: item.identity.position)
    if header != [item.identity.name for item in ordered]:
        raise ValueError("Source schema differs from the bound dataset version.")
    source_rows = list(reader)
    if any(len(row) != len(header) for row in source_rows):
        raise ValueError("Source rows differ from the bound dataset schema.")
    preperiod, window_manifest = _preperiod_access(frame, context, source_rows, selected_treatments, scientific_version) if longitudinal else (None, None)
    if window_manifest is not None:
        registry = variables(frame.target, frame.treatments, frame.roles)
        for window in window_manifest["windows"]:
            for identity in window["assignment_variable_ids"]:
                column = registry[identity].column_ids[0]
                # Only the verified assignment map grants this metadata access.
                # It does not turn an indicator into a baseline covariate or a
                # general operational identifier, and position() rejects actual
                # and qualified outcome lineage before reaching this branch.
                selected = sorted(set(selected) | {column})
                access[column] = {"rule": "verified_adoption_assignment_metadata", "window": window}
    for column in historical_columns:
        access[column] = {"rule": "supplied_target_history_before_adoption", "window": window_manifest}
    identities = [item.identity.column_id for item in ordered]
    indices = [identities.index(cid) for cid in selected]
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(selected)
    rows = 0
    groups: dict[str, dict[str, str]] = {cid: {} for cid in group_keys & set(selected)}
    for row_number, row in enumerate(source_rows):
        values = []
        for cid, index in zip(selected, indices, strict=True):
            value = row[index]
            if cid in historical_columns and (preperiod is None or not preperiod[row_number]):
                value = ""
            if cid in groups and value:
                value = groups[cid].setdefault(value, f"unit_{len(groups[cid])}")
            values.append(value)
        writer.writerow(values)
        rows += 1
    raw = output.getvalue().encode()
    return raw, {
        "columns": selected,
        "withheld": sorted(set(identities) - set(selected)),
        "rows": rows,
        "sha256": __import__("hashlib").sha256(raw).hexdigest(),
        "graph_hash": frame.graph_hash,
        "policy": "purpose_scoped_treatment_baseline_and_operational_metadata",
        "purpose": purpose,
        "effect_ids": sorted(selected_effects),
        "access": {column: access[column] for column in selected},
        "pseudonymized_group_keys": sorted(groups),
        "preperiod_window": window_manifest,
    }


def validate_program(program: Program) -> dict[str, Any]:
    ast.parse(program.code)
    if "facts.json" not in program.code:
        raise code_checks.ProgramCheckError("The script must write facts.json.")
    return code_checks.verify(program.code)


def validate_repair(previous: str, proposed: str) -> None:
    """Conservatively lock numerical expressions and control flow during code-only repair.

    This supplements scientific review, not a proof of Python equivalence. Repairs that
    require altering these surfaces must stop rather than silently redesign the program.
    """

    def signature(code: str) -> list[str]:
        tree = ast.parse(code)
        # A missing literal column manifest is a binding repair, not a numerical rule.
        tree.body = [
            node
            for node in tree.body
            if not (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "input_columns"
                and isinstance(node.value, (ast.List, ast.Dict, ast.Tuple))
            )
        ]
        locked: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.Compare, ast.BoolOp, ast.IfExp)):
                locked.append(ast.dump(node))
            elif isinstance(node, (ast.If, ast.While, ast.Assert)):
                locked.append(type(node).__name__ + ast.dump(node.test))
            elif isinstance(node, (ast.For, ast.comprehension)):
                locked.append(type(node).__name__ + ast.dump(node.iter))
            elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                locked.append(repr(node.value))
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                locked.append(ast.dump(node.func))
        return sorted(locked)

    if signature(previous) != signature(proposed):
        raise ValueError(
            "Code-only repair changed numerical expressions, constants, method calls or control flow. "
            "Preserve these exactly; fix only the reported programming failure. Statistical robustness changes require a different scientific specification."
        )


def execution_error(facts: dict[str, float], effects: Treatments, plan: Plan | None = None) -> str:
    """Validate declared result alternatives without inventing point estimates or ordinary intervals."""
    requirements = {effect.effect_id: effect for effect in plan.effects} if plan else {}
    for effect in effects.effects:
        requirement = requirements.get(effect.id)
        kind = requirement.result_kind if requirement else "estimate"
        keys = {name: f"{effect.id}_{name}" for name in ("estimate", "ci_low", "ci_high", "n")}
        if any(not math.isfinite(value) for key, value in facts.items() if key.startswith(effect.id + "_")):
            return f"Non-finite declared result for effect {effect.id}."
        if kind == "structural_zero":
            if facts.get(keys["estimate"]) != 0 or keys["ci_low"] in facts or keys["ci_high"] in facts:
                return f"Effect {effect.id} requires its graph-implied zero without a fabricated sampling interval."
            continue
        if keys["n"] not in facts or facts[keys["n"]] <= 0 or not float(facts[keys["n"]]).is_integer():
            return f"Invalid sample count for effect {effect.id}."
        if kind == "confidence_set":
            prefix = effect.id + "_"
            status, count = facts.get(prefix + "confidence_status"), facts.get(prefix + "region_count")
            if status not in {0, 1, 2, 3} or count is None or not float(count).is_integer() or not 0 <= count <= 100:
                return f"Invalid confidence-set status or region count for effect {effect.id}."
            if (status == 0 and count != 1) or (status == 1 and count < 2) or (status == 2 and count != 0):
                return f"Confidence-set regions disagree with the declared shape for effect {effect.id}."
            previous: float | None = None
            for index in range(int(count)):
                lo, hi = facts.get(f"{prefix}region_{index}_low"), facts.get(f"{prefix}region_{index}_high")
                if lo is None or hi is None or lo > hi or (previous is not None and lo <= previous):
                    return f"Confidence-set regions must be ordered and disjoint for effect {effect.id}."
                previous = hi
            if status == 3:
                low, high = facts.get(prefix + "searched_low"), facts.get(prefix + "searched_high")
                if low is None or high is None or low >= high:
                    return f"An unresolved-tail confidence set needs its finite searched range for effect {effect.id}."
                if count and (facts[prefix + "region_0_low"] < low or facts[f"{prefix}region_{int(count) - 1}_high"] > high):
                    return f"Confidence-set regions exceed the searched range for effect {effect.id}."
            if keys["ci_low"] in facts or keys["ci_high"] in facts:
                return f"Effect {effect.id} must preserve its confidence set instead of an ordinary interval."
            continue
        if keys["estimate"] not in facts:
            return f"Missing declared results for effect {effect.id}."
        if kind == "estimate_without_inference":
            if keys["ci_low"] in facts or keys["ci_high"] in facts:
                return f"Effect {effect.id} declared unavailable inference but returned an interval."
            continue
        if not {keys["ci_low"], keys["ci_high"]} <= facts.keys():
            return f"Missing declared results for effect {effect.id}."
        if facts[keys["ci_low"]] > facts[keys["ci_high"]]:
            return f"Invalid sample count or interval for effect {effect.id}."
    return ""


def repair_error(execution: dict[str, Any]) -> str | None:
    """An allowlist of programming failures; no result values or traceback locals reach repair."""
    if execution.get("ok") or execution.get("facts"):
        return None
    lines = str(execution.get("stderr", "")).splitlines()
    for line in reversed(lines):
        match = re.match(
            r"^(SyntaxError|IndentationError|NameError|AttributeError|ImportError|ModuleNotFoundError|TypeError):",
            line.strip(),
        )
        if match:
            return (
                match.group(1)
                + ": programming failure. Inspect the approved code against the fixed specification; runtime output and values are withheld."
            )
    return None


def graph_hash(causal: CausalAccount) -> str:
    return fingerprint(causal.model_dump(mode="json"))
