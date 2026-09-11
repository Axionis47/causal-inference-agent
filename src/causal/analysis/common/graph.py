"""Bounded navigation projected from the installed method definitions."""
from __future__ import annotations

import base64
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast, get_args

from causal.analysis.common.candidate import (
    CandidateDraft,
    CapabilityExploration,
    CapabilityGraph,
    CapabilityRequestError,
    GraphCoverage,
    GraphEdge,
    GraphNode,
    OptionState,
    Relation,
)
from causal.analysis.common.catalog import METHODS, method_module
from causal.analysis.common.definitions import (
    DiagnosticDefinition,
    MethodDefinition,
    RequirementDefinition,
    RoleDefinition,
    SensitivityDefinition,
)
from causal.analysis.common.evaluation import digest, evaluate_candidate, fixed_value, schema_values
from causal.analysis.common.models import Model

COMMON_REQUIREMENTS = {
    "input": "Partial candidates retain unknown values and reject invalid submitted input.",
    "method": "Nominate a supported method; inspecting a method never selects it.",
    "population": "Declare the scientific population; data preparation must preserve it.",
    "unit_grain": "Declare what one observational unit and row represent.",
    "outcome.column": "Bind the declared outcome to its expected executable column.",
    "outcome.kind": "The implemented outcome kind is continuous or binary.",
    "outcome.units": "State the measurement units and interpretation of the outcome.",
    "seed": "The mechanical default seed is 0; an explicit seed must be in [0, 2**32 - 2].",
    "facts": "Each assertion has one name, a value or explicit unknown, and declared support.",
    "estimand": "The desired scientific estimand and executable estimand must agree.",
    "role_separation": "The outcome cannot also be assigned to an input role.",
    "bindings": "Declared source bindings and prepared aliases must match executable variables.",
    "population_policy": "Any caller-declared population policy is frozen with the complete candidate.",
    "missingness_policy": "Any caller-declared scientific missingness policy is frozen before preparation.",
}


def validate_definition(definition: MethodDefinition) -> None:
    """Reject ambiguous IDs, dangling dependency references and prerequisite cycles."""
    requirements = {rule.id: rule for rule in definition.requirements}
    if len(requirements) != len(definition.requirements):
        raise ValueError("Duplicate requirement identifiers.")
    for collection in (definition.roles, definition.diagnostics, definition.sensitivities, definition.fixed_policies):
        if len({entry.id for entry in collection}) != len(collection):
            raise ValueError("Duplicate capability identifiers.")
    entries: tuple[RequirementDefinition | RoleDefinition | DiagnosticDefinition | SensitivityDefinition, ...] = (
        *definition.requirements, *definition.roles, *definition.diagnostics, *definition.sensitivities)
    for entry in entries:
        if any(dependency not in requirements for dependency in entry.dependencies):
            raise ValueError(f"Dangling requirement dependency on {entry.id!r}.")
    fields = definition.specification.model_fields
    if len(dict(definition.option_dependencies)) != len(definition.option_dependencies):
        raise ValueError("Duplicate option dependency declarations.")
    for name, dependencies in definition.option_dependencies:
        if name.split("=", 1)[0] not in fields or any(dep not in requirements for dep in dependencies):
            raise ValueError(f"Dangling option dependency on {name!r}.")
    for role in definition.roles:
        if role.field.removeprefix("configuration.") not in fields:
            raise ValueError(f"Unknown role field {role.field!r}.")
    for rule in definition.requirements:
        for field in (*rule.fields, *rule.scope_fields):
            if field.startswith("configuration.") and field.removeprefix("configuration.") not in fields:
                raise ValueError(f"Dangling requirement target field {field!r}.")
    active: set[str] = set()
    done: set[str] = set()

    def visit(identifier: str) -> None:
        if identifier in active:
            raise ValueError("Prerequisite cycles cannot define an enterable branch.")
        if identifier in done:
            return
        active.add(identifier)
        for dependency in requirements[identifier].dependencies:
            visit(dependency)
        active.remove(identifier)
        done.add(identifier)

    for identifier in requirements:
        visit(identifier)


def build_graph(definitions: Sequence[MethodDefinition] | None = None) -> CapabilityGraph:
    installed = tuple(definitions) if definitions is not None else tuple(
        cast(MethodDefinition, method_module(method).DEFINITION) for method in METHODS)
    versions = tuple((definition.method, definition.version) for definition in installed)
    graph_version = digest({"schema": "analysis-graph.v1", "methods": versions})
    nodes = [GraphNode(node_id="analysis", type="root", title="Analysis capabilities",
                       description="Explore supported causal decisions before nominating a method. A path is not a complete design.",
                       capability_version=graph_version)]
    edges: list[GraphEdge] = []

    def edge(source: str, target: str, relation: Relation) -> None:
        edges.append(GraphEdge(source=source, target=target, relation=relation))

    for name, explanation in COMMON_REQUIREMENTS.items():
        identifier = f"analysis:requirement:{name}"
        nodes.append(GraphNode(node_id=identifier, type="requirement", title=name,
                               description=explanation, capability_version="analysis-evaluation.v1",
                               fields=(name,), boundary="design"))
        edge("analysis", identifier, "requires")
    for definition in installed:
        validate_definition(definition)
        method, version = definition.method, definition.version
        entry = f"method:{method}"
        schema = definition.specification.model_json_schema()
        policies = tuple(f"{field}={description.get('default')}" for field, description in schema["properties"].items()
                         if fixed_value(description))
        policies += tuple(f"{policy.id}={policy.value}" for policy in definition.fixed_policies)
        nodes.append(GraphNode(node_id=entry, type="method", title=definition.title,
                               description=definition.summary + " Fixed implementation policies: " + "; ".join(policies)
                               + ". Unsupported: " + "; ".join(f"{name}: {reason}" for name, reason in definition.unsupported),
                               capability_version=version))
        edge("analysis", entry, "offers")
        for policy in definition.fixed_policies:
            identifier = f"{method}:policy:{policy.id}"
            nodes.append(GraphNode(node_id=identifier, type="policy", title=policy.id.replace("_", " "),
                                   description=policy.description + " This implementation policy cannot be overridden.",
                                   capability_version=version, fields=(f"policy.{policy.id}",),
                                   permitted_values=(policy.value,), input_schema={"const": policy.value},
                                   boundary="design"))
            edge(entry, identifier, "reveals")
        for name in COMMON_REQUIREMENTS:
            edge(entry, f"analysis:requirement:{name}", "requires")
        for rule in definition.requirements:
            identifier = f"{method}:requirement:{rule.id}"
            expectation = None
            if rule.fact_name:
                expectation = ("Declared source references; " + ("explicit assumption permitted" if rule.permits_assumption
                               else "factual evidence required; an assumption cannot substitute"))
                if rule.scope_fields:
                    expectation += "; support scoped to columns in " + ", ".join(rule.scope_fields)
            nodes.append(GraphNode(node_id=identifier, type="requirement", title=rule.id.replace("_", " "),
                                   description=rule.explanation + " Expected: " + rule.expected,
                                   capability_version=version, fields=rule.fields, boundary="design",
                                   evidence_expectation=expectation, requirement_kind=rule.kind,
                                   expected_condition=rule.expected, failure_category=rule.failure_category,
                                   explanation_reference=identifier))
            edge(entry, identifier, "requires")
            for dependency in rule.dependencies:
                edge(identifier, f"{method}:requirement:{dependency}", "requires")
        for name in ("schema", "diagnostics", "sensitivities"):
            identifier = f"{method}:requirement:{name}"
            nodes.append(GraphNode(node_id=identifier, type="requirement", title=name,
                                   description="Use declared fields and unique supported computations.",
                                   capability_version=version, boundary="design"))
            edge(entry, identifier, "requires")
        dependencies = dict(definition.option_dependencies)
        for field, description in schema["properties"].items():
            identifier = f"{method}:decision:{field}"
            values = schema_values(description)
            nodes.append(GraphNode(node_id=identifier, type="policy" if fixed_value(description) else "decision",
                                   title=field.replace("_", " "), description=description.get("description",
                                   "Fixed implementation policy." if fixed_value(description) else "Supported configuration selection or variable binding."),
                                   capability_version=version, fields=(f"configuration.{field}",),
                                   permitted_values=values, input_schema=description, boundary="design"))
            edge(entry, identifier, "offers")
            schema_id = f"{method}:requirement:schema.{field}"
            nodes.append(GraphNode(node_id=schema_id, type="requirement", title=f"{field} schema",
                                   description="This schema constrains submitted values; invalid explicit values are never replaced by defaults.",
                                   capability_version=version, input_schema=description,
                                   fields=(f"configuration.{field}",), boundary="design"))
            edge(identifier, schema_id, "requires")
            for rule in definition.requirements:
                if f"configuration.{field}" in rule.fields:
                    edge(identifier, f"{method}:requirement:{rule.id}", "requires")
            for dependency in dependencies.get(field, ()):
                edge(identifier, f"{method}:requirement:{dependency}", "requires")
            for value in values:
                key = f"{field}={value}"
                option_id = f"{method}:option:{key}"
                nodes.append(GraphNode(node_id=option_id, type="option", title=key,
                                       description="A supported value under its declared prerequisites.",
                                       capability_version=version, fields=(f"configuration.{field}",), permitted_values=(value,)))
                edge(identifier, option_id, "offers")
                for rule in definition.requirements:
                    if f"configuration.{field}" in rule.fields:
                        edge(option_id, f"{method}:requirement:{rule.id}", "requires")
                for dependency in dependencies.get(key, dependencies.get(field, ())):
                    edge(option_id, f"{method}:requirement:{dependency}", "requires")
        for role in definition.roles:
            identifier = f"{method}:role:{role.id}"
            nodes.append(GraphNode(node_id=identifier, type="role", title=role.id.replace("_", " "),
                                   description=role.description, capability_version=version,
                                   fields=(role.field,), input_schema={"kind": role.kind},
                                   minimum=role.minimum, maximum=role.maximum))
            edge(entry, identifier, "offers")
            edge(f"{method}:decision:{role.field.removeprefix('configuration.')}", identifier, "reveals")
            required = f"{method}:requirement:role.{role.id}"
            nodes.append(GraphNode(node_id=required, type="requirement", title=f"{role.id} cardinality",
                                   description=role.description, capability_version=version, boundary="design",
                                   fields=(role.field,), minimum=role.minimum, maximum=role.maximum))
            edge(identifier, required, "requires")
            for dependency in role.dependencies:
                edge(identifier, f"{method}:requirement:{dependency}", "requires")
        nodes.append(GraphNode(node_id=f"{method}:role:outcome", type="role", title="Outcome",
                               description="The declared scientific outcome measurement.", capability_version=version,
                               fields=("outcome.column",), minimum=1, maximum=1, input_schema={"kind": "numeric"}))
        edge(entry, f"{method}:role:outcome", "offers")
        for family, branches in (("diagnostic", definition.diagnostics), ("sensitivity", definition.sensitivities)):
            for branch in branches:
                identifier = f"{method}:{family}:{branch.id}"
                nodes.append(GraphNode(node_id=identifier, type="check" if family == "diagnostic" else "option",
                                       title=branch.title, description=branch.purpose, capability_version=version,
                                       boundary="execution", applicability_boundary="design", measurement_boundary="execution"))
                edge(entry, identifier, "checked_by" if family == "diagnostic" else "offers")
                for dependency in branch.dependencies:
                    edge(identifier, f"{method}:requirement:{dependency}", "requires")
        nodes.append(GraphNode(node_id=f"{method}:check:data", type="check", title="Prepared frame runnability",
                               description="Check the exact frame's roles, types, missingness and structural support; no effects are fitted.",
                               capability_version=version, boundary="data_preflight",
                               applicability_boundary="design", measurement_boundary="data_preflight"))
        edge(entry, f"{method}:check:data", "checked_by")
        for name, reason in definition.unsupported:
            identifier = f"{method}:exclusion:{name}"
            nodes.append(GraphNode(node_id=identifier, type="exclusion", title=name, description=reason, capability_version=version))
            edge(entry, identifier, "excludes")
    identifiers = {node.node_id for node in nodes}
    if len(identifiers) != len(nodes) or any(e.source not in identifiers or e.target not in identifiers for e in edges):
        raise ValueError("Graph identifiers must be unique and all links must resolve.")
    unique_edges = tuple({(e.source, e.target, e.relation): e for e in edges}.values())
    return CapabilityGraph(version=graph_version, nodes=tuple(nodes), edges=unique_edges)


def explore_capabilities(draft: CandidateDraft | Mapping[str, Any] | Model | None = None,
                         at: str | None = None, relations: Sequence[str] | None = None,
                         limit: int = 30, cursor: str | None = None) -> CapabilityExploration:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 200:
        raise CapabilityRequestError("limit", "A neighborhood limit must be an integer between 1 and 200.")
    allowed = get_args(Relation)
    if relations is not None and (isinstance(relations, str) or any(r not in allowed for r in relations)):
        raise CapabilityRequestError("relations", "Unknown relation kind; use declared graph relations.")
    selected_relations = tuple(sorted(set(allowed if relations is None else relations)))
    graph = build_graph()
    by_id = {node.node_id: node for node in graph.nodes}
    target = at or "analysis"
    if target not in by_id:
        raise CapabilityRequestError("at", "Unknown capability node; follow a returned stable identifier.")
    evaluation = evaluate_candidate(draft)
    query = digest({"at": target, "relations": selected_relations, "limit": limit,
                    "graph": graph.version, "candidate": evaluation.candidate_fingerprint})
    offset = 0
    if cursor is not None:
        try:
            page = json.loads(base64.urlsafe_b64decode(cursor.encode()))
            if page["query"] != query or type(page["offset"]) is not int or page["offset"] < 0:
                raise ValueError("Stale or mismatched cursor")
            offset = page["offset"]
        except (ValueError, KeyError, TypeError, UnicodeError) as error:
            raise CapabilityRequestError("cursor", "The cursor does not match this candidate, graph version and query.") from error
    adjacent = tuple(edge for edge in graph.edges if edge.relation in selected_relations
                     and target in (edge.source, edge.target))
    neighbors = sorted({edge.target if edge.source == target else edge.source for edge in adjacent} - {target})
    if offset > len(neighbors):
        raise CapabilityRequestError("cursor", "Cursor offset exceeds this neighborhood.")
    page_ids = neighbors[offset:offset + limit]
    visible = {target, *page_ids}
    next_offset = offset + len(page_ids)
    next_cursor = (base64.urlsafe_b64encode(json.dumps({"query": query, "offset": next_offset},
                                                     separators=(",", ":")).encode()).decode()
                   if next_offset < len(neighbors) else None)
    options = [option for option in evaluation.options if option.node_id in visible]
    overlay_ids = {option.node_id for option in options}
    for identifier in sorted(visible - overlay_ids):
        node = by_id[identifier]
        if node.type in ("decision", "option", "check", "exclusion"):
            options.append(OptionState(node_id=identifier, state="unsupported" if node.type == "exclusion" else "conditional",
                                       explanation=node.description if node.type == "exclusion" else
                                       "Inspecting this unselected branch does not nominate it. Submit a hypothetical candidate to evaluate its choices."))
    blockers = tuple(row.requirement_id for row in evaluation.requirements
                     if row.status in ("unresolved", "violated") and row.boundary == "design")
    return CapabilityExploration(
        graph_version=graph.version, candidate_fingerprint=evaluation.candidate_fingerprint,
        at=by_id[target], nodes=tuple(by_id[node] for node in page_ids),
        edges=tuple(edge for edge in adjacent if edge.source in visible and edge.target in visible),
        options=tuple(options), requirements=tuple(row for row in evaluation.requirements if row.requirement_id in visible),
        selections=tuple(row for row in evaluation.selections if row.node_id in visible),
        status=evaluation.status, blocker_count=len(blockers), blocker_references=blockers,
        coverage=GraphCoverage(total_neighbors=len(neighbors), returned_neighbors=len(page_ids), offset=offset,
                               truncated=next_cursor is not None, next_cursor=next_cursor))
