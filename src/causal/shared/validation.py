"""Common validation machinery shared by every stage validator (SC §5.4, §14.1; D-048)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Final, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from causal.shared.contracts import (
    REFERENCE_KIND_SCHEMA_KEY,
    REFERENCE_ROLE_SCHEMA_KEY,
    Identity,
    ReferenceKind,
    ReferenceRole,
)
from causal.shared.envelope import AgentTaskResultV1, EvidenceClass
from causal.shared.registry import RegistryError

__all__ = [
    "ACTIONS",
    "ASK_ACTIONS",
    "DROP_ACTIONS",
    "EVIDENCE_ACTIONS",
    "EVIDENCE_CLASS_PREFIXES",
    "FIX_ACTIONS",
    "ReferenceField",
    "ReferenceKind",
    "ReferenceRole",
    "ValidationIssueV1",
    "ValidationReport",
    "ValidationRuleV1",
    "as_tuple",
    "collect_ids",
    "constrain_reference_schema",
    "evidence_class",
    "has_cycle",
    "load_rules",
    "make_issue",
    "parse_strict",
    "reference_projection",
    "reference_snapshot",
    "rewrite_references",
    "shape_report",
    "unresolved_issues",
    "validate_references",
    "walk_reference_fields",
]

# The §16.4 correction actions, offered as four bundles.
FIX_ACTIONS: Final = ("revise_field",)
ASK_ACTIONS: Final = ("revise_field", "request_context")
EVIDENCE_ACTIONS: Final = ("add_evidence", "relabel_epistemic_status", "request_context")
DROP_ACTIONS: Final = ("revise_field", "remove_claim")
ACTIONS: Final = tuple(sorted({*FIX_ACTIONS, *ASK_ACTIONS, *EVIDENCE_ACTIONS, *DROP_ACTIONS}))
# §10.2 evidence-class resolution by id family; a measured pointer wins over every prefix.
EVIDENCE_CLASS_PREFIXES: Final = (
    ("ua:", EvidenceClass.USER_CONFIRMATION), ("ev:kaggle/column/", EvidenceClass.DATA_DICTIONARY),
    ("ev:kaggle/file/", EvidenceClass.DATA_DICTIONARY), ("ev:doc/", EvidenceClass.SOURCE_STATEMENT),
    ("ev:kaggle/dataset/", EvidenceClass.SOURCE_STATEMENT))

_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)


@dataclass(frozen=True)
class ReferenceField:
    """One metadata-declared reference field reached through a model's JSON schema."""

    kind: ReferenceKind
    role: ReferenceRole
    path: str
    schema: dict[str, Any]
    value: Any

    def values(self) -> tuple[tuple[str, str], ...]:
        """String values with exact JSON-pointer paths, including sequence indexes."""
        if isinstance(self.value, str):
            return ((self.path, self.value),)
        if isinstance(self.value, list | tuple):
            return tuple(
                (f"{self.path}/{index}", value)
                for index, value in enumerate(self.value)
                if isinstance(value, str)
            )
        return ()


_MISSING: Final = object()


def _pointer(parts: tuple[str, ...]) -> str:
    encoded = (part.replace("~", "~0").replace("/", "~1") for part in parts)
    return "/" + "/".join(encoded) if parts else ""


def _resolve_schema(schema: Mapping[str, Any], root: Mapping[str, Any]) -> Mapping[str, Any]:
    reference = schema.get("$ref")
    if not isinstance(reference, str) or not reference.startswith("#/"):
        return schema
    target: Any = root
    for token in reference[2:].split("/"):
        target = target[token.replace("~1", "/").replace("~0", "~")]
    return cast(Mapping[str, Any], target)


def walk_reference_fields(
    model_or_schema: type[BaseModel] | Mapping[str, Any], value: Any = _MISSING,
    *, prefix: str = "",
) -> tuple[ReferenceField, ...]:
    """Walk every typed reference/declaration in one schema, optionally paired with a value."""
    root = (model_or_schema.model_json_schema()
            if isinstance(model_or_schema, type) and issubclass(model_or_schema, BaseModel)
            else dict(model_or_schema))
    found: list[ReferenceField] = []

    def visit(schema: Mapping[str, Any], held: Any, parts: tuple[str, ...]) -> None:
        kind_value = schema.get(REFERENCE_KIND_SCHEMA_KEY)
        if isinstance(kind_value, str):
            found.append(ReferenceField(
                kind=ReferenceKind(kind_value),
                role=ReferenceRole(str(schema.get(
                    REFERENCE_ROLE_SCHEMA_KEY, ReferenceRole.REFERENCE.value))),
                path=prefix + _pointer(parts), schema=cast(dict[str, Any], schema),
                value=None if held is _MISSING else held,
            ))
            return
        resolved = _resolve_schema(schema, root)
        if resolved is not schema:
            visit(resolved, held, parts)
            return
        for choice in schema.get("anyOf", ()):
            if isinstance(choice, Mapping):
                visit(choice, held, parts)
        properties = schema.get("properties")
        if isinstance(properties, Mapping):
            values = held if isinstance(held, Mapping) else {}
            for name, child in properties.items():
                if isinstance(child, Mapping):
                    visit(child, values.get(name, _MISSING), (*parts, str(name)))
        items = schema.get("items")
        if isinstance(items, Mapping):
            if isinstance(held, list | tuple):
                for index, item in enumerate(held):
                    visit(items, item, (*parts, str(index)))
            elif held is _MISSING:
                visit(items, _MISSING, (*parts, "*"))

    visit(root, value, ())
    unique = {(row.kind, row.role, row.path): row for row in found}
    return tuple(unique[key] for key in sorted(
        unique, key=lambda item: (item[2], item[0].value, item[1].value)))


class ValidationIssueV1(BaseModel):
    """One failed rule and what the responsible agent may do about it (PRD-002 §16.4)."""

    model_config = _MODEL_CONFIG

    code: Identity
    json_path: str
    rule_id: Identity
    artifact_ids: tuple[Identity, ...]
    allowed_actions: tuple[Identity, ...]
    user_resolvable: bool
    # Human-readable near-miss text; a code alone leaves a deterministic model repeating itself.
    detail: str = ""


class ValidationReport(BaseModel):
    """Every issue one wall found; the run stops at the first wall that fails."""

    model_config = _MODEL_CONFIG

    wall: Annotated[int, Field(ge=1, le=15)]
    issues: tuple[ValidationIssueV1, ...]

    @property
    def passed(self) -> bool:
        return not self.issues


class ValidationRuleV1(BaseModel):
    """One declarative wall row; an engine reads `params` by `kind`."""

    model_config = _MODEL_CONFIG

    rule_id: Identity
    wall: int
    kind: Identity
    params: dict[str, Any]
    code: Identity
    allowed_actions: tuple[Identity, ...]
    user_resolvable: bool

    def issue(self, path: str, ids: tuple[str, ...] = ()) -> ValidationIssueV1:
        return make_issue(self.code, path, self.rule_id, self.allowed_actions,
                          self.user_resolvable, ids)


def make_issue(code: str, path: str, rule: str, actions: tuple[str, ...],
               resolvable: bool = False, ids: tuple[str, ...] = (),
               detail: str = "") -> ValidationIssueV1:
    return ValidationIssueV1(code=code, json_path=path, rule_id=rule, artifact_ids=ids,
                             allowed_actions=actions, user_resolvable=resolvable, detail=detail)


def load_rules(path: Path, *, registry_version: str, kinds: tuple[str, ...],
               max_wall: int = 10) -> tuple[ValidationRuleV1, ...]:
    """Load declarative wall rows; an unreadable file, unknown kind, or wall fails closed."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document["registry_version"] != registry_version:
            raise ValueError(f"registry version {document['registry_version']!r}")
        rules = tuple(parse_strict(ValidationRuleV1, row) for row in document["rules"])
    except (OSError, LookupError, TypeError, ValueError, ValidationError) as error:
        raise RegistryError(f"invalid rule registry {path}: {error}",
                            "invalid_registry_file") from error
    for rule in rules:
        if rule.kind not in kinds:
            raise RegistryError(f"{rule.rule_id} unknown kind {rule.kind!r}", "unknown_rule_kind")
        if not 1 <= rule.wall <= max_wall:
            raise RegistryError(f"{rule.rule_id} unknown wall {rule.wall}", "unknown_wall")
    return rules


# Strict payload models take JSON arrays, not python lists, so payloads parse as JSON.
def parse_strict[Model: BaseModel](model_cls: type[Model], payload: Mapping[str, object]) -> Model:
    return model_cls.model_validate_json(json.dumps(payload, default=str))


def evidence_class(evidence_id: str) -> EvidenceClass | None:
    """Resolve one evidence id to its §10.2 class, or None when no family claims it."""
    if "#/columns/" in evidence_id:
        return EvidenceClass.MEASURED_OBSERVATION
    return next((f for p, f in EVIDENCE_CLASS_PREFIXES if evidence_id.startswith(p)), None)


def as_tuple(value: Any) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, list | tuple) else (value,)


def collect_ids(node: Any, keys: tuple[str, ...]) -> set[str]:
    """Every string held (directly or in sequences) under any of `keys`, at any depth."""
    if isinstance(node, Mapping):
        here = {value for key, held in node.items() if key in keys
                for value in as_tuple(held) if isinstance(value, str)}
        return here | {found for held in node.values() for found in collect_ids(held, keys)}
    if isinstance(node, list | tuple):
        return {found for item in node for found in collect_ids(item, keys)}
    return set()


def _catalogs_with_declarations(
    fields: Sequence[ReferenceField], catalogs: Mapping[ReferenceKind, Sequence[str]],
) -> dict[ReferenceKind, frozenset[str]]:
    declared = {
        kind: frozenset(value for field in fields
                        if field.kind is kind and field.role is ReferenceRole.DECLARATION
                        for _, value in field.values())
        for kind in ReferenceKind
    }
    return {
        kind: frozenset(catalogs.get(kind, ())) | declared[kind]
        for kind in ReferenceKind
    }


def constrain_reference_schema(
    schema: dict[str, Any], catalogs: Mapping[ReferenceKind, Sequence[str]],
) -> None:
    """Close metadata-declared references in a response schema to task-owned catalogs."""
    fields = walk_reference_fields(schema)
    declaration_kinds = {field.kind for field in fields
                         if field.role is ReferenceRole.DECLARATION}
    for field in fields:
        if field.kind not in catalogs:
            continue
        legal = tuple(dict.fromkeys(str(item) for item in catalogs[field.kind]))
        target = cast(dict[str, Any], field.schema.get("items")
                      if field.schema.get("type") == "array" else field.schema)
        if legal:
            target["enum"] = list(legal)
        elif field.kind not in declaration_kinds and field.schema.get("type") == "array":
            field.schema["maxItems"] = 0


def validate_references(
    model_or_schema: type[BaseModel] | Mapping[str, Any], value: Any,
    catalogs: Mapping[ReferenceKind, Sequence[str]], *, prefix: str = "",
) -> tuple[ValidationIssueV1, ...]:
    """Reject every typed reference outside its exact external or local-declaration catalog."""
    fields = walk_reference_fields(model_or_schema, value, prefix=prefix)
    legal = _catalogs_with_declarations(fields, catalogs)
    issues: list[ValidationIssueV1] = []
    for field in fields:
        allowed = frozenset(catalogs.get(field.kind, ()))
        # A declaration is locally authoritative unless an upstream catalog explicitly closes it.
        if field.role is ReferenceRole.DECLARATION and field.kind not in catalogs:
            continue
        accepted = allowed if field.role is ReferenceRole.DECLARATION else legal[field.kind]
        for path, item in field.values():
            if item in accepted:
                continue
            choices = f"; allowed: {', '.join(sorted(accepted))}" if accepted else ""
            issues.append(make_issue(
                f"unresolved_{field.kind.value}", path, "wall2.typed_reference",
                FIX_ACTIONS, False, (item,),
                f"{item!r} is not a task-local {field.kind.value} reference{choices}",
            ))
    return tuple(issues)


def _path_tokens(path: str) -> tuple[str, ...]:
    if not path:
        return ()
    return tuple(token.replace("~1", "/").replace("~0", "~")
                 for token in path.removeprefix("/").split("/"))


def _delete_path(node: Any, path: str) -> None:
    tokens = _path_tokens(path)
    if not tokens:
        return
    parent = node
    for token in tokens[:-1]:
        parent = parent[int(token)] if isinstance(parent, list) else parent[token]
    if isinstance(parent, list):
        parent.pop(int(tokens[-1]))
    else:
        parent.pop(tokens[-1], None)


def _set_path(node: Any, path: str, value: str) -> None:
    tokens = _path_tokens(path)
    parent = node
    for token in tokens[:-1]:
        parent = parent[int(token)] if isinstance(parent, list) else parent[token]
    if isinstance(parent, list):
        parent[int(tokens[-1])] = value
    else:
        parent[tokens[-1]] = value


def reference_projection(model_or_schema: type[BaseModel] | Mapping[str, Any], value: Any) -> Any:
    """Return the semantic decision with only typed reference fields removed."""
    projected = deepcopy(value)
    paths = {field.path for field in walk_reference_fields(model_or_schema, value)
             if field.role is ReferenceRole.REFERENCE}
    for path in sorted(paths, key=lambda item: item.count("/"), reverse=True):
        _delete_path(projected, path)
    return projected


def reference_snapshot(
    model_or_schema: type[BaseModel] | Mapping[str, Any], value: Any, *, prefix: str = "",
) -> dict[str, str]:
    """Return exact paths and values for every typed reference in a model decision."""
    return {
        path: item
        for field in walk_reference_fields(model_or_schema, value, prefix=prefix)
        if field.role is ReferenceRole.REFERENCE
        for path, item in field.values()
    }


def rewrite_references(
    model_or_schema: type[BaseModel] | Mapping[str, Any], value: Any,
    rewrite: Any,
) -> Any:
    """Apply a deterministic normalizer only at metadata-declared reference value paths."""
    rewritten = deepcopy(value)
    for field in walk_reference_fields(model_or_schema, value):
        if field.role is not ReferenceRole.REFERENCE:
            continue
        for path, item in field.values():
            replacement = rewrite(field.kind, item)
            if isinstance(replacement, str) and replacement != item:
                _set_path(rewritten, path, replacement)
    return rewritten


def unresolved_issues(ids: set[str], code: str, prefix: str,
                      actions: tuple[str, ...] = ASK_ACTIONS,
                      legal: Sequence[str] = (), user_resolvable: bool = True) -> Iterator[ValidationIssueV1]:
    """One issue per unresolved id, each naming the offending value and the legal set (D-103).

    The correction budget is two attempts, and `detail` used to be empty on every issue: a
    worker was told a code and a path and had to guess what the harness would accept.
    """
    allowed = f"; allowed: {', '.join(sorted(legal))}" if legal else ""
    return (make_issue(code, f"{prefix}/{i}", f"wall2.{code}", actions, user_resolvable, (i,),
                       f"{i!r} is not a known {prefix.strip('/')} value{allowed}")
            for i in sorted(ids))


def shape_report(model_cls: type[BaseModel], result: AgentTaskResultV1,
                 actions: tuple[str, ...] = FIX_ACTIONS) -> ValidationReport:
    """Wall 1: the payload is exactly the declared schema; errors map to JSON paths."""
    try:
        parse_strict(model_cls, result.payload)
    except ValidationError as error:
        return ValidationReport(wall=1, issues=tuple(
            make_issue("shape_invalid", "/payload/" + "/".join(str(p) for p in item["loc"]),
                       "wall1.shape", actions, detail=item["msg"][:200])
            for item in error.errors()))
    return ValidationReport(wall=1, issues=())


# Kahn peeling; whatever survives sits on a cycle.
def has_cycle(pairs: Iterable[tuple[str, str]]) -> bool:
    live = list(pairs)
    nodes = {node for pair in live for node in pair}
    while nodes:
        roots = nodes - {target for _, target in live}
        if not roots:
            return True
        nodes -= roots
        live = [pair for pair in live if pair[0] not in roots]
    return False
