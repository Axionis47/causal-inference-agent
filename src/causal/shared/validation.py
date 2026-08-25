"""Common validation machinery shared by every stage validator (SC §5.4, §14.1; D-048)."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Annotated, Any, Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from causal.shared.contracts import Identity
from causal.shared.envelope import AgentTaskResultV1, EvidenceClass
from causal.shared.registry import RegistryError

__all__ = [
    "ACTIONS", "ASK_ACTIONS", "DROP_ACTIONS", "EVIDENCE_ACTIONS", "EVIDENCE_CLASS_PREFIXES",
    "FIX_ACTIONS", "ValidationIssueV1", "ValidationReport", "ValidationRuleV1", "as_tuple",
    "collect_ids", "evidence_class", "has_cycle", "load_rules", "make_issue", "parse_strict",
    "self_citation_issues", "shape_report", "unresolved_issues",
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

    wall: Annotated[int, Field(ge=1, le=10)]
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


def unresolved_issues(ids: set[str], code: str, prefix: str,
                      actions: tuple[str, ...] = ASK_ACTIONS) -> Iterator[ValidationIssueV1]:
    return (make_issue(code, f"{prefix}/{i}", f"wall2.{code}", actions, True, (i,))
            for i in sorted(ids))


def self_citation_issues(raw: Mapping[str, Any],
                         actions: tuple[str, ...] = DROP_ACTIONS) -> Iterator[ValidationIssueV1]:
    """A claim citing itself as evidence is never acceptable (SC §16.2)."""
    for claim in (item for item in raw.get("claims") or () if isinstance(item, Mapping)):
        cited = (*(claim.get("supporting_evidence_ids") or ()),
                 *(claim.get("contrary_evidence_ids") or ()))
        if claim.get("claim_id") in cited:
            yield make_issue("claim_cites_itself", f"/claims/{claim['claim_id']}", "wall2.self",
                             actions)


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
