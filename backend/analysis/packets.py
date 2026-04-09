"""Bounded views of pinned evidence. The full context and role ledger remain authoritative."""

import json
from copy import deepcopy
from typing import Any

from backend.analysis.authority import source_predicates, source_selectable
from backend.analysis.references import handles
from backend.context.contracts import ContextContent, DatasetStatement, statement_entries
from backend.context.domain import project

EVIDENCE_LIMIT = 32_000
MODEL_INPUT_LIMIT = 96_000


class PacketOverflow(ValueError):
    pass


def size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True))


def evidence_records(context: ContextContent) -> list[dict[str, Any]]:
    fields = {
        "value",
        "basis",
        "qualifications",
        "status",
        "reason",
        "citations",
        "column_ids",
        "participants",
        "timing",
        "conditions",
        "kind",
        "directness",
        "absence_scope",
        "conflicts_with",
    }
    entries = [item for item in sorted(statement_entries(context)) if item[1].status != "withdrawn"]
    aliases = handles(("e", key) for key, _ in entries)
    return [
        {"reference": aliases[key], "key": key, "source_predicates": source_predicates(key, statement),
         "source_selectable": source_selectable(statement),
         "statement": statement.model_dump(mode="json", include=fields)}
        for key, statement in entries
    ]


def scope(context: ContextContent, columns: set[str]) -> ContextContent:
    # Global study facts are shared. Column-specific facts and incident relationships
    # travel with their columns. project preserves cross-column dependencies and caveats.
    keys = [
        key
        for key, statement in statement_entries(context)
        if isinstance(statement, DatasetStatement) and not statement.column_ids
    ]
    keys.extend(f"columns.{cid}" for cid in sorted(columns))
    if not keys:
        return ContextContent()
    # project has a per-request key limit, not an evidence truncation rule.
    if len(keys) > 100:
        raise PacketOverflow(
            "Study context needs more than 100 initial keys. Narrow the question or review the keyed context."
        )
    return project(context, keys)[0]


def exposure(context: ContextContent, total: int, columns: set[str]) -> dict[str, Any]:
    records = evidence_records(context)
    return {
        "total_columns": total,
        "task_columns": sorted(columns),
        "included_columns": sorted(context.columns),
        "evidence_records": len(records),
        "evidence_characters": size(records),
        "evidence_limit": EVIDENCE_LIMIT,
        "model_input_limit": MODEL_INPUT_LIMIT,
    }


def bounded(context: ContextContent) -> ContextContent:
    count = size(evidence_records(context))
    if count > EVIDENCE_LIMIT:
        raise PacketOverflow(
            f"This task needs {count} evidence characters, above its {EVIDENCE_LIMIT} limit. "
            "No evidence was truncated. Review the scoped context or narrow the task."
        )
    return context


def selected(prior: dict[str, Any]) -> set[str]:
    target = prior.get("target", {}).get("outcome") or {}
    return set(target.get("column_ids", [])) | {
        cid
        for treatment in prior.get("treatments", {}).get("treatments", [])
        for cid in treatment["variable"]["column_ids"]
    }


def downstream_role(item: dict[str, Any]) -> bool:
    # New-policy inactive interpretations have already completed coverage. They
    # stay inspectable in their receipt, without entering causal/program packets.
    if item.get("proposed_roles") is not None and set(item["roles"]) <= {"unknown"} and not item.get("operational_use"):
        return False
    return bool(item.get("operational_use")) or not set(item["roles"]) <= {"irrelevant", "identifier"}


def active(prior: dict[str, Any]) -> set[str]:
    return selected(prior) | {
        cid
        for item in prior.get("roles", {}).get("assignments", [])
        if downstream_role(item)
        for cid in item["variable"]["column_ids"]
    }


def decisions(prior: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(prior)
    if "roles" in result:
        roles = result["roles"]
        # Keep unresolved roles, all assumptions and their dependencies. Only accepted
        # exclusions leave the working packet; the exact ledger remains in decision:roles.
        roles["assignments"] = [
            item for item in roles["assignments"]
            if set(item["variable"]["column_ids"]) & selected(prior)
            or downstream_role(item)
        ]
        for field in ("summary", "changes", "permits"):
            roles[field] = (
                "Relevant and unresolved roles are shown. Full exclusions remain in decision:roles."
            )
        roles["citations"] = []
        # Excluded decisions remain in their own receipts, including their uncertainty.
        # Only assumptions actually used by retained claims travel downstream.
        from backend.analysis.domain import references
        needed = references(roles["assignments"])
        ledger = {a["id"]: a for a in roles.get("assumptions", [])}
        while True:
            expanded = needed | {ref for aid, a in ledger.items() if f"assumption:{aid}" in needed for ref in a.get("depends_on", [])}
            if expanded == needed:
                break
            needed = expanded
        roles["assumptions"] = [a for aid, a in ledger.items() if f"assumption:{aid}" in needed]
    return result


def science(snapshot: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(snapshot)
    frame = result.get("frame", result)
    reduced = decisions(frame)
    frame.update(reduced)
    return result


def assumption_ids(prior: dict[str, Any]) -> set[str]:
    """Only explicitly offered accepted ledgers, never rejected correction prose."""
    return {item["id"] for value in prior.values() if isinstance(value, dict)
            for item in value.get("assumptions", [])}


def diagnostic_view(value: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(value)
    manifest = result.get("manifest", result)
    if "withheld" in manifest:
        manifest["withheld_count"] = len(manifest.pop("withheld"))
        manifest["full_manifest_reference"] = "diagnostic_manifest"
    return result


def model_view(value: Any) -> Any:
    """Bounded provider view of admitted science, never its rejected audit prose.

    Build reference decoders from the canonical packet first. Their registries
    retain full immutable objects for exact hydration; this view is not a record.
    """
    if isinstance(value, list):
        return [model_view(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: model_view(item) for key, item in value.items()
              if key not in {"admission", "original_proposal", "admission_reason", "proposed_roles"}}
    admission = value.get("admission")
    if isinstance(admission, dict) and admission.get("status") != "admitted":
        for key in ("statement", "explanation"):
            if key in result:
                result[key] = "Unresolved interpretation; this claim has no admitted scientific authority."
    if {"status", "summary", "changes", "permits"} <= value.keys():
        for key in ("summary", "changes", "permits"):
            result.pop(key, None)
    if {"nodes", "edges", "identifications"} <= value.keys():
        result.pop("identification", None)
    if isinstance(value.get("assumptions"), list) and all(isinstance(item, dict) and "status" in item for item in value["assumptions"]):
        result["assumptions"] = [model_view(item) for item in value["assumptions"]
                                 if item.get("status") == "relied_upon"]
    return result
