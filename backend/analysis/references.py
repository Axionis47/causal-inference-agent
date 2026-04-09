"""Stable scoped selections and bounded feedback, never fuzzy reference repair."""

from collections.abc import Iterable
from hashlib import sha256
from typing import Any


def handle(kind: str, identity: str) -> str:
    """Durable identity determines the handle, never packet order or membership."""
    return kind + sha256(f"{kind}\0{identity}".encode()).hexdigest()[:12]


def key_kind(key: str) -> str:
    """A retrievable statement keeps its evidence handle before and after reading it."""
    parts = key.split(".")
    statement = (parts[0], len(parts)) in {
        ("dataset", 3), ("columns", 4), ("concepts", 2), ("relationships", 2)
    }
    return "e" if statement else "k"


def handles(identities: Iterable[tuple[str, str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    reverse: dict[str, str] = {}
    for kind, identity in identities:
        alias = handle(kind, identity)
        if (identity in mapping and mapping[identity] != alias) or (alias in reverse and reverse[alias] != identity):
            raise ValueError("Reference handle collision; no ambiguous request can be submitted.")
        mapping[identity] = alias
        reverse[alias] = identity
    return mapping


def translate(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return mapping.get(value, value)
    if isinstance(value, list):
        return [translate(item, mapping) for item in value]
    if isinstance(value, dict):
        # Exact data literals are operands, not reference identifiers. A text
        # category may happen to equal a durable ID or an offered short handle.
        return {mapping.get(key, key): item if key == "literal" else translate(item, mapping) for key, item in value.items()}
    return value


def selections(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str], dict[str, list[str]]]:
    columns = [item["column_id"] for item in payload.get("column_identities", [])]
    mapping = handles([
        *(("c", identity) for identity in columns),
        *(("e", item["key"]) for item in payload.get("evidence_records", [])),
        *((key_kind(key), key) for key in payload.get("available_context", [])),
    ])
    encoded = translate(payload, mapping)
    # Physical storage order and total dataset width are audit metadata, not
    # evidence for this scoped semantic decision. The source packet retains both.
    for item in encoded.get("column_identities", []):
        item.pop("position", None)
    if "evidence_scope" in encoded:
        encoded["evidence_scope"].pop("total_columns", None)
    for item in encoded.get("evidence_records", []):
        item["reference"] = item["key"]
    reverse = {alias: key for key, alias in mapping.items()}
    encoded["reference_handles"] = {
        alias: {"kind": "column" if key in columns else "context_key" if key_kind(key) == "k" else "evidence"}
        for alias, key in reverse.items()
    }
    choices = {
        "column_ids": [mapping[cid] for cid in columns],
        "citations": ["question", *payload.get("decision_refs", []), *[mapping[x["key"]] for x in payload.get("evidence_records", [])]],
        "prerequisites": list(payload.get("decision_refs", [])),
        "requested_keys": [mapping.get(key, key) for key in payload.get("available_context", [])],
        "_supplied_references": ["question", *[mapping[x["key"]] for x in payload.get("evidence_records", []) if x["statement"].get("basis") == "source_statement" and x["statement"].get("status") not in {"withdrawn", "unresolved"}]],
        "_assumption_namespace": [str(payload.get("assumption_namespace", "new_"))],
    }
    for field in ("unit_variable_id", "time_variable_id", "adoption_variable_id", "assignment_variable_id", "running_variable_id", "history_variables"):
        choices[field] = [mapping[cid] for cid in columns]
    prior = payload.get("decisions", {})
    effects = prior.get("treatments", {}).get("effects", []) or prior.get("target", {}).get("requested_effects", [])
    if effects:
        choices["effect_ids"] = choices["effect_id"] = [item["id"] for item in effects]
    treatments = prior.get("treatments", {}).get("treatments", [])
    if treatments:
        choices["treatment_ids"] = [mapping.get(item["variable"]["id"], item["variable"]["id"]) for item in treatments]
    values = prior.get("treatments", {}).get("values", [])
    if values:
        for field in ("value_ids", "cohort_value_id", "comparison_value_ids", "treatment_value_ids", "assignment_value_ids"):
            choices[field] = [item["id"] for item in values]
    return encoded, reverse, choices


def correction(output: dict[str, Any], checks: list[dict[str, Any]], reference: str) -> dict[str, Any]:
    """Raw rejected output stays in its receipt. Never carry it into the next context."""
    from backend.analysis import packets

    result: dict[str, Any] = {
        "rejected_artifact": reference,
        "instruction": "Submit a complete corrected small decision from the original evidence. The rejected output is not evidence.",
        "gate_feedback": [
            {"rule": str(c.get("rule", "validation")), "detail": str(c.get("detail", ""))[:1800]}
            for c in checks if not c.get("passed", False)
        ][:12],
    }
    # Registration itself succeeds even when it records an inactive claim. Its
    # per-field reasons must survive a later scientific-gate rejection, without
    # replaying the proposal or admitting it as new evidence.
    claim_feedback: list[dict[str, Any]] = []

    def visit(value: Any, path: str = "") -> None:
        if len(claim_feedback) >= 12:
            return
        if isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif isinstance(value, dict):
            admission = value.get("admission")
            if isinstance(admission, dict) and admission.get("status") != "admitted":
                claim_feedback.append({"path": path, "rule": admission.get("rule"),
                                       "detail": str(admission.get("reason", ""))[:700]})
                return
            for key, item in value.items():
                if key not in {"assumptions", "admission", "original_proposal"}:
                    visit(item, f"{path}.{key}" if path else key)

    visit(output)
    if claim_feedback:
        result["claim_feedback"] = claim_feedback
    if "invalid_submission" not in output and packets.size(output) <= 4000:
        result["rejected_fields"] = output
    return result


def output_limit(name: str) -> int:
    if "code" in name or "program" in name or "repair" in name:
        return 4096 if "review" in name else 16384
    if any(item in name for item in ("screen", "binding", "review", "relation", "reconcile")):
        return 4096
    return 8192
