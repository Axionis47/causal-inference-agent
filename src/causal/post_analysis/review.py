"""Source-aware code checks supplement, and never impersonate, model review."""
from __future__ import annotations

import hashlib
from typing import Any

from causal.post_analysis.contracts import EvidencePacket, ReportDraft
from causal.post_analysis.store import Store
from causal.shared.canonical import content_hash
from causal.shared.persistence import PersistenceError


def select(value: Any, pointer: str) -> Any:
    if not pointer:
        return value
    if not pointer.startswith("/"):
        raise ValueError("citation selector must be a JSON Pointer")
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, (list, tuple)):
            if not token.isascii() or not token.isdigit() or (token.startswith("0") and token != "0"):
                raise ValueError("array selector must be a nonnegative canonical index")
            value = value[int(token)]
        else:
            value = value[token]
    return value


def check_draft(draft: ReportDraft, packet: EvidencePacket,
                visuals: dict[str, dict[str, Any]]) -> tuple[str, ...]:
    issues: list[str] = []
    if set(draft.coverage) != set(packet.required_evidence):
        issues.append("coverage must account for exactly every required evidence item")
    cited: set[str] = set()
    selected: list[str] = []
    for section in draft.sections:
        selected.extend(section.visual_ids)
        for statement in section.statements:
            for citation in statement.citations:
                try:
                    select(packet.evidence[citation.evidence_id], citation.selector)
                    cited.add(citation.evidence_id)
                except (KeyError, ValueError, IndexError, TypeError):
                    issues.append(f"invalid citation: {citation.evidence_id}{citation.selector}")
    if missing := set(packet.required_evidence) - cited:
        issues.append(f"required evidence is not discussed with citations: {sorted(missing)}")
    if not selected:
        issues.append("report requires a visual or a source table")
    if len(set(selected)) != len(selected):
        issues.append("a visual may appear only once in the report")
    for visual_id in selected:
        if visual_id not in visuals or visuals[visual_id].get("status") != "rendered":
            issues.append(f"visual is unavailable: {visual_id}")
    return tuple(issues)


def verify_objects(store: Store, artifact: dict[str, Any]) -> None:
    for name, locator in artifact["objects"].items():
        data = store.deps.objects.get(locator)
        if hashlib.sha256(data).hexdigest() != artifact["object_hashes"][name]:
            raise PersistenceError(f"export object changed: {name}", "artifact_hash_mismatch")


def review_binding(context: Any, draft: Any, visuals: Any, export: Any) -> str:
    """A review is valid only for this exact dependency set, including final page bytes."""
    return content_hash({"context": context, "draft": draft, "visuals": visuals, "export": export})
