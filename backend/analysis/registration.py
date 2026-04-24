"""Register explicit model premises without interpreting or promoting their authority."""

from copy import deepcopy
from typing import Any

from backend.analysis import domain
from backend.analysis.contracts import Assumption, Decision, GateCheck
from backend.context.contracts import ContextContent
from backend.context.service import fingerprint


def normalize[T: Decision](
    value: T, context: ContextContent, prior: dict[str, Any], namespace: str, *, scientific_version: int = 18,
) -> tuple[T, list[str]]:
    """Separate exact decision dependencies and register already-declared assumptions.

    No semantic claim, basis, graph endpoint or source reference is invented. A
    missing/unknown/mixed evidence reference still reaches the original gates.
    Existing assumption references and ledger definitions are never repaired.
    """
    if scientific_version >= 19:
        return value, []
    roots = domain.supplied(context)
    decisions = {"decision:" + key for key in prior}
    output = deepcopy(value.model_dump(mode="json"))
    existing = {item["id"]: item for item in output["assumptions"]}
    prior_ledger = {item["id"]: item for decision in prior.values() if isinstance(decision, dict)
                    for item in decision.get("assumptions", [])}
    added: list[dict[str, Any]] = []
    notes: list[str] = []

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, list):
            for index, item in enumerate(obj):
                walk(item, f"{path}[{index}]")
            return
        if not isinstance(obj, dict):
            return
        claim = "statement" in obj and {"basis", "references", "prerequisites"} <= obj.keys()
        edge = {"source", "target", "basis", "references", "explanation", "prerequisites"} <= obj.keys()
        if claim or edge:
            refs = list(obj["references"])
            direct = [ref for ref in refs if ref in roots]
            dependencies = [ref for ref in refs if ref in decisions]
            # A direct source must already exist. Decision-only claims remain
            # rejected; moving a receipt never creates supplied evidence.
            if obj["basis"] in {"supplied", "assumption"} and direct and dependencies and set(refs) <= roots | decisions:
                obj["references"] = direct
                obj["prerequisites"] = list(dict.fromkeys([*obj["prerequisites"], *dependencies]))
                refs = direct
                notes.append(f"{path}: separated accepted decision dependencies from its existing direct sources.")
            # Registration is eligible only for an explicitly conditional model
            # proposal with exclusively offered direct sources and no existing
            # ledger reference. No missing rationale or premise is synthesized.
            if obj["basis"] == "assumption" and refs and set(refs) <= roots:
                statement = obj["explanation"] if edge else obj["statement"]
                location = {"source": obj["source"], "target": obj["target"]} if edge else {"path": path}
                identity = namespace + "registered_" + fingerprint({
                    "location": location, "statement": statement, "references": sorted(set(refs)),
                    "prerequisites": sorted(set(obj["prerequisites"])),
                })[:32]
                entry = Assumption(
                    id=identity, statement=statement, rationale=statement,
                    depends_on=sorted(set([*refs, *obj["prerequisites"]])),
                    affects=[obj["source"], obj["target"]] if edge else [path],
                ).model_dump(mode="json")
                saved = existing.get(identity, prior_ledger.get(identity))
                if saved is not None and saved != entry:
                    raise ValueError("Registered assumption identity conflicts with an existing ledger definition.")
                if identity not in existing and identity not in prior_ledger:
                    existing[identity] = entry
                    added.append(entry)
                obj["references"] = [*refs, "assumption:" + identity]
                notes.append(f"{path}: registered its explicit conditional statement as assumption:{identity}; original sources retained.")
        for key, item in obj.items():
            if key != "assumptions":
                walk(item, path + "." + key)

    walk(output, "decision")
    output["assumptions"].extend(added)
    return type(value).model_validate(output), notes


def gate(notes: list[str]) -> GateCheck:
    return GateCheck(
        rule="evidence_and_assumption_registration", kind="deterministic", passed=True,
        detail=" ".join(notes) if notes else "No evidence registration was necessary. Original authority and scientific gates still apply.",
    )
