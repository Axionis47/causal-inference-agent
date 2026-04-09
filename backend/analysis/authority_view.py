"""Read-only projections of saved claim decisions, with every attempt's origin."""

from typing import Any


def inspect(receipts: list[dict[str, Any]]) -> dict[str, Any]:
    claims: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    seen: set[str] = set()
    for receipt in receipts:
        identity = receipt.get("id")
        if not isinstance(identity, str) or identity in seen:
            continue
        seen.add(identity)
        origin = {"receipt_id": identity, "checkpoint": receipt["checkpoint"], "receipt_status": receipt["status"]}

        def walk(value: Any, path: str, origin: dict[str, Any] = origin) -> None:
            if isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, f"{path}[{index}]")
            elif isinstance(value, dict):
                if isinstance(value.get("admission"), dict):
                    claims.append({**origin, "path": path,
                                   "statement": value.get("statement", value.get("explanation")),
                                   "admission": value["admission"]})
                for key, item in value.items():
                    # Original proposals are evidence about the attempt, never a
                    # second set of admitted decisions or a ledger transition.
                    if key not in {"admission", "original_proposal", "assumptions"}:
                        walk(item, f"{path}.{key}" if path else key)

        output = receipt.get("output", {})
        walk(output, "")
        for entry in output.get("assumptions", []):
            if entry.get("status") is not None:
                ledger.append({**origin, "entry": entry})
    return {"claims": claims, "ledger": ledger}


def summary(output: dict[str, Any]) -> dict[str, Any]:
    view = inspect([{"id": "summary", "checkpoint": "summary", "status": "pending", "output": output}])
    return {
        "claims": {status: sum(item["admission"]["status"] == status for item in view["claims"])
                   for status in ("admitted", "candidate", "rejected")},
        "ledger": {status: [item["entry"]["id"] for item in view["ledger"] if item["entry"]["status"] == status]
                   for status in ("candidate", "relied_upon", "rejected")},
    }
