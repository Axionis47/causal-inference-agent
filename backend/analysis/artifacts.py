"""The same immutable analysis artifacts serve downloads and bound chat selections."""

import hashlib
import json
import re
from typing import Any

from backend.context.service import fingerprint
from backend.handoffs import AnalysisBinding
from backend.models import AnalysisRun
from backend.storage import files


def catalogue(run: AnalysisRun) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    if frozen := (run.inputs or {}).get("frozen_context"):
        result["frozen-context.json"] = {"inline": json.dumps(frozen, indent=2)}
    if run.snapshot:
        result["snapshot.json"] = {"inline": json.dumps(run.snapshot, indent=2)}
    for key, value in run.receipts.items():
        if not isinstance(value, dict):
            continue
        if (run.inputs or {}).get("frozen_context") and key.startswith("checkpoint:") and value.get("id"):
            result[f"checkpoint-{value['id']}.json"] = {"inline": json.dumps(value, indent=2)}
        if key.startswith("program:"):
            result[f"program-{value['revision']}.py"] = {"inline": value["program"]["code"]}
        if key.startswith("review_rejection:"):
            result[f"review-rejection-{value['revision']}.json"] = {"inline": json.dumps(value, indent=2)}
        if key.startswith("execution:") or key == "execution":
            prefix = f"execution-{key.split(':')[1]}-" if ":" in key else ""
            for name, record in value.get("files", {}).items():
                result[prefix + name] = record
        if key.startswith("diagnostic:"):
            result[f"diagnostic-{key.split(':')[1]}.py"] = {"inline": value["code"]}
    return result


def read(run: AnalysisRun, name: str) -> bytes:
    record = catalogue(run).get(name)
    if record is None:
        raise ValueError("The artifact is not part of the bound analysis.")
    if "inline" in record:
        return str(record["inline"]).encode()
    return files().read(str(record["key"]), str(record["sha256"]))


def binding(run: AnalysisRun) -> AnalysisBinding:
    """Freeze the available artifact catalogue at chat submission, before future execution."""
    return {
        "snapshot_hash": run.snapshot_hash,
        "result_hash": fingerprint(run.result) if run.result is not None else None,
        "status": run.status,
        "files": {
            name: hashlib.sha256(str(record["inline"]).encode()).hexdigest()
            if "inline" in record
            else record["sha256"]
            for name, record in catalogue(run).items()
        },
    }


def numerical_facts(run: AnalysisRun, name: str) -> dict[str, Any]:
    """Only validated execution facts are computed evidence; arbitrary JSON is not."""
    match = re.fullmatch(r"execution-(\d+)-facts\.json", name)
    if not match:
        return {}
    receipt = run.receipts.get(f"execution:{match[1]}", {})
    if not isinstance(receipt, dict) or receipt.get("valid") is not True or name not in catalogue(run):
        return {}
    return dict(receipt["facts"])
