"""Read-only program and approval projections from authoritative analysis receipts."""

from typing import Any

from backend.models import AnalysisRun


def programs(run: AnalysisRun) -> list[dict[str, Any]]:
    return sorted(
        [
            dict(value)
            for key, value in run.receipts.items()
            if key.startswith("program:") and isinstance(value, dict)
        ],
        key=lambda value: value["revision"],
    )


def approval_hash(run: AnalysisRun) -> str | None:
    saved = programs(run)
    if not saved:
        return None
    latest = saved[-1]
    approval = run.receipts.get(f"approval:{latest['revision']}")
    return (
        latest["hash"]
        if isinstance(approval, dict) and approval.get("program_hash") == latest["hash"]
        else None
    )


def requested_revision(run: AnalysisRun) -> str | None:
    saved = programs(run)
    if not saved:
        return None
    latest = saved[-1]
    request = run.receipts.get(f"revision_request:{latest['revision']}")
    return latest["hash"] if isinstance(request, dict) and request.get("program_hash") == latest["hash"] else None


