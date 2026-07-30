"""Local monitoring records that remain useful with or without LangSmith."""
from __future__ import annotations

import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MONITOR_SCHEMA_VERSION = "1.1.0"
INTERACTION_SCHEMA_VERSION = "1.0.0"
INTERACTION_DB = Path(__file__).parent.parent / "studio_events.sqlite"
SENSITIVE_KEY = re.compile(r"password|secret|token|email|phone|address|ssn|passport", re.I)


def event(kind: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema_version": MONITOR_SCHEMA_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        **fields,
    }


def _safe_value(value: Any, *, key: str = "") -> Any:
    """Make an interaction payload JSON-safe without storing row-level data."""
    if SENSITIVE_KEY.search(key):
        return "[redacted]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, dict):
        return {
            str(child_key)[:100]: _safe_value(child_value, key=str(child_key))
            for child_key, child_value in list(value.items())[:40]
        }
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item, key=key) for item in list(value)[:40]]
    return str(value)[:500]


def append_interaction(
    *,
    session_id: str,
    dataset_id: str,
    kind: str,
    stage: str,
    payload: dict[str, Any] | None = None,
    run_id: str = "",
    parent_event_id: str = "",
    path: Path = INTERACTION_DB,
) -> dict[str, Any]:
    """Persist one sanitized, append-only server interaction event."""
    record = {
        "schema_version": INTERACTION_SCHEMA_VERSION,
        "event_id": uuid.uuid4().hex,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": str(session_id),
        "dataset_id": str(dataset_id),
        "run_id": str(run_id),
        "stage": str(stage),
        "kind": str(kind),
        "parent_event_id": str(parent_event_id),
        "payload": _safe_value(payload or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=5) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS interaction_events (
                event_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                kind TEXT NOT NULL,
                parent_event_id TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO interaction_events
            (event_id, timestamp, session_id, dataset_id, run_id, stage, kind,
             parent_event_id, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["event_id"], record["timestamp"], record["session_id"],
                record["dataset_id"], record["run_id"], record["stage"],
                record["kind"], record["parent_event_id"],
                json.dumps(record["payload"], sort_keys=True),
            ),
        )
    return record


def read_interactions(
    session_id: str, *, path: Path = INTERACTION_DB, limit: int = 500
) -> list[dict[str, Any]]:
    """Read one session's event chain in creation order."""
    if not path.exists():
        return []
    with sqlite3.connect(path, timeout=5) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='interaction_events'"
        ).fetchone()
        if not exists:
            return []
        rows = connection.execute(
            """
            SELECT event_id, timestamp, session_id, dataset_id, run_id, stage,
                   kind, parent_event_id, payload_json
            FROM interaction_events
            WHERE session_id = ?
            ORDER BY timestamp, event_id
            LIMIT ?
            """,
            (str(session_id), int(limit)),
        ).fetchall()
    return [
        {
            "event_id": row[0], "timestamp": row[1], "session_id": row[2],
            "dataset_id": row[3], "run_id": row[4], "stage": row[5],
            "kind": row[6], "parent_event_id": row[7],
            "payload": json.loads(row[8]),
        }
        for row in rows
    ]


def save_design_contract(
    contract: dict[str, Any], *, path: Path = INTERACTION_DB
) -> None:
    """Persist one immutable design-contract revision by its content hash."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=5) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS design_contracts (
                contract_hash TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                revision INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO design_contracts
            (contract_hash, dataset_id, revision, created_at, payload_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(contract["contract_hash"]), str(contract["dataset_id"]),
                int(contract["revision"]), datetime.now(timezone.utc).isoformat(),
                json.dumps(contract, sort_keys=True, default=str),
            ),
        )


def read_design_contracts(
    dataset_id: str, *, path: Path = INTERACTION_DB
) -> list[dict[str, Any]]:
    """Load immutable revisions for one exact dataset/table fingerprint."""
    if not path.exists():
        return []
    with sqlite3.connect(path, timeout=5) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='design_contracts'"
        ).fetchone()
        if not exists:
            return []
        rows = connection.execute(
            """
            SELECT payload_json FROM design_contracts
            WHERE dataset_id = ? ORDER BY revision, created_at
            """,
            (str(dataset_id),),
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


def save_analysis_run(
    record: dict[str, Any], *, path: Path = INTERACTION_DB
) -> None:
    """Persist minimal run lineage so post-result revisions survive UI restarts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path, timeout=5) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_runs (
                run_id TEXT PRIMARY KEY,
                dataset_id TEXT NOT NULL,
                data_version_id TEXT NOT NULL,
                contract_hash TEXT NOT NULL,
                parent_run_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO analysis_runs
            (run_id, dataset_id, data_version_id, contract_hash, parent_run_id,
             status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET status = excluded.status
            """,
            (
                str(record["run_id"]), str(record["dataset_id"]),
                str(record.get("data_version_id", "")),
                str(record.get("contract_hash", "")),
                str(record.get("parent_run_id", "")),
                str(record.get("status", "estimated")),
                str(record.get("created_at") or datetime.now(timezone.utc).isoformat()),
            ),
        )


def read_analysis_runs(
    dataset_id: str, *, path: Path = INTERACTION_DB
) -> list[dict[str, Any]]:
    """Read immutable run lineage for one exact bundle/table fingerprint."""
    if not path.exists():
        return []
    with sqlite3.connect(path, timeout=5) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='analysis_runs'"
        ).fetchone()
        if not exists:
            return []
        rows = connection.execute(
            """
            SELECT run_id, dataset_id, data_version_id, contract_hash,
                   parent_run_id, status, created_at
            FROM analysis_runs WHERE dataset_id = ?
            ORDER BY created_at, run_id
            """,
            (str(dataset_id),),
        ).fetchall()
    return [
        {
            "run_id": row[0], "dataset_id": row[1], "data_version_id": row[2],
            "contract_hash": row[3], "parent_run_id": row[4],
            "status": row[5], "created_at": row[6],
        }
        for row in rows
    ]


def summarize(state: dict[str, Any]) -> dict[str, Any]:
    prep = state.get("preparation") or {}
    diagnostics = state.get("diagnostics") or []
    preflight = state.get("preflight") or []
    contract = state.get("design_contract") or {}
    repairs = state.get("repairs") or []
    context = state.get("context") or {}
    quality = state.get("data_quality") or {}
    data_version = state.get("data_version") or {}
    before = next((r.get("rows_before") for r in repairs if r.get("rows_before")), None)
    after = repairs[-1].get("rows_after") if repairs else before
    row_loss = 0.0
    if before and after is not None:
        row_loss = max(0.0, (before - after) / before)
    return {
        "schema_version": MONITOR_SCHEMA_VERSION,
        "run_id": state.get("run_id", ""),
        "source": state.get("source", ""),
        "parent_run_id": state.get("parent_run_id", ""),
        "data_version_id": data_version.get("version_id", ""),
        "data_version_revision": data_version.get("revision"),
        "prepared_fingerprint": data_version.get("prepared_fingerprint", ""),
        "repair_manifest_hash": data_version.get("manifest_hash", ""),
        "lane": state.get("lane", ""),
        "preparation_mode": prep.get("mode", "manual"),
        "preparation_provider": prep.get("provider", ""),
        "preparation_project": prep.get("project", ""),
        "preparation_location": prep.get("location", ""),
        "preparation_model": prep.get("model", ""),
        "prompt_versions": state.get("prompt_versions", {}),
        "protocol_version": contract.get("protocol_version", ""),
        "contract_revision": contract.get("revision"),
        "contract_hash": contract.get("contract_hash", ""),
        "tables_seen": prep.get("tables_seen", 1),
        "tool_calls": len(prep.get("trace", [])),
        "repairs_applied": len(repairs),
        "row_loss_fraction": row_loss,
        "possible_pii_columns": len(quality.get("possible_pii_columns", [])),
        "assignment": context.get("assignment", ""),
        "diagnostic_failures": sum(d.get("verdict") == "fail" for d in diagnostics),
        "diagnostic_untestable": sum(d.get("verdict") == "untestable" for d in diagnostics),
        "preflight_failures": sum(d.get("verdict") == "fail" for d in preflight),
        "preflight_reviews": sum(d.get("verdict") in {"review", "warn", "untestable"} for d in preflight),
        "policy_decision": (state.get("policy") or {}).get("decision", "not_evaluated"),
        "analysis_error": state.get("error", ""),
    }


def alerts(summary: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if summary.get("row_loss_fraction", 0) > 0.20:
        out.append({"severity": "critical", "metric": "row_loss_fraction", "message": "Preparation removed more than 20% of rows."})
    elif summary.get("row_loss_fraction", 0) > 0.05:
        out.append({"severity": "warning", "metric": "row_loss_fraction", "message": "Preparation removed more than 5% of rows."})
    if summary.get("diagnostic_failures", 0):
        out.append({"severity": "critical", "metric": "diagnostic_failures", "message": "One or more causal diagnostics failed."})
    if summary.get("preflight_failures", 0):
        out.append({"severity": "critical", "metric": "preflight_failures", "message": "One or more pre-estimation design checks failed."})
    if summary.get("possible_pii_columns", 0):
        out.append({"severity": "warning", "metric": "possible_pii_columns", "message": "Possible PII remains in the analysis-ready dataset."})
    if summary.get("analysis_error"):
        out.append({"severity": "critical", "metric": "analysis_error", "message": "The selected lane failed to execute."})
    return out
