"""Static provider-field classification and source_field_index rows (§5.7; D-030)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from causal.intake.contracts import ContextClass, SemanticStatus

__all__ = ["FieldIndexRow", "classify_capture", "load_field_classes"]

FieldClasses = dict[tuple[str, str], ContextClass]


@dataclass(frozen=True)
class FieldIndexRow:
    scope_kind: str  # dataset|table|column
    table_name: str | None
    column_name: str | None
    field_or_slot_name: str
    context_class: ContextClass
    status: SemanticStatus
    evidence_count: int
    json_pointer: str


def load_field_classes(path: Path) -> FieldClasses:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("registry_version") != "kaggle-field-classes.v1":
        raise ValueError(f"unsupported field-class registry at {path}")
    return {
        (str(row["scope"]), str(row["field"])): ContextClass(row["context_class"])
        for row in document["fields"]
    }


def escape_pointer(token: str) -> str:
    """RFC 6901 token escaping for JSON pointers into immutable capture objects."""
    return token.replace("~", "~0").replace("/", "~1")


def _is_empty(value: object) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _status_for(context_class: ContextClass, value: object) -> SemanticStatus:
    if context_class in (ContextClass.OPERATIONAL, ContextClass.POPULARITY):
        return SemanticStatus.NOT_APPLICABLE
    if context_class is ContextClass.WITHHELD:
        return SemanticStatus.WITHHELD
    return SemanticStatus.EMPTY if _is_empty(value) else SemanticStatus.EVIDENCED


def _row(
    classes: FieldClasses, scope: str, table: str | None, column: str | None,
    field: str, value: object, pointer: str,
) -> FieldIndexRow:
    # Unlisted provider fields default to operational: fail-closed away from
    # model context (D-030), but still indexed with exactly one class.
    context_class = classes.get((scope, field), ContextClass.OPERATIONAL)
    status = _status_for(context_class, value)
    evidence = 1 if (
        context_class is ContextClass.SEMANTIC and status is SemanticStatus.EVIDENCED
    ) else 0
    return FieldIndexRow(scope, table, column, field, context_class, status, evidence, pointer)


def classify_capture(
    payload: dict[str, object], classes: FieldClasses
) -> tuple[FieldIndexRow, ...]:
    """One row per provider field in the frozen capture; deterministic (§5.7)."""
    rows: list[FieldIndexRow] = []
    metadata = payload.get("metadata_response")
    if isinstance(metadata, dict):
        for field, value in metadata.items():
            pointer = f"/metadata_response/{escape_pointer(field)}"
            rows.append(_row(classes, "dataset", None, None, field, value, pointer))
    files_response = payload.get("files_response")
    files = files_response.get("files") if isinstance(files_response, dict) else None
    for index, file_entry in enumerate(files if isinstance(files, list) else []):
        if not isinstance(file_entry, dict):
            continue
        table = str(file_entry.get("name", f"file-{index}"))
        for field, value in file_entry.items():
            if field == "columns":
                continue
            pointer = f"/files_response/files/{index}/{escape_pointer(field)}"
            rows.append(_row(classes, "table", table, None, field, value, pointer))
        columns = file_entry.get("columns")
        for col_index, column_entry in enumerate(columns if isinstance(columns, list) else []):
            if not isinstance(column_entry, dict):
                continue
            column = str(column_entry.get("name", f"column-{col_index}"))
            for field, value in column_entry.items():
                pointer = (f"/files_response/files/{index}/columns/{col_index}/"
                           f"{escape_pointer(field)}")
                rows.append(_row(classes, "column", table, column, field, value, pointer))
    return tuple(rows)
