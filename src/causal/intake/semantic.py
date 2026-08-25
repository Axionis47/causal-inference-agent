"""Deterministic EvidenceBundle and SemanticMap builders (PRD-001 §7, §8; D-032)."""

from __future__ import annotations

from typing import Final

from causal.intake.contracts import (
    COLUMN_SLOTS,
    DATASET_SLOTS,
    ColumnSemanticsV1,
    ContextClass,
    DatasetSemanticsV1,
    SemanticSlotV1,
    SemanticStatus,
)
from causal.intake.fields import FieldClasses, FieldIndexRow, escape_pointer

__all__ = [
    "build_evidence_bundle",
    "build_semantic_map",
    "measured_index_rows",
    "slot_index_rows",
]

MAX_DOCUMENT_BYTES: Final = 1024 * 1024


def _column_descriptions(capture: dict[str, object]) -> dict[tuple[str, str], str | None]:
    """(table, column) -> provider description; None when offered blank."""
    descriptions: dict[tuple[str, str], str | None] = {}
    files_response = capture.get("files_response")
    files = files_response.get("files") if isinstance(files_response, dict) else None
    for file_entry in files if isinstance(files, list) else []:
        if not isinstance(file_entry, dict):
            continue
        table = str(file_entry.get("name", ""))
        columns = file_entry.get("columns")
        for column_entry in columns if isinstance(columns, list) else []:
            if not isinstance(column_entry, dict) or "description" not in column_entry:
                continue
            column = str(column_entry.get("name", ""))
            value = column_entry.get("description")
            text = str(value).strip() if value is not None else ""
            descriptions[(table, column)] = text or None
    return descriptions


def build_evidence_bundle(
    capture: dict[str, object], documents: dict[str, str], classes: FieldClasses
) -> dict[str, object]:
    """Cited semantic texts with stable evidence IDs; nothing interpreted."""
    items: list[dict[str, object]] = []

    def add(evidence_id: str, scope: str, table: str | None, column: str | None,
            source_field: str, value: str) -> None:
        items.append({
            "evidence_id": evidence_id, "scope_kind": scope, "table_name": table,
            "column_name": column, "source_field": source_field, "value": value})

    metadata = capture.get("metadata_response")
    for field, value in (metadata.items() if isinstance(metadata, dict) else []):
        semantic = classes.get(("dataset", field)) is ContextClass.SEMANTIC
        if semantic and value not in (None, "", [], {}):
            add(f"ev:kaggle/dataset/{field}", "dataset", None, None, field, str(value))
    files_response = capture.get("files_response")
    files = files_response.get("files") if isinstance(files_response, dict) else None
    for file_entry in files if isinstance(files, list) else []:
        if not isinstance(file_entry, dict):
            continue
        table = str(file_entry.get("name", ""))
        description = file_entry.get("description")
        if description not in (None, ""):
            add(f"ev:kaggle/file/{table}/description", "table", table, None,
                "description", str(description))
    for (table, column), text in sorted(_column_descriptions(capture).items()):
        if text is not None:
            add(f"ev:kaggle/column/{table}/{column}/description", "column", table,
                column, "description", text)
    for name in sorted(documents):
        text = documents[name]
        if len(text.encode("utf-8")) > MAX_DOCUMENT_BYTES:
            text = text.encode("utf-8")[:MAX_DOCUMENT_BYTES].decode("utf-8", "ignore")
        add(f"ev:doc/{name}", "dataset", None, None, name, text)
    return {"schema_version": "evidence-bundle.v1", "items": items}


def _meaning_slot(description: str | None, offered: bool, table: str, column: str) -> SemanticSlotV1:
    if description:
        evidence_id = f"ev:kaggle/column/{table}/{column}/description"
        return SemanticSlotV1(
            status=SemanticStatus.EVIDENCED, value=description, evidence_ids=(evidence_id,)
        )
    if offered:
        return SemanticSlotV1(status=SemanticStatus.EMPTY)
    return SemanticSlotV1(status=SemanticStatus.NOT_OFFERED)


def _hypothesis_slot(profile_column: dict[str, object], kind: str) -> SemanticSlotV1 | None:
    hypotheses = profile_column.get("hypotheses")
    for hypothesis in hypotheses if isinstance(hypotheses, list) else []:
        if isinstance(hypothesis, dict) and hypothesis.get("kind") == kind:
            return SemanticSlotV1(
                status=SemanticStatus.HYPOTHESIS, value=str(hypothesis.get("detail", kind))
            )
    return None


def build_semantic_map(
    capture: dict[str, object], profiles: dict[str, dict[str, object]]
) -> dict[str, object]:
    """v0 conservative slot mapping (D-032); every slot present with one status."""
    not_offered = SemanticSlotV1(status=SemanticStatus.NOT_OFFERED)
    dataset = DatasetSemanticsV1(slots=dict.fromkeys(DATASET_SLOTS, not_offered))
    descriptions = _column_descriptions(capture)
    columns: list[dict[str, object]] = []
    for table in sorted(profiles):
        profile_columns = profiles[table].get("columns")
        if not isinstance(profile_columns, dict):
            continue
        for column in profile_columns:
            offered = (table, column) in descriptions
            slots: dict[str, SemanticSlotV1] = dict.fromkeys(COLUMN_SLOTS, not_offered)
            slots["meaning"] = _meaning_slot(
                descriptions.get((table, column)), offered, table, column
            )
            for slot_name, kind in (("missing_sentinel", "missing_sentinel"),
                                    ("kind", "identifier")):
                hypothesis = _hypothesis_slot(profile_columns[column], kind)
                if hypothesis is not None:
                    slots[slot_name] = hypothesis
            semantics = ColumnSemanticsV1(column_name=column, slots=slots)
            columns.append({"table_name": table, **semantics.model_dump(mode="json")})
    return {
        "schema_version": "semantic-map.v1",
        "dataset": dataset.model_dump(mode="json"),
        "columns": columns,
    }


def measured_index_rows(
    profiles: dict[str, dict[str, object]], profile_artifact_ids: dict[str, str]
) -> tuple[FieldIndexRow, ...]:
    """One measured-class row per profiled column, pointing at its profile (D-037)."""
    rows: list[FieldIndexRow] = []
    for table in sorted(profiles):
        columns = profiles[table].get("columns")
        for column in columns if isinstance(columns, dict) else {}:
            pointer = f"{profile_artifact_ids[table]}#/columns/{escape_pointer(column)}"
            rows.append(FieldIndexRow(
                "column", table, column, "profile", ContextClass.MEASURED,
                SemanticStatus.EVIDENCED, 0, pointer))
    return tuple(rows)


def slot_index_rows(semantic_map: dict[str, object]) -> tuple[FieldIndexRow, ...]:
    """One source_field_index row per semantic slot, pointing into the map object."""
    rows: list[FieldIndexRow] = []

    def add(scope: str, table: str | None, column: str | None, slot_name: str,
            slot: dict[str, object], pointer: str) -> None:
        evidence_ids = slot.get("evidence_ids")
        rows.append(FieldIndexRow(
            scope, table, column, slot_name, ContextClass.SEMANTIC,
            SemanticStatus(str(slot["status"])),
            len(evidence_ids) if isinstance(evidence_ids, list) else 0, pointer))

    dataset = semantic_map["dataset"]
    if isinstance(dataset, dict) and isinstance(dataset.get("slots"), dict):
        for slot_name, slot in dataset["slots"].items():
            add("dataset", None, None, slot_name, slot,
                f"/dataset/slots/{escape_pointer(slot_name)}")
    columns = semantic_map["columns"]
    for index, entry in enumerate(columns if isinstance(columns, list) else []):
        if not isinstance(entry, dict) or not isinstance(entry.get("slots"), dict):
            continue
        table, column = str(entry.get("table_name")), str(entry.get("column_name"))
        for slot_name, slot in entry["slots"].items():
            add("column", table, column, slot_name, slot,
                f"/columns/{index}/slots/{escape_pointer(slot_name)}")
    return tuple(rows)
