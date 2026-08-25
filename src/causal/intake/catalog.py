"""CatalogStore: the intake-owned `catalog` schema (PRD-001 §9.3, §9.4; D-029)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from psycopg import Connection

from causal.intake.fields import FieldIndexRow

__all__ = ["CatalogStore", "RunRow"]


@dataclass(frozen=True)
class RunRow:
    analysis_id: str
    stage_run_id: str
    submission_hash: str
    intake_status: str | None
    intake_outcome_artifact_id: str | None


class CatalogStore:
    def __init__(self, conn: Connection[Any]) -> None:
        self._conn = conn

    def _write(self, sql: str, params: tuple[object, ...]) -> None:
        with self._conn.transaction():
            self._conn.execute(sql, params)

    def _run_row(self, where: str, value: str) -> RunRow | None:
        row = self._conn.execute(
            "SELECT analysis_id, stage_run_id, submission_hash, intake_status,"
            f" intake_outcome_artifact_id FROM catalog.runs WHERE {where} = %s",
            (value,)).fetchone()
        return None if row is None else RunRow(
            str(row[0]), str(row[1]), str(row[2]), row[3], row[4])

    def find_run(self, idempotency_key: str) -> RunRow | None:
        return self._run_row("idempotency_key", idempotency_key)

    def find_run_by_analysis(self, analysis_id: str) -> RunRow | None:
        return self._run_row("analysis_id", analysis_id)

    def count_stage_runs(self, analysis_id: str) -> int:
        row = self._conn.execute(
            "SELECT count(*) FROM causal.stage_runs WHERE analysis_id = %s",
            (analysis_id,)).fetchone()
        return int(row[0]) if row else 0

    def create_run(
        self, analysis_id: str, stage_run_id: str, question_artifact_id: str,
        idempotency_key: str, submission_hash: str, created_at_utc: datetime,
    ) -> None:
        self._write(
            "INSERT INTO catalog.runs (analysis_id, stage_run_id, question_artifact_id,"
            " idempotency_key, submission_hash, created_at_utc) VALUES (%s, %s, %s, %s, %s, %s)",
            (analysis_id, stage_run_id, question_artifact_id, idempotency_key,
             submission_hash, created_at_utc))

    def reassign_run(self, analysis_id: str, stage_run_id: str) -> None:
        self._write("UPDATE catalog.runs SET stage_run_id = %s WHERE analysis_id = %s",
                    (stage_run_id, analysis_id))

    def set_run_dataset(self, analysis_id: str, dataset_id: str) -> None:
        self._write("UPDATE catalog.runs SET dataset_id = %s WHERE analysis_id = %s",
                    (dataset_id, analysis_id))

    def upsert_dataset(self, dataset: dict[str, object], capture_artifact_id: str) -> None:
        self._write(
            "INSERT INTO catalog.datasets (dataset_id, provider, owner, slug, version,"
            " provider_status, capture_artifact_id) VALUES (%s, %s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (dataset_id) DO NOTHING",
            (dataset["dataset_id"], dataset["provider"], dataset["owner"], dataset["slug"],
             dataset["version"], dataset["status"], capture_artifact_id))

    def set_source_manifest(self, dataset_id: str, artifact_id: str) -> None:
        self._write(
            "UPDATE catalog.datasets SET source_manifest_artifact_id = %s"
            " WHERE dataset_id = %s", (artifact_id, dataset_id))

    def insert_resource(
        self, resource_id: str, dataset_id: str, kind: str, logical_name: str,
        object_sha256: str, object_key: str, media_type: str | None, byte_size: int,
        parse_status: str, reason: str | None,
    ) -> None:
        self._write(
            "INSERT INTO catalog.resources VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
            " ON CONFLICT (dataset_id, logical_name) DO NOTHING",
            (resource_id, dataset_id, kind, logical_name, object_sha256, object_key,
             media_type, byte_size, parse_status, reason))

    def replace_field_rows(
        self, dataset_id: str, capture_artifact_id: str,
        semantic_map_artifact_id: str | None, rows: tuple[FieldIndexRow, ...],
    ) -> None:
        """Rebuild the read index for one dataset in one transaction (§9.3)."""
        with self._conn.transaction():
            self._conn.execute(
                "DELETE FROM catalog.source_field_index WHERE dataset_id = %s", (dataset_id,))
            for row in rows:
                self._conn.execute(
                    "INSERT INTO catalog.source_field_index VALUES"
                    " (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (dataset_id, capture_artifact_id, semantic_map_artifact_id,
                     row.scope_kind, row.table_name, row.column_name,
                     row.field_or_slot_name, row.context_class.value, row.status.value,
                     row.evidence_count, row.json_pointer))

    def finalize_run(
        self, analysis_id: str, intake_status: str, outcome_artifact_id: str
    ) -> None:
        """§11.1(7): status and outcome pointer commit in the same transaction."""
        self._write(
            "UPDATE catalog.runs SET intake_status = %s, intake_outcome_artifact_id = %s"
            " WHERE analysis_id = %s", (intake_status, outcome_artifact_id, analysis_id))
