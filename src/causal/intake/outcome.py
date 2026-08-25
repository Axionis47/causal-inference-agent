"""IntakeOutcome payload and the §11.1 handoff opening (T-008; D-033, D-037)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from causal.intake.catalog import CatalogStore
from causal.intake.contracts import AVAILABLE_STATUSES
from causal.intake.fields import FieldIndexRow
from causal.shared.contracts import ArtifactEnvelopeV1, ArtifactRef, HandoffManifestV1
from causal.shared.persistence import PersistenceError, ProductStore

__all__ = [
    "HANDOFF_UNAVAILABLE",
    "IDEMPOTENCY_CONFLICT",
    "IntakeError",
    "IntakeResult",
    "open_handoff",
    "outcome_payload",
]

IDEMPOTENCY_CONFLICT: Final = "idempotency_conflict"
HANDOFF_UNAVAILABLE: Final = "handoff_unavailable"
SURFACES: Final = (
    "structural_manifest", "semantic_available", "semantic_missing",
    "measured_fact_manifest", "provenance_manifest",
)


class IntakeError(ValueError):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class IntakeResult:
    analysis_id: str
    stage_run_id: str
    status: str
    outcome_artifact_id: str | None
    replayed: bool


def outcome_payload(
    *, status: str, analysis_id: str, stage_run_id: str, dataset_id: str | None,
    question: ArtifactEnvelopeV1, manifest: ArtifactEnvelopeV1 | None,
    profile_envs: tuple[ArtifactEnvelopeV1, ...], evidence: ArtifactEnvelopeV1 | None,
    map_env: ArtifactEnvelopeV1 | None, tables: tuple[str, ...],
    slot_rows: tuple[FieldIndexRow, ...], counts: dict[str, int],
    refusal_reason: str | None,
) -> dict[str, object]:
    """Identifiers and summary counts only (PRD-001 §10)."""
    available = sum(1 for row in slot_rows if row.status in AVAILABLE_STATUSES)
    missing = [
        {"scope_kind": row.scope_kind, "table_name": row.table_name,
         "column_name": row.column_name, "slot": row.field_or_slot_name,
         "status": row.status.value}
        for row in slot_rows if row.status not in AVAILABLE_STATUSES
    ]
    return {
        "schema_version": "intake-outcome.v1", "analysis_id": analysis_id,
        "stage_run_id": stage_run_id, "dataset_id": dataset_id,
        "handoff_contract_version": "handoff.v1",
        "field_classification_registry_version": "kaggle-field-classes.v1",
        "question_artifact_id": question.artifact_id,
        "source_manifest_artifact_id": manifest.artifact_id if manifest else None,
        "candidate_tables": list(tables),
        "table_profile_artifact_ids": [env.artifact_id for env in profile_envs],
        "evidence_bundle_artifact_id": evidence.artifact_id if evidence else None,
        "semantic_map_artifact_id": map_env.artifact_id if map_env else None,
        "retrieval_surfaces": {name: "catalog-view.v1" for name in SURFACES},
        "available_slot_count": available,
        "unavailable_slot_count": len(slot_rows) - available,
        "missing_semantic_slots": missing,
        "excluded_resource_count": counts.get("excluded", 0),
        "unreadable_resource_count": counts.get("unreadable", 0),
        "failed_resource_count": counts.get("failed", 0),
        "status": status, "refusal_reason": refusal_reason,
    }


def open_handoff(
    catalog: CatalogStore,
    products: ProductStore,
    analysis_id: str,
    intake_outcome_artifact_id: str,
    receiving_stage_run_id: str,
    clock: Callable[[], datetime],
) -> HandoffManifestV1:
    """§11.1: build the PRD-002 handoff from exactly these two identifiers.

    Never persisted here — the T-006 gate records the manifest at receipt
    (D-037). Refused, missing, or mismatched outcomes never open.
    """
    run = catalog.find_run_by_analysis(analysis_id)
    if (
        run is None
        or run.intake_status not in ("usable", "partial")
        or run.intake_outcome_artifact_id != intake_outcome_artifact_id
    ):
        raise IntakeError(f"no consumable handoff for {analysis_id!r}", HANDOFF_UNAVAILABLE)
    try:
        outcome = products.load_envelope(intake_outcome_artifact_id)
    except PersistenceError as error:
        raise IntakeError(
            f"outcome artifact unreadable for {analysis_id!r}", HANDOFF_UNAVAILABLE
        ) from error
    if outcome.artifact_type != "IntakeOutcome" or outcome.analysis_id != analysis_id:
        raise IntakeError(
            f"outcome artifact mismatched for {analysis_id!r}", HANDOFF_UNAVAILABLE
        )
    return HandoffManifestV1(
        handoff_id=f"ho:{analysis_id}:{outcome.content_hash[:16]}",
        schema_version="handoff.v1", analysis_id=analysis_id,
        producing_stage_run_id=run.stage_run_id,
        receiving_stage_run_id=receiving_stage_run_id,
        entries=(ArtifactRef(artifact_id=outcome.artifact_id,
                             content_hash=outcome.content_hash),),
        originating_outcome=str(run.intake_status), approval_ids=(),
        registry_version="artifact-types.v1", compatibility_version="handoff.v1",
        receiver_validation_result=None, receiver_error_codes=(),
        created_at_utc=clock(), accepted_at_utc=None)
