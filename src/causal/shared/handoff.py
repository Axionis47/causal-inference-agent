"""Handoff persistence and the fail-closed acceptance gate (T-006; D-021, D-022)."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

from psycopg import Connection

from causal.shared.contracts import ArtifactRef, HandoffManifestV1
from causal.shared.events import EventEmitter, OperationalEventV1
from causal.shared.persistence import ObjectStore, PersistenceError, ProductStore
from causal.shared.registry import ArtifactTypeRegistry, RegistryError

__all__ = ["HandoffGate", "HandoffResult", "HandoffStore"]

MISSING_ARTIFACT: Final = "missing_artifact"
ENTRY_HASH_MISMATCH: Final = "entry_hash_mismatch"
MISSING_OBJECT: Final = "missing_object"
WRONG_OUTCOME: Final = "wrong_outcome"
READER_NOT_ALLOWED: Final = "reader_not_allowed"
INCOMPLETE_LINEAGE: Final = "incomplete_lineage"
UNSUPPORTED_VERSION: Final = "unsupported_version"
UNKNOWN_HANDOFF: Final = "unknown_handoff"
DUPLICATE_HANDOFF: Final = "duplicate_handoff"

SUPPORTED_SCHEMA_VERSION: Final = "handoff.v1"

EventFactory = Callable[[str, tuple[str, ...]], OperationalEventV1]


@dataclass(frozen=True)
class HandoffResult:
    accepted: bool
    error_codes: tuple[str, ...]


class HandoffStore:
    """PostgreSQL visibility for handoff manifests (§8.1)."""

    def __init__(self, conn: Connection[Any]) -> None:
        self._conn = conn

    def record(self, manifest: HandoffManifestV1) -> None:
        if self._row(manifest.handoff_id) is not None:
            raise PersistenceError(
                f"handoff {manifest.handoff_id!r} already recorded", DUPLICATE_HANDOFF
            )
        with self._conn.transaction():
            self._conn.execute(
                "INSERT INTO causal.handoffs VALUES"
                " (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    manifest.handoff_id, manifest.schema_version, manifest.analysis_id,
                    manifest.producing_stage_run_id, manifest.receiving_stage_run_id,
                    manifest.originating_outcome, list(manifest.approval_ids),
                    manifest.registry_version, manifest.compatibility_version,
                    manifest.receiver_validation_result,
                    list(manifest.receiver_error_codes), manifest.created_at_utc,
                    manifest.accepted_at_utc,
                ),
            )
            for index, entry in enumerate(manifest.entries):
                self._conn.execute(
                    "INSERT INTO causal.handoff_entries VALUES (%s, %s, %s, %s)",
                    (manifest.handoff_id, index, entry.artifact_id, entry.content_hash),
                )

    def _row(self, handoff_id: str) -> tuple[Any, ...] | None:
        return self._conn.execute(
            "SELECT handoff_id, schema_version, analysis_id, producing_stage_run_id,"
            " receiving_stage_run_id, originating_outcome, approval_ids, registry_version,"
            " compatibility_version, receiver_validation_result, receiver_error_codes,"
            " created_at_utc, accepted_at_utc FROM causal.handoffs WHERE handoff_id = %s",
            (handoff_id,),
        ).fetchone()

    def load(self, handoff_id: str) -> HandoffManifestV1:
        row = self._row(handoff_id)
        if row is None:
            raise PersistenceError(f"unknown handoff {handoff_id!r}", UNKNOWN_HANDOFF)
        entries = self._conn.execute(
            "SELECT artifact_id, content_hash FROM causal.handoff_entries"
            " WHERE handoff_id = %s ORDER BY entry_index",
            (handoff_id,),
        ).fetchall()
        return HandoffManifestV1(
            handoff_id=str(row[0]), schema_version=str(row[1]), analysis_id=str(row[2]),
            producing_stage_run_id=str(row[3]), receiving_stage_run_id=str(row[4]),
            originating_outcome=str(row[5]), approval_ids=tuple(row[6]),
            registry_version=str(row[7]), compatibility_version=str(row[8]),
            receiver_validation_result=row[9], receiver_error_codes=tuple(row[10]),
            created_at_utc=row[11], accepted_at_utc=row[12],
            entries=tuple(
                ArtifactRef(artifact_id=str(e[0]), content_hash=str(e[1])) for e in entries
            ),
        )

    def mark(self, handoff_id: str, result: str, error_codes: tuple[str, ...]) -> None:
        accepted_at = datetime.now(tz=UTC) if result == "accepted" else None
        with self._conn.transaction():
            updated = self._conn.execute(
                "UPDATE causal.handoffs SET receiver_validation_result = %s,"
                " receiver_error_codes = %s, accepted_at_utc = %s WHERE handoff_id = %s",
                (result, list(error_codes), accepted_at, handoff_id),
            )
            if updated.rowcount != 1:
                raise PersistenceError(f"unknown handoff {handoff_id!r}", UNKNOWN_HANDOFF)


class HandoffGate:
    """The §3 receiver: opens a handoff only when every check passes; fails closed."""

    def __init__(
        self,
        object_store: ObjectStore,
        product_store: ProductStore,
        handoff_store: HandoffStore,
        registry: ArtifactTypeRegistry,
        emitter: EventEmitter,
    ) -> None:
        self._objects = object_store
        self._products = product_store
        self._handoffs = handoff_store
        self._registry = registry
        self._emitter = emitter

    def _check_entry(self, entry: ArtifactRef, receiving_component: str) -> set[str]:
        codes: set[str] = set()
        try:
            envelope = self._products.load_envelope(entry.artifact_id)
        except PersistenceError:
            return {MISSING_ARTIFACT}
        if envelope.content_hash != entry.content_hash:
            codes.add(ENTRY_HASH_MISMATCH)
        try:
            data = self._objects.get(envelope.payload_locator)
            if hashlib.sha256(data).hexdigest() != entry.content_hash:
                codes.add(MISSING_OBJECT)
        except Exception:  # noqa: BLE001 -- any retrieval failure is a missing object
            codes.add(MISSING_OBJECT)
        try:
            registration = self._registry.lookup(envelope.artifact_type)
        except RegistryError:
            return codes | {READER_NOT_ALLOWED}
        if receiving_component not in registration.allowed_reader_components:
            codes.add(READER_NOT_ALLOWED)
        committed_parent_types = set(
            self._products.artifact_types_of(envelope.parent_artifacts).values()
        )
        if set(registration.required_parent_types) - committed_parent_types:
            codes.add(INCOMPLETE_LINEAGE)
        return codes

    def accept(
        self,
        manifest: HandoffManifestV1,
        receiving_component: str,
        allowed_outcomes: frozenset[str],
        event_factory: EventFactory,
    ) -> HandoffResult:
        codes: set[str] = set()
        if manifest.schema_version != SUPPORTED_SCHEMA_VERSION:
            codes.add(UNSUPPORTED_VERSION)
        if manifest.originating_outcome not in allowed_outcomes:
            codes.add(WRONG_OUTCOME)
        for entry in manifest.entries:
            codes |= self._check_entry(entry, receiving_component)
        result = HandoffResult(accepted=not codes, error_codes=tuple(sorted(codes)))
        self._handoffs.record(manifest)
        verdict = "accepted" if result.accepted else "rejected"
        self._handoffs.mark(manifest.handoff_id, verdict, result.error_codes)
        self._emitter.emit(event_factory(verdict, result.error_codes))
        return result
