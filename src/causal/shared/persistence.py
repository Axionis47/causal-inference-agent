"""Persistence layer and §8.2 artifact commit protocol (T-005; D-017..D-020)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Final, Protocol

from psycopg import Connection

from causal.shared.canonical import canonical_bytes, content_hash
from causal.shared.contracts import ArtifactEnvelopeV1, ArtifactRef, SensitivityClass
from causal.shared.events import EventEmitter, OperationalEventV1
from causal.shared.registry import ArtifactTypeRegistry
from causal.shared.tracing import TracerProtocol

__all__ = [
    "ArtifactCommitter",
    "ObjectStore",
    "PersistenceError",
    "ProductStore",
    "S3ClientProtocol",
    "apply_migrations",
    "build_envelope",
]

INTEGRITY_CONFLICT: Final = "integrity_conflict"
ARTIFACT_HASH_MISMATCH: Final = "artifact_hash_mismatch"
LOCATOR_MISMATCH: Final = "locator_mismatch"
REGISTRY_VIOLATION: Final = "registry_violation"
MISSING_REQUIRED_PARENT: Final = "missing_required_parent"
REOPEN_VALIDATION_FAILED: Final = "reopen_validation_failed"
ILLEGAL_STATE_TRANSITION: Final = "illegal_state_transition"
UNKNOWN_STAGE_RUN: Final = "unknown_stage_run"

# Legal run-state transitions (SYSTEM-CONTRACT §4; D-018).
RUN_STATE_TRANSITIONS: Final[dict[str, frozenset[str]]] = {
    "created": frozenset({"tracing_preflight"}),
    "tracing_preflight": frozenset({"running", "failed_observability", "failed"}),
    "running": frozenset(
        {"waiting_for_user", "committing", "completed", "failed_observability", "failed"}
    ),
    "waiting_for_user": frozenset({"running", "failed"}),
    "committing": frozenset({"running", "completed", "failed_observability", "failed"}),
    "completed": frozenset(),
    "failed": frozenset(),
    "failed_observability": frozenset(),
}


class PersistenceError(ValueError):
    """A persistence operation failed. `code` is a stable contract value."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class S3ClientProtocol(Protocol):
    """The subset of a boto3 S3 client the object store uses (D-017)."""

    def head_object(self, *, Bucket: str, Key: str) -> Any: ...

    def put_object(self, *, Bucket: str, Key: str, Body: bytes) -> Any: ...

    def get_object(self, *, Bucket: str, Key: str) -> Any: ...


class ObjectStore:
    """Content-addressed immutable object storage (D-019 key layout)."""

    def __init__(self, client: S3ClientProtocol, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    @staticmethod
    def locator_for(digest: str) -> str:
        return f"objects/{digest}"

    def put_if_absent(self, digest: str, data: bytes) -> str:
        key = self.locator_for(digest)
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return key  # immutable content-addressed object already present
        except Exception:  # noqa: BLE001 -- absent (or unreachable; the put would fail too)
            self._client.put_object(Bucket=self._bucket, Key=key, Body=data)
            return key

    def get(self, locator: str) -> bytes:
        response = self._client.get_object(Bucket=self._bucket, Key=locator)
        return bytes(response["Body"].read())


def build_envelope(
    registry: ArtifactTypeRegistry,
    artifact_type: str,
    payload: dict[str, object],
    *,
    analysis_id: str,
    stage_run_id: str,
    producer_version: str,
    parents: tuple[ArtifactEnvelopeV1, ...],
    created_at_utc: Any,
    producer_component: str | None = None,
) -> ArtifactEnvelopeV1:
    """Deterministic envelope from a registration and payload (D-031 identities)."""
    registration = registry.lookup(artifact_type)
    digest = content_hash(payload)
    return ArtifactEnvelopeV1(
        artifact_id=f"{artifact_type.lower()}:{analysis_id}:{digest[:16]}",
        artifact_type=artifact_type,
        schema_version=registration.schema_version,
        content_hash=digest,
        analysis_id=analysis_id,
        stage_run_id=stage_run_id,
        producer_component=producer_component or registration.producer_component,
        producer_version=producer_version,
        parent_artifacts=tuple(
            ArtifactRef(artifact_id=p.artifact_id, content_hash=p.content_hash)
            for p in parents
        ),
        sensitivity_class=registration.sensitivity_class,
        created_at_utc=created_at_utc,
        payload_locator=ObjectStore.locator_for(digest),
    )


def apply_migrations(conn: Connection[Any], migrations_dir: Path) -> None:
    """Execute every *.sql file in name order inside one transaction."""
    with conn.transaction():
        for path in sorted(migrations_dir.glob("*.sql")):
            conn.execute(path.read_text(encoding="utf-8"))


class ProductStore:
    """PostgreSQL authority for identities, pointers, lineage, and run state (§8.1)."""

    def __init__(self, conn: Connection[Any]) -> None:
        self._conn = conn

    def create_stage_run(self, stage_run_id: str, analysis_id: str, stage: str) -> None:
        with self._conn.transaction():
            self._conn.execute(
                "INSERT INTO causal.stage_runs VALUES (%s, %s, %s, 'created', now(), now())",
                (stage_run_id, analysis_id, stage),
            )

    def get_stage_run_state(self, stage_run_id: str) -> str:
        row = self._conn.execute(
            "SELECT run_state FROM causal.stage_runs WHERE stage_run_id = %s", (stage_run_id,)
        ).fetchone()
        if row is None:
            raise PersistenceError(f"unknown stage run {stage_run_id!r}", UNKNOWN_STAGE_RUN)
        return str(row[0])

    def transition_stage_run(self, stage_run_id: str, new_state: str) -> None:
        with self._conn.transaction():
            row = self._conn.execute(
                "SELECT run_state FROM causal.stage_runs WHERE stage_run_id = %s FOR UPDATE",
                (stage_run_id,),
            ).fetchone()
            if row is None:
                raise PersistenceError(f"unknown stage run {stage_run_id!r}", UNKNOWN_STAGE_RUN)
            current = str(row[0])
            if new_state not in RUN_STATE_TRANSITIONS[current]:
                raise PersistenceError(
                    f"illegal transition {current!r} -> {new_state!r} for {stage_run_id!r}",
                    ILLEGAL_STATE_TRANSITION,
                )
            self._conn.execute(
                "UPDATE causal.stage_runs SET run_state = %s, updated_at_utc = now()"
                " WHERE stage_run_id = %s",
                (new_state, stage_run_id),
            )

    def find_artifact_hash(self, artifact_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT content_hash FROM causal.artifacts WHERE artifact_id = %s", (artifact_id,)
        ).fetchone()
        return None if row is None else str(row[0])

    def artifact_types_of(self, refs: tuple[ArtifactRef, ...]) -> dict[str, str]:
        if not refs:
            return {}
        ids = [ref.artifact_id for ref in refs]
        rows = self._conn.execute(
            "SELECT artifact_id, artifact_type FROM causal.artifacts WHERE artifact_id = ANY(%s)",
            (ids,),
        ).fetchall()
        return {str(row[0]): str(row[1]) for row in rows}

    def insert_artifact(self, envelope: ArtifactEnvelopeV1) -> None:
        with self._conn.transaction():
            self._conn.execute(
                "INSERT INTO causal.artifacts VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                (
                    envelope.artifact_id, envelope.artifact_type, envelope.schema_version,
                    envelope.content_hash, envelope.analysis_id, envelope.stage_run_id,
                    envelope.producer_component, envelope.producer_version,
                    envelope.sensitivity_class.value, envelope.created_at_utc,
                    envelope.payload_locator,
                ),
            )
            for index, parent in enumerate(envelope.parent_artifacts):
                self._conn.execute(
                    "INSERT INTO causal.artifact_parents VALUES (%s, %s, %s, %s)",
                    (envelope.artifact_id, index, parent.artifact_id, parent.content_hash),
                )

    def load_envelope(self, artifact_id: str) -> ArtifactEnvelopeV1:
        row = self._conn.execute(
            "SELECT artifact_id, artifact_type, schema_version, content_hash, analysis_id,"
            " stage_run_id, producer_component, producer_version, sensitivity_class,"
            " created_at_utc, payload_locator FROM causal.artifacts WHERE artifact_id = %s",
            (artifact_id,),
        ).fetchone()
        if row is None:
            raise PersistenceError(f"unknown artifact {artifact_id!r}", UNKNOWN_STAGE_RUN)
        parents = self._conn.execute(
            "SELECT parent_artifact_id, parent_content_hash FROM causal.artifact_parents"
            " WHERE artifact_id = %s ORDER BY parent_index",
            (artifact_id,),
        ).fetchall()
        return ArtifactEnvelopeV1(
            artifact_id=str(row[0]), artifact_type=str(row[1]), schema_version=str(row[2]),
            content_hash=str(row[3]), analysis_id=str(row[4]), stage_run_id=str(row[5]),
            producer_component=str(row[6]), producer_version=str(row[7]),
            sensitivity_class=SensitivityClass(row[8]), created_at_utc=row[9],
            payload_locator=str(row[10]),
            parent_artifacts=tuple(
                ArtifactRef(artifact_id=str(p[0]), content_hash=str(p[1])) for p in parents
            ),
        )


class ArtifactCommitter:
    """The §8.2 commit protocol; an attached tracer must acknowledge a flush (T-010)."""

    def __init__(
        self,
        object_store: ObjectStore,
        product_store: ProductStore,
        registry: ArtifactTypeRegistry,
        emitter: EventEmitter,
        *,
        tracer: TracerProtocol | None = None,
    ) -> None:
        self._objects = object_store
        self._products = product_store
        self._registry = registry
        self._emitter = emitter
        self._tracer = tracer
        self._flushes = 0

    @property
    def spans_acknowledged(self) -> bool:
        """Whether this run's trace spans reached LangSmith.

        LangGraph's node instrumentation fills the trace queue (PRD-002 §20.4, D-097) and each
        commit flushes it, raising `ObservabilityError` on a failure. So an attached tracer plus
        one completed flush means nothing was lost. No tracer means the run required none —
        composition is fail-closed when tracing is required (SC §10.2).
        """
        return self._tracer is None or self._flushes > 0

    def _validate(self, envelope: ArtifactEnvelopeV1, payload: dict[str, object]) -> bytes:
        digest = content_hash(payload)
        if digest != envelope.content_hash:
            raise PersistenceError(
                f"payload hash {digest} != envelope hash {envelope.content_hash}",
                ARTIFACT_HASH_MISMATCH,
            )
        if envelope.payload_locator != ObjectStore.locator_for(digest):
            raise PersistenceError(
                f"locator {envelope.payload_locator!r} violates D-019", LOCATOR_MISMATCH
            )
        registration = self._registry.lookup(envelope.artifact_type)
        mismatches = [
            name
            for name, actual, expected in (
                # D-086: any registered producer may stamp itself, and only itself.
                ("producer_component", envelope.producer_component in {
                    registration.producer_component, *registration.also_produced_by}, True),
                ("schema_version", envelope.schema_version, registration.schema_version),
                ("sensitivity_class", envelope.sensitivity_class,
                 registration.sensitivity_class),
            )
            if actual != expected
        ]
        if mismatches:
            raise PersistenceError(
                f"envelope disagrees with registration on {mismatches}", REGISTRY_VIOLATION
            )
        parent_types = set(
            self._products.artifact_types_of(envelope.parent_artifacts).values()
        )
        missing = set(registration.required_parent_types) - parent_types
        if missing:
            raise PersistenceError(
                f"required parent types not present: {sorted(missing)}", MISSING_REQUIRED_PARENT
            )
        return canonical_bytes(payload)

    def commit(
        self,
        envelope: ArtifactEnvelopeV1,
        payload: dict[str, object],
        event: OperationalEventV1,
    ) -> ArtifactEnvelopeV1:
        data = self._validate(envelope, payload)
        existing_hash = self._products.find_artifact_hash(envelope.artifact_id)
        if existing_hash is not None:
            if existing_hash == envelope.content_hash:
                return self._products.load_envelope(envelope.artifact_id)  # §8.2 replay no-op
            raise PersistenceError(
                f"artifact {envelope.artifact_id!r} exists with hash {existing_hash}",
                INTEGRITY_CONFLICT,
            )
        self._objects.put_if_absent(envelope.content_hash, data)
        self._products.insert_artifact(envelope)
        reopened = self._objects.get(envelope.payload_locator)
        if hashlib.sha256(reopened).hexdigest() != envelope.content_hash:
            raise PersistenceError(
                f"reopened object hash mismatch for {envelope.artifact_id!r}",
                REOPEN_VALIDATION_FAILED,
            )
        self._emitter.emit(event)
        if self._tracer is not None:
            # SC §8.2: the commit completes only after an acknowledged flush. The row and the
            # object are already durable here, so ObservabilityError propagates to the caller
            # (which maps it to failed_observability) over a preserved artifact.
            self._tracer.flush()
            self._flushes += 1
        return envelope
