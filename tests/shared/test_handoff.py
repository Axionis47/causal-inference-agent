"""Integration tests for the handoff store and acceptance gate (T-006)."""

from __future__ import annotations

import io
import json
import uuid
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef, HandoffManifestV1
from causal.shared.events import EventEmitter, OperationalEventV1
from causal.shared.handoff import HandoffGate, HandoffStore
from causal.shared.persistence import ArtifactCommitter, ObjectStore, PersistenceError, ProductStore
from causal.shared.registry import ArtifactTypeRegistry
from tests.shared.conftest import requires_docker
from tests.shared.test_persistence import (
    committed_event,
    envelope_for,
    make_committer,
    registration,
)

NOW = datetime(2026, 8, 24, 16, 0, 0, 0, tzinfo=UTC)

pytestmark = requires_docker

ALLOWED = frozenset({"usable", "partial"})


def manifest_for(entries: tuple[ArtifactRef, ...], **overrides: object) -> HandoffManifestV1:
    base: dict[str, object] = {
        "handoff_id": f"h-{uuid.uuid4().hex[:8]}",
        "schema_version": "handoff.v1",
        "analysis_id": "an-1",
        "producing_stage_run_id": "run-1",
        "receiving_stage_run_id": "run-2",
        "entries": entries,
        "originating_outcome": "usable",
        "approval_ids": (),
        "registry_version": "artifact-types.v1",
        "compatibility_version": "compat.v1",
        "receiver_validation_result": None,
        "receiver_error_codes": (),
        "created_at_utc": NOW,
        "accepted_at_utc": None,
    }
    base.update(overrides)
    return HandoffManifestV1(**base)  # type: ignore[arg-type]


def gate_event(verdict: str, codes: tuple[str, ...]) -> OperationalEventV1:
    return committed_event(
        event_name=f"handoff.{verdict}",
        status=verdict,
        required_eval_ids=("EV-SYS-006",),
        error_code=codes[0] if codes else None,
    )


def build_gate(
    object_store: ObjectStore,
    conn: psycopg.Connection[Any],
    registrations: tuple[Any, ...] = (),
) -> tuple[HandoffGate, ArtifactCommitter, ProductStore, io.StringIO]:
    committer, products, _ = make_committer(object_store, conn)
    sink = io.StringIO()
    gate = HandoffGate(
        object_store, products, HandoffStore(conn),
        ArtifactTypeRegistry(registrations or (registration(),)), EventEmitter(sink),
    )
    return gate, committer, products, sink


def commit_entry(
    committer: ArtifactCommitter, *, artifact_id: str = "art-1"
) -> tuple[ArtifactRef, dict[str, object]]:
    payload: dict[str, object] = {"id": artifact_id}
    committer.commit(
        envelope_for(payload, artifact_id=artifact_id), payload, committed_event()
    )
    return ArtifactRef(artifact_id=artifact_id, content_hash=content_hash(payload)), payload


class TestHandoffStore:
    def test_record_load_roundtrip(self, conn: psycopg.Connection[Any]) -> None:
        store = HandoffStore(conn)
        manifest = manifest_for(
            (ArtifactRef(artifact_id="a", content_hash="0" * 64),), handoff_id="h-rt"
        )
        store.record(manifest)
        assert store.load("h-rt") == manifest

    def test_duplicate_rejected(self, conn: psycopg.Connection[Any]) -> None:
        store = HandoffStore(conn)
        manifest = manifest_for(
            (ArtifactRef(artifact_id="a", content_hash="0" * 64),), handoff_id="h-dup"
        )
        store.record(manifest)
        with pytest.raises(PersistenceError) as excinfo:
            store.record(manifest)
        assert excinfo.value.code == "duplicate_handoff"

    def test_unknown_handoff(self, conn: psycopg.Connection[Any]) -> None:
        with pytest.raises(PersistenceError) as excinfo:
            HandoffStore(conn).load("h-none")
        assert excinfo.value.code == "unknown_handoff"


class TestHandoffGate:
    def test_happy_path(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        gate, committer, _, sink = build_gate(object_store, conn)
        ref, _ = commit_entry(committer)
        manifest = manifest_for((ref,))
        result = gate.accept(manifest, "design-harness", ALLOWED, gate_event)
        assert result.accepted and result.error_codes == ()
        stored = HandoffStore(conn).load(manifest.handoff_id)
        assert stored.receiver_validation_result == "accepted"
        assert stored.accepted_at_utc is not None
        assert json.loads(sink.getvalue())["event_name"] == "handoff.accepted"

    def test_missing_artifact(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        gate, _, _, sink = build_gate(object_store, conn)
        ghost = ArtifactRef(artifact_id="ghost", content_hash="0" * 64)
        result = gate.accept(manifest_for((ghost,)), "design-harness", ALLOWED, gate_event)
        assert not result.accepted and result.error_codes == ("missing_artifact",)
        assert json.loads(sink.getvalue())["event_name"] == "handoff.rejected"

    def test_entry_hash_mismatch(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        gate, committer, _, _ = build_gate(object_store, conn)
        ref, _ = commit_entry(committer)
        lying = ArtifactRef(artifact_id=ref.artifact_id, content_hash="f" * 64)
        result = gate.accept(manifest_for((lying,)), "design-harness", ALLOWED, gate_event)
        assert not result.accepted
        assert "entry_hash_mismatch" in result.error_codes
        assert "missing_object" in result.error_codes  # object re-hash also disagrees

    def test_wrong_outcome(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        gate, committer, _, _ = build_gate(object_store, conn)
        ref, _ = commit_entry(committer)
        manifest = manifest_for((ref,), originating_outcome="refused")
        result = gate.accept(manifest, "design-harness", ALLOWED, gate_event)
        assert result.error_codes == ("wrong_outcome",)

    def test_reader_not_allowed(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        gate, committer, _, _ = build_gate(object_store, conn)
        ref, _ = commit_entry(committer)
        result = gate.accept(manifest_for((ref,)), "estimation-harness", ALLOWED, gate_event)
        assert result.error_codes == ("reader_not_allowed",)

    def test_unsupported_version(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        gate, committer, _, _ = build_gate(object_store, conn)
        ref, _ = commit_entry(committer)
        manifest = manifest_for((ref,), schema_version="handoff.v2")
        result = gate.accept(manifest, "design-harness", ALLOWED, gate_event)
        assert result.error_codes == ("unsupported_version",)

    def test_incomplete_lineage(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        # The committer refuses to create this state, so insert directly: an
        # artifact whose registration requires a parent it does not have.
        needs_parent = registration(
            artifact_type="Child", schema_version="child.v1",
            required_parent_types=("TestArtifact",),
        )
        gate, _, products, _ = build_gate(
            object_store, conn, (registration(), needs_parent)
        )
        payload: dict[str, object] = {"orphan": True}
        envelope = envelope_for(
            payload, artifact_id="orphan-1", artifact_type="Child",
            schema_version="child.v1",
        )
        products.insert_artifact(envelope)
        object_store.put_if_absent(envelope.content_hash, b'{"orphan":true}')
        ref = ArtifactRef(artifact_id="orphan-1", content_hash=envelope.content_hash)
        result = gate.accept(manifest_for((ref,)), "design-harness", ALLOWED, gate_event)
        assert "incomplete_lineage" in result.error_codes

    def test_multiple_failures_all_reported(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        gate, _, _, _ = build_gate(object_store, conn)
        ghost = ArtifactRef(artifact_id="ghost", content_hash="0" * 64)
        manifest = manifest_for(
            (ghost,), originating_outcome="refused", schema_version="handoff.v2"
        )
        result = gate.accept(manifest, "design-harness", ALLOWED, gate_event)
        assert result.error_codes == (
            "missing_artifact", "unsupported_version", "wrong_outcome"
        )
