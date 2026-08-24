"""Integration tests for the persistence layer and commit protocol (T-005)."""

from __future__ import annotations

import io
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import pytest

from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactEnvelopeV1, ArtifactRef, SensitivityClass
from causal.shared.events import EventEmitter, OperationalEventV1, Severity, Stage
from causal.shared.persistence import (
    ArtifactCommitter,
    ObjectStore,
    PersistenceError,
    ProductStore,
)
from causal.shared.registry import ArtifactTypeRegistrationV1, ArtifactTypeRegistry, RegistryError
from tests.shared.conftest import requires_docker

MIGRATIONS = Path(__file__).resolve().parents[2] / "migrations"
NOW = datetime(2026, 8, 24, 15, 0, 0, 0, tzinfo=UTC)

pytestmark = requires_docker


def registration(**overrides: object) -> ArtifactTypeRegistrationV1:
    base: dict[str, object] = {
        "artifact_type": "TestArtifact",
        "schema_version": "test-artifact.v1",
        "producer_component": "intake-coordinator",
        "allowed_reader_components": ("design-harness",),
        "required_parent_types": (),
        "optional_parent_types": (),
        "sensitivity_class": SensitivityClass.INTERNAL,
        "terminal_statuses": ("usable",),
        "destinations": ("design",),
        "validator_version": "test-validator.v1",
    }
    base.update(overrides)
    return ArtifactTypeRegistrationV1(**base)  # type: ignore[arg-type]


def envelope_for(
    payload: dict[str, object], *, artifact_id: str = "art-1", **overrides: object
) -> ArtifactEnvelopeV1:
    digest = content_hash(payload)
    base: dict[str, object] = {
        "artifact_id": artifact_id,
        "artifact_type": "TestArtifact",
        "schema_version": "test-artifact.v1",
        "content_hash": digest,
        "analysis_id": "an-1",
        "stage_run_id": "run-1",
        "producer_component": "intake-coordinator",
        "producer_version": "0.1.0",
        "parent_artifacts": (),
        "sensitivity_class": SensitivityClass.INTERNAL,
        "created_at_utc": NOW,
        "payload_locator": ObjectStore.locator_for(digest),
    }
    base.update(overrides)
    return ArtifactEnvelopeV1(**base)  # type: ignore[arg-type]


def committed_event(**overrides: object) -> OperationalEventV1:
    base: dict[str, object] = {
        "schema_version": "operational-event.v1",
        "occurred_at_utc": NOW,
        "severity": Severity.INFO,
        "event_name": "artifact.committed",
        "event_id": f"ev-{uuid.uuid4().hex[:8]}",
        "parent_event_id": None,
        "analysis_id": "an-1",
        "stage": Stage.INTAKE,
        "stage_run_id": "run-1",
        "graph_thread_id": None,
        "task_id": None,
        "attempt_id": None,
        "attempt_number": None,
        "component_id": "intake-coordinator",
        "component_version": "0.1.0",
        "versions": {},
        "status": "committed",
        "error_code": None,
        "retryable": None,
        "duration_ms": None,
        "token_usage": {},
        "cost": None,
        "artifact_refs": (),
        "required_eval_ids": (),
        "evaluation_run_id": None,
        "evaluation_case_id": None,
        "evaluation_fixture_hash": None,
        "evaluator_version": None,
        "evaluation_gate_status": None,
        "exception_class": None,
        "exception_fingerprint": None,
        "safe_dimensions": {},
    }
    base.update(overrides)
    return OperationalEventV1(**base)  # type: ignore[arg-type]


def make_committer(
    object_store: ObjectStore,
    conn: psycopg.Connection[Any],
    registrations: tuple[ArtifactTypeRegistrationV1, ...] = (),
) -> tuple[ArtifactCommitter, ProductStore, io.StringIO]:
    sink = io.StringIO()
    products = ProductStore(conn)
    products.create_stage_run("run-1", "an-1", "intake")
    committer = ArtifactCommitter(
        object_store, products,
        ArtifactTypeRegistry(registrations or (registration(),)),
        EventEmitter(sink),
    )
    return committer, products, sink


class TestObjectStore:
    def test_roundtrip_and_idempotent_put(self, object_store: ObjectStore) -> None:
        data = b'{"k":"v"}'
        import hashlib

        digest = hashlib.sha256(data).hexdigest()
        locator = object_store.put_if_absent(digest, data)
        assert locator == f"objects/{digest}"
        assert object_store.put_if_absent(digest, data) == locator
        assert object_store.get(locator) == data


class TestStageRuns:
    def test_lifecycle_and_illegal_transition(self, conn: psycopg.Connection[Any]) -> None:
        store = ProductStore(conn)
        store.create_stage_run("run-x", "an-1", "intake")
        assert store.get_stage_run_state("run-x") == "created"
        store.transition_stage_run("run-x", "tracing_preflight")
        store.transition_stage_run("run-x", "running")
        with pytest.raises(PersistenceError) as excinfo:
            store.transition_stage_run("run-x", "created")
        assert excinfo.value.code == "illegal_state_transition"

    def test_terminal_state_refuses_transitions(self, conn: psycopg.Connection[Any]) -> None:
        store = ProductStore(conn)
        store.create_stage_run("run-y", "an-1", "design")
        for state in ("tracing_preflight", "running", "failed"):
            store.transition_stage_run("run-y", state)
        with pytest.raises(PersistenceError) as excinfo:
            store.transition_stage_run("run-y", "running")
        assert excinfo.value.code == "illegal_state_transition"

    def test_unknown_stage_run(self, conn: psycopg.Connection[Any]) -> None:
        with pytest.raises(PersistenceError) as excinfo:
            ProductStore(conn).get_stage_run_state("nope")
        assert excinfo.value.code == "unknown_stage_run"


class TestCommitProtocol:
    def test_happy_path(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        committer, products, sink = make_committer(object_store, conn)
        payload: dict[str, object] = {"fact": "value", "n": 1}
        envelope = envelope_for(payload)
        committed = committer.commit(envelope, payload, committed_event())
        assert committed == envelope
        loaded = products.load_envelope("art-1")
        assert loaded == envelope
        assert object_store.get(envelope.payload_locator) == b'{"fact":"value","n":1}'
        line = sink.getvalue()
        assert line.endswith("\n") and json.loads(line)["event_name"] == "artifact.committed"

    def test_parents_persist_in_order(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        committer, products, _ = make_committer(object_store, conn)
        first: dict[str, object] = {"p": 1}
        second: dict[str, object] = {"p": 2}
        committer.commit(envelope_for(first, artifact_id="p-1"), first, committed_event())
        committer.commit(envelope_for(second, artifact_id="p-2"), second, committed_event())
        parents = (
            ArtifactRef(artifact_id="p-2", content_hash=content_hash(second)),
            ArtifactRef(artifact_id="p-1", content_hash=content_hash(first)),
        )
        child: dict[str, object] = {"c": 1}
        committer.commit(
            envelope_for(child, artifact_id="c-1", parent_artifacts=parents),
            child, committed_event(),
        )
        assert products.load_envelope("c-1").parent_artifacts == parents

    def test_replay_same_id_and_hash_is_noop(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        committer, _, sink = make_committer(object_store, conn)
        payload: dict[str, object] = {"k": "v"}
        envelope = envelope_for(payload)
        committer.commit(envelope, payload, committed_event())
        replayed = committer.commit(envelope, payload, committed_event())
        assert replayed == envelope
        assert sink.getvalue().count("\n") == 1  # no second event for the no-op

    def test_same_id_different_hash_is_integrity_conflict(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        committer, _, _ = make_committer(object_store, conn)
        first: dict[str, object] = {"k": "v"}
        committer.commit(envelope_for(first), first, committed_event())
        second: dict[str, object] = {"k": "other"}
        with pytest.raises(PersistenceError) as excinfo:
            committer.commit(envelope_for(second), second, committed_event())
        assert excinfo.value.code == "integrity_conflict"

    def test_hash_mismatch_rejected(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        committer, _, _ = make_committer(object_store, conn)
        payload: dict[str, object] = {"k": "v"}
        wrong = envelope_for({"k": "other"})
        with pytest.raises(PersistenceError) as excinfo:
            committer.commit(wrong, payload, committed_event())
        assert excinfo.value.code == "artifact_hash_mismatch"

    def test_locator_mismatch_rejected(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        committer, _, _ = make_committer(object_store, conn)
        payload: dict[str, object] = {"k": "v"}
        envelope = envelope_for(payload, payload_locator="objects/" + "0" * 64)
        with pytest.raises(PersistenceError) as excinfo:
            committer.commit(envelope, payload, committed_event())
        assert excinfo.value.code == "locator_mismatch"

    def test_unregistered_type_bubbles_unsupported_schema(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        committer, _, _ = make_committer(object_store, conn)
        payload: dict[str, object] = {"k": "v"}
        envelope = envelope_for(payload, artifact_type="Mystery")
        with pytest.raises(RegistryError) as excinfo:
            committer.commit(envelope, payload, committed_event())
        assert excinfo.value.code == "unsupported_schema"

    def test_producer_mismatch_is_registry_violation(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        committer, _, _ = make_committer(object_store, conn)
        payload: dict[str, object] = {"k": "v"}
        envelope = envelope_for(payload, producer_component="design-harness")
        with pytest.raises(PersistenceError) as excinfo:
            committer.commit(envelope, payload, committed_event())
        assert excinfo.value.code == "registry_violation"

    def test_missing_required_parent(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        needs_parent = registration(
            artifact_type="Child", schema_version="child.v1",
            required_parent_types=("TestArtifact",),
        )
        committer, _, _ = make_committer(object_store, conn, (registration(), needs_parent))
        payload: dict[str, object] = {"k": "v"}
        envelope = envelope_for(
            payload, artifact_id="child-1", artifact_type="Child", schema_version="child.v1"
        )
        with pytest.raises(PersistenceError) as excinfo:
            committer.commit(envelope, payload, committed_event())
        assert excinfo.value.code == "missing_required_parent"

    def test_reopen_validation_failure(
        self, object_store: ObjectStore, conn: psycopg.Connection[Any]
    ) -> None:
        class CorruptingStore(ObjectStore):
            def get(self, locator: str) -> bytes:
                return b"corrupted"

        corrupting = CorruptingStore(object_store._client, object_store._bucket)
        committer, _, _ = make_committer(corrupting, conn)
        payload: dict[str, object] = {"k": "v"}
        with pytest.raises(PersistenceError) as excinfo:
            committer.commit(envelope_for(payload), payload, committed_event())
        assert excinfo.value.code == "reopen_validation_failed"
