"""End-to-end intake coordinator tests (T-008; EV-P1-001/002/005 unit layer)."""

from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import pytest

from causal.intake.catalog import CatalogStore
from causal.intake.contracts import IntakeSubmissionV1
from causal.intake.coordinator import IntakeCoordinator, IntakeError
from causal.intake.fields import load_field_classes
from causal.shared.events import EventEmitter
from causal.shared.handoff import HandoffGate, HandoffStore
from causal.shared.persistence import ArtifactCommitter, ObjectStore, ProductStore
from causal.shared.registry import load_artifact_type_registry
from tests.infrastructure import requires_docker
from tests.intake.conftest import CSV, README, FailingKaggleClient, FrozenKaggleClient
from tests.shared.test_persistence import committed_event

pytestmark = requires_docker

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = load_artifact_type_registry(ROOT / "registries" / "artifact-types.v1.json")
CLASSES = load_field_classes(ROOT / "registries" / "kaggle-field-classes.v1.json")
NOW = datetime(2026, 8, 24, 12, 0, 0, 0, tzinfo=UTC)
ALLOWED = frozenset({"usable", "partial"})


def make_coordinator(
    conn: psycopg.Connection[Any],
    object_store: ObjectStore,
    client: FrozenKaggleClient,
) -> tuple[IntakeCoordinator, io.StringIO]:
    sink = io.StringIO()
    emitter = EventEmitter(sink)
    products = ProductStore(conn)
    committer = ArtifactCommitter(object_store, products, REGISTRY, emitter)
    coordinator = IntakeCoordinator(
        client, committer, products, CatalogStore(conn), object_store,
        REGISTRY, CLASSES, emitter, clock=lambda: NOW,
    )
    return coordinator, sink


def submission(
    key: str = "key-1", question: str = "Does the program raise earnings?",
    context: str | None = None,
) -> IntakeSubmissionV1:
    return IntakeSubmissionV1(
        schema_version="intake-submission.v1", question_text=question,
        context_text=context, kaggle_ref="lalonde/nsw", idempotency_key=key,
    )


def outcome_payload(
    conn: psycopg.Connection[Any], object_store: ObjectStore, artifact_id: str
) -> dict[str, object]:
    envelope = ProductStore(conn).load_envelope(artifact_id)
    data = object_store.get(envelope.payload_locator)
    payload = json.loads(data)
    assert isinstance(payload, dict)
    return payload


class TestUsableRun:
    def test_end_to_end_usable(self, conn: Any, object_store: ObjectStore) -> None:
        coordinator, _ = make_coordinator(conn, object_store, FrozenKaggleClient())
        result = coordinator.run(submission())
        assert result.status == "usable" and not result.replayed
        run = conn.execute(
            "SELECT intake_status, intake_outcome_artifact_id, dataset_id"
            " FROM catalog.runs WHERE analysis_id = %s", (result.analysis_id,),
        ).fetchone()
        assert run == ("usable", result.outcome_artifact_id, "kaggle:lalonde/nsw@3")
        payload = outcome_payload(conn, object_store, str(result.outcome_artifact_id))
        assert payload["status"] == "usable"
        assert payload["candidate_tables"] == ["nsw.csv"]
        assert payload["available_slot_count"] == 3

    def test_the_question_is_citable_evidence(self, conn: Any, object_store: ObjectStore) -> None:
        """Intent claims about the question need an allowlisted id to cite (D-065)."""
        coordinator, _ = make_coordinator(conn, object_store, FrozenKaggleClient())
        result = coordinator.run(submission(question="Does training raise earnings?"))
        payload = outcome_payload(conn, object_store, str(result.outcome_artifact_id))
        bundle = outcome_payload(
            conn, object_store, str(payload["evidence_bundle_artifact_id"]))
        items = bundle["items"]
        assert isinstance(items, list)
        found = [row for row in items if row["evidence_id"] == "ua:question/text"]
        assert found == [{"evidence_id": "ua:question/text", "scope_kind": "dataset",
                          "table_name": None, "column_name": None,
                          "source_field": "question_text",
                          "value": "Does training raise earnings?"}]

    def test_user_context_is_citable_evidence(self, conn: Any, object_store: ObjectStore) -> None:
        coordinator, _ = make_coordinator(conn, object_store, FrozenKaggleClient())
        result = coordinator.run(submission(context="Treatment was randomized."))
        payload = outcome_payload(conn, object_store, str(result.outcome_artifact_id))
        bundle = outcome_payload(conn, object_store, str(payload["evidence_bundle_artifact_id"]))
        assert [row for row in bundle["items"] if row["evidence_id"] == "ua:context/text"] == [{
            "evidence_id": "ua:context/text", "scope_kind": "dataset", "table_name": None,
            "column_name": None, "source_field": "context_text",
            "value": "Treatment was randomized.",
        }]

    def test_every_resource_has_a_terminal_row(
        self, conn: Any, object_store: ObjectStore
    ) -> None:
        coordinator, _ = make_coordinator(conn, object_store, FrozenKaggleClient())
        result = coordinator.run(submission())
        rows = conn.execute(
            "SELECT logical_name, parse_status FROM catalog.resources"
            " WHERE dataset_id = %s ORDER BY logical_name",
            ("kaggle:lalonde/nsw@3",),
        ).fetchall()
        assert dict(rows) == {
            "kaggle_capture": "parsed", "source_archive": "parsed",
            "nsw.csv": "parsed", "readme.md": "parsed",
        }
        assert result.status == "usable"

    def test_narrow_views_answer_availability(
        self, conn: Any, object_store: ObjectStore
    ) -> None:
        coordinator, _ = make_coordinator(conn, object_store, FrozenKaggleClient())
        coordinator.run(submission())
        available = conn.execute(
            "SELECT field_or_slot_name FROM catalog.semantic_available"
            " WHERE scope_kind = 'dataset'"
        ).fetchall()
        assert ("title" in {row[0] for row in available})
        missing = conn.execute(
            "SELECT status FROM catalog.semantic_missing WHERE column_name = 'group'"
            " AND field_or_slot_name = 'meaning'"
        ).fetchone()
        assert missing == ("not_offered",)
        for view in ("structural_manifest", "measured_fact_manifest",
                     "provenance_manifest"):
            count = conn.execute(f"SELECT count(*) FROM catalog.{view}").fetchone()
            assert count is not None and count[0] > 0
        with pytest.raises(psycopg.errors.UndefinedTable):
            conn.execute("SELECT count(*) FROM catalog.all_context")

    def test_no_credential_anywhere(self, conn: Any, object_store: ObjectStore) -> None:
        client = FrozenKaggleClient()
        coordinator, sink = make_coordinator(conn, object_store, client)
        result = coordinator.run(submission())
        assert client.api_token not in sink.getvalue()
        payload = outcome_payload(conn, object_store, str(result.outcome_artifact_id))
        assert client.api_token not in json.dumps(payload)


class TestIdempotency:
    def test_repeat_returns_same_identities(
        self, conn: Any, object_store: ObjectStore
    ) -> None:
        coordinator, _ = make_coordinator(conn, object_store, FrozenKaggleClient())
        first = coordinator.run(submission())
        second = coordinator.run(submission())
        assert second.replayed
        assert (second.analysis_id, second.outcome_artifact_id) == (
            first.analysis_id, first.outcome_artifact_id,
        )

    def test_conflicting_duplicate_raises_blocker(
        self, conn: Any, object_store: ObjectStore
    ) -> None:
        coordinator, sink = make_coordinator(conn, object_store, FrozenKaggleClient())
        coordinator.run(submission())
        with pytest.raises(IntakeError) as excinfo:
            coordinator.run(submission(question="A different question entirely?"))
        assert excinfo.value.code == "idempotency_conflict"
        assert '"event_name":"blocker.raised"' in sink.getvalue()
        count = conn.execute("SELECT count(*) FROM catalog.runs").fetchone()
        assert count is not None and count[0] == 1


class TestRefusalPaths:
    def test_refused_archive_records_every_member_without_profiles(
        self, conn: Any, object_store: ObjectStore,
    ) -> None:
        client = FrozenKaggleClient(files={"nsw.csv": CSV, "model.pkl": b"x"})
        coordinator, _ = make_coordinator(conn, object_store, client)
        result = coordinator.run(submission())
        assert result.status == "refused"
        rows = conn.execute(
            "SELECT logical_name, parse_status FROM catalog.resources"
            " WHERE logical_name IN ('nsw.csv', 'model.pkl')").fetchall()
        assert dict(rows) == {"nsw.csv": "excluded", "model.pkl": "unsafe"}
        assert conn.execute(
            "SELECT count(*) FROM causal.artifacts WHERE artifact_type = 'TableProfile'"
        ).fetchone() == (0,)

    @pytest.mark.parametrize("corrupt_member", [False, True])
    def test_corrupt_archive_and_member_have_committed_outcomes(
        self, conn: Any, object_store: ObjectStore, corrupt_member: bool,
    ) -> None:
        class CorruptClient(FrozenKaggleClient):
            def download_archive(self, owner: str, slug: str, version: str) -> bytes:
                if not corrupt_member:
                    return b"not a zip archive"
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
                    archive.writestr("nsw.csv", CSV)
                    archive.writestr("readme.md", b"original document")
                return buffer.getvalue().replace(b"original document", b"tampered document")

        coordinator, _ = make_coordinator(conn, object_store, CorruptClient())
        result = coordinator.run(submission())
        payload = outcome_payload(conn, object_store, str(result.outcome_artifact_id))
        assert result.status == ("partial" if corrupt_member else "refused")
        assert conn.execute(
            "SELECT run_state FROM causal.stage_runs WHERE stage_run_id = %s",
            (result.stage_run_id,)).fetchone() == ("completed",)
        if corrupt_member:
            assert payload["candidate_tables"] == ["nsw.csv"]
            assert payload["failed_resource_count"] == 1
            assert conn.execute(
                "SELECT parse_status FROM catalog.resources WHERE logical_name = 'readme.md'"
            ).fetchone() == ("failed",)

    def test_non_finite_table_still_commits_a_profile(
        self, conn: Any, object_store: ObjectStore,
    ) -> None:
        client = FrozenKaggleClient(files={"nsw.csv": b"value\n1.0\nNaN\ninf\n-inf\n"})
        coordinator, _ = make_coordinator(conn, object_store, client)
        result = coordinator.run(submission())
        assert result.status == "usable"
        payload = outcome_payload(conn, object_store, str(result.outcome_artifact_id))
        profile = outcome_payload(conn, object_store, payload["table_profile_artifact_ids"][0])
        assert profile["columns"]["value"]["numeric"]["non_finite_count"] == 3
        assert profile["columns"]["value"]["numeric"]["mean"] == 1.0

    def test_unsafe_archive_refused_without_handoff(
        self, conn: Any, object_store: ObjectStore
    ) -> None:
        client = FrozenKaggleClient(files={"nsw.csv": CSV, "model.pkl": b"x"})
        coordinator, _ = make_coordinator(conn, object_store, client)
        result = coordinator.run(submission())
        assert result.status == "refused"
        with pytest.raises(IntakeError) as excinfo:
            coordinator.open_handoff(
                result.analysis_id, str(result.outcome_artifact_id), "sr:design"
            )
        assert excinfo.value.code == "handoff_unavailable"
        row = conn.execute(
            "SELECT parse_status FROM catalog.resources WHERE logical_name = %s",
            ("source_archive",),
        ).fetchone()
        assert row == ("unsafe",)

    def test_no_table_refused(self, conn: Any, object_store: ObjectStore) -> None:
        client = FrozenKaggleClient(files={"readme.md": README})
        coordinator, _ = make_coordinator(conn, object_store, client)
        result = coordinator.run(submission())
        payload = outcome_payload(conn, object_store, str(result.outcome_artifact_id))
        assert result.status == "refused"
        assert payload["refusal_reason"] == "no supported table profiled"

    def test_capture_failure_refused(self, conn: Any, object_store: ObjectStore) -> None:
        coordinator, sink = make_coordinator(conn, object_store, FailingKaggleClient())
        result = coordinator.run(submission())
        assert result.status == "refused"
        assert '"event_name":"tool.failed"' in sink.getvalue()

    def test_partial_when_resource_unreadable(
        self, conn: Any, object_store: ObjectStore
    ) -> None:
        client = FrozenKaggleClient(
            files={"nsw.csv": CSV, "readme.md": README, "chart.pdf": b"%PDF"}
        )
        coordinator, _ = make_coordinator(conn, object_store, client)
        result = coordinator.run(submission())
        assert result.status == "partial"
        payload = outcome_payload(conn, object_store, str(result.outcome_artifact_id))
        assert payload["unreadable_resource_count"] == 1


class TestHandoff:
    def test_gate_accepts_usable_handoff(
        self, conn: Any, object_store: ObjectStore
    ) -> None:
        coordinator, _ = make_coordinator(conn, object_store, FrozenKaggleClient())
        result = coordinator.run(submission())
        manifest = coordinator.open_handoff(
            result.analysis_id, str(result.outcome_artifact_id),
            f"sr:{result.analysis_id}:design",
        )
        gate = HandoffGate(
            object_store, ProductStore(conn), HandoffStore(conn), REGISTRY,
            EventEmitter(io.StringIO()),
        )
        outcome = gate.accept(
            manifest, "design-harness", ALLOWED,
            lambda verdict, codes: committed_event(
                event_name=f"handoff.{verdict}", status=verdict,
                required_eval_ids=("EV-P1-005",),
            ),
        )
        assert outcome.accepted and outcome.error_codes == ()
        recorded = conn.execute(
            "SELECT receiver_validation_result FROM causal.handoffs"
            " WHERE handoff_id = %s", (manifest.handoff_id,),
        ).fetchone()
        assert recorded == ("accepted",)

    def test_open_handoff_needs_matching_outcome(
        self, conn: Any, object_store: ObjectStore
    ) -> None:
        coordinator, _ = make_coordinator(conn, object_store, FrozenKaggleClient())
        result = coordinator.run(submission())
        with pytest.raises(IntakeError):
            coordinator.open_handoff(result.analysis_id, "not-the-outcome", "sr:d")
        with pytest.raises(IntakeError):
            coordinator.open_handoff(
                "an-unknown", str(result.outcome_artifact_id), "sr:d"
            )
