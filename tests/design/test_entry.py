"""Design entry gate: candidates, selection routing, manifest compilation (T-011 §6, §8)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest

from causal.design import entry
from causal.design.contracts import (
    REGISTRY_VERSION_KEYS,
    SelectionSource,
    TableSelectionDecisionV1,
    TableSelectionV1,
)
from causal.shared.contracts import (
    ArtifactEnvelopeV1,
    ArtifactRef,
    HandoffManifestV1,
    SensitivityClass,
)
from tests.infrastructure import requires_docker

NOW = datetime(2026, 8, 24, 12, 0, 0, tzinfo=UTC)
HASH = "a" * 64
OTHER_HASH = "b" * 64
DATASET = "ds-1"
REF = ArtifactRef(artifact_id="art-1", content_hash=HASH)
REGISTRY_VERSIONS = dict.fromkeys(REGISTRY_VERSION_KEYS, "v1")
SELECTION = TableSelectionV1(
    dataset_id=DATASET, logical_name="nsw.csv", resource_object_locator="objects/nsw.csv",
    resource_sha256=HASH, candidate_count=1,
    selection_source=SelectionSource.ONLY_CANDIDATE, decision_artifact_id=None,
)


def seed_dataset(conn: psycopg.Connection[Any]) -> None:
    """The stage run, capture artifact, and dataset row the catalog FKs point at."""
    conn.execute(
        "INSERT INTO causal.stage_runs (stage_run_id, analysis_id, stage, run_state,"
        " created_at_utc, updated_at_utc)"
        " VALUES ('run-1', 'an-1', 'intake', 'running', %s, %s)", (NOW, NOW))
    conn.execute(
        "INSERT INTO causal.artifacts VALUES ('cap-1', 'KaggleCapture', 'kaggle-capture.v1',"
        " %s, 'an-1', 'run-1', 'intake-coordinator', '0.1.0', 'internal', %s, %s)",
        (HASH, NOW, f"objects/{HASH}"))
    conn.execute(
        "INSERT INTO catalog.datasets VALUES (%s, 'kaggle', 'lalonde', 'nsw', '3', 'ready',"
        " 'cap-1', NULL)", (DATASET,))


def add_resource(
    conn: psycopg.Connection[Any], logical_name: str, media_type: str, *,
    kind: str = "table", parse_status: str = "parsed", reason: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO catalog.resources VALUES (%s, %s, %s, %s, %s, %s, %s, 10, %s, %s)",
        (f"res:{DATASET}:{logical_name}", DATASET, kind, logical_name, HASH,
         f"objects/{logical_name}", media_type, parse_status, reason))


def add_field(
    conn: psycopg.Connection[Any], scope_kind: str, table: str | None, column: str | None,
    slot: str, context_class: str, *, status: str = "evidenced", evidence_count: int = 0,
    pointer: str = "/x",
) -> None:
    conn.execute(
        "INSERT INTO catalog.source_field_index VALUES"
        " (%s, 'cap-1', NULL, %s, %s, %s, %s, %s, %s, %s, %s)",
        (DATASET, scope_kind, table, column, slot, context_class, status,
         evidence_count, pointer))


def decision(selected_table: str) -> TableSelectionDecisionV1:
    return TableSelectionDecisionV1(
        interrupt_id="int-1", expected_interrupt_hash=HASH, expected_revision=1,
        selected_table=selected_table, idempotency_key="idem-1")


@requires_docker
class TestCandidateRouting:
    def test_only_admitted_csv_is_a_candidate(self, conn: psycopg.Connection[Any]) -> None:
        seed_dataset(conn)
        add_resource(conn, "nsw.csv", "csv")
        add_resource(conn, "meta.json", "json", kind="metadata")
        add_resource(conn, "notes.yaml", "yaml", kind="document",
                     parse_status="excluded", reason="yaml parsing deferred")
        reader = entry.PsycopgCatalogReader(conn)
        candidates = entry.list_csv_candidates(reader, DATASET)
        assert [row.logical_name for row in candidates] == ["nsw.csv"]
        assert candidates[0].sha256 == HASH
        assert candidates[0].resource_object_locator == "objects/nsw.csv"
        assert candidates[0].parse_status == "parsed"
        assert entry.list_admitted_non_csv(reader, DATASET) == ("meta.json",)

    def test_zero_csv_refuses(self, conn: psycopg.Connection[Any]) -> None:
        seed_dataset(conn)
        add_resource(conn, "meta.json", "json", kind="metadata")
        reader = entry.PsycopgCatalogReader(conn)
        with pytest.raises(entry.EntryError) as error:
            entry.resolve_selection(entry.list_csv_candidates(reader, DATASET), None, DATASET)
        assert error.value.code == "NO_ANALYSIS_CSV"

    def test_single_candidate_needs_no_decision(self, conn: psycopg.Connection[Any]) -> None:
        seed_dataset(conn)
        add_resource(conn, "nsw.csv", "csv")
        reader = entry.PsycopgCatalogReader(conn)
        selection = entry.resolve_selection(
            entry.list_csv_candidates(reader, DATASET), None, DATASET)
        assert isinstance(selection, TableSelectionV1)
        assert selection.selection_source is SelectionSource.ONLY_CANDIDATE
        assert selection.candidate_count == 1
        assert selection.media_type == "text/csv"
        assert selection.decision_artifact_id is None

    def test_two_candidates_without_a_decision_require_selection(
        self, conn: psycopg.Connection[Any]
    ) -> None:
        seed_dataset(conn)
        add_resource(conn, "nsw.csv", "csv")
        add_resource(conn, "psid.csv", "text/csv")
        reader = entry.PsycopgCatalogReader(conn)
        result = entry.resolve_selection(
            entry.list_csv_candidates(reader, DATASET), None, DATASET)
        assert isinstance(result, entry.SelectionRequired)
        assert [row.logical_name for row in result.candidates] == ["nsw.csv", "psid.csv"]

    def test_decision_selects_the_second_candidate(self, conn: psycopg.Connection[Any]) -> None:
        seed_dataset(conn)
        add_resource(conn, "nsw.csv", "csv")
        add_resource(conn, "psid.csv", "csv")
        reader = entry.PsycopgCatalogReader(conn)
        selection = entry.resolve_selection(
            entry.list_csv_candidates(reader, DATASET), decision("psid.csv"), DATASET,
            decision_artifact_id="dec-1")
        assert isinstance(selection, TableSelectionV1)
        assert selection.logical_name == "psid.csv"
        assert selection.selection_source is SelectionSource.USER_DECISION
        assert selection.candidate_count == 2
        assert selection.decision_artifact_id == "dec-1"

    def test_decision_naming_a_non_csv_or_unknown_resource_is_refused(
        self, conn: psycopg.Connection[Any]
    ) -> None:
        seed_dataset(conn)
        add_resource(conn, "nsw.csv", "csv")
        add_resource(conn, "psid.csv", "csv")
        add_resource(conn, "meta.json", "json", kind="metadata")
        reader = entry.PsycopgCatalogReader(conn)
        candidates = entry.list_csv_candidates(reader, DATASET)
        others = entry.list_admitted_non_csv(reader, DATASET)
        with pytest.raises(entry.EntryError) as refused:
            entry.resolve_selection(
                candidates, decision("meta.json"), DATASET, other_admitted=others)
        assert refused.value.code == "UNSUPPORTED_ANALYSIS_FORMAT_V1"
        with pytest.raises(entry.EntryError) as unknown:
            entry.resolve_selection(
                candidates, decision("gone.csv"), DATASET, other_admitted=others)
        assert unknown.value.detail_codes == ("unknown_selected_table",)


@requires_docker
class TestManifestCompilation:
    def test_manifest_matches_the_seeded_views(self, conn: psycopg.Connection[Any]) -> None:
        seed_dataset(conn)
        add_resource(conn, "nsw.csv", "csv")
        for slot in ("name", "type", "order"):
            add_field(conn, "column", "nsw.csv", "unit_id", slot, "structural")
        for slot in ("name", "order"):
            add_field(conn, "column", "nsw.csv", "earnings", slot, "structural")
        add_field(conn, "column", "psid.csv", "z", "name", "structural")
        add_field(conn, "column", "nsw.csv", "earnings", "meaning", "semantic",
                  evidence_count=1, pointer="/columns/0/slots/meaning")
        add_field(conn, "column", "nsw.csv", "unit_id", "encoding", "semantic",
                  status="not_offered")
        add_field(conn, "column", "psid.csv", "z", "meaning", "semantic", evidence_count=1)
        add_field(conn, "column", "nsw.csv", "earnings", "profile", "measured",
                  pointer="tp-1#/columns/earnings")
        add_field(conn, "dataset", None, None, "licenseName", "provenance")
        reader = entry.PsycopgCatalogReader(conn)
        selection = entry.resolve_selection(
            entry.list_csv_candidates(reader, DATASET), None, DATASET)
        assert isinstance(selection, TableSelectionV1)
        manifest = entry.compile_manifest(
            reader, selection, question_ref=REF, outcome_ref=REF, selection_ref=REF,
            design_revision=2, registry_versions=REGISTRY_VERSIONS)
        # One ordered structural row per column of the selected table, and no other table.
        assert [
            (row.column_name, row.ordinal, row.dtype) for row in manifest.structural_inventory
        ] == [("earnings", 0, "undeclared"), ("unit_id", 1, "declared")]
        assert [row.field_or_slot_name for row in manifest.semantic_available] == ["meaning"]
        assert manifest.semantic_available[0].evidence_count == 1
        assert [row.field_or_slot_name for row in manifest.semantic_missing] == ["encoding"]
        # The semantic_missing view carries neither counter nor pointer (0003_catalog.sql).
        assert manifest.semantic_missing[0].evidence_count == 0
        assert manifest.semantic_missing[0].json_pointer == ""
        assert [row.json_pointer for row in manifest.measured_surface] == [
            "tp-1#/columns/earnings"]
        assert [row.scope_kind for row in manifest.provenance_surface] == ["dataset"]
        assert manifest.selected_table == "nsw.csv"
        assert manifest.design_revision == 2
        assert manifest.retrieval_surfaces == entry.RETRIEVAL_SURFACES
        assert manifest.registry_versions == REGISTRY_VERSIONS


def envelope(artifact_id: str, artifact_type: str, digest: str = HASH) -> ArtifactEnvelopeV1:
    return ArtifactEnvelopeV1(
        artifact_id=artifact_id, artifact_type=artifact_type, schema_version="v1",
        content_hash=digest, analysis_id="an-1", stage_run_id="run-1",
        producer_component="intake-coordinator", producer_version="0.1.0",
        parent_artifacts=(), sensitivity_class=SensitivityClass.INTERNAL,
        created_at_utc=NOW, payload_locator=f"objects/{digest}")


class FakeProducts:
    """A ProductsReader over a fixed set of committed envelopes."""

    def __init__(self, envelopes: tuple[ArtifactEnvelopeV1, ...]) -> None:
        self._by_id = {item.artifact_id: item for item in envelopes}

    def load_envelope(self, artifact_id: str) -> ArtifactEnvelopeV1:
        if artifact_id not in self._by_id:
            raise KeyError(artifact_id)
        return self._by_id[artifact_id]


def handoff(entry_hash: str = HASH) -> HandoffManifestV1:
    return HandoffManifestV1(
        handoff_id="ho-1", schema_version="handoff.v1", analysis_id="an-1",
        producing_stage_run_id="run-1", receiving_stage_run_id="run-2",
        entries=(ArtifactRef(artifact_id="out-1", content_hash=entry_hash),),
        originating_outcome="usable", approval_ids=(), registry_version="artifact-types.v1",
        compatibility_version="handoff.v1", receiver_validation_result=None,
        receiver_error_codes=(), created_at_utc=NOW, accepted_at_utc=None)


def outcome(**overrides: object) -> dict[str, Any]:
    return {
        "status": "usable", "dataset_id": DATASET, "handoff_contract_version": "handoff.v1",
        "question_artifact_id": "q-1", "source_manifest_artifact_id": "sm-1",
        "table_profile_artifact_ids": ["tp-1"], "evidence_bundle_artifact_id": "eb-1",
        "semantic_map_artifact_id": "map-1",
        "retrieval_surfaces": dict.fromkeys(entry.RETRIEVAL_SURFACES, "catalog-view.v1"),
    } | overrides


COMMITTED = (
    envelope("out-1", "IntakeOutcome"), envelope("q-1", "QuestionRecord"),
    envelope("sm-1", "SourceManifest"), envelope("tp-1", "TableProfile"),
    envelope("eb-1", "EvidenceBundle"), envelope("map-1", "SemanticMap"),
)


class TestValidateEntry:
    def test_clean_handoff_passes(self) -> None:
        entry.validate_entry(handoff(), outcome(), FakeProducts(COMMITTED), selection=SELECTION)

    def test_refused_outcome_and_hash_mismatch_report_together(self) -> None:
        with pytest.raises(entry.EntryError) as error:
            entry.validate_entry(
                handoff(OTHER_HASH), outcome(status="refused"), FakeProducts(COMMITTED),
                selection=None)
        assert error.value.code == "entry_validation_failed"
        assert error.value.detail_codes == (
            "artifact_hash_mismatch", "no_table_selection", "unsupported_intake_status")

    def test_missing_artifact_unreadable_question_and_bad_surfaces(self) -> None:
        products = FakeProducts((COMMITTED[0], envelope("q-1", "SourceManifest")))
        with pytest.raises(entry.EntryError) as error:
            entry.validate_entry(
                handoff(), outcome(retrieval_surfaces={"structural_manifest": "catalog-view.v1"}),
                products, selection=SELECTION)
        assert error.value.detail_codes == (
            "missing_artifact", "question_unreadable", "unsupported_retrieval_surface")

    def test_selection_from_another_dataset_is_refused(self) -> None:
        with pytest.raises(entry.EntryError) as error:
            entry.validate_entry(
                handoff(), outcome(dataset_id="ds-2"), FakeProducts(COMMITTED),
                selection=SELECTION)
        assert error.value.detail_codes == ("no_table_selection",)
