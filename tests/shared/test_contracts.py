"""Tests for shared contract models (T-002, EV-SYS-001 unit layer)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from causal.shared.canonical import content_hash
from causal.shared.contracts import (
    ArtifactEnvelopeV1,
    ArtifactRef,
    HandoffManifestV1,
    SensitivityClass,
)

HASH = "a" * 64
NOW = datetime(2026, 8, 24, 12, 30, 45, 123456, tzinfo=UTC)


def envelope_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "artifact_id": "art-1",
        "artifact_type": "IntakeOutcome",
        "schema_version": "artifact.v1",
        "content_hash": HASH,
        "analysis_id": "an-1",
        "stage_run_id": "run-1",
        "producer_component": "intake-coordinator",
        "producer_version": "0.1.0",
        "parent_artifacts": (ArtifactRef(artifact_id="p-1", content_hash="b" * 64),),
        "sensitivity_class": SensitivityClass.INTERNAL,
        "created_at_utc": NOW,
        "payload_locator": "objects/ab/cd",
    }
    base.update(overrides)
    return base


class TestArtifactEnvelope:
    def test_valid_roundtrip(self) -> None:
        envelope = ArtifactEnvelopeV1(**envelope_kwargs())  # type: ignore[arg-type]
        again = ArtifactEnvelopeV1.model_validate(envelope.model_dump())
        assert again == envelope

    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactEnvelopeV1(**envelope_kwargs(surprise="x"))  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", ["", "A" * 64, "abc", "g" * 64])
    def test_bad_content_hash_rejected(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            ArtifactEnvelopeV1(**envelope_kwargs(content_hash=bad))  # type: ignore[arg-type]

    def test_naive_datetime_rejected(self) -> None:
        naive = datetime(2026, 8, 24, 12, 0, 0)  # noqa: DTZ001 -- naivety is what we test
        with pytest.raises(ValidationError):
            ArtifactEnvelopeV1(**envelope_kwargs(created_at_utc=naive))  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad",
        ["objects/x?sig=1", "objects/x&y", "https://s3/o/X-Amz-Signature=1", "", "x" * 1025],
    )
    def test_bad_payload_locator_rejected(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            ArtifactEnvelopeV1(**envelope_kwargs(payload_locator=bad))  # type: ignore[arg-type]

    def test_long_locator_within_1024_accepted(self) -> None:
        envelope = ArtifactEnvelopeV1(**envelope_kwargs(payload_locator="x" * 1024))  # type: ignore[arg-type]
        assert len(envelope.payload_locator) == 1024

    def test_identity_over_200_chars_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ArtifactEnvelopeV1(**envelope_kwargs(artifact_id="x" * 201))  # type: ignore[arg-type]

    def test_timestamp_serializes_six_digit_z_form(self) -> None:
        envelope = ArtifactEnvelopeV1(**envelope_kwargs())  # type: ignore[arg-type]
        payload = envelope.canonical_payload()
        assert payload["created_at_utc"] == "2026-08-24T12:30:45.123456Z"

    def test_frozen(self) -> None:
        envelope = ArtifactEnvelopeV1(**envelope_kwargs())  # type: ignore[arg-type]
        with pytest.raises(ValidationError):
            envelope.artifact_id = "other"  # type: ignore[misc]

    def test_canonical_payload_hash_stable(self) -> None:
        data = ArtifactEnvelopeV1(**envelope_kwargs()).model_dump()  # type: ignore[arg-type]
        reordered = dict(reversed(list(data.items())))
        first = ArtifactEnvelopeV1.model_validate(data)
        second = ArtifactEnvelopeV1.model_validate(reordered)
        assert content_hash(first.canonical_payload()) == content_hash(second.canonical_payload())


def manifest_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "handoff_id": "h-1",
        "schema_version": "handoff.v1",
        "analysis_id": "an-1",
        "producing_stage_run_id": "run-1",
        "receiving_stage_run_id": "run-2",
        "entries": (ArtifactRef(artifact_id="art-1", content_hash=HASH),),
        "originating_outcome": "usable",
        "approval_ids": (),
        "registry_version": "registry.v1",
        "compatibility_version": "compat.v1",
        "receiver_validation_result": None,
        "receiver_error_codes": (),
        "created_at_utc": NOW,
        "accepted_at_utc": None,
    }
    base.update(overrides)
    return base


class TestHandoffManifest:
    def test_valid_manifest(self) -> None:
        manifest = HandoffManifestV1(**manifest_kwargs())  # type: ignore[arg-type]
        assert manifest.entries[0].artifact_id == "art-1"

    def test_empty_entries_rejected(self) -> None:
        with pytest.raises(ValidationError):
            HandoffManifestV1(**manifest_kwargs(entries=()))  # type: ignore[arg-type]

    def test_none_accepted_at_serializes_null(self) -> None:
        manifest = HandoffManifestV1(**manifest_kwargs())  # type: ignore[arg-type]
        payload = manifest.canonical_payload()
        assert payload["accepted_at_utc"] is None
        assert content_hash(payload) == content_hash(payload)

    def test_accepted_at_serializes_six_digit_z_form(self) -> None:
        manifest = HandoffManifestV1(**manifest_kwargs(accepted_at_utc=NOW))  # type: ignore[arg-type]
        assert manifest.canonical_payload()["accepted_at_utc"] == "2026-08-24T12:30:45.123456Z"
