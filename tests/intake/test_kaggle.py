"""Tests for the Kaggle capture layer (T-008; PRD-001 §5; EV-P1-002 unit layer)."""

from __future__ import annotations

import hashlib

import pytest

from causal.intake.kaggle import KaggleError, capture
from causal.shared.canonical import canonical_bytes
from tests.intake.conftest import FailingKaggleClient, FrozenKaggleClient


class TestCapture:
    def test_resolves_version_and_identity(self) -> None:
        result = capture(FrozenKaggleClient(), "lalonde/nsw")
        dataset = result.payload["dataset"]
        assert isinstance(dataset, dict)
        assert dataset["version"] == "3"
        assert dataset["dataset_id"] == "kaggle:lalonde/nsw@3"
        assert dataset["status"] == "ready"
        assert result.payload["schema_version"] == "kaggle-capture.v1"

    def test_archive_hash_matches_bytes(self) -> None:
        result = capture(FrozenKaggleClient(), "lalonde/nsw")
        assert result.payload["archive_sha256"] == hashlib.sha256(
            result.archive_bytes
        ).hexdigest()

    def test_raw_responses_kept_verbatim(self) -> None:
        result = capture(FrozenKaggleClient(), "lalonde/nsw")
        metadata = result.payload["metadata_response"]
        assert isinstance(metadata, dict)
        assert metadata["usabilityRating"] == 0.88  # popularity captured, not dropped

    def test_provider_failure_is_fetch_failed(self) -> None:
        with pytest.raises(KaggleError) as excinfo:
            capture(FailingKaggleClient(), "lalonde/nsw")
        assert excinfo.value.code == "fetch_failed"

    def test_missing_version_is_unresolved(self) -> None:
        client = FrozenKaggleClient(status={"status": "ready"}, metadata={"title": "x"})
        with pytest.raises(KaggleError) as excinfo:
            capture(client, "lalonde/nsw")
        assert excinfo.value.code == "version_unresolved"

    def test_no_credential_in_payload(self) -> None:
        client = FrozenKaggleClient()
        result = capture(client, "lalonde/nsw")
        assert client.api_token.encode() not in canonical_bytes(result.payload)
