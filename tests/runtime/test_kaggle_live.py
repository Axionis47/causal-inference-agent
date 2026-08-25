"""The live kaggle==2.2.4 adapter, driven by a faked SDK client (T-014 §3; D-034)."""

from __future__ import annotations

import inspect
import io
import json
import os
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

import pytest

from causal.intake.kaggle import KaggleError, capture
from causal.runtime.kaggle_live import LiveKaggleClient
from tests.intake.conftest import CSV, README, build_zip

TOKEN: Final = "kaggle-secret-token-canary"
ARCHIVE: Final = build_zip({"nsw.csv": CSV, "readme.md": README})
FILES: Final[dict[str, Any]] = {
    "datasetFiles": [{"name": "nsw.csv", "description": "main analysis table",
                      "totalBytes": 90, "columns": [{"name": "unit_id", "type": "integer"}]}],
    "nextPageToken": None}


class _Files:
    """A kagglesdk response object: the adapter must reach it through `to_dict`."""

    def to_dict(self) -> dict[str, Any]:
        return dict(FILES)


class FakeApi:
    """`KaggleApi` with the 2.2.4 method names and return shapes, and no network."""

    def __init__(self, failing: str | None = None) -> None:
        self.failing = failing
        self.refs: dict[str, str] = {}

    def _record(self, operation: str, dataset: str) -> None:
        self.refs[operation] = dataset
        if self.failing == operation:
            raise RuntimeError(f"401 Unauthorized: Authorization=Bearer {TOKEN}")

    def dataset_status(self, dataset: str, format: str | None = None) -> str:
        self._record("dataset_status", dataset)
        return json.dumps({"status": "ready", "current_version_number": 3})

    def dataset_metadata(self, dataset: str, path: str) -> str:
        self._record("dataset_metadata", dataset)
        target = Path(path) / "dataset-metadata.json"
        target.write_text(json.dumps({"title": "NSW earnings", "licenseName": "CC0-1.0"}))
        return str(target)

    def dataset_list_files(self, dataset: str) -> _Files:
        self._record("dataset_list_files", dataset)
        return _Files()

    def dataset_download_files(self, dataset: str, path: str | None = None, force: bool = False,
                               quiet: bool = True, unzip: bool = False,
                               licenses: list[str] | None = None) -> None:
        self._record("dataset_download_files", dataset)
        (Path(str(path)) / "nsw.zip").write_bytes(ARCHIVE)


def client(failing: str | None = None) -> tuple[LiveKaggleClient, FakeApi]:
    api = FakeApi(failing)
    return LiveKaggleClient(api_factory=lambda: api), api


class TestProtocolShape:
    def test_every_response_is_a_plain_json_dict_in_contract_keys(self) -> None:
        live, _ = client()
        assert live.dataset_status("lalonde", "nsw") == {
            "status": "ready", "currentVersionNumber": 3}
        assert live.dataset_metadata("lalonde", "nsw")["licenseName"] == "CC0-1.0"
        files = live.dataset_files("lalonde", "nsw")
        declared = files["files"]
        assert isinstance(declared, list)
        assert declared[0]["name"] == "nsw.csv" and declared[0]["totalBytes"] == 90

    def test_the_archive_round_trips_as_readable_zip_bytes(self) -> None:
        live, api = client()
        data = live.download_archive("lalonde", "nsw", "3")
        assert zipfile.ZipFile(io.BytesIO(data)).namelist() == ["nsw.csv", "readme.md"]
        assert api.refs["dataset_download_files"] == "lalonde/nsw/3"  # the pinned version

    def test_the_capture_layer_resolves_the_version_through_the_adapter(self) -> None:
        live, _ = client()
        result = capture(live, "lalonde/nsw")
        dataset = result.payload["dataset"]
        assert isinstance(dataset, dict)
        assert dataset["version"] == "3" and dataset["status"] == "ready"
        assert result.archive_bytes == ARCHIVE


class TestSecrets:
    @pytest.mark.parametrize(
        "operation", ["dataset_status", "dataset_metadata", "dataset_list_files",
                      "dataset_download_files"])
    def test_a_provider_failure_loses_its_body(self, operation: str) -> None:
        live, _ = client(failing=operation)
        calls: dict[str, Callable[[], object]] = {
            "dataset_status": lambda: live.dataset_status("lalonde", "nsw"),
                 "dataset_metadata": lambda: live.dataset_metadata("lalonde", "nsw"),
                 "dataset_list_files": lambda: live.dataset_files("lalonde", "nsw"),
                 "dataset_download_files": lambda: live.download_archive("lalonde", "nsw", "3")}
        with pytest.raises(KaggleError) as raised:
            calls[operation]()
        message = str(raised.value)
        assert TOKEN not in message and "Bearer" not in message
        assert "lalonde/nsw" in message and "RuntimeError" in message
        assert raised.value.code == "fetch_failed"
        # No chained cause: a traceback must not reprint the provider's body either.
        assert raised.value.__cause__ is None and raised.value.__context__ is None

    def test_no_method_accepts_a_credential_parameter(self) -> None:
        for name in ("__init__", "dataset_status", "dataset_metadata", "dataset_files",
                     "download_archive"):
            parameters = inspect.signature(getattr(LiveKaggleClient, name)).parameters
            assert not [p for p in parameters
                        if any(word in p for word in ("key", "token", "secret", "credential"))]


@pytest.mark.skipif(os.environ.get("RUN_LIVE_KAGGLE") != "1",
                    reason="the live Kaggle smoke is opt-in (RUN_LIVE_KAGGLE=1)")
def test_live_metadata_smoke() -> None:
    """Metadata only: authenticates from ~/.kaggle/kaggle.json and downloads no archive."""
    live = LiveKaggleClient()
    status = live.dataset_status("uciml", "iris")
    assert str(status["currentVersionNumber"]).strip()
    assert isinstance(live.dataset_files("uciml", "iris")["files"], list)
