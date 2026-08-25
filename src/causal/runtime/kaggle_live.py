"""The live `kaggle==2.2.4` adapter for `KaggleClientProtocol` (T-014 §3; D-034).

No method here takes, returns, or logs a credential: the SDK client reads
`~/.kaggle/kaggle.json` itself when the adapter is constructed, and every provider
failure is re-raised as a `KaggleError` naming only the dataset and the exception
class, never a response body that could embed a token.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from causal.intake.kaggle import FETCH_FAILED, KaggleError

__all__ = ["LiveKaggleClient", "build_kaggle_api"]

METADATA_FILE: Final = "dataset-metadata.json"


def build_kaggle_api() -> Any:
    """The authenticated SDK client; imported here because import-time auth is a hazard."""
    from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore[import-untyped]

    api = KaggleApi()
    api.authenticate()  # reads ~/.kaggle/kaggle.json itself (D-034)
    return api


def _as_dict(value: Any) -> dict[str, Any]:
    """One SDK response object as a plain JSON dict; `to_dict` emits the wire names."""
    converted = value.to_dict() if hasattr(value, "to_dict") else vars(value)
    return {str(key): item for key, item in dict(converted).items()}


class LiveKaggleClient:
    """kaggle 2.2.4 behind the intake Protocol; responses become plain JSON dicts."""

    def __init__(self, api_factory: Callable[[], Any] = build_kaggle_api) -> None:
        self._api = api_factory()

    def dataset_status(self, owner: str, slug: str) -> dict[str, object]:
        """The contract's `currentVersionNumber`, resolved through `dataset_list`.

        Kaggle retired `GetDatasetStatus`, so 2.2.4's `api.dataset_status` answers 404 for
        every dataset; a search row still carries `current_version_number`. The protocol
        method name stays as it is because the capture contract is frozen.
        """
        def fetch(ref: str) -> dict[str, object]:
            rows = [row for row in self._api.dataset_list(search=slug) if str(row.ref) == ref]
            version = str(rows[0].current_version_number or "") if rows else ""
            if not version.strip():
                raise LookupError("no search row carries a version for this exact ref")
            return {"currentVersionNumber": version}

        return self._call("dataset_status", owner, slug, fetch)

    def dataset_metadata(self, owner: str, slug: str) -> dict[str, object]:
        """The SDK writes `dataset-metadata.json` to a directory; read it back and drop it."""
        def fetch(ref: str) -> dict[str, object]:
            with tempfile.TemporaryDirectory() as directory:
                written = self._api.dataset_metadata(ref, directory)
                path = Path(str(written)) if written else Path(directory) / METADATA_FILE
                loaded: object = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                return {"metadata": loaded}
            # The live file nests title/subtitle/description under `info`; capture is flat (D-065).
            info = loaded.pop("info", None)
            return loaded | info if isinstance(info, dict) else loaded

        return self._call("dataset_metadata", owner, slug, fetch)

    def dataset_files(self, owner: str, slug: str) -> dict[str, object]:
        """The declared file and column metadata under the contract's `files` key."""
        def fetch(ref: str) -> dict[str, object]:
            response = self._api.dataset_list_files(ref)
            body = _as_dict(response)
            declared = body.get("datasetFiles") or []
            return {"files": [dict(entry) for entry in declared],
                    "nextPageToken": body.get("nextPageToken")}

        return self._call("dataset_files", owner, slug, fetch)

    def download_archive(self, owner: str, slug: str, version: str) -> bytes:
        """The exact pinned version's archive, downloaded to a temporary directory."""
        def fetch(ref: str) -> bytes:
            with tempfile.TemporaryDirectory() as directory:
                self._api.dataset_download_files(
                    f"{ref}/{version}", path=directory, force=True, quiet=True, unzip=False)
                found = sorted(Path(directory).glob("*.zip"))
                if not found:
                    raise FileNotFoundError("the provider wrote no archive")
                return found[0].read_bytes()

        return self._call("download_archive", owner, slug, fetch)

    def _call[T](self, operation: str, owner: str, slug: str, call: Callable[[str], T]) -> T:
        """Every provider failure loses its body: only the dataset and the class survive."""
        try:
            return call(f"{owner}/{slug}")
        except Exception as error:  # noqa: BLE001 -- any provider failure is one KaggleError
            failure = type(error).__name__
        # Raised outside the handler so the original exception is neither the cause nor the
        # context: a provider body may carry a token or a signed URL, and a traceback prints it.
        raise KaggleError(
            f"kaggle {operation} failed for {owner}/{slug}: {failure}", FETCH_FAILED)
