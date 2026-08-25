"""Frozen Kaggle fixtures for intake tests (T-008; EV-P1-002 fixture focus)."""

from __future__ import annotations

import copy
import io
import zipfile
from typing import Final

CSV: Final = (
    b"unit_id,earnings,group\n"
    b"1,100.5,treated\n"
    b"2,999,control\n"
    b"3,250.0,treated\n"
    b"4,999,control\n"
    b"5,120.0,control\n"
)
README: Final = b"# NSW\nEarnings study of a job training program.\n"

STATUS_RESPONSE: Final[dict[str, object]] = {"status": "ready", "currentVersionNumber": 3}
METADATA_RESPONSE: Final[dict[str, object]] = {
    "title": "NSW earnings",
    "subtitle": "",
    "description": "Job training program and yearly earnings",
    "licenseName": "CC0-1.0",
    "usabilityRating": 0.88,
    "viewCount": 120,
    "keywords": ["economics"],
    "mysteryField": "unclassified",
}
FILES_RESPONSE: Final[dict[str, object]] = {
    "files": [
        {
            "name": "nsw.csv",
            "description": "main analysis table",
            "totalBytes": 90,
            "columns": [
                {"name": "unit_id", "description": "participant id",
                 "type": "integer", "order": 0},
                {"name": "earnings", "description": "", "type": "number", "order": 1},
                {"name": "group", "type": "string", "order": 2},
            ],
        }
    ]
}

DEFAULT_FILES: Final[dict[str, bytes]] = {"nsw.csv": CSV, "readme.md": README}


def build_zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


class FrozenKaggleClient:
    """Replays frozen provider responses. The token is a canary: it must never
    appear in any artifact, event line, or catalog row."""

    api_token = "kaggle-secret-token-canary"

    def __init__(
        self,
        files: dict[str, bytes] | None = None,
        status: dict[str, object] | None = None,
        metadata: dict[str, object] | None = None,
        files_response: dict[str, object] | None = None,
    ) -> None:
        self._files = dict(DEFAULT_FILES if files is None else files)
        self._status = status if status is not None else dict(STATUS_RESPONSE)
        self._metadata = metadata if metadata is not None else dict(METADATA_RESPONSE)
        self._files_response = (
            files_response if files_response is not None
            else copy.deepcopy(FILES_RESPONSE)
        )

    def dataset_status(self, owner: str, slug: str) -> dict[str, object]:
        return copy.deepcopy(self._status)

    def dataset_metadata(self, owner: str, slug: str) -> dict[str, object]:
        return copy.deepcopy(self._metadata)

    def dataset_files(self, owner: str, slug: str) -> dict[str, object]:
        return copy.deepcopy(self._files_response)

    def download_archive(self, owner: str, slug: str, version: str) -> bytes:
        return build_zip(self._files)


class FailingKaggleClient(FrozenKaggleClient):
    def dataset_status(self, owner: str, slug: str) -> dict[str, object]:
        raise RuntimeError("provider unreachable")
