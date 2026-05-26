"""Download subsystem.

Owns Kaggle credentials, dataset acquisition, and on-disk dataset
storage. Analysis stages read a DownloadRecord via get_record(); they
do not call Kaggle, do not see credentials, and do not assume a
particular storage layout beyond what the record reports.

Public surface
--------------

Profile (Kaggle credentials):
    set_profile_key(profile_id, username, key, *, root, encryption_key)
    get_profile(profile_id, *, root) -> KaggleProfile | None
    clear_profile_key(profile_id, *, root)

Downloads (the lifecycle):
    start_download(req, *, root, encryption_key, bus) -> download_id
    get_record(download_id, *, root) -> DownloadRecord | None
    list_records(*, root, profile_id=None) -> list[DownloadRecord]

HTTP integration:
    router         FastAPI APIRouter mounted at /api/v1/download

Event bus:
    get_event_bus() -> DownloadEventBus  (process-wide, lazy)
"""
from src.domain.download import (
    DownloadEvent,
    DownloadRecord,
    DownloadRequest,
    DownloadStatus,
    FileEntry,
    KaggleMetadata,
    KaggleProfile,
)

from src.download.api import router
from src.download.download_store import (
    list_records,
    load as get_record,
)
from src.download.events import get_bus as get_event_bus
from src.download.profile_store import (
    clear_kaggle_key as clear_profile_key,
    get_profile,
    set_kaggle_key as set_profile_key,
)
from src.download.pull import start_download

__all__ = [
    # Domain types
    "DownloadEvent",
    "DownloadRecord",
    "DownloadRequest",
    "DownloadStatus",
    "FileEntry",
    "KaggleMetadata",
    "KaggleProfile",
    # Profile
    "set_profile_key",
    "get_profile",
    "clear_profile_key",
    # Downloads
    "start_download",
    "get_record",
    "list_records",
    # HTTP / events
    "router",
    "get_event_bus",
]
