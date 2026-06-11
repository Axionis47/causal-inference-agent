"""Minimal job state for the download / data-review slice.

The single source of truth for the slice's types. The kept shell (api, storage,
jobs) imports these from here, replacing the deleted analysis package.
"""
from .dataset import DatasetInfo, FileEntry
from .profile import DataProfile
from .state import SCHEMA_VERSION, AnalysisState, StaleParkedState
from .status import JobStatus

__all__ = [
    "AnalysisState",
    "DataProfile",
    "DatasetInfo",
    "FileEntry",
    "JobStatus",
    "SCHEMA_VERSION",
    "StaleParkedState",
]
