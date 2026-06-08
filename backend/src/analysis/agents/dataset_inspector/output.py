"""Output types for dataset_inspector.

The inspector writes `state.file_profiles: dict[str, DataProfile]` (one
entry per candidate) and copies the winner into `state.data_profile`. No
new type is introduced — the per-file profiles are just DataProfile,
which is what the rest of the pipeline already reads.

Re-exported here so consumers can `from dataset_inspector import
DataProfile` matching the convention used by data_profiler / eda /
causal_discovery.
"""
from __future__ import annotations

from src.analysis.agents.data_profiler import DataProfile
from src.domain.relational import FileRelational, RelationalProfile

__all__ = ["DataProfile", "FileRelational", "RelationalProfile"]
