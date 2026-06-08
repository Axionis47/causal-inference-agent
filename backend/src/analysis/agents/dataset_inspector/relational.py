"""Build the bundle's RelationalProfile from per-file profiles + the parquets.

Pure functions, no agent state. Schema identity and counts come from the
DataProfile objects the inspector already produced; candidate keys are computed
by the caller from the FULL candidate frames (never a sample, or a non-unique
key can look unique) and passed in. No cross-file foreign-key / containment pass
lives here: that is O(files^2 x columns), can blow up memory, and is
anticonservative if wrong, so it is deferred to a later confirmed stage.
"""
from __future__ import annotations

import pandas as pd

from src.analysis.agents.data_profiler import DataProfile
from src.domain.relational import FileRelational, RelationalProfile, ShapeHint


def candidate_keys(df: pd.DataFrame) -> list[str]:
    """Columns unique over the FULL frame with no nulls (candidate primary keys).

    Uniqueness must be checked on the whole file: a column that is unique in a
    sample but not in the file would be a false key, exactly the silent error
    this report exists to avoid.
    """
    n = len(df)
    if n == 0:
        return []
    keys: list[str] = []
    for col in df.columns:
        series = df[col]
        if series.notna().all() and series.nunique(dropna=False) == n:
            keys.append(str(col))
    return keys


def _schema_signature(profile: DataProfile) -> tuple[tuple[str, str], ...]:
    """Identity of a file's schema: its (column, dtype) set, order-independent."""
    return tuple(
        sorted(
            (str(name), str(profile.feature_types.get(name, "")))
            for name in profile.feature_names
        )
    )


def _classify(n_files: int, groups: dict[object, list[str]]) -> ShapeHint:
    if n_files <= 1:
        return "single"
    if len(groups) == 1:
        # Every file shares one schema: pieces of one logical table.
        return "same_schema_shards"
    if all(len(members) == 1 for members in groups.values()):
        # No two files share a schema.
        return "unrelated"
    # Mixed: some files share a schema, some do not. Deliberately coarse.
    return "multiple_files"


def build_relational_profile(
    profiles: dict[str, DataProfile],
    key_candidates_by_file: dict[str, list[str]],
) -> RelationalProfile:
    """Assemble the descriptive RelationalProfile. Pure, no I/O.

    profiles: every candidate's DataProfile (schema + counts).
    key_candidates_by_file: per-file candidate keys the caller computed from the
    full frames; missing entries default to empty (file not re-read).
    """
    signatures: dict[str, tuple] = {}
    files: list[FileRelational] = []
    for name, profile in profiles.items():
        signatures[name] = _schema_signature(profile)
        files.append(
            FileRelational(
                file=name,
                n_rows=profile.n_samples,
                n_cols=profile.n_features,
                key_candidates=key_candidates_by_file.get(name, []),
            )
        )

    groups: dict[tuple, list[str]] = {}
    for name, sig in signatures.items():
        groups.setdefault(sig, []).append(name)
    same_schema_groups = sorted(
        (sorted(members) for members in groups.values() if len(members) > 1),
        key=lambda g: g[0],
    )

    return RelationalProfile(
        shape_hint=_classify(len(profiles), groups),
        files=sorted(files, key=lambda f: f.file),
        same_schema_groups=same_schema_groups,
    )
