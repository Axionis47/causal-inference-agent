"""Build the analyzable frame from a multi-file bundle, per an AssemblyPlan.

Pure and deterministic: the same manifest + plan always yield the same frame.
`load_analysis_frame` calls this once at run start, and the generated notebook
imports and calls it too, so the analysis and its replay read identical data.
The plan does no reshape; wide->long stays in the DiD lane this pass.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from src.analysis_v2.spec import AssemblyPlan
from src.domain.dataset_manifest import DatasetManifest
from src.storage.job_data import job_normalized_dir, job_raw_dir


class DatasetUnavailable(RuntimeError):
    """A confirmed job's dataset, or a file an AssemblyPlan references, could
    not be located on disk."""


def _read_manifest_file(
    job_id: str, manifest: DatasetManifest, name: str
) -> pd.DataFrame:
    """One file in the bundle, by manifest name: normalized parquet preferred,
    raw CSV the fallback for pre-normalization bundles. The single source of
    'manifest name -> frame on disk', shared by the loader and the executor."""
    entry = next((f for f in manifest.files if f.name == name), None)
    if entry is None:
        raise DatasetUnavailable(f"file {name!r} is not in the manifest file list")
    if entry.normalized_path:
        parquet = job_normalized_dir(job_id) / Path(entry.normalized_path).name
        if parquet.exists():
            return pd.read_parquet(parquet)
    raw = job_raw_dir(job_id) / name
    if not raw.exists():
        raise DatasetUnavailable(f"no normalized or raw file for {name!r}")
    return pd.read_csv(raw)


def assemble_from(
    read_file: Callable[[str], pd.DataFrame], plan: AssemblyPlan
) -> pd.DataFrame:
    """Build the analyzable frame per the plan, resolving each file name through
    `read_file`: the base file, same-schema shards concatenated onto it, sibling
    joins applied in order, then column drops. Pure given the reader. The
    backend and the generated notebook share this one implementation (with
    different readers) so the assembled frame cannot drift between them."""
    frame = read_file(plan.base_file)

    if plan.concat_files:
        shards = [frame] + [read_file(name) for name in plan.concat_files]
        frame = pd.concat(shards, ignore_index=True)

    for join in plan.joins:
        right = read_file(join.right_file)
        missing_left = [c for c in join.on if c not in frame.columns]
        missing_right = [c for c in join.on if c not in right.columns]
        if missing_left or missing_right:
            raise DatasetUnavailable(
                f"join on {join.on} needs keys in both frames; "
                f"missing left={missing_left} right={missing_right}"
            )
        frame = frame.merge(right, on=join.on, how=join.how)

    drop = [c for c in plan.drop_columns if c in frame.columns]
    if drop:
        frame = frame.drop(columns=drop)
    return frame


def assemble_frame(
    job_id: str, manifest: DatasetManifest, plan: AssemblyPlan
) -> pd.DataFrame:
    """`assemble_from` with the backend reader: each name resolves to its
    on-disk frame via the manifest. The notebook replays the same plan with a
    relative-path reader, so the analysis and its verification read identically."""
    return assemble_from(
        lambda name: _read_manifest_file(job_id, manifest, name), plan
    )
