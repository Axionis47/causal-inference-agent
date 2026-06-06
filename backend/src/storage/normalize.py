"""Normalise a job's raw bundle to canonical parquet.

A Kaggle download can ship several files in several formats (csv, tsv, txt,
json, xlsx/xls, parquet). At the data-review gate every tabular file is
converted once to one parquet under ``{job_id}/normalized/`` so that everything
downstream, the row preview at the gate and the analysis after approval, reads a
single format with zero per-format branching. Files that are not tabular (an
image, a pdf, a freeform readme that does not parse) are left in ``raw/`` only
and flagged so the UI can list them without offering a preview.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.logging_config.structured import get_logger
from src.storage.job_data import job_normalized_dir, job_raw_dir

logger = get_logger(__name__)

# Extensions we attempt to read into a DataFrame. Anything else is non-tabular.
_TABULAR_EXTS = {".csv", ".tsv", ".txt", ".json", ".xlsx", ".xls", ".parquet"}


@dataclass
class NormalizeResult:
    """Outcome of normalising one raw file.

    ``normalized_name`` is the parquet file written under normalized/ (or None
    when the file is not tabular or failed to parse). ``tabular`` is True only
    when a parquet was written, so the UI can decide previewability from it.
    """

    name: str  # the raw file name
    normalized_name: str | None
    columns: list[str] = field(default_factory=list)
    n_rows: int | None = None
    tabular: bool = False
    error: str | None = None


def _read_tabular(path: Path) -> pd.DataFrame:
    """Read one raw file into a DataFrame, dispatched on extension.

    txt is read with delimiter sniffing; a file that does not parse as a table
    raises, which the caller turns into a non-tabular result.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".txt":
        return pd.read_csv(path, sep=None, engine="python")
    if suffix == ".json":
        return pd.read_json(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"unsupported extension {suffix}")


def normalize_file(raw_path: Path, out_dir: Path) -> NormalizeResult:
    """Convert one raw file to parquet under out_dir, or flag it non-tabular.

    The parquet is named ``{raw_name}.parquet`` so two raw files cannot collide
    on a shared stem (e.g. ``a.csv`` and ``a.json``).
    """
    name = raw_path.name
    if raw_path.suffix.lower() not in _TABULAR_EXTS:
        return NormalizeResult(name, None)

    try:
        df = _read_tabular(raw_path)
    except Exception as exc:
        logger.warning("normalize_read_failed", file=name, error=str(exc)[:200])
        return NormalizeResult(name, None, error=str(exc)[:200])

    columns = [str(c) for c in df.columns]
    out_dir.mkdir(parents=True, exist_ok=True)
    normalized_name = f"{name}.parquet"
    try:
        df.to_parquet(out_dir / normalized_name, index=False)
    except Exception as exc:
        logger.warning("normalize_write_failed", file=name, error=str(exc)[:200])
        return NormalizeResult(name, None, columns, int(df.shape[0]), error=str(exc)[:200])

    return NormalizeResult(name, normalized_name, columns, int(df.shape[0]), tabular=True)


def normalize_bundle(job_id: str) -> list[NormalizeResult]:
    """Normalise every file in a job's raw dir to parquet, return one result
    per file. The normalized dir is cleared first so a re-run leaves no stale
    parquet from a previous bundle."""
    raw = job_raw_dir(job_id)
    out = job_normalized_dir(job_id)
    for stale in out.iterdir():
        if stale.is_file():
            stale.unlink()

    results: list[NormalizeResult] = []
    for path in sorted(raw.iterdir()):
        if path.is_file():
            results.append(normalize_file(path, out))
    return results
