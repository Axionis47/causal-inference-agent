"""Dataset ingestion and reversible preparation for the Streamlit studio.

This module owns *data mechanics*, not causal meaning.  It can describe a
table, propose conservative repairs, and apply only the repairs a person has
selected.  It never imputes outcomes, removes outliers, or overwrites the raw
upload: those choices can change a causal estimate and belong in a reviewed
analysis plan.
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PII_NAME = re.compile(
    r"(^|_)(name|email|phone|address|ssn|social_security|passport|dob|"
    r"date_of_birth|ip_address|credit_card)($|_)",
    re.I,
)


def fingerprint(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:16]


def _read_named_tables(filename: str, payload: bytes) -> dict[str, pd.DataFrame]:
    """Read one supported file, preserving every Excel sheet as a table."""
    suffix = Path(filename).suffix.lower()
    source = io.BytesIO(payload)
    if suffix == ".csv":
        return {filename: pd.read_csv(source)}
    if suffix in {".tsv", ".txt"}:
        return {filename: pd.read_csv(source, sep="\t")}
    if suffix in {".xlsx", ".xls"}:
        sheets = pd.read_excel(source, sheet_name=None)
        return {f"{filename}::{sheet}": frame for sheet, frame in sheets.items()}
    if suffix in {".parquet", ".pq"}:
        return {filename: pd.read_parquet(source)}
    raise ValueError(f"Unsupported table file {filename!r}")


def read_uploaded(filename: str, payload: bytes) -> pd.DataFrame:
    """Read a single upload and return its first table for compatibility."""
    tables = _read_named_tables(filename, payload)
    return next(iter(tables.values()))


def read_bundle(files: list[tuple[str, bytes]]) -> dict[str, pd.DataFrame]:
    """Read uploaded tables or Kaggle zip files without extracting to disk."""
    tables: dict[str, pd.DataFrame] = {}
    supported = {".csv", ".tsv", ".txt", ".xlsx", ".xls", ".parquet", ".pq"}
    for filename, payload in files:
        if len(payload) > 250 * 1024 * 1024:
            raise ValueError(f"{filename} exceeds the 250 MB MVP per-file limit")
        if Path(filename).suffix.lower() == ".zip":
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                members = [
                    member for member in archive.infolist()
                    if not member.is_dir()
                    and Path(member.filename).suffix.lower() in supported
                    and member.file_size <= 250 * 1024 * 1024
                ]
                if len(members) > 30:
                    raise ValueError("Bundle contains more than 30 supported tables; narrow it first")
                for member in members:
                    # Reading directly avoids zip-slip/path-traversal extraction.
                    inner = Path(member.filename).name
                    key = f"{Path(filename).stem}/{inner}"
                    for table_name, frame in _read_named_tables(key, archive.read(member)).items():
                        tables[table_name] = frame
        else:
            for table_name, frame in _read_named_tables(filename, payload).items():
                tables[table_name] = frame
    if not tables:
        raise ValueError("No supported table was found in the upload bundle")
    return tables


def bundle_fingerprint(files: list[tuple[str, bytes]]) -> str:
    digest = hashlib.sha256()
    for name, payload in sorted(files, key=lambda pair: pair[0]):
        digest.update(name.encode())
        digest.update(payload)
    return digest.hexdigest()[:16]


def _unique_columns(columns: list[Any]) -> list[str]:
    """Strip column names while preserving every column through collisions."""
    seen: dict[str, int] = {}
    result: list[str] = []
    for raw in columns:
        base = str(raw).strip() or "unnamed"
        count = seen.get(base, 0)
        seen[base] = count + 1
        result.append(base if count == 0 else f"{base}__{count + 1}")
    return result


def propose_repairs(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Return a reviewable repair plan; nothing is applied here."""
    proposals: list[dict[str, Any]] = []

    renamed = _unique_columns(list(df.columns))
    if renamed != [str(c) for c in df.columns]:
        proposals.append({
            "id": "normalize_column_names",
            "label": "Normalize column names",
            "detail": "Trim surrounding whitespace and make collisions explicit.",
            "affected": sum(a != b for a, b in zip(map(str, df.columns), renamed)),
            "safe_default": True,
        })

    text = df.select_dtypes(include=["object", "string"])
    blank_count = 0
    for column in text.columns:
        blank_count += int(text[column].astype("string").str.strip().eq("").sum())
    if blank_count:
        proposals.append({
            "id": "blank_strings_to_null",
            "label": "Treat blank strings as missing",
            "detail": "Replace empty/whitespace-only cells with null; no rows are removed.",
            "affected": blank_count,
            "safe_default": True,
        })

    numeric = df.select_dtypes(include=[np.number])
    infinity_count = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()) if not numeric.empty else 0
    if infinity_count:
        proposals.append({
            "id": "infinity_to_null",
            "label": "Treat ±infinity as missing",
            "detail": "Replace non-finite numeric values with null; no rows are removed.",
            "affected": infinity_count,
            "safe_default": True,
        })

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        proposals.append({
            "id": "drop_exact_duplicates",
            "label": "Drop exact duplicate rows",
            "detail": "Keep the first byte-for-byte identical row. This changes row count.",
            "affected": duplicate_count,
            "safe_default": False,
        })

    for column in text.columns:
        raw = text[column].astype("string").str.strip()
        present = raw.notna() & raw.ne("")
        if present.sum() < 20:
            continue
        parsed = pd.to_numeric(raw.str.replace(",", "", regex=False), errors="coerce")
        success = float(parsed[present].notna().mean())
        if success >= 0.95 and parsed[present].nunique() > 2:
            proposals.append({
                "id": f"coerce_numeric::{column}",
                "label": f"Parse `{column}` as numeric",
                "detail": f"{success:.0%} of present values parse as numbers. Review IDs and leading zeros.",
                "affected": int(present.sum()),
                "safe_default": False,
            })

    return proposals


def apply_repairs(
    df: pd.DataFrame, proposals: list[dict[str, Any]], selected: list[str]
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply selected repairs to a copy and return an append-only audit log."""
    clean = df.copy(deep=True)
    by_id = {p["id"]: p for p in proposals}
    log: list[dict[str, Any]] = []

    for repair_id in selected:
        proposal = by_id.get(repair_id)
        if not proposal:
            continue
        before_rows = len(clean)
        if repair_id == "normalize_column_names":
            clean.columns = _unique_columns(list(clean.columns))
        elif repair_id == "blank_strings_to_null":
            for column in clean.select_dtypes(include=["object", "string"]).columns:
                values = clean[column].astype("string")
                clean[column] = values.mask(values.str.strip().eq(""), pd.NA)
        elif repair_id == "infinity_to_null":
            clean.replace([np.inf, -np.inf], np.nan, inplace=True)
        elif repair_id == "drop_exact_duplicates":
            clean = clean.drop_duplicates(keep="first").copy()
        elif repair_id.startswith("coerce_numeric::"):
            column = repair_id.split("::", 1)[1]
            if column in clean.columns:
                clean[column] = pd.to_numeric(
                    clean[column].astype("string").str.replace(",", "", regex=False),
                    errors="coerce",
                )
        else:
            continue
        log.append({
            "id": repair_id,
            "label": proposal["label"],
            "affected": proposal["affected"],
            "rows_before": before_rows,
            "rows_after": len(clean),
        })
    return clean, log


def quality_summary(df: pd.DataFrame) -> dict[str, Any]:
    missing = df.isna().mean().sort_values(ascending=False)
    constants = [str(c) for c in df.columns if df[c].nunique(dropna=True) <= 1]
    pii = [str(c) for c in df.columns if PII_NAME.search(str(c))]
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "high_missing_columns": [str(c) for c, rate in missing.items() if rate > 0.30],
        "constant_columns": constants,
        "possible_pii_columns": pii,
    }


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Hash the exact canonical CSV representation used by analysis runs."""
    payload = df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_fingerprint(path: str | Path) -> str:
    """Hash a persisted analysis artifact without loading it into graph state."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_data_version(
    raw: pd.DataFrame,
    prepared: pd.DataFrame,
    *,
    cohort: dict[str, Any] | None,
    repairs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a content-addressed version for one approved prepared dataset."""
    manifest = {
        "schema_version": "1.0.0",
        "cohort": cohort,
        "repair_ids": [
            str(item.get("id")) for item in repairs
            if item.get("id") and item.get("id") != "analysis_cohort"
        ],
    }
    manifest_json = json.dumps(manifest, sort_keys=True, default=str, separators=(",", ":"))
    raw_fingerprint = dataframe_fingerprint(raw)
    prepared_fingerprint = dataframe_fingerprint(prepared)
    manifest_hash = hashlib.sha256(manifest_json.encode()).hexdigest()
    identity = json.dumps(
        {
            "raw_fingerprint": raw_fingerprint,
            "prepared_fingerprint": prepared_fingerprint,
            "manifest_hash": manifest_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        **manifest,
        "version_id": hashlib.sha256(identity.encode()).hexdigest()[:16],
        "raw_fingerprint": raw_fingerprint,
        "prepared_fingerprint": prepared_fingerprint,
        "manifest_hash": manifest_hash,
        "rows": int(len(prepared)),
        "columns": int(len(prepared.columns)),
    }


REQUIRED_CONTEXT = (
    "question",
    "description",
    "unit",
    "assignment",
    "timing",
    "population",
    "intended_use",
    "outcome",
)


def context_readiness(context: dict[str, Any]) -> tuple[bool, list[str]]:
    missing = [key for key in REQUIRED_CONTEXT if not str(context.get(key, "")).strip()]
    return not missing, missing
