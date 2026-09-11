"""Deterministic table profiler: observations, not interpretations (PRD-001 §6; D-025)."""

from __future__ import annotations

import hashlib
import io
import math
from typing import Final

import polars as pl

__all__ = ["profile_table"]

LEVEL_CAP: Final = 20
SENTINEL_CANDIDATES: Final = (9, 99, 999, 9999, -9, -99, -999, -9999)
SENTINEL_MIN_RATE: Final = 0.01
QUANTILES: Final = (0.25, 0.5, 0.75)


def _read(data: bytes, media_type: str) -> pl.DataFrame:
    buffer = io.BytesIO(data)
    if media_type in ("csv", "tsv"):
        return pl.read_csv(buffer, separator="\t" if media_type == "tsv" else ",",
                           try_parse_dates=True)
    if media_type == "parquet":
        return pl.read_parquet(buffer)
    raise ValueError(f"unsupported media type for profiling: {media_type!r}")


def _numeric_stats(series: pl.Series) -> dict[str, object]:
    def as_float(value: object) -> float | None:
        if value is None:
            return None
        number = float(value)  # type: ignore[arg-type]
        # Finite inputs can still overflow a derived statistic.
        return number if math.isfinite(number) else None

    non_finite = 0
    if series.dtype.is_float():
        flags = series.is_nan() | series.is_infinite()
        non_finite = int(flags.sum() or 0)
        # Summarize usable numbers without changing the source's measured facts.
        series = series.filter(series.is_finite())
    return {
        "min": as_float(series.min()),
        "max": as_float(series.max()),
        "mean": as_float(series.mean()),
        "std": as_float(series.std()),
        "quantiles": {
            str(q): as_float(series.quantile(q, interpolation="linear")) for q in QUANTILES
        },
        "non_finite_count": non_finite,
    }


def _hypotheses(series: pl.Series, row_count: int, cardinality: int) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    if row_count > 0 and cardinality == row_count and (
        series.dtype.is_integer() or series.dtype == pl.String
    ):
        found.append(
            {"kind": "identifier", "detail": "every value is unique; may be an identifier"}
        )
    if series.dtype.is_numeric():
        non_null = row_count - series.null_count()
        minimum, maximum = series.min(), series.max()
        for candidate in SENTINEL_CANDIDATES:
            count = int((series == candidate).sum() or 0)
            if (
                count > 0
                and non_null > 0
                and count >= SENTINEL_MIN_RATE * non_null
                and (minimum == candidate or maximum == candidate)
            ):
                found.append(
                    {
                        "kind": "missing_sentinel",
                        "detail": f"{candidate} may be a missing-value sentinel",
                    }
                )
    return found


def _column_profile(series: pl.Series, row_count: int) -> dict[str, object]:
    cardinality = int(series.drop_nulls().n_unique())
    null_count = int(series.null_count())
    profile: dict[str, object] = {
        "dtype": str(series.dtype),
        "null_count": null_count,
        "null_rate": (null_count / row_count) if row_count else 0.0,
        "cardinality": cardinality,
        "all_null": row_count > 0 and null_count == row_count,
        "constant": row_count > 0 and series.drop_nulls().n_unique() <= 1,
        "hypotheses": _hypotheses(series, row_count, cardinality),
    }
    if series.dtype.is_numeric():
        profile["numeric"] = _numeric_stats(series)
    if series.dtype == pl.String and 0 < cardinality <= LEVEL_CAP:
        counts = series.drop_nulls().value_counts().sort(series.name)
        profile["levels"] = {
            str(row[0]): int(row[1]) for row in counts.iter_rows()
        }
    if series.dtype.is_temporal():
        minimum, maximum = series.min(), series.max()
        profile["temporal"] = {
            "min": None if minimum is None else str(minimum),
            "max": None if maximum is None else str(maximum),
        }
    return profile


def profile_table(data: bytes, media_type: str, profiler_version: str) -> dict[str, object]:
    """Canonical-ready measured-facts payload; identical bytes give identical output."""
    frame = _read(data, media_type)
    row_count = frame.height
    columns = {name: _column_profile(frame[name], row_count) for name in frame.columns}
    unique_single_columns = sorted(
        name
        for name, profile in columns.items()
        if row_count > 0 and profile["cardinality"] == row_count and not profile["all_null"]
    )
    return {
        "schema_version": "table-profile.v1",
        "profiler_version": profiler_version,
        "media_type": media_type,
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "row_count": row_count,
        "column_count": frame.width,
        "duplicate_row_count": row_count - frame.unique().height,
        "unique_single_columns": unique_single_columns,
        "columns": columns,
    }
