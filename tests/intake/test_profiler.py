"""Tests for the deterministic table profiler (T-007; PRD-001 §6; EV-P1-004 unit layer)."""

from __future__ import annotations

import hashlib
import io
import json
import math

import polars as pl
import pytest

from causal.intake.profiler import profile_table
from causal.shared.canonical import canonical_bytes, content_hash

CSV = (
    b"unit_id,earnings,group,visit_date,notes\n"
    b"1,100.5,treated,2020-01-01,a\n"
    b"2,999,control,2020-02-01,\n"
    b"3,250.0,treated,2020-03-01,c\n"
    b"4,999,control,2020-04-01,d\n"
    b"5,,control,2020-05-01,e\n"
)


@pytest.fixture(scope="module")
def profile() -> dict[str, object]:
    return profile_table(CSV, "csv", "profiler.v1")


class TestTableFacts:
    def test_shape_and_metadata(self, profile: dict[str, object]) -> None:
        assert profile["row_count"] == 5
        assert profile["column_count"] == 5
        assert profile["duplicate_row_count"] == 0
        assert profile["profiler_version"] == "profiler.v1"
        assert profile["schema_version"] == "table-profile.v1"

    def test_unique_single_columns(self, profile: dict[str, object]) -> None:
        assert "unit_id" in profile["unique_single_columns"]  # type: ignore[operator]

    def test_null_accounting(self, profile: dict[str, object]) -> None:
        columns = profile["columns"]
        assert isinstance(columns, dict)
        assert columns["earnings"]["null_count"] == 1
        assert columns["earnings"]["null_rate"] == pytest.approx(0.2)
        assert columns["earnings"]["cardinality"] == 3

    def test_numeric_stats(self, profile: dict[str, object]) -> None:
        numeric = profile["columns"]["earnings"]["numeric"]  # type: ignore[index]
        assert numeric["min"] == 100.5
        assert numeric["max"] == 999.0
        assert numeric["non_finite_count"] == 0
        assert set(numeric["quantiles"]) == {"0.25", "0.5", "0.75"}

    def test_categorical_levels(self, profile: dict[str, object]) -> None:
        levels = profile["columns"]["group"]["levels"]  # type: ignore[index]
        assert levels == {"control": 3, "treated": 2}

    def test_temporal_min_max(self, profile: dict[str, object]) -> None:
        temporal = profile["columns"]["visit_date"]["temporal"]  # type: ignore[index]
        assert temporal["min"] == "2020-01-01"
        assert temporal["max"] == "2020-05-01"


class TestHypotheses:
    def test_identifier_hypothesis(self, profile: dict[str, object]) -> None:
        kinds = [h["kind"] for h in profile["columns"]["unit_id"]["hypotheses"]]  # type: ignore[index]
        assert kinds == ["identifier"]

    def test_sentinel_hypothesis(self, profile: dict[str, object]) -> None:
        hypotheses = profile["columns"]["earnings"]["hypotheses"]  # type: ignore[index]
        assert any(
            h["kind"] == "missing_sentinel" and "999" in h["detail"] for h in hypotheses
        )

    def test_no_causal_role_language(self, profile: dict[str, object]) -> None:
        text = str(profile).lower()
        for forbidden in ("treatment", "outcome", "confounder", "instrument", "mediator"):
            assert forbidden not in text


class TestDeterminismAndFormats:
    def test_identical_bytes_identical_hash(self) -> None:
        first = profile_table(CSV, "csv", "profiler.v1")
        second = profile_table(CSV, "csv", "profiler.v1")
        assert content_hash(first) == content_hash(second)

    def test_input_hash_recorded(self, profile: dict[str, object]) -> None:
        assert profile["input_sha256"] == hashlib.sha256(CSV).hexdigest()

    @pytest.mark.parametrize("media_type", ["csv", "tsv", "parquet"])
    def test_non_finite_values_preserve_evidence_and_canonical_roundtrip(
        self, media_type: str,
    ) -> None:
        data = (
            b"mixed,nonfinite,finite\n"
            b"NaN,NaN,1\ninf,inf,2\n-inf,-inf,3\n"
            b"1,NaN,4\n3,inf,5\n,-inf,6\n"
        )
        if media_type == "tsv":
            data = data.replace(b",", b"\t")
        elif media_type == "parquet":
            buffer = io.BytesIO()
            pl.read_csv(io.BytesIO(data)).write_parquet(buffer)
            data = buffer.getvalue()

        result = profile_table(data, media_type, "profiler.v1")
        columns = result["columns"]
        assert isinstance(columns, dict)
        assert result["input_sha256"] == hashlib.sha256(data).hexdigest()
        assert result["row_count"] == 6
        assert columns["mixed"]["null_count"] == 1
        assert columns["mixed"]["cardinality"] == 5
        assert columns["mixed"]["numeric"] == {
            "min": 1.0, "max": 3.0, "mean": 2.0,
            "std": pytest.approx(math.sqrt(2)),
            "quantiles": {"0.25": 1.5, "0.5": 2.0, "0.75": 2.5},
            "non_finite_count": 3,
        }
        assert columns["nonfinite"]["null_count"] == 0
        assert columns["nonfinite"]["all_null"] is False
        assert columns["nonfinite"]["numeric"] == {
            "min": None, "max": None, "mean": None, "std": None,
            "quantiles": {"0.25": None, "0.5": None, "0.75": None},
            "non_finite_count": 6,
        }
        assert columns["finite"]["numeric"] == {
            "min": 1.0, "max": 6.0, "mean": 3.5,
            "std": pytest.approx(math.sqrt(3.5)),
            "quantiles": {"0.25": 2.25, "0.5": 3.5, "0.75": 4.75},
            "non_finite_count": 0,
        }
        encoded = canonical_bytes(result)
        assert json.loads(encoded) == result
        assert canonical_bytes(json.loads(encoded)) == encoded
        assert content_hash(result) == hashlib.sha256(encoded).hexdigest()
        assert content_hash(result) == content_hash(
            profile_table(data, media_type, "profiler.v1")
        )

    def test_finite_values_with_overflowed_statistics_are_canonical(self) -> None:
        result = profile_table(b"x\n1e308\n1e308\n", "csv", "profiler.v1")
        numeric = result["columns"]["x"]["numeric"]  # type: ignore[index]
        assert numeric["min"] == numeric["max"] == 1e308
        assert numeric["non_finite_count"] == 0
        assert numeric["mean"] is None
        assert numeric["std"] is None
        assert json.loads(canonical_bytes(result)) == result

    def test_tsv(self) -> None:
        tsv = CSV.replace(b",", b"\t")
        assert profile_table(tsv, "tsv", "profiler.v1")["row_count"] == 5

    def test_parquet(self) -> None:
        frame = pl.read_csv(io.BytesIO(CSV))
        buffer = io.BytesIO()
        frame.write_parquet(buffer)
        result = profile_table(buffer.getvalue(), "parquet", "profiler.v1")
        assert result["row_count"] == 5 and result["media_type"] == "parquet"

    def test_unsupported_media_type(self) -> None:
        with pytest.raises(ValueError, match="unsupported media type"):
            profile_table(b"x", "xlsx", "profiler.v1")

    def test_all_null_and_constant_flags(self) -> None:
        data = b"a,b\n,x\n,x\n"
        columns = profile_table(data, "csv", "profiler.v1")["columns"]
        assert columns["a"]["all_null"] is True  # type: ignore[index]
        assert columns["a"]["cardinality"] == 0  # type: ignore[index]
        assert columns["b"]["constant"] is True  # type: ignore[index]
