"""Validation tests for the RepairRecord and DataRepairs typed models."""

import pytest
from pydantic import ValidationError

from src.analysis.agents.data_repair import DataRepairs, RepairRecord


class TestRepairRecord:
    def test_missing_record(self):
        r = RepairRecord(
            type="missing", strategy="median", columns=["education"], before=8, after=0
        )
        assert r.type == "missing"
        assert r.before == 8
        assert r.after == 0
        assert r.rows_dropped is None

    def test_outliers_record(self):
        r = RepairRecord(type="outliers", strategy="winsorize", columns=["income"])
        assert r.type == "outliers"
        assert r.strategy == "winsorize"

    def test_invalid_type_rejected(self):
        """type is restricted to the three known repair categories."""
        with pytest.raises(ValidationError):
            RepairRecord(type="unknown", strategy="x", columns=[])

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            RepairRecord(strategy="median", columns=[])


class TestDataRepairs:
    def test_defaults(self):
        d = DataRepairs()
        assert d.records == []
        assert d.quality_assessment == ""
        assert d.cautions == []
        assert d.original_shape is None
        assert d.final_shape is None

    def test_round_trip(self):
        """model_dump / model_validate must round-trip cleanly for persistence."""
        d1 = DataRepairs(
            records=[RepairRecord(type="missing", strategy="mode", columns=["region"])],
            quality_assessment="Acceptable",
            cautions=["watch for selection bias"],
            original_shape=(100, 8),
            final_shape=(95, 7),
        )
        d2 = DataRepairs.model_validate(d1.model_dump())
        assert d2 == d1
