"""Validation tests for the DataProfile and TreatmentEncoding output models."""

import pytest
from pydantic import ValidationError

from src.analysis.agents.data_profiler import DataProfile, TreatmentEncoding


class TestDataProfile:
    def test_minimum_required_fields(self):
        """DataProfile requires shape + types/missing/stats; everything else defaults."""
        p = DataProfile(
            n_samples=10,
            n_features=2,
            feature_names=["a", "b"],
            feature_types={"a": "binary", "b": "numeric"},
            missing_values={"a": 0, "b": 1},
            numeric_stats={"b": {"mean": 1.0, "std": 0.5, "min": 0, "max": 2, "median": 1}},
            categorical_stats={},
        )
        assert p.n_samples == 10
        assert p.treatment_candidates == []
        assert p.outcome_candidates == []
        assert p.has_time_dimension is False
        assert p.time_column is None

    def test_full_construction(self):
        p = DataProfile(
            n_samples=100,
            n_features=3,
            feature_names=["t", "y", "x"],
            feature_types={"t": "binary", "y": "numeric", "x": "numeric"},
            missing_values={"t": 0, "y": 0, "x": 0},
            numeric_stats={},
            categorical_stats={},
            treatment_candidates=["t"],
            outcome_candidates=["y"],
            potential_confounders=["x"],
            has_time_dimension=True,
            time_column="t",
        )
        assert p.treatment_candidates == ["t"]
        assert p.has_time_dimension is True

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            DataProfile(n_samples=10)


class TestTreatmentEncoding:
    def test_label_encode(self):
        enc = TreatmentEncoding(
            original_type="binary",
            strategy="label_encode",
            value_mapping={"control": 0, "treated": 1},
        )
        assert enc.strategy == "label_encode"
        assert enc.control_value is None
        assert enc.value_mapping == {"control": 0, "treated": 1}

    def test_collapse_to_binary(self):
        enc = TreatmentEncoding(
            original_type="multi_categorical",
            strategy="collapse_to_binary",
            control_value="No E-Mail",
            value_mapping={"No E-Mail": 0, "Mens E-Mail": 1, "Womens E-Mail": 1},
        )
        assert enc.control_value == "No E-Mail"
        assert sum(enc.value_mapping.values()) == 2

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            TreatmentEncoding(strategy="label_encode")
