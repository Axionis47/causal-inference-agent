"""Validation tests for the ConfounderAnalysis and ConfounderRecord output models."""

import pytest
from pydantic import ValidationError

from src.analysis.agents.confounder_discovery import ConfounderAnalysis, ConfounderRecord


class TestConfounderRecord:
    def test_full_construction(self):
        r = ConfounderRecord(
            variable="age",
            corr_with_treatment=0.3,
            corr_with_outcome=0.55,
            balance_p_value=0.02,
            strength=0.55,
            reasons=["classic", "prognostic"],
        )
        assert r.variable == "age"
        assert r.source == "statistical_fallback"

    def test_optional_balance_p(self):
        r = ConfounderRecord(
            variable="x", corr_with_treatment=0.1, corr_with_outcome=0.2, strength=0.2
        )
        assert r.balance_p_value is None
        assert r.reasons == []

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            ConfounderRecord(variable="x", corr_with_outcome=0.2, strength=0.2)


class TestConfounderAnalysis:
    def test_defaults(self):
        a = ConfounderAnalysis()
        assert a.ranked_confounders == []
        assert a.excluded_variables == []
        assert a.adjustment_strategy == ""
        assert a.investigation_log == []
        assert a.statistical_fallback_details is None

    def test_round_trip(self):
        """model_dump / model_validate must round-trip cleanly for persistence."""
        a1 = ConfounderAnalysis(
            ranked_confounders=["age", "education"],
            excluded_variables=["mediator"],
            adjustment_strategy="three-criterion scan",
            statistical_fallback_details=[
                ConfounderRecord(
                    variable="age",
                    corr_with_treatment=0.3,
                    corr_with_outcome=0.55,
                    strength=0.55,
                    reasons=["classic"],
                )
            ],
        )
        a2 = ConfounderAnalysis.model_validate(a1.model_dump())
        assert a2 == a1
