"""Validation tests for the DomainKnowledge output model."""

import pytest
from pydantic import ValidationError

from src.analysis.agents.domain_knowledge import DomainKnowledge, Hypothesis, Uncertainty


class TestHypothesis:
    def test_minimum_fields(self):
        h = Hypothesis(claim="x is treatment")
        assert h.claim == "x is treatment"
        assert h.confidence == "medium"
        assert h.evidence == ""
        assert h.revised is False
        assert h.superseded_by is None

    def test_full_fields(self):
        h = Hypothesis(
            claim="y is outcome",
            confidence="high",
            evidence="explicit in description",
            revised=True,
            superseded_by="z is outcome",
        )
        assert h.revised is True
        assert h.superseded_by == "z is outcome"

    def test_extra_field_rejected(self):
        """extra='forbid' must reject unknown fields so the LLM cannot drift the schema."""
        with pytest.raises(ValidationError):
            Hypothesis(claim="x", strength="strong")

    def test_frozen_after_construction(self):
        h = Hypothesis(claim="x")
        with pytest.raises(ValidationError):
            h.claim = "y"


class TestUncertainty:
    def test_required_fields(self):
        u = Uncertainty(issue="control source unclear", impact="confounding risk")
        assert u.issue == "control source unclear"
        assert u.impact == "confounding risk"

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            Uncertainty(issue="x")

    def test_frozen_after_construction(self):
        u = Uncertainty(issue="x", impact="y")
        with pytest.raises(ValidationError):
            u.issue = "z"


class TestDomainKnowledge:
    def test_empty_defaults(self):
        """All collections default to empty; investigation_complete defaults False."""
        dk = DomainKnowledge()
        assert dk.hypotheses == []
        assert dk.all_hypotheses == []
        assert dk.uncertainties == []
        assert dk.temporal_understanding is None
        assert dk.immutable_vars == []
        assert dk.investigation_complete is False

    def test_round_trip_through_dict(self):
        """A constructed DomainKnowledge must survive model_dump -> model_validate
        because storage adapters round-trip it as JSON."""
        original = DomainKnowledge(
            hypotheses=[Hypothesis(claim="t is treatment", confidence="high")],
            all_hypotheses=[Hypothesis(claim="t is treatment", confidence="high")],
            uncertainties=[Uncertainty(issue="i", impact="im")],
            temporal_understanding="baseline -> treatment -> outcome",
            immutable_vars=["age", "race"],
            investigation_complete=True,
        )
        restored = DomainKnowledge.model_validate(original.model_dump())
        assert restored == original

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            DomainKnowledge(hypotheses=[], total_count=0)

    def test_frozen_after_construction(self):
        dk = DomainKnowledge()
        with pytest.raises(ValidationError):
            dk.investigation_complete = True
