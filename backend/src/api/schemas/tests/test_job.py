"""Validation tests for the create-job request schema.

The causal question is mandatory at submit (the analyst states what they want to
know; treatment and outcome are derived later, not named here). These tests pin
that contract: a request without a non-blank causal question is rejected at the
API boundary, and the question is free text (a real sentence), not a column name.
"""
import pytest
from pydantic import ValidationError

from src.api.schemas.job import CreateJobRequest

VALID_URL = "https://www.kaggle.com/datasets/owner/dataset-name"


def _payload(**overrides):
    base = {
        "kaggle_url": VALID_URL,
        "causal_question": "Does job training increase income?",
    }
    base.update(overrides)
    return base


def test_valid_request_parses():
    req = CreateJobRequest(**_payload())
    assert req.causal_question == "Does job training increase income?"


def test_causal_question_is_required():
    payload = _payload()
    del payload["causal_question"]
    with pytest.raises(ValidationError):
        CreateJobRequest(**payload)


def test_empty_causal_question_rejected():
    with pytest.raises(ValidationError):
        CreateJobRequest(**_payload(causal_question=""))


def test_whitespace_only_causal_question_rejected():
    with pytest.raises(ValidationError):
        CreateJobRequest(**_payload(causal_question="   "))


def test_surrounding_whitespace_is_stripped():
    req = CreateJobRequest(**_payload(causal_question="  Does X cause Y?  "))
    assert req.causal_question == "Does X cause Y?"


def test_free_text_question_with_spaces_and_punctuation_is_allowed():
    # Unlike the old treatment/outcome column names, the question is prose: a
    # sentence with spaces and punctuation must not be rejected.
    q = "Did the 2020 policy reduce unemployment, and for whom?"
    req = CreateJobRequest(**_payload(causal_question=q))
    assert req.causal_question == q


def test_treatment_and_outcome_are_not_part_of_the_contract():
    # They are no longer submit inputs. Passing them is ignored (extra keys), and
    # a request carrying only them (no question) is rejected for the missing
    # required question.
    with pytest.raises(ValidationError):
        CreateJobRequest(
            kaggle_url=VALID_URL, treatment_variable="t", outcome_variable="y"
        )


def test_user_context_still_optional():
    req = CreateJobRequest(**_payload())
    assert req.user_context is None
