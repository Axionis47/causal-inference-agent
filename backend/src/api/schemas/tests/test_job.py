"""Validation tests for the create-job request schema.

Treatment and outcome are mandatory at submit (the analyst names the columns so
the pipeline never has to guess them). These tests pin that contract: a request
without a real, non-blank treatment/outcome is rejected at the API boundary.
"""
import pytest
from pydantic import ValidationError

from src.api.schemas.job import CreateJobRequest

VALID_URL = "https://www.kaggle.com/datasets/owner/dataset-name"


def _payload(**overrides):
    base = {
        "kaggle_url": VALID_URL,
        "treatment_variable": "treat",
        "outcome_variable": "re78",
    }
    base.update(overrides)
    return base


def test_valid_request_parses():
    req = CreateJobRequest(**_payload())
    assert req.treatment_variable == "treat"
    assert req.outcome_variable == "re78"


def test_treatment_variable_is_required():
    payload = _payload()
    del payload["treatment_variable"]
    with pytest.raises(ValidationError):
        CreateJobRequest(**payload)


def test_outcome_variable_is_required():
    payload = _payload()
    del payload["outcome_variable"]
    with pytest.raises(ValidationError):
        CreateJobRequest(**payload)


def test_empty_treatment_variable_rejected():
    with pytest.raises(ValidationError):
        CreateJobRequest(**_payload(treatment_variable=""))


def test_whitespace_only_outcome_variable_rejected():
    with pytest.raises(ValidationError):
        CreateJobRequest(**_payload(outcome_variable="   "))


def test_surrounding_whitespace_is_stripped():
    req = CreateJobRequest(**_payload(treatment_variable="  treat  "))
    assert req.treatment_variable == "treat"


def test_invalid_character_set_rejected():
    with pytest.raises(ValidationError):
        CreateJobRequest(**_payload(outcome_variable="re 78!"))


def test_user_context_still_optional():
    req = CreateJobRequest(**_payload())
    assert req.user_context is None
