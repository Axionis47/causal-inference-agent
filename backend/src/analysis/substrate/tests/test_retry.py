"""RetryPolicy + should_retry: declarative retry decisions with veto."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis.substrate.retry import (
    Criticality,
    RetryPolicy,
    should_retry,
)


def test_retry_returns_true_when_exception_class_in_retry_on():
    policy = RetryPolicy(max_attempts=3, retry_on=("ValueError",))
    assert should_retry(ValueError("boom"), policy, attempts_so_far=1) is True


def test_retry_returns_false_when_exception_not_in_either_list():
    policy = RetryPolicy(max_attempts=3, retry_on=("ValueError",))
    assert should_retry(KeyError("boom"), policy, attempts_so_far=0) is False


def test_no_retry_on_vetoes_even_when_also_in_retry_on():
    policy = RetryPolicy(
        max_attempts=3,
        retry_on=("ValueError",),
        no_retry_on=("ValueError",),
    )
    assert should_retry(ValueError("boom"), policy, attempts_so_far=0) is False


def test_subclass_of_retry_on_entry_matches_via_mro():
    class MyValueError(ValueError):
        pass

    policy = RetryPolicy(max_attempts=3, retry_on=("ValueError",))
    assert should_retry(MyValueError("boom"), policy, attempts_so_far=0) is True


def test_parent_class_in_no_retry_on_vetoes_subclass():
    class TransientError(Exception):
        pass

    class FatalSubError(TransientError):
        pass

    policy = RetryPolicy(
        max_attempts=3,
        retry_on=("TransientError",),
        no_retry_on=("FatalSubError",),
    )
    assert should_retry(FatalSubError("boom"), policy, attempts_so_far=0) is False


def test_attempts_so_far_at_max_returns_false_even_for_retryable():
    policy = RetryPolicy(max_attempts=3, retry_on=("ValueError",))
    assert should_retry(ValueError("boom"), policy, attempts_so_far=3) is False


def test_attempts_so_far_past_max_returns_false():
    policy = RetryPolicy(max_attempts=2, retry_on=("ValueError",))
    assert should_retry(ValueError("boom"), policy, attempts_so_far=5) is False


def test_attempts_so_far_below_max_with_no_match_returns_false():
    # Whitelist semantics: silence is not consent.
    policy = RetryPolicy(max_attempts=3, retry_on=("TimeoutError",))
    assert should_retry(RuntimeError("boom"), policy, attempts_so_far=0) is False


def test_substring_does_not_match_class_name():
    # retry_on uses exact class names, not substrings.
    policy = RetryPolicy(max_attempts=3, retry_on=("Value",))
    assert should_retry(ValueError("boom"), policy, attempts_so_far=0) is False


def test_policy_is_frozen():
    policy = RetryPolicy(max_attempts=3, retry_on=("ValueError",))
    with pytest.raises(ValidationError):
        policy.max_attempts = 5  # type: ignore[misc]


def test_policy_rejects_max_attempts_below_one():
    with pytest.raises(ValidationError):
        RetryPolicy(max_attempts=0)


def test_policy_rejects_max_attempts_above_ten():
    with pytest.raises(ValidationError):
        RetryPolicy(max_attempts=11)


def test_policy_rejects_extra_fields():
    with pytest.raises(ValidationError):
        RetryPolicy(max_attempts=3, jitter=True)  # type: ignore[call-arg]


def test_criticality_literal_values():
    # Documents the intended enum domain. Used by the orchestrator in S6
    # to decide between job-abort vs slot-null-and-continue.
    valid: list[Criticality] = ["critical", "advisory"]
    assert valid == ["critical", "advisory"]
