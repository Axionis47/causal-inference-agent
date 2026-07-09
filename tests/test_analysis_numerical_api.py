"""Probability API regressions are checked without executing generated code."""

import pytest

from backend.analysis.numerical_api import confidence_arguments


@pytest.fixture(autouse=True)
def isolated():
    yield


PLAN = {"effects": [{"inference": "Two-sided 95% Welch t confidence interval"}]}


@pytest.mark.parametrize("code", [
    "from scipy import stats\ncoverage=0.95\nalpha=1-coverage\nlo,hi=stats.t.interval(alpha, 42, loc=estimate, scale=se)",
    "from scipy.stats import t as student\nif ready:\n coverage=.95\n alpha=1-coverage\n lo,hi=student.interval(confidence=alpha, df=42)",
    "import scipy.stats as s\ns.norm.interval(.05)",
])
def test_confidence_coverage_is_not_error_probability(code):
    with pytest.raises(ValueError, match="confidence coverage"):
        confidence_arguments(code, PLAN)


@pytest.mark.parametrize("code", [
    "from scipy import stats\ncoverage=.95\nalpha=1-coverage\nlo,hi=stats.t.interval(1-alpha, 42, loc=estimate, scale=se)",
    "from scipy.stats import t as student\nstudent.interval(confidence=.95, df=42)",
    "import scipy\nscipy.stats.norm.interval(95/100)",
])
def test_explicit_correct_coverage_is_accepted(code):
    result = confidence_arguments(code, PLAN)
    assert result["sites"][0]["coverage"] == pytest.approx(.95)
    assert not result["target_data_access"]


@pytest.mark.parametrize("code", [
    "from scipy import stats\ncoverage=read_configuration()\nstats.t.interval(coverage, 42)",
    "from scipy import stats\ncoverage=.95\ncoverage=.05\nstats.t.interval(coverage, 42)",
    "from scipy import stats\nstats=custom_library\nstats.t.interval(.05, 42)",
])
def test_unknown_or_rebound_expressions_are_not_claimed_as_numerically_verified(code):
    result = confidence_arguments(code, PLAN)
    assert all(site["coverage"] is None for site in result["sites"])


def test_an_explicit_low_coverage_request_is_not_silently_replaced():
    result = confidence_arguments("from scipy import stats\nstats.t.interval(.05, 42)", {"effects": [{"inference": "Two-sided 5% interval"}]})
    assert result["sites"][0]["expected"] == .05


def test_conditional_placeholder_cannot_masquerade_as_a_computed_estimate():
    code = "if n < 2:\n facts['effect_estimate'] = 0.0\nelse:\n facts['effect_estimate'] = difference\n"
    with pytest.raises(ValueError, match="Do not fabricate"):
        confidence_arguments(code, PLAN)


@pytest.mark.parametrize("code", [
    "if n < 2:\n raise ValueError('Undefined')\nfacts['effect_estimate'] = difference\n",
    "facts['structural_zero_estimate'] = 0.0\nfacts['effect_estimate'] = difference\n",
    "facts['effect_estimate'] = 0.0\nfacts['effect_estimate'] = difference\n",
])
def test_valid_errors_structural_constants_and_overwritten_initializers_remain_reviewable(code):
    confidence_arguments(code, PLAN)
