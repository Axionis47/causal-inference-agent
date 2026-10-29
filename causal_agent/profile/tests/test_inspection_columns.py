"""Phase 3: original field text and supported interpretations."""

import pytest

from causal_agent.profile import inspection as P


def test_original_tokens_and_sentinels():
    col = P.column(["NA", "NA", "unknown", "", "  "], " code ", 0)
    assert col.empty_count == 1 and col.whitespace_only_count == 1 and col.distinct_count == 3
    assert col.top_values[0].value == "NA" and col.top_values[0].count == 2
    sentinel = next(i for i in col.issues if i.code == "suspected_sentinel")
    assert sentinel.affected_count == 4
    assert {i.code for i in col.issues} >= {"whitespace", "header_whitespace"}


@pytest.mark.parametrize(
    "values,dtype",
    [
        (["00123", "00456"], "text"),
        (["1", "2"], "integer"),
        (["0", "1"], "integer"),
        (["1.5", "2e0"], "decimal"),
        (["yes", "NO"], "boolean"),
        (["2025-01-02"], "date"),
        (["2025-01-02T12:30:00Z"], "datetime"),
        (["a", "b"], "text"),
        (["", ""], "unknown"),
        (["01/02/2025"], "text"),
    ],
)
def test_supported_types(values, dtype):
    original = list(values)
    col = P.column(values, "x", 0)
    assert col.suggested_dtype == dtype and values == original
    if dtype in {"text", "unknown", "date", "datetime", "boolean"}:
        assert col.numeric is None


def test_numeric_values_and_empty_fields():
    col = P.column(["1", "2", "3", ""], "x", 0)
    assert col.numeric.model_dump() == {"min": 1, "max": 3, "mean": 2, "p01": 1.02, "p25": 1.5, "p50": 2, "p75": 2.5, "p99": 2.98}
    assert col.numeric_parse_count == 3 and col.numeric_exception_count == 0


def test_numeric_failures_are_not_discarded():
    col = P.column([str(i) for i in range(99)] + ["oops"], "amount", 0)
    assert col.numeric is None, "Invalid non-empty values must prevent a numeric summary"
    assert col.numeric_parse_count == 99 and col.numeric_exception_count == 1
    assert col.status == "partial" and col.suggested_dtype == "mixed"
    assert next(i for i in col.issues if i.code == "numeric_exceptions").examples == ["oops"]


@pytest.mark.parametrize("token", ["NaN", "inf", "-inf", "1e999", "$12", "12%", "1,234", " 12", "9007199254740993"])
def test_unsafe_numeric_summaries_unavailable(token):
    col = P.column([token], "x", 0)
    assert col.numeric is None and col.status == "partial"


def test_bounded_examples_and_top_values():
    col = P.column([f"{i} " + "x" * 200 for i in range(20)], "x", 0)
    assert len(col.top_values) == 8 and all(len(v.value) <= 120 and v.truncated for v in col.top_values)
    assert all(len(i.examples) <= 5 and all(len(e) <= 120 for e in i.examples) for i in col.issues)
