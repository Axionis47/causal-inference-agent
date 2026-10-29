"""Bounded descriptions of original CSV text, without publishing conversions.

The old analysis profile contract remains separate. These reports do not seed
memory or establish row meaning. All library conversions are temporary probes.
"""

import csv
import hashlib
import io
import json
import logging
import math
import re
from collections import Counter
from datetime import date, datetime
from itertools import combinations
from typing import BinaryIO

import numpy as np

from causal_agent.common.addresses import key
from causal_agent.profile.profiler import PLACEHOLDER_NUMBERS, PLACEHOLDER_STRINGS
from causal_agent.profile.report import Check, ColumnResult, ComputedProfile, DatasetResult, Frequency, Issue, NumericSummary, Status

VERSION = "1.0.0"
POLICY = {"top": 8, "examples": 5, "example_chars": 120, "pair_columns": 12, "interpretation": "strict-text-v1"}
FINGERPRINT = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()
log = logging.getLogger(__name__)
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
LEADING_ZERO = re.compile(r"[+-]?0[0-9]+\Z")
BOOL = {"true", "false", "yes", "no", "y", "n", "t", "f"}


def issue(check: str, code: str, reason: str, values: list[str] | None, position: int | None = None) -> Issue:
    samples = list(dict.fromkeys(values or []))
    return Issue(
        id=f"{'dataset' if position is None else f'column:{position}'}:{check}:{code}",
        check=check,
        code=code,
        reason=reason,
        column_positions=[] if position is None else [position],
        affected_count=len(values) if values is not None else None,
        examples=[v[:120] for v in samples[:5]],
        examples_truncated=len(samples) > 5 or any(len(v) > 120 for v in samples[:5]),
    )


def status(checks: list[Check]) -> Status:
    if checks and all(c.status == "failed" for c in checks):
        return "failed"
    return "partial" if any(c.status != "computed" for c in checks) else "complete"


def numeric_summary(values: list[float]) -> NumericSummary:
    data = np.asarray(values, dtype=float)
    with np.errstate(over="raise", invalid="raise"):
        q = np.quantile(data, [0.01, 0.25, 0.5, 0.75, 0.99])
        mean = float(np.mean(data))
    numbers = [float(data.min()), float(data.max()), mean, *map(float, q)]
    if not all(math.isfinite(n) for n in numbers):
        raise ValueError("Non-finite numerical summary")
    return NumericSummary(**dict(zip(["min", "max", "mean", "p01", "p25", "p50", "p75", "p99"], numbers)))


def iso_kind(value: str) -> str | None:
    try:
        if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
            date.fromisoformat(value)
            return "date"
        if re.match(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}", value):
            datetime.fromisoformat(value)
            return "datetime"
    except ValueError:
        pass
    return None


def column(values: list[str], name: str, position: int) -> ColumnResult:
    nonempty = [v for v in values if v != ""]
    counts = Counter(nonempty)
    result = ColumnResult(
        position=position,
        name=name,
        key=key(name),
        status="complete",
        empty_count=len(values) - len(nonempty),
        whitespace_only_count=sum(v != "" and v.isspace() for v in values),
        distinct_count=len(counts),
        top_values=[Frequency(value=v[:120], count=n, truncated=len(v) > 120) for v, n in counts.most_common(8)],
        checks=[Check(name="contents", status="computed")],
    )
    spaces = [v for v in nonempty if v != v.strip()]
    if spaces:
        result.issues.append(issue("format", "whitespace", "Values contain leading or trailing whitespace; retained unchanged.", spaces, position))
    if name != name.strip():
        result.issues.append(issue("format", "header_whitespace", "Header whitespace retained unchanged.", [name], position))
    # Aggregate candidate counts, not an unbounded list of distinct values.
    sentinels = [v for v in nonempty if v.strip().lower() in PLACEHOLDER_STRINGS or (NUMBER.fullmatch(v) and float(v) in PLACEHOLDER_NUMBERS)]
    result.suspected_sentinels = [Frequency(value=v[:120], count=n, truncated=len(v) > 120) for v, n in Counter(sentinels).most_common(8)]
    result.sentinel_values_truncated = len(set(sentinels)) > 8
    if sentinels:
        result.issues.append(issue("sentinels", "suspected_sentinel", "Possible placeholder values, not confirmed missing values.", sentinels, position))
    result.checks.extend([Check(name="format", status="computed"), Check(name="sentinels", status="computed")])
    parsed = [float(v) if NUMBER.fullmatch(v) else None for v in nonempty]
    valid = [v for v in parsed if v is not None and math.isfinite(v)]
    result.numeric_parse_count = len(valid)
    result.numeric_exception_count = len(nonempty) - len(valid)
    leading = [v for v in nonempty if LEADING_ZERO.fullmatch(v)]
    dates = [iso_kind(v) for v in nonempty]
    reason = None
    if not nonempty:
        reason = "No non-empty values."
    elif leading:
        result.suggested_dtype = "text"
        reason = "Leading-zero integer strings may be identifiers."
        result.issues.append(issue("dtype", "leading_zero", reason, leading, position))
    elif len(valid) == len(nonempty):
        result.suggested_dtype = "integer" if all(re.fullmatch(r"[+-]?[0-9]+", v) for v in nonempty) else "decimal"
        # Float summaries cannot faithfully represent large integral values.
        if result.suggested_dtype == "integer" and any(abs(int(v)) > 2**53 for v in nonempty):
            reason = "Integer values exceed the exact range supported by numerical summaries."
        else:
            try:
                result.numeric = numeric_summary(valid)
            except Exception:
                log.exception("Numeric summary failed")
                result.checks.append(Check(name="numeric", status="failed", reason="Numerical calculation failed."))
                result.issues.append(issue("numeric", "calculation_failed", "Numerical calculation failed; no summary published.", None, position))
    elif all(v.lower() in BOOL for v in nonempty):
        result.suggested_dtype = "boolean"
    elif all(v == "date" for v in dates):
        result.suggested_dtype = "date"
    elif all(v == "datetime" for v in dates):
        result.suggested_dtype = "datetime"
    elif valid or any(dates):
        result.suggested_dtype = "mixed"
        reason = "Values do not share one supported interpretation."
    else:
        result.suggested_dtype = "text"
        if any(re.search(r"[0-9]", v) or v.lower() in {"nan", "inf", "infinity", "+inf", "-inf"} for v in nonempty):
            reason = "Numeric/date-looking values have no unambiguous supported interpretation."
    if result.numeric_exception_count and (valid or reason):
        failures = [v for v, n in zip(nonempty, parsed) if n is None or not math.isfinite(n)]
        result.issues.append(
            issue("dtype", "numeric_exceptions", "Non-empty values not interpreted as finite numbers; none were discarded.", failures, position)
        )
    result.checks.append(Check(name="dtype", status="unavailable" if reason else "computed", reason=reason))
    if not any(c.name == "numeric" for c in result.checks) and (result.numeric is not None or reason):
        result.checks.append(Check(name="numeric", status="unavailable" if reason else "computed", reason=reason))
    result.status = status(result.checks)
    return result


def candidate_keys(rows: list[list[str]], width: int) -> list[list[int]]:
    columns = list(zip(*rows))
    eligible = [i for i in range(width) if "" not in columns[i]]
    cardinality = {i: len(set(columns[i])) for i in eligible}
    singles = [i for i in eligible if cardinality[i] == len(rows)]
    candidates = sorted((i for i in eligible if i not in singles and cardinality[i] > 1), key=lambda i: (-cardinality[i], i))[:12]
    return [[i] for i in singles] + [sorted([a, b]) for a, b in combinations(candidates, 2) if len(set(zip(columns[a], columns[b]))) == len(rows)]


def inspect(stream: BinaryIO) -> ComputedProfile:
    """Input was bounded and structurally validated by the server. No writes."""
    stream.seek(0)
    wrapper = io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")
    try:
        reader = csv.reader(wrapper, strict=True)
        headers = next(reader)
        rows = list(reader)
    finally:
        wrapper.detach()
    if not rows or any(len(row) != len(headers) for row in rows):
        raise ValueError("Unexpected CSV structure")
    dataset = DatasetResult(
        rows=len(rows), columns=len(headers), duplicate_rows=len(rows) - len({tuple(row) for row in rows}), checks=[Check(name="structure", status="computed")]
    )
    try:
        dataset.candidate_keys = candidate_keys(rows, len(headers))
        dataset.checks.append(Check(name="candidate_keys", status="computed"))
    except Exception:
        log.exception("Candidate key check failed")
        dataset.checks.append(Check(name="candidate_keys", status="failed", reason="Candidate key calculation failed."))
        dataset.issues.append(issue("candidate_keys", "calculation_failed", "Candidate key calculation failed.", None))
    results = []
    for position, name in enumerate(headers):
        try:
            results.append(column([row[position] for row in rows], name, position))
        except Exception:
            log.exception("Column inspection failed at position %s", position)
            results.append(
                ColumnResult(
                    position=position,
                    name=name,
                    key=key(name),
                    status="failed",
                    checks=[Check(name="column", status="failed", reason="Column inspection failed.")],
                    issues=[issue("column", "calculation_failed", "Column inspection failed.", None, position)],
                )
            )
    return ComputedProfile(status=status(dataset.checks + [check for col in results for check in col.checks]), dataset=dataset, columns=results)
