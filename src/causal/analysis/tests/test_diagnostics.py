"""Diagnostic execution policy and the evidence-constrained judgment ceiling."""
from __future__ import annotations

import pytest

from causal.analysis.common import legacy_diagnostics as diagnostic_policy
from causal.analysis.integration import contracts as ec
from causal.analysis.tests.support.builders import (
    PACK,
    base,
)

THRESHOLDS: ec.ValueMap = {"max_overall_attrition": 0.2, "minimum_clusters": 8}


@pytest.mark.parametrize(("severity", "values", "expected"), [
    ("descriptive", {"overall_attrition": 0.9}, "descriptive"),
    ("required_blocking", {"overall_attrition": 0.1, "clusters": 9}, "acceptable"),
    ("required_blocking", {"overall_attrition": 0.9}, "warning"),
    ("qualification_guard", {"clusters": 2}, "warning"),
    ("invalidation_guard", {"clusters": 2}, "invalidating"),
    ("invalidation_guard", {"clusters": 9}, "acceptable")])
def test_the_severity_applier_covers_every_registered_severity(
    severity: str, values: ec.ValueMap, expected: str
) -> None:
    # §14.2: severity is fixed before execution and no value may reinterpret it.
    assert diagnostic_policy.severity_result(severity, values, THRESHOLDS) == expected


def test_every_required_diagnostic_reaches_a_visible_terminal_result() -> None:
    # A harvested row computes, an empty harvest fails, an absent row is not computable —
    # and every registered row appears exactly once, in the pack's order.
    harvest = {"baseline_balance": {"standardized_difference": 0.02},
               "model_convergence_integrity": {}}
    results = diagnostic_policy.run_diagnostics(PACK, harvest, base())
    assert tuple(row.diagnostic_id for row in results) == tuple(
        row.diagnostic_id for row in PACK.required_diagnostics)
    reported = {row.diagnostic_id: row for row in results}
    assert reported["baseline_balance"].execution_status == "computed"
    assert reported["model_convergence_integrity"].execution_status == "failed"
    assert reported["covariance_cluster_adequacy"].execution_status == "not_computable"
    assert reported["covariance_cluster_adequacy"].warnings == (diagnostic_policy.DIAGNOSTIC_NOT_COMPUTED,)

