"""Small frozen numerical plans and evidence builders; no database or upstream stage execution."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import polars as pl

from causal.analysis.integration import RESOURCE_ROOT
from causal.analysis.integration import contracts as ec
from causal.analysis.integration.packs import load_estimation_packs
from causal.shared.contracts import ArtifactRef

REGISTRIES = Path(__file__).resolve().parents[5] / "registries"


PACKS = load_estimation_packs(RESOURCE_ROOT / "method-pack-estimation.v1.json",
                              REGISTRIES / "method-packs.v1.json")


PACK = PACKS.get("randomized_experiment", "randomized-experiment-pack.v1")


CONTRAST = "arm_b_vs_control"


ROLES = {"outcome": "y", "treatment": "arm", "unit_identifier": "uid",
         "running_variable": "score", "group": "arm", "time": "period"}


FRAME = pl.DataFrame({"y": [1.0, None, 3.0, 4.0], "arm": ["a", "b", "a", "b"],
                      "uid": [1, 2, 3, 4], "score": [0.1, 0.9, 1.4, 2.0],
                      "period": [1, 1, 2, 2], "spare": ["x", "x", "x", "x"]})


def digest(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def ref(name: str) -> ArtifactRef:
    return ArtifactRef(artifact_id=name, content_hash=digest(name))


ROW_HASH = digest("rows")


MASK_OBJECT = ec.ObjectRefV1(object_locator=f"objects/{digest('bits')}",
                             content_hash=digest("bits"))


def make_plan(**over: Any) -> ec.EstimationPlanV1:
    # One frozen §6.1 plan; every test varies only the field it is about.
    fields: dict[str, Any] = {
        "method_id": PACK.method_id, "method_pack_version": PACK.pack_version,
        "estimand_id": "att", "population_id": "enrolled", "timeframe_id": "wave_one",
        "comparator_id": "control", "outcome_id": "completion", "unit_id": "participant",
        "role_columns": dict(ROLES),
        "row_set_hash": ROW_HASH, "contrast_ids": (CONTRAST,),
        "required_sensitivity_ids": tuple(r.branch_id for r in PACK.sensitivity_branches),
        "figure_builder_ids": PACK.figure_builder_ids, "capacity_report": ref("capacity"),
        "numerical_tolerances": {"sensitivity_magnitude": 0.25},
        "parents": (ref("manifest"),), "versions": {"schema": "estimation-plan.v1"},
        "context_manifest": ref("manifest"), "plan_revision": 1, "seed": 7,
        "estimator_id": PACK.estimator_id, "estimator_version": PACK.estimator_version,
        "outcome_scale": "difference", "multiplicity_policy_id": None,
        "primary_mask_rule_id": "outcome_observed", "confidence_level": PACK.confidence_level,
        "uncertainty_method": PACK.uncertainty_method,
        "finite_sample_correction": PACK.finite_sample_correction,
        "estimator_parameters": {"threads": 2, "processes": 1}, "nuisance_profile_id": None,
        "fold_count": None, "fold_assignment_rule_id": None, "preprocessing_recipe_ids": (),
        "required_diagnostics": PACK.severities(),
        "numerical_failure_rule_ids": PACK.not_estimable_rule_ids}
    return ec.EstimationPlanV1(**(fields | over))


def make_item(estimate: float = 1.0, lower: float = 0.5, upper: float = 1.5, *,
              contrast: str = CONTRAST, convergence: str = "converged",
              ) -> ec.PrimaryContrastResultV1:
    return ec.PrimaryContrastResultV1(
        contrast_id=contrast, estimand_id="att", estimand_label="ATT", estimate=estimate,
        estimate_units="probability", comparator_id="control",
        effect_direction="higher_is_treated", standard_error=0.25, confidence_level=0.95,
        interval_lower=lower, interval_upper=upper, p_value=0.04,
        uncertainty_method=PACK.uncertainty_method, contributing_counts={"row": 100},
        finite_sample_correction=PACK.finite_sample_correction, contribution_mask=ref("mask"),
        estimator_id=PACK.estimator_id, estimator_version=PACK.estimator_version,
        estimator_parameters={}, adapter_version="rct.v1", convergence=convergence,
        method_quantities={})


def make_result(*items: ec.PrimaryContrastResultV1) -> ec.PrimaryAnalysisResultV1:
    rows = items or (make_item(),)
    return ec.PrimaryAnalysisResultV1(
        parents=(ref("plan"),), versions={"schema": "primary-analysis-result.v1"},
        plan=ref("plan"), method_id=PACK.method_id, estimator_id=PACK.estimator_id,
        outcome_id="completion", estimand_family="att", primary_items=rows,
        contrast_order=tuple(row.contrast_id for row in rows), multiplicity_result=None,
        complete=True)


def base(**over: Any) -> dict[str, Any]:
    return {"parents": (ref("plan"),), "versions": {"schema": "diagnostic-result.v1"},
            "plan": ref("plan"), "primary_result": ref("primary"),
            "denominators": {"row": 100}, "contribution_mask_hash": digest("mask"),
            "numerical_environment": ref("env")} | over


def make_diagnostic(diagnostic_id: str, severity: str, *, status: str = "computed",
                    policy: str = "acceptable") -> ec.DiagnosticResultV1:
    return ec.DiagnosticResultV1(
        **base(), diagnostic_id=diagnostic_id, diagnostic_version="v1", severity=severity,
        threshold_context={}, execution_status=status, policy_result=policy, values={},
        warnings=(), interpreting_rule_id=diagnostic_id, implementation_version="v1")


