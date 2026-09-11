"""Read-only approval disclosure of user input and the selected registered analysis policy.

Registry rows are data here. No estimation-stage module or sample calculation belongs in
design review; parity with the downstream plan compiler is checked at the test boundary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _rct_explanation(policy: Mapping[str, Any]) -> str:
    parameters = policy["estimator_parameters"]
    roles = {row["role"]: row["columns"] for row in policy["role_bindings"]}
    stratum = roles.get("stratum", []) if parameters["stratum_handling"] == "fixed_effects" else []
    precision = roles.get("precision_covariate", [])
    adjusted = (precision and precision != stratum
                and parameters["specification"] != "difference_in_means")
    parts = [("Unweighted OLS on one outcome-observed row per approved unit for each approved "
              "treatment contrast; no population-size or voter-count weights.")]
    if stratum:
        parts.append(f"Absorbed stratum fixed effects: {', '.join(stratum)}.")
        if not adjusted:
            parts.append("Without a separate precision covariate, the coefficient is the "
                         "normalized weighted average of within-stratum mean differences, "
                         "with weights n1_s*n0_s/n_s among contributing rows; single-arm strata "
                         "have zero contrast weight. Unequal stratum sizes or treatment "
                         "fractions can make this differ from an equal-unit average effect "
                         "when effects vary across strata.")
    if adjusted:
        parts.append(f"Additional approved precision adjustment: {', '.join(precision)}; "
                     "the coefficient also residualizes this covariate.")
    profile = parameters["cluster_covariance"]
    cluster = roles.get("cluster", [])
    covariance = ("CRV3" if profile == "cluster_robust_cr3" else "CRV1") if cluster else (
        "HC3" if profile == "cluster_robust_cr3" else "HC2")
    if cluster:
        parts.append(f"One-way cluster-robust {covariance} covariance on {', '.join(cluster)}; "
                     "Student-t reference with G-1 degrees of freedom, where G is the number "
                     "of contributing clusters.")
        if covariance == "CRV1":
            parts.append("The CRV1 default applies the implementation's cluster and regression "
                         "degrees-of-freedom corrections.")
    else:
        parts.append(f"Heteroskedasticity-robust {covariance} covariance; Student-t reference "
                     "with N-K degrees of freedom, using the fitted model's effective K.")
    if profile == "cluster_robust_cr2":
        parts.append("The legacy cluster_robust_cr2 name maps to CRV1 (or HC2 without a "
                     "cluster role); it does not implement CR2.")
    parts.append("Pre-repair arm-based power/precision diagnostics are descriptive and do not "
                 "represent the final fixed-effects regression's uncertainty.")
    if not precision:
        parts.append("No approved precision covariate is bound: baseline balance is unavailable, "
                     "not a passed balance check.")
    parts.append(f"{100 * policy['confidence_level']:g}% intervals are approximate regression "
                 "inference, not exact randomization inference. Sample-dependent degrees of "
                 "freedom and intervals are determined only after preparation and fitting. "
                 "Target interpretation remains the approved experimental population and "
                 "unit, without an unsupported individual-level or population-wide effect.")
    return " ".join(parts)


def _selected_profile(rows: list[dict[str, Any]], profile_id: object) -> dict[str, Any] | None:
    if not rows:
        return None
    matches = [row for row in rows if row["profile_id"] == profile_id]
    if len(matches) != 1:
        raise ValueError("approval disclosure requires an exact registered analysis profile")
    return matches[0]


def approval_disclosures(*, design: Mapping[str, Any], design_ref: ArtifactRef,
                         question: Mapping[str, Any], question_ref: ArtifactRef,
                         registry_path: Path) -> dict[str, str]:
    """Bind the preview to exact input/design hashes and the full selected registry file."""
    if content_hash(design) != design_ref.content_hash or content_hash(question) != question_ref.content_hash:
        raise ValueError("approval disclosure artifact hash mismatch")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    matches = [row for row in registry["packs"] if (
        row["method_id"], row["pack_version"]) == (
            design["method_id"], design["method_pack_version"])]
    if len(matches) != 1:
        raise ValueError("approval disclosure requires one exact registered estimation pack")
    pack = matches[0]
    structure = (design.get("preparation") or {}).get("method_structure") or {}
    parameters = dict(pack["parameter_defaults"]) | dict(structure) | {"estimand": design["estimand"]}
    nuisance = _selected_profile(pack["nuisance_profiles"], parameters.get("nuisance_profile_id"))
    adoption = _selected_profile(pack["estimator_profiles"], parameters.get("adoption_profile_id"))
    policy = {
        "scope": "planned policy only; no sample-dependent estimates or uncertainty results",
        "compiled_design": design_ref.model_dump(mode="json"),
        "registry_file": registry_path.name, "registry_version": registry["registry_version"],
        "registry_content_hash": content_hash(registry),
        "method_id": pack["method_id"], "method_pack_version": pack["pack_version"],
        "estimator_id": pack["estimator_id"], "estimator_version": pack["estimator_version"],
        "parameter_defaults": pack["parameter_defaults"], "approved_method_structure": structure,
        "estimator_parameters": parameters, "confidence_level": pack["confidence_level"],
        "uncertainty_method": pack["uncertainty_method"],
        "finite_sample_correction": pack["finite_sample_correction"],
        "fold_count_default": pack["fold_count_default"],
        "selected_nuisance_profile": nuisance, "selected_adoption_profile": adoption,
        "profile_estimator_parameters": (dict(adoption["parameter_defaults"]) | parameters
                                         if adoption is not None else parameters),
        "multiplicity_policy_id": (pack["parameter_defaults"].get("multiplicity_policy_id")
                                   if len(design["primary_contrasts"]) > 1 else None),
        "role_bindings": design["role_bindings"],
    }
    summary = {
        "original_user_input": _json({
            "provenance": "Original user-submitted QuestionRecord; user input, not inferred evidence",
            "artifact": question_ref.model_dump(mode="json"),
            "question_text": question["question_text"], "context_text": question.get("context_text")}),
        "registered_analysis_policy": _json(policy),
    }
    if design["method_id"] == "randomized_experiment":
        summary["analysis_policy_explanation"] = _rct_explanation(policy)
    return summary
