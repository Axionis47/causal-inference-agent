"""Approval must expose exact user intent and the analysis policy before anyone consents."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from causal.analysis.integration import RESOURCE_ROOT, plancompile
from causal.analysis.integration.tests.support.plans import REGISTRY, manifest, ref
from causal.design import graph
from causal.design.review_policy import approval_disclosures
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = RESOURCE_ROOT / "method-pack-estimation.v1.json"


def captured() -> dict[str, Any]:
    return json.loads((Path(__file__).parent / "fixtures/rct_approval_policy.json").read_text())


def disclosures(design: dict[str, Any], *, path: Path = REGISTRY_PATH) -> dict[str, str]:
    case = captured()
    return approval_disclosures(design=design, design_ref=ArtifactRef(
        artifact_id="compiled:preview", content_hash=content_hash(design)),
        question=case["question"], question_ref=ArtifactRef.model_validate(case["question_ref"]),
        registry_path=path)


def test_exact_saved_approval_interrupt_includes_original_input_and_bound_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = captured()
    previous = case["packet"]["payload"]
    assert "original_user_input" not in previous["review_summary"]
    assert "registered_analysis_policy" not in previous["review_summary"]
    harness = object.__new__(graph._Harness)
    harness._cache = {row["envelope"]["artifact_id"]: row["payload"]
                      for row in case["artifacts"].values()}
    harness.deps = SimpleNamespace(prompts_root=ROOT, estimation_registry_path=REGISTRY_PATH)
    monkeypatch.setattr(harness, "_emit", lambda *args, **kwargs: None)
    state = {"design_revision": case["packet"]["interrupt"]["expected_revision"],
             "artifacts": {kind: row["envelope"]["artifact_id"]
                           for kind, row in case["artifacts"].items()},
             "hashes": {row["envelope"]["artifact_id"]: row["envelope"]["content_hash"]
                        for row in case["artifacts"].values()}}
    opened: dict[str, Any] = {}

    class Paused(Exception):
        pass

    def interrupt(payload: dict[str, Any]) -> None:
        opened.update(payload)
        raise Paused

    monkeypatch.setattr(graph, "interrupt", interrupt)
    with pytest.raises(Paused):
        harness.approval(state)
    for key in ("design", "compiled_design", "diagnostic_report", "capacity_report"):
        assert opened[key] == previous[key]
    assert opened["interrupt_hash"] == case["packet"]["interrupt"]["expected_interrupt_hash"]
    summary = opened["review_summary"]
    assert all(isinstance(value, str) for value in summary.values())
    user_input = json.loads(summary["original_user_input"])
    assert user_input["artifact"] == case["question_ref"]
    assert user_input["question_text"] == case["question"]["question_text"]
    assert user_input["context_text"] == case["question"]["context_text"]
    assert "user input, not inferred evidence" in user_input["provenance"]
    policy = json.loads(summary["registered_analysis_policy"])
    assert policy["compiled_design"] == previous["capacity_report"]["compiled_design"]
    assert policy["role_bindings"] == previous["compiled_design"]["role_bindings"]
    assert policy["registry_version"] == "method-pack-estimation.v1"
    assert policy["registry_content_hash"] == content_hash(json.loads(REGISTRY_PATH.read_text()))
    assert "n1_s*n0_s/n_s" in summary["analysis_policy_explanation"]
    assert "G-1 degrees of freedom" in summary["analysis_policy_explanation"]
    assert "not exact randomization inference" in summary["analysis_policy_explanation"]
    assert "95% intervals" in summary["analysis_policy_explanation"]
    assert "not a passed balance check" in summary["analysis_policy_explanation"]
    assert "do not represent the final fixed-effects" in summary["analysis_policy_explanation"]


@pytest.mark.parametrize("use_structure", (False, True))
@pytest.mark.parametrize(("method", "version", "structure", "estimand"), (
    ("randomized_experiment", "randomized-experiment-pack.v1",
     {"specification": "difference_in_means"}, "itt"),
    ("aipw", "aipw-pack.v1", {"propensity_bound": 0.02}, "att"),
    ("did", "did-pack.v1", {"adoption_profile_id": "staggered", "adoption_time": 4}, "att"),
    ("sharp_rdd", "sharp-rdd-pack.v1", {"cutoff": 59.1984, "polynomial_order": 2}, "local_ate"),
))
def test_each_method_preview_matches_actual_plan_compiler(
    method: str, version: str, structure: dict[str, Any], estimand: str, use_structure: bool,
) -> None:
    structure = structure if use_structure else {}
    design = deepcopy(captured()["packet"]["payload"]["compiled_design"])
    design.update(method_id=method, method_pack_version=version, estimand=estimand,
                  primary_contrasts=["1_vs_0", "2_vs_0"])
    design["preparation"]["method_structure"] = structure
    policy = json.loads(disclosures(design)["registered_analysis_policy"])
    context = manifest().model_copy(update={"method_id": method, "method_pack_version": version,
        "method_structure": structure, "estimand_id": estimand,
        "contrast_ids": tuple(design["primary_contrasts"])})
    plan = plancompile.compile_plan(context, REGISTRY.get(method, version), ref("manifest"))
    for key in ("estimator_id", "estimator_version", "estimator_parameters", "confidence_level",
                "uncertainty_method", "finite_sample_correction", "multiplicity_policy_id"):
        assert policy[key] == getattr(plan, key)
    assert policy["fold_count_default"] == plan.fold_count
    pack = REGISTRY.get(method, version)
    if method == "aipw":
        profile = pack.primary_nuisance_profile()
        assert profile is not None
        assert policy["selected_nuisance_profile"] == profile.model_dump(mode="json")
        assert policy["selected_nuisance_profile"]["profile_id"] == plan.nuisance_profile_id
        assert policy["selected_nuisance_profile"]["hyperparameters"]["regularization_c"] == 1.0
    elif method == "did":
        profile = pack.estimator_profile(str(plan.estimator_parameters["adoption_profile_id"]))
        assert policy["selected_adoption_profile"] == profile.model_dump(mode="json")
        assert policy["profile_estimator_parameters"] == (
            dict(profile.parameter_defaults) | dict(plan.estimator_parameters))


def test_rct_disclosure_handles_distinct_precision_covariate_and_legacy_covariance() -> None:
    design = deepcopy(captured()["packet"]["payload"]["compiled_design"])
    design["role_bindings"].append({"role": "precision_covariate", "columns": ["baseline"]})
    design["preparation"]["method_structure"] = {"cluster_covariance": "cluster_robust_cr2"}
    text = disclosures(design)["analysis_policy_explanation"]
    assert "Additional approved precision adjustment: baseline" in text
    assert "n1_s*n0_s/n_s" not in text
    assert "legacy cluster_robust_cr2 name maps to CRV1" in text
    design["role_bindings"] = [row for row in design["role_bindings"] if row["role"] != "cluster"]
    text = disclosures(design)["analysis_policy_explanation"]
    assert "HC2" in text and "N-K degrees of freedom" in text
    assert "G-1 degrees of freedom" not in text


def test_registry_path_is_configured_and_hash_changes_with_registered_policy(tmp_path: Path) -> None:
    design = captured()["packet"]["payload"]["compiled_design"]
    registry = json.loads(REGISTRY_PATH.read_text())
    registry["packs"][0]["confidence_level"] = 0.90
    changed = tmp_path / REGISTRY_PATH.name
    changed.write_text(json.dumps(registry))
    result = disclosures(design, path=changed)
    policy = json.loads(result["registered_analysis_policy"])
    assert policy["registry_content_hash"] == content_hash(registry)
    assert "90% intervals" in result["analysis_policy_explanation"]
    assert policy["estimator_parameters"] == json.loads(disclosures(design)[
        "registered_analysis_policy"])["estimator_parameters"]


def test_disclosure_rejects_mismatched_artifact_binding() -> None:
    case = captured()
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        approval_disclosures(design=case["packet"]["payload"]["compiled_design"],
            design_ref=ref("wrong"), question=case["question"],
            question_ref=ArtifactRef.model_validate(case["question_ref"]), registry_path=REGISTRY_PATH)


@pytest.mark.parametrize(("method", "version", "name"), (
    ("aipw", "aipw-pack.v1", "nuisance_profile_id"),
    ("did", "did-pack.v1", "adoption_profile_id"),
))
def test_profile_selection_has_no_silent_fallback(method: str, version: str, name: str) -> None:
    design = deepcopy(captured()["packet"]["payload"]["compiled_design"])
    design.update(method_id=method, method_pack_version=version)
    design["preparation"]["method_structure"] = {name: "unregistered_profile"}
    with pytest.raises(ValueError, match="exact registered analysis profile"):
        disclosures(design)
