# The estimation method-pack registry and the appended artifact-type rows (T-022 §2; PRD-004 §8).

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from causal.analysis.integration import RESOURCE_ROOT
from causal.analysis.integration import packs as ep
from causal.analysis.integration.contracts import EstimationError
from causal.shared.registry import load_artifact_type_registry

REGISTRIES = Path(__file__).resolve().parents[5] / "registries"
PACKS_PATH = RESOURCE_ROOT / "method-pack-estimation.v1.json"
DESIGN_PATH = REGISTRIES / "method-packs.v1.json"
METHOD_KEYS = (
    ("randomized_experiment", "randomized-experiment-pack.v1"), ("aipw", "aipw-pack.v1"),
    ("did", "did-pack.v1"), ("sharp_rdd", "sharp-rdd-pack.v1"),
)
REGISTRY = ep.load_estimation_packs(PACKS_PATH, DESIGN_PATH)
PACKS = REGISTRY.all()
ARTIFACT_TYPES = load_artifact_type_registry(REGISTRIES / "artifact-types.v1.json")
ESTIMATION_TYPES = (
    "EstimationContextManifest", "EstimationPlan", "AnalysisContributionMask",
    "CrossFitAssignment", "PrimaryAnalysisResult", "MultiplicityResult", "DiagnosticResult",
    "SensitivityResult", "FigureDataArtifact", "EstimationEvidenceBundle", "JudgmentCeiling",
    "NumericalEnvironmentManifest", "EstimationBundle", "EstimationOutcome",
)
# T-025 registers ClaimJudgment; the bundle names it now because this registry is append-only.
FORWARD_PARENTS = ("ClaimJudgment",)
RESTRICTED_TYPES = ("AnalysisContributionMask", "CrossFitAssignment")
Mutation = Callable[[dict[str, Any]], None]


def _strip_guard_threshold(document: dict[str, Any]) -> None:
    for row in document["packs"][3]["required_diagnostics"]:
        if row["diagnostic_id"] == "density_manipulation_test":
            row["threshold_params"] = {}


FAILURES: tuple[tuple[Mutation, str], ...] = (
    (lambda doc: doc["packs"][0].update(invented_key="x"), ep.INVALID_PACK_FILE),
    (lambda doc: doc["packs"][0]["required_diagnostics"][0].pop("severity"), ep.INVALID_PACK_FILE),
    (_strip_guard_threshold, ep.INVALID_PACK_FILE),
    (lambda doc: doc["packs"].append(doc["packs"][0]), ep.DUPLICATE_PACK_ID),
    (lambda doc: doc["packs"][0]["required_diagnostics"].append(
        doc["packs"][0]["required_diagnostics"][0]), ep.DUPLICATE_PACK_ID),
    (lambda doc: doc["packs"][1]["sensitivity_branches"][0]["parameter_delta"].update(
        nuisance_profile_id="invented"), ep.UNREGISTERED_PACK_REFERENCE),
    (lambda doc: doc["packs"][2]["parameter_defaults"].update(adoption_profile_id="invented"),
     ep.UNREGISTERED_PACK_REFERENCE),
    (lambda doc: doc["packs"][1]["nuisance_profiles"][0].update(role="sensitivity"),
     ep.UNREGISTERED_PACK_REFERENCE),
    (lambda doc: doc["packs"][0].update(method_id="invented_method"), ep.UNKNOWN_ESTIMATION_PACK),
    (lambda doc: doc["packs"].pop(), ep.UNKNOWN_ESTIMATION_PACK),
)


def tamper(tmp_path: Path, mutate: Mutation) -> Path:
    document: dict[str, Any] = json.loads(PACKS_PATH.read_text(encoding="utf-8"))
    mutate(document)
    target = tmp_path / PACKS_PATH.name
    target.write_text(json.dumps(document), encoding="utf-8")
    return target


def expect_code(path: Path, code: str) -> None:
    with pytest.raises(EstimationError) as excinfo:
        ep.load_estimation_packs(path, DESIGN_PATH)
    assert excinfo.value.code == code


def test_the_registry_covers_all_four_methods() -> None:
    assert len(REGISTRY) == 4
    assert tuple((row.method_id, row.pack_version) for row in PACKS) == METHOD_KEYS


@pytest.mark.parametrize("pack", PACKS, ids=[row.method_id for row in PACKS])
class TestEveryPack:
    def test_declares_an_estimator_uncertainty_and_masks(
        self, pack: ep.EstimationPackV1
    ) -> None:
        assert pack.estimator_id and pack.estimator_version and pack.uncertainty_method
        assert pack.finite_sample_correction and pack.confidence_level == 0.95
        assert pack.not_estimable_rule_ids and pack.invalidation_rule_ids
        assert pack.parameter_defaults["mask_rule_id"] in pack.allowed_mask_rule_ids
        assert {"treatment", "outcome", "unit_identifier"} <= set(pack.estimator_input_schema)
        assert all(dtypes for dtypes in pack.estimator_input_schema.values())

    def test_severities_carry_a_threshold_wherever_a_guard_decides(
        self, pack: ep.EstimationPackV1
    ) -> None:
        severities = pack.severities()
        assert severities == {row.diagnostic_id: row.severity for row in pack.required_diagnostics}
        assert "required_blocking" in severities.values()
        for row in pack.required_diagnostics:
            assert row.threshold_params or row.severity not in (
                "invalidation_guard", "qualification_guard")

    def test_every_branch_states_a_delta_and_a_comparison_rule(
        self, pack: ep.EstimationPackV1
    ) -> None:
        assert pack.sensitivity_branches
        for row in pack.sensitivity_branches:
            assert pack.branch(row.branch_id) is row
            assert row.parameter_delta and row.comparison_rule_id


def test_the_aipw_pack_cross_fits_a_primary_glm_with_a_boosted_branch() -> None:
    pack = REGISTRY.get("aipw", "aipw-pack.v1")
    assert pack.fold_count_default == 5
    primary = pack.primary_nuisance_profile()
    assert primary is not None and primary.profile_id == "regularized_glm"
    assert primary.propensity_learner == "sklearn_logistic_regression_l2"
    assert pack.branch("alternative_nuisance_profile").parameter_delta[
        "nuisance_profile_id"] == "histogram_gradient_boosting"
    assert [row.role for row in pack.nuisance_profiles] == ["primary", "sensitivity"]


def test_the_did_pack_registers_both_adoption_profiles() -> None:
    pack = REGISTRY.get("did", "did-pack.v1")
    assert [row.profile_id for row in pack.estimator_profiles] == ["simultaneous", "staggered"]
    assert pack.parameter_defaults["adoption_profile_id"] == "simultaneous"
    staggered = pack.estimator_profile("staggered")
    assert staggered.estimator_id == "sun_abraham_event_study"
    assert "declared_panel" in staggered.required_structure_ids
    assert pack.estimator_profile("simultaneous").estimator_id == "common_adoption_did"


def test_the_rdd_and_rct_packs_prespecify_their_own_branches_and_guards() -> None:
    rdd = REGISTRY.get("sharp_rdd", "sharp-rdd-pack.v1")
    assert rdd.parameter_defaults["kernel"] == "triangular"
    assert rdd.branch("half_bandwidth").parameter_delta["bandwidth_multiplier"] == 0.5
    assert rdd.branch("local_quadratic").parameter_delta["polynomial_order"] == 2
    assert rdd.severities()["density_manipulation_test"] == "invalidation_guard"
    rct = REGISTRY.get("randomized_experiment", "randomized-experiment-pack.v1")
    assert rct.severities()["multiplicity_handling"] == "required_blocking"
    assert rct.nuisance_profiles == () and rct.fold_count_default is None
    assert rct.primary_nuisance_profile() is None


def test_an_unregistered_accessor_id_fails_closed() -> None:
    pack = REGISTRY.get("did", "did-pack.v1")
    for call in (lambda: pack.branch("invented"), lambda: pack.estimator_profile("invented")):
        with pytest.raises(EstimationError) as excinfo:
            call()
        assert excinfo.value.code == ep.UNREGISTERED_PACK_REFERENCE
    with pytest.raises(EstimationError) as missing:
        REGISTRY.get("aipw", "aipw-pack.v2")
    assert missing.value.code == ep.UNKNOWN_ESTIMATION_PACK


@pytest.mark.parametrize(("mutate", "code"), FAILURES)
def test_the_loader_fails_closed(tmp_path: Path, mutate: Mutation, code: str) -> None:
    expect_code(tamper(tmp_path, mutate), code)


def test_an_unreadable_registry_file_fails_closed(tmp_path: Path) -> None:
    expect_code(tmp_path / "absent.json", ep.INVALID_PACK_FILE)


@pytest.mark.parametrize("artifact_type", ESTIMATION_TYPES)
class TestEstimationArtifactRow:
    def test_is_produced_and_read_by_the_estimation_harness(self, artifact_type: str) -> None:
        row = ARTIFACT_TYPES.lookup(artifact_type)
        assert row.producer_component == "estimation-harness"
        assert "estimation-harness" in row.allowed_reader_components
        assert (row.sensitivity_class == "restricted") is (artifact_type in RESTRICTED_TYPES)

    def test_every_parent_type_resolves(self, artifact_type: str) -> None:
        row = ARTIFACT_TYPES.lookup(artifact_type)
        for parent in (*row.required_parent_types, *row.optional_parent_types):
            if parent not in FORWARD_PARENTS:
                assert ARTIFACT_TYPES.lookup(parent).artifact_type == parent

    def test_validator_version_follows_the_schema_stem(self, artifact_type: str) -> None:
        row = ARTIFACT_TYPES.lookup(artifact_type)
        assert row.validator_version == f"{row.schema_version.removesuffix('.v1')}-validator.v1"


def test_presentation_reads_the_handoff_types_only() -> None:
    assert len(ARTIFACT_TYPES) == 70
    for artifact_type in ("EstimationBundle", "EstimationEvidenceBundle", "FigureDataArtifact"):
        assert "presentation-coordinator" in ARTIFACT_TYPES.lookup(
            artifact_type).allowed_reader_components
    for artifact_type in ("EstimationPlan", "AnalysisContributionMask", "CrossFitAssignment"):
        assert "presentation-coordinator" not in ARTIFACT_TYPES.lookup(
            artifact_type).allowed_reader_components
