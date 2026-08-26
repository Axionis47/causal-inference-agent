"""The method-pack preparation overlay and its binding to the frozen packs (T-015 §1.3; PRD-003 §13)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from causal.preparation.packs import PreparationPackV1, load_preparation_packs
from causal.preparation.plans import FitScope
from causal.shared.registry import RegistryError

REGISTRIES = Path(__file__).resolve().parents[2] / "registries"
OVERLAY_PATH = REGISTRIES / "method-pack-preparation.v1.json"
PACKS_PATH = REGISTRIES / "method-packs.v1.json"
METHOD_KEYS = (
    ("randomized_experiment", "randomized-experiment-pack.v1"),
    ("aipw", "aipw-pack.v1"),
    ("did", "did-pack.v1"),
    ("sharp_rdd", "sharp-rdd-pack.v1"),
)
DESIGN_PACKS: dict[str, dict[str, Any]] = {
    pack["method_id"]: pack
    for pack in json.loads(PACKS_PATH.read_text(encoding="utf-8"))["packs"]
}
REGISTRY = load_preparation_packs(OVERLAY_PATH, PACKS_PATH)


def tamper(tmp_path: Path, source: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    """Copy one registry file, apply a mutation, and return the tampered path."""
    document: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    mutate(document)
    target = tmp_path / source.name
    target.write_text(json.dumps(document), encoding="utf-8")
    return target


def expect_code(overlay: Path, packs: Path, code: str) -> None:
    with pytest.raises(RegistryError) as excinfo:
        load_preparation_packs(overlay, packs)
    assert excinfo.value.code == code


def test_the_overlay_covers_all_four_methods() -> None:
    assert len(REGISTRY) == 4
    assert tuple((row.method_id, row.pack_version) for row in REGISTRY.all()) == METHOD_KEYS


@pytest.mark.parametrize(("method_id", "pack_version"), METHOD_KEYS)
class TestOverlayRow:
    def row(self, method_id: str, pack_version: str) -> PreparationPackV1:
        return REGISTRY.get(method_id, pack_version)

    def test_impact_dimensions_match_the_design_pack(
        self, method_id: str, pack_version: str
    ) -> None:
        design = DESIGN_PACKS[method_id]["deletion_impact_dimensions"]
        assert list(self.row(method_id, pack_version).dimension_impact_dimensions) == design

    def test_postrepair_diagnostics_match_the_design_pack(
        self, method_id: str, pack_version: str
    ) -> None:
        design = DESIGN_PACKS[method_id]["required_postrepair_diagnostic_ids"]
        assert list(self.row(method_id, pack_version).required_postrepair_diagnostic_ids) == design

    def test_protected_roles_cover_every_forbidden_imputation_role(
        self, method_id: str, pack_version: str
    ) -> None:
        forbidden = set(DESIGN_PACKS[method_id]["imputation_forbidden_roles"])
        assert forbidden <= set(self.row(method_id, pack_version).protected_roles)

    def test_imputation_targets_stay_inside_the_eligible_roles(
        self, method_id: str, pack_version: str
    ) -> None:
        row = self.row(method_id, pack_version)
        eligible = set(DESIGN_PACKS[method_id]["imputation_eligible_roles"])
        assert {target.role for target in row.permitted_imputation_targets} <= eligible

    def test_a_numeric_median_target_requires_its_missingness_indicator(
        self, method_id: str, pack_version: str
    ) -> None:
        # PRD-003 §11.1 rule 3: median imputation always ships an indicator.
        for target in self.row(method_id, pack_version).permitted_imputation_targets:
            median = target.strategy_id == "numeric_median_with_indicator"
            assert target.requires_missingness_indicator is median


def test_aipw_confounders_are_fitted_inside_cross_fitting() -> None:
    row = REGISTRY.get("aipw", "aipw-pack.v1")
    scopes = {target.fit_scope for target in row.permitted_imputation_targets}
    assert scopes == {FitScope.CROSS_FIT_TRAINING_FOLD}


def test_did_imputes_from_pre_treatment_information_only() -> None:
    row = REGISTRY.get("did", "did-pack.v1")
    scopes = {target.fit_scope for target in row.permitted_imputation_targets}
    assert scopes == {FitScope.PRE_TREATMENT_ONLY}


def test_rdd_protects_the_running_variable_and_pools_blinded() -> None:
    row = REGISTRY.get("sharp_rdd", "sharp-rdd-pack.v1")
    assert "running_variable" in row.protected_roles
    scopes = {target.fit_scope for target in row.permitted_imputation_targets}
    assert scopes == {FitScope.FROZEN_FRAME_BLINDED}


class TestFailsClosed:
    def test_an_unknown_pack_version_is_rejected(self, tmp_path: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["packs"][0]["pack_version"] = "aipw-pack.v9"

        expect_code(tamper(tmp_path, OVERLAY_PATH, mutate), PACKS_PATH, "unknown_method_pack")

    def test_a_missing_overlay_row_is_rejected(self, tmp_path: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["packs"] = document["packs"][:3]

        expect_code(tamper(tmp_path, OVERLAY_PATH, mutate), PACKS_PATH, "unknown_method_pack")

    def test_a_duplicate_overlay_row_is_rejected(self, tmp_path: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["packs"].append(document["packs"][0])

        expect_code(tamper(tmp_path, OVERLAY_PATH, mutate), PACKS_PATH, "invalid_registry_file")

    def test_a_wrong_registry_version_is_rejected(self, tmp_path: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["registry_version"] = "method-pack-preparation.v2"

        expect_code(tamper(tmp_path, OVERLAY_PATH, mutate), PACKS_PATH, "invalid_registry_file")

    def test_an_imputable_protected_role_is_rejected(self, tmp_path: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["packs"][0]["permitted_imputation_targets"][0]["role"] = "treatment"

        expect_code(tamper(tmp_path, OVERLAY_PATH, mutate), PACKS_PATH, "invalid_registry_file")

    def test_a_tampered_method_pack_file_fails_closed(self, tmp_path: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["packs"][1]["pack_version"] = "aipw-pack.v9"

        expect_code(OVERLAY_PATH, tamper(tmp_path, PACKS_PATH, mutate), "unknown_method_pack")

    def test_a_structurally_broken_method_pack_file_fails_closed(self, tmp_path: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            del document["packs"]

        expect_code(OVERLAY_PATH, tamper(tmp_path, PACKS_PATH, mutate), "invalid_registry_file")

    def test_a_missing_overlay_file_fails_closed(self, tmp_path: Path) -> None:
        expect_code(tmp_path / "absent.json", PACKS_PATH, "invalid_registry_file")
