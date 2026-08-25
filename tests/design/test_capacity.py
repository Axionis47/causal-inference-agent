"""Delivery-capacity preflight over the shipped template registry (T-012 §5, §6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from causal.design.capacity import (
    CapacityRegistryError,
    CapacityRegistryV1,
    VisualizationTemplateV1,
    check_capacity,
    load_capacity_registry,
)
from causal.design.frame import CAPACITY_DIMENSIONS, CapacityStatus, DeliveryCapacityCheckV1
from causal.design.packs import METHOD_IDS, MethodPackV1, load_method_packs

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "registries" / "delivery-capacity.v1.json"
REGISTRY = load_capacity_registry(REGISTRY_PATH)
PACKS = load_method_packs(ROOT / "registries" / "method-packs.v1.json")
# A modest but realistic design: two arms, two contrasts, four periods, twenty evidence rows.
CARDINALITIES = {
    "arms": 2, "contrasts": 2, "subgroups": 2, "cohorts": 2, "periods": 4,
    "event_times": 4, "cutoff_sides": 2, "series": 4, "evidence_items": 20,
}


def check(pack: MethodPackV1, **overrides: int) -> DeliveryCapacityCheckV1:
    return check_capacity(
        pack, CARDINALITIES | overrides, pack.required_visual_evidence_ids, registry=REGISTRY)


class TestRegistry:
    def test_the_shipped_registry_pins_both_versions_and_the_concurrency_cap(self) -> None:
        assert REGISTRY.visualization_catalog_version == "visualization-catalog.v1"
        assert REGISTRY.capacity_registry_version == "delivery-capacity.v1"
        assert REGISTRY.accessible_table_max_rows == 200
        assert REGISTRY.max_concurrency == 8

    def test_the_catalog_covers_every_pack_visual_evidence_id(self) -> None:
        covered = {
            evidence_id
            for template in REGISTRY.templates
            for evidence_id in template.visual_evidence_ids
        }
        for pack in PACKS.all():
            assert set(pack.required_visual_evidence_ids) <= covered

    def test_a_missing_registry_file_fails_closed(self, tmp_path: Path) -> None:
        with pytest.raises(CapacityRegistryError) as error:
            load_capacity_registry(tmp_path / "absent.json")
        assert error.value.code == "invalid_registry_file"

    def test_an_unknown_field_fails_closed(self, tmp_path: Path) -> None:
        path = tmp_path / "capacity.json"
        path.write_text(
            REGISTRY_PATH.read_text(encoding="utf-8").replace(
                '"max_concurrency": 8', '"max_concurrency": 8, "invented": 1'),
            encoding="utf-8")
        with pytest.raises(CapacityRegistryError) as error:
            load_capacity_registry(path)
        assert error.value.code == "invalid_registry_file"

    def test_a_repeated_template_id_fails_closed(self, tmp_path: Path) -> None:
        path = tmp_path / "capacity.json"
        path.write_text(
            REGISTRY_PATH.read_text(encoding="utf-8").replace(
                '"assignment_flow.v1"', '"balance_overview.v1"', 1),
            encoding="utf-8")
        with pytest.raises(CapacityRegistryError) as error:
            load_capacity_registry(path)
        assert error.value.code == "invalid_registry_file"


@pytest.mark.parametrize("method_id", METHOD_IDS)
class TestPassPerMethod:
    def test_the_shipped_catalog_carries_the_method(self, method_id: str) -> None:
        pack = PACKS.get(method_id)
        result = check(pack)
        assert result.status is CapacityStatus.PASS
        assert result.failure_codes == ()
        assert result.compatible_templates
        assert result.method_id == method_id
        assert result.method_profile_id == pack.pack_version
        assert result.method_registry_version == "method-packs.v1"
        assert set(result.cardinalities) == set(CAPACITY_DIMENSIONS)
        assert result.accessible_table_capacity == 200
        assert (result.execution_concurrency, result.render_concurrency) == (8, 8)

    def test_every_compatible_template_publishes_its_limits(self, method_id: str) -> None:
        result = check(PACKS.get(method_id))
        for template_id in result.compatible_templates:
            assert result.template_limits[f"{template_id}:max_panels"] >= 1
            assert result.template_limits[f"{template_id}:max_series"] >= 1


class TestFailures:
    def test_an_unregistered_visual_evidence_id_has_no_template(self) -> None:
        result = check_capacity(
            PACKS.get("aipw"), CARDINALITIES, ("overlap", "invented_evidence"), registry=REGISTRY)
        assert result.status is CapacityStatus.FAIL
        assert result.failure_codes == ("no_template:invented_evidence",)

    def test_cardinalities_beyond_every_template_limit_are_over_limit(self) -> None:
        result = check_capacity(
            PACKS.get("aipw"), CARDINALITIES | {"arms": 50}, ("primary_estimate",),
            registry=REGISTRY)
        assert result.status is CapacityStatus.FAIL
        assert "over_limit:estimate_forest.v1:arms" in result.failure_codes
        assert "over_limit:single_contrast.v1:arms" in result.failure_codes

    def test_one_fitting_template_is_enough(self) -> None:
        result = check_capacity(
            PACKS.get("aipw"), CARDINALITIES | {"arms": 3}, ("primary_estimate",),
            registry=REGISTRY)
        assert result.status is CapacityStatus.PASS
        assert result.compatible_templates == ("estimate_forest.v1",)

    def test_evidence_rows_beyond_the_accessible_table_fail(self) -> None:
        result = check(PACKS.get("did"), evidence_items=500)
        assert result.status is CapacityStatus.FAIL
        assert "over_limit:accessible_table:evidence_items" in result.failure_codes

    def test_an_unknown_cardinality_dimension_fails_closed(self) -> None:
        with pytest.raises(CapacityRegistryError) as error:
            check_capacity(PACKS.get("did"), {"invented": 1}, ("trends",), registry=REGISTRY)
        assert error.value.code == "unknown_dimension"

    def test_an_unnamed_dimension_defaults_to_zero(self) -> None:
        result = check_capacity(PACKS.get("did"), {"arms": 2}, ("trends",), registry=REGISTRY)
        assert result.cardinalities["event_times"] == 0
        assert result.status is CapacityStatus.PASS


def test_concurrency_never_exceeds_eight() -> None:
    generous = CapacityRegistryV1(
        visualization_catalog_version="visualization-catalog.v1",
        capacity_registry_version="delivery-capacity.v1",
        accessible_table_max_rows=200, max_concurrency=64,
        templates=(VisualizationTemplateV1(
            template_id="wide.v1", visual_evidence_ids=("trends",), max_panels=4,
            max_series=12, max_labels=24, max_annotations=12),))
    result = check_capacity(PACKS.get("did"), CARDINALITIES, ("trends",), registry=generous)
    assert (result.execution_concurrency, result.render_concurrency) == (8, 8)
    frugal = generous.model_copy(update={"max_concurrency": 4})
    modest = check_capacity(PACKS.get("did"), CARDINALITIES, ("trends",), registry=frugal)
    assert (modest.execution_concurrency, modest.render_concurrency) == (4, 4)
