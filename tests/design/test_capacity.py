"""Scientific cardinalities remain independent of historical presentation limits."""

from __future__ import annotations

from pathlib import Path

import pytest

from causal.design.capacity import (
    CapacityRegistryError,
    CapacityRegistryV1,
    VisualizationTemplateV1,
    compile_capacity_report,
    load_capacity_registry,
)
from causal.design.packs import METHOD_IDS, MethodPackV1, load_method_packs
from causal.design.semantics import RoleName
from causal.design.v2 import DesignFactSetV2, RoleBindingV2
from causal.shared.contracts import ArtifactRef
from causal.shared.envelope import EpistemicStatus

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "registries" / "delivery-capacity.v1.json"
REGISTRY = load_capacity_registry(REGISTRY_PATH)
PACKS = load_method_packs(ROOT / "registries" / "method-packs.v1.json")
REF = ArtifactRef(artifact_id="artifact:1", content_hash="a" * 64)


def binding(role: RoleName, column: str) -> RoleBindingV2:
    return RoleBindingV2(
        role=role, columns=(column,), concept_id=f"concept:{role.value}",
        source_artifact_ids=("ledger:1",), epistemic_status=EpistemicStatus.EVIDENCED)


FACTS = DesignFactSetV2(
    selected_csv=REF, grain="one_row_per_group_time", facts=(), conflicts=(),
    role_bindings=(binding(RoleName.TREATMENT, "treat"),
                   binding(RoleName.GROUP, "group"), binding(RoleName.TIME, "time"),
                   binding(RoleName.RUNNING_VARIABLE, "score")))
PROFILE = {"columns": {
    "treat": {"cardinality": 2}, "group": {"cardinality": 3},
    "time": {"cardinality": 4}, "score": {"cardinality": 20}}}


def compile_for(pack: MethodPackV1, *, registry: CapacityRegistryV1 = REGISTRY):
    roles = {
        "randomized_experiment": {RoleName.TREATMENT},
        "aipw": {RoleName.TREATMENT},
        "did": {RoleName.TREATMENT, RoleName.GROUP, RoleName.TIME},
        "sharp_rdd": {RoleName.TREATMENT, RoleName.RUNNING_VARIABLE},
    }[pack.method_id]
    facts = FACTS.model_copy(update={
        "role_bindings": tuple(row for row in FACTS.role_bindings if row.role in roles)})
    return compile_capacity_report(
        pack=pack, facts=facts, profile=PROFILE, contrast_count=2,
        compiled_design=REF, registry=registry)


class TestRegistry:
    def test_shipped_registry_versions_and_catalog_coverage(self) -> None:
        assert REGISTRY.visualization_catalog_version == "visualization-catalog.v1"
        assert REGISTRY.capacity_registry_version == "delivery-capacity.v1"
        covered = {evidence for template in REGISTRY.templates
                   for evidence in template.visual_evidence_ids}
        for pack in PACKS.all():
            assert set(pack.required_visual_evidence_ids) <= covered

    def test_missing_invalid_and_duplicate_registries_fail_closed(self, tmp_path: Path) -> None:
        with pytest.raises(CapacityRegistryError):
            load_capacity_registry(tmp_path / "absent.json")
        invalid = tmp_path / "invalid.json"
        invalid.write_text(REGISTRY_PATH.read_text().replace(
            '"max_concurrency": 8', '"max_concurrency": 8, "invented": 1'))
        with pytest.raises(CapacityRegistryError):
            load_capacity_registry(invalid)
        duplicate = tmp_path / "duplicate.json"
        duplicate.write_text(REGISTRY_PATH.read_text().replace(
            '"assignment_flow.v1"', '"balance_overview.v1"', 1))
        with pytest.raises(CapacityRegistryError):
            load_capacity_registry(duplicate)


@pytest.mark.parametrize("method_id", METHOD_IDS)
def test_scientific_cardinalities_do_not_prescribe_visuals(
        method_id: str) -> None:
    report = compile_for(PACKS.get(method_id))
    assert report.status == "pass"
    assert report.compiled_design == REF
    assert report.compatible_template_ids == ()
    values = {row.dimension: row for row in report.dimensions}
    assert values["contrasts"].value == 2
    if method_id == "did":
        assert values["cohorts"].value == 3
    elif method_id == "sharp_rdd":
        assert values["cutoff_sides"].value == 2
    else:
        assert values["arms"].value == 2
    assert values["subgroups"].applicability == "not_applicable"
    assert values["subgroups"].value is None
    assert all(values[name].applicability == "not_applicable"
               for name in ("series", "evidence_items"))


def test_missing_template_cannot_reject_a_scientific_design() -> None:
    registry = REGISTRY.model_copy(update={"templates": (REGISTRY.templates[0],)})
    report = compile_for(PACKS.get("aipw"), registry=registry)
    assert report.status == "pass"
    assert report.issues == report.compatible_template_ids == ()


def test_display_limit_overflow_preserves_the_scientific_cardinality() -> None:
    tight = CapacityRegistryV1(
        visualization_catalog_version="visualization-catalog.v1",
        capacity_registry_version="delivery-capacity.v1", accessible_table_max_rows=200,
        max_concurrency=8,
        templates=tuple(VisualizationTemplateV1(
            template_id=row.template_id, visual_evidence_ids=row.visual_evidence_ids,
            max_panels=row.max_panels, max_series=1, max_labels=row.max_labels,
            max_annotations=row.max_annotations) for row in REGISTRY.templates))
    report = compile_for(PACKS.get("aipw"), registry=tight)
    assert report.status == "pass"
    assert report.issues == report.compatible_template_ids == ()
    assert next(row.value for row in report.dimensions if row.dimension == "arms") == 2


def test_rdd_capacity_ignores_incidental_optional_group_role() -> None:
    report = compile_capacity_report(
        pack=PACKS.get("sharp_rdd"), facts=FACTS, profile=PROFILE | {"columns":
            PROFILE["columns"] | {"group": {"cardinality": 1352}}}, contrast_count=1,
        compiled_design=REF, registry=REGISTRY)
    values = {row.dimension: row for row in report.dimensions}
    assert report.status == "pass"
    assert values["cohorts"].applicability == "not_applicable"
    assert values["cutoff_sides"].value == 2
    assert values["series"].value is None


def test_missing_cardinality_stays_unknown_without_a_layout_blocker() -> None:
    report = compile_capacity_report(
        pack=PACKS.get("randomized_experiment"), facts=FACTS, profile={}, contrast_count=1,
        compiled_design=REF, registry=REGISTRY)
    value = next(row for row in report.dimensions if row.dimension == "arms")
    assert value.value is None and value.applicability == "unknown"
    assert report.status == "pass" and not report.issues
