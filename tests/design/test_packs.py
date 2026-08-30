"""Tests for method packs, requirement templates, and the tool allowlist (T-011)."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from causal.design.packs import (
    METHOD_IDS,
    PREREPAIR_DIAGNOSTIC_IDS,
    TASK_KINDS,
    PackRegistryError,
    load_method_packs,
    load_requirement_templates,
    load_tool_registry,
    verify_requirement_references,
)
from causal.design.semantics import RoleName

REGISTRIES = Path(__file__).resolve().parents[2] / "registries"
PACKS_PATH = REGISTRIES / "method-packs.v1.json"
REQUIREMENTS_PATH = REGISTRIES / "context-requirements.v1.json"
TOOLS_PATH = REGISTRIES / "design-tools.v1.json"

ROLE_FIELDS = (
    "required_roles", "optional_roles", "forbidden_adjustment_roles",
    "imputation_forbidden_roles", "imputation_eligible_roles",
)
# PRD-002 §10.1 — one requirement template per common semantic bullet.
REQUIRED_TEMPLATE_IDS = (
    "design.causal_question", "design.treatment_meaning", "design.outcome_window",
    "design.population_comparator", "design.unit_identity", "design.table_grain",
    "dataset.sampling_mechanism", "design.assignment_mechanism", "column.meaning",
    "column.encoding", "column.missing_meaning", "column.measurement_timing",
    "column.source_process", "design.concept_mapping", "design.treatment_descendants",
    "design.selection_variables", "design.conflict_resolution",
)
# SC §5.4 — the five PRD-002 model-task rows, verbatim.
# D-100: no design tool has a handler any more, so the registry advertises none
# and every envelope's `allowed_tool_ids` is empty.
EXPECTED_RECIPIENT_MAP: dict[str, tuple[str, ...]] = dict.fromkeys(TASK_KINDS, ())

HARNESS_ONLY_TOOLS = ("validate_experiment_design", "render_causal_graph", "request_user_context")


def tamper(tmp_path: Path, source: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    """Copy one registry file, apply a mutation, and return the tampered path."""
    document: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    mutate(document)
    target = tmp_path / source.name
    target.write_text(json.dumps(document), encoding="utf-8")
    return target


def expect_code(path: Path, code: str) -> None:
    with pytest.raises(PackRegistryError) as excinfo:
        load_method_packs(path)
    assert excinfo.value.code == code


class TestMethodPacks:
    def test_four_packs_with_exact_method_ids(self) -> None:
        packs = load_method_packs(PACKS_PATH)
        assert len(packs) == 4
        assert tuple(pack.method_id for pack in packs.all()) == METHOD_IDS

    def test_every_pack_holds_the_manifest_invariants(self) -> None:
        known = {role.value for role in RoleName}
        for pack in load_method_packs(PACKS_PATH).all():
            assert pack.pack_version.endswith("-pack.v1")
            assert pack.reserved_estimator_id.endswith("-estimator.v1")
            assert pack.runnable_frame_schema == "runnable-frame-contract.v1"
            for field in ROLE_FIELDS:
                cited: tuple[str, ...] = getattr(pack, field)
                assert set(cited) <= known, f"{pack.method_id}.{field}"
            forbidden = set(pack.imputation_forbidden_roles)
            assert {RoleName.TREATMENT.value, RoleName.OUTCOME.value} <= forbidden
            assert not forbidden & set(pack.imputation_eligible_roles)
            assert {RoleName.MEDIATOR.value, RoleName.COLLIDER.value} <= set(
                pack.forbidden_adjustment_roles
            )
            allowed = set(pack.allowed_prerepair_diagnostic_ids)
            assert allowed <= set(PREREPAIR_DIAGNOSTIC_IDS)
            assert set(pack.required_postrepair_diagnostic_ids) <= allowed

    def test_method_specific_design_facts(self) -> None:
        packs = load_method_packs(PACKS_PATH)
        rct = packs.get("randomized_experiment")
        assert rct.compatible_assignment_mechanisms == ("randomized",)
        assert rct.supported_estimands[0] == "itt"
        assert "arms_ge_2" in rct.structural_requirements
        aipw = packs.get("aipw")
        assert aipw.supported_estimands == ("ate", "att")
        assert "one_row_per_unit" in aipw.structural_requirements
        assert "treatment_binary" in aipw.structural_requirements
        did = packs.get("did")
        assert did.supported_estimands == ("att_group_time_aggregate",)
        assert RoleName.CLUSTER.value in did.required_roles
        assert "unit_time_or_group_time_rows" in did.structural_requirements
        rdd = packs.get("sharp_rdd")
        assert rdd.supported_estimands == ("late_at_cutoff",)
        assert RoleName.RUNNING_VARIABLE.value in rdd.required_roles
        assert "support_on_both_sides_of_cutoff" in rdd.structural_requirements

    def test_unsupported_method_lookup(self) -> None:
        with pytest.raises(PackRegistryError) as excinfo:
            load_method_packs(PACKS_PATH).get("synthetic_control")
        assert excinfo.value.code == "unsupported_method"


class TestTamperedPacks:
    def test_wrong_registry_version(self, tmp_path: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["registry_version"] = "method-packs.v2"

        expect_code(tamper(tmp_path, PACKS_PATH, mutate), "invalid_registry_file")

    def test_duplicate_method(self, tmp_path: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["packs"].append(document["packs"][0])

        expect_code(tamper(tmp_path, PACKS_PATH, mutate), "duplicate_method")

    def test_three_packs(self, tmp_path: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["packs"] = document["packs"][:3]

        expect_code(tamper(tmp_path, PACKS_PATH, mutate), "wrong_pack_count")

    def test_unknown_role(self, tmp_path: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["packs"][0]["optional_roles"].append("chief_scientist")

        expect_code(tamper(tmp_path, PACKS_PATH, mutate), "unknown_role")

    def test_unknown_diagnostic_id(self, tmp_path: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["packs"][1]["allowed_prerepair_diagnostic_ids"].append("eyeball_the_data")

        expect_code(tamper(tmp_path, PACKS_PATH, mutate), "unknown_diagnostic")

    def test_postrepair_diagnostic_outside_allowed_set(self, tmp_path: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["packs"][2]["required_postrepair_diagnostic_ids"].append("arm_counts")

        expect_code(tamper(tmp_path, PACKS_PATH, mutate), "unknown_diagnostic")

    def test_imputable_outcome_rejected(self, tmp_path: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["packs"][0]["imputation_forbidden_roles"].remove("outcome")

        expect_code(tamper(tmp_path, PACKS_PATH, mutate), "invalid_registry_file")


class TestRequirementTemplates:
    def test_every_common_requirement_has_a_row(self) -> None:
        templates = load_requirement_templates(REQUIREMENTS_PATH)
        assert set(REQUIRED_TEMPLATE_IDS) <= set(templates)
        timing = templates["column.measurement_timing"]
        assert timing.scope_kind.value == "column"
        assert timing.criticality.value == "blocking"
        assert timing.required_support.value == "direct_or_corroborated"
        assert timing.missing_action.value == "ask_user"
        assert "timestamp_relationship" in [e.value for e in timing.acceptable_evidence_types]
        for template in templates.values():
            assert set(template.methods_required_for) <= set(METHOD_IDS)
            assert template.user_may_know is True

    def test_pack_requirement_ids_resolve(self) -> None:
        packs = load_method_packs(PACKS_PATH)
        templates = load_requirement_templates(REQUIREMENTS_PATH)
        verify_requirement_references(packs, templates)
        for pack in packs.all():
            assert set(pack.required_context_requirement_ids) <= set(templates)

    def test_unknown_requirement_id_fails_closed(self, tmp_path: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["packs"][3]["required_context_requirement_ids"].append("design.vibes")

        packs = load_method_packs(tamper(tmp_path, PACKS_PATH, mutate))
        with pytest.raises(PackRegistryError) as excinfo:
            verify_requirement_references(packs, load_requirement_templates(REQUIREMENTS_PATH))
        assert excinfo.value.code == "unknown_requirement"

    def test_wrong_requirement_registry_version(self, tmp_path: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["registry_version"] = "context-requirements.v2"

        path = tamper(tmp_path, REQUIREMENTS_PATH, mutate)
        with pytest.raises(PackRegistryError) as excinfo:
            load_requirement_templates(path)
        assert excinfo.value.code == "invalid_registry_file"


class TestToolRegistry:
    def test_eleven_tools_and_recipient_map_matches_the_audit_ledger(self) -> None:
        registry = load_tool_registry(TOOLS_PATH)
        assert len(registry) == 11
        assert registry.recipient_map() == EXPECTED_RECIPIENT_MAP
        assert tuple(registry.recipient_map()) == TASK_KINDS

    def test_harness_only_tools_reach_no_task_kind(self) -> None:
        registry = load_tool_registry(TOOLS_PATH)
        for tool_id in HARNESS_ONLY_TOOLS:
            row = registry.lookup(tool_id)
            assert row is not None
            assert row.allowed_task_kinds == ()
            assert row.registered is False
        assert registry.lookup("run_arbitrary_code") is None

    def test_no_tool_is_registered_and_none_reaches_a_task_kind(self) -> None:
        """The handlers were deleted unrun, so the registry may not advertise them (D-100)."""
        registry = load_tool_registry(TOOLS_PATH)
        for tool_id in ("validate_causal_model", "run_preflight_diagnostic",
                        "preview_eligibility_impact", "list_intake_inventory",
                        "get_semantic_evidence", "get_measured_facts", "get_provenance",
                        "get_method_contract"):
            row = registry.lookup(tool_id)
            assert row is not None
            assert (row.registered, row.allowed_task_kinds) == (False, ())

    def test_unknown_task_kind_fails_closed(self, tmp_path: Path) -> None:
        def mutate(document: dict[str, Any]) -> None:
            document["tools"][0]["allowed_task_kinds"].append("free_for_all")

        path = tamper(tmp_path, TOOLS_PATH, mutate)
        with pytest.raises(PackRegistryError) as excinfo:
            load_tool_registry(path)
        assert excinfo.value.code == "invalid_registry_file"
