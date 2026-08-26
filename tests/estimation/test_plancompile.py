# The estimation entry gate, deterministic plan compiler, and exact capacity recheck
# (T-023 §2; PRD-004 §4, §6.1, §6.5).

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from causal.estimation import plancompile as pc
from causal.estimation.contracts import ESTIMATION_REGISTRY_KEYS, EstimationError
from causal.estimation.packs import load_estimation_packs
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef

REGISTRIES = Path(__file__).resolve().parents[2] / "registries"
CAPACITY_PATH = REGISTRIES / "delivery-capacity.v1.json"
REGISTRY = load_estimation_packs(REGISTRIES / "method-pack-estimation.v1.json",
                                 REGISTRIES / "method-packs.v1.json")
PACK = REGISTRY.get("randomized_experiment", "randomized-experiment-pack.v1")


def digest(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def ref(name: str) -> ArtifactRef:
    return ArtifactRef(artifact_id=name, content_hash=digest(name))


DECLARED = {key: ref(key) for key in pc.ENTRY_KEYS}
ROW_HASH = digest("rows")
CARDINALITIES = {"arms": 2, "contrasts": 1, "subgroups": 0, "cohorts": 0, "periods": 0,
                 "event_times": 0, "cutoff_sides": 0, "series": 2, "evidence_items": 12}
VERSIONS = dict.fromkeys(ESTIMATION_REGISTRY_KEYS, "pinned.v1") | {
    "capacity": "delivery-capacity.v1", "visualization_catalog": "visualization-catalog.v1"}


# A T-019-shaped prepared handoff: the four §4 entries plus the PRD-003 stabilization record.
def payloads() -> dict[str, dict[str, Any]]:
    return {
        "outcome": {"status": "prepared",
                    "prepared_bundle": DECLARED["prepared_bundle"].model_dump()},
        "bundle": {"experiment_design": DECLARED["experiment_design"].model_dump(),
                   "runnable_frame_contract": DECLARED["runnable_frame_contract"].model_dump(),
                   "capacity_check": DECLARED["capacity_check"].model_dump(),
                   "row_set_hash": ROW_HASH, "stabilized_frame_row_set_hash": ROW_HASH,
                   "prepared_frame_row_set_hash": ROW_HASH},
        "design": {"method_id": PACK.method_id, "method_pack_version": PACK.pack_version,
                   "estimand": "att", "comparator": "control", "unit": "participant",
                   "frame": {"population": "enrolled", "timeframe": "wave_one",
                             "outcome": "completion", "treatment": "offer"},
                   "primary_contrasts": ["arm_b_vs_control"],
                   "required_postrepair_diagnostics": ["baseline_balance"],
                   "registry_versions": {"schema": "prepared-frame.v1"},
                   "visualization_catalog_version": "visualization-catalog.v1",
                   "capacity_registry_version": "delivery-capacity.v1",
                   "capacity_check": DECLARED["capacity_check"].model_dump()},
        "contract": {"estimator_input_schema": "prepared-frame.v1",
                     "required_roles": ["treatment", "outcome"]},
        "capacity": {"status": "pass", "method_id": PACK.method_id,
                     "cardinalities": dict(CARDINALITIES),
                     "visualization_catalog_version": "visualization-catalog.v1",
                     "capacity_registry_version": "delivery-capacity.v1"},
        "record": {"freeze": {"row_set_hash": ROW_HASH},
                   "dispositions": {"counts": [{"disposition": "retained", "row_count": 100}]},
                   "source_row_index": {"row_count": 100}}}


def inputs(**over: Any) -> pc.EntryInputs:
    fields: dict[str, Any] = payloads() | {
        "declared": dict(DECLARED), "committed": dict(DECLARED),
        "estimator_input_types": {"treatment": "categorical", "outcome": "numeric"},
        "role_columns": {"treatment": "arm", "outcome": "finished"},
        "postrepair_statuses": {"baseline_balance": "pass"}}
    return pc.EntryInputs(**(fields | over))


def policy(**over: Any) -> pc.EntryPolicy:
    fields: dict[str, Any] = {"pack": PACK, "registry_versions": dict(VERSIONS),
                              "numerical_tolerances": {"absolute": 1e-9},
                              "recipient_map": {"estimation": ("estimation-harness",)}}
    return pc.EntryPolicy(**(fields | over))


def manifest() -> Any:
    return pc.compile_context_manifest(inputs(), policy())


def plan(**over: Any) -> Any:
    return pc.compile_plan(manifest(), PACK, ref("context_manifest"), **over)


BASE = payloads()
# One violation per §4 entry condition, in the PRD's order, with the code it must report.
VIOLATIONS: tuple[tuple[str, pc.EntryInputs, pc.EntryPolicy], ...] = (
    (pc.PREPARATION_NOT_PREPARED, inputs(outcome=BASE["outcome"] | {"status": "failed"}), policy()),
    (pc.ENTRY_HASH_MISMATCH,
     inputs(committed=dict(DECLARED) | {"capacity_check": ref("moved")}), policy()),
    (pc.BUNDLE_BINDING_MISMATCH,
     inputs(bundle=BASE["bundle"] | {"experiment_design": ref("other").model_dump()}), policy()),
    (pc.ROW_SET_HASH_MISMATCH,
     inputs(bundle=BASE["bundle"] | {"prepared_frame_row_set_hash": digest("drift")}), policy()),
    (pc.ROW_ACCOUNTING_INCOMPLETE,
     inputs(record=BASE["record"] | {"source_row_index": {"row_count": 7}}), policy()),
    (pc.ESTIMATOR_SCHEMA_MISMATCH,
     inputs(estimator_input_types={"treatment": "numeric", "outcome": "numeric"}), policy()),
    (pc.POSTREPAIR_DIAGNOSTIC_UNHANDLED,
     inputs(postrepair_statuses={"baseline_balance": "failed"}), policy()),
    (pc.UNREGISTERED_PREPROCESSING, inputs(preprocessing_recipe_ids=("unknown_recipe",)), policy()),
    (pc.AMBIGUOUS_METHOD_SELECTION, inputs(design=BASE["design"] | {"method_id": "did"}), policy()),
    (pc.UNSUPPORTED_REGISTRY_VERSION, inputs(), policy(registry_versions={"schema": "v1"})),
    (pc.UPSTREAM_ESTIMATE_MARKER, inputs(design=BASE["design"] | {"estimate": 0.4}), policy()),
    (pc.CAPACITY_CHECK_NOT_PASS, inputs(capacity=BASE["capacity"] | {"status": "fail"}), policy()))


def structure(**over: Any) -> pc.PreparedStructureV1:
    fields: dict[str, Any] = {"cardinalities": dict(CARDINALITIES),
                              "required_visual_evidence": ("primary_contrast_estimates",),
                              "registry_path": CAPACITY_PATH}
    return pc.PreparedStructureV1(**(fields | over))


def test_the_golden_handoff_compiles_a_closed_context_manifest() -> None:
    assert pc.entry_codes(inputs(), policy()) == ()
    found = manifest()
    assert found.method_id == PACK.method_id and found.row_set_hash == ROW_HASH
    assert found.contrast_ids == ("arm_b_vs_control",)
    assert found.capacity_check == DECLARED["capacity_check"]
    assert set(found.registry_versions) == set(ESTIMATION_REGISTRY_KEYS)
    assert found.required_diagnostic_ids == tuple(PACK.severities())
    assert 0 <= found.seed < 2**31


@pytest.mark.parametrize(("code", "bad", "rules"), VIOLATIONS, ids=[row[0] for row in VIOLATIONS])
def test_each_entry_condition_reports_its_stable_code(
        code: str, bad: pc.EntryInputs, rules: pc.EntryPolicy) -> None:
    assert code in pc.entry_codes(bad, rules)
    with pytest.raises(EstimationError) as caught:
        pc.compile_context_manifest(bad, rules)
    assert caught.value.code == pc.ENTRY_VALIDATION_FAILED and code in caught.value.detail_codes


def test_a_missing_entry_artifact_is_reported() -> None:
    broken = inputs(committed=dict(DECLARED) | {"prepared_bundle": None})
    assert pc.MISSING_ARTIFACT in pc.entry_codes(broken, policy())


def test_the_plan_is_deterministic_and_every_field_change_rehashes_it() -> None:
    first, second = plan(), plan()
    assert content_hash(first.canonical_payload()) == content_hash(second.canonical_payload())
    assert first.seed == second.seed and 0 <= first.seed < 2**31
    revised = plan(plan_revision=2)
    assert content_hash(revised.canonical_payload()) != content_hash(first.canonical_payload())
    assert revised.seed != first.seed


def test_the_plan_copies_the_approved_selection_and_the_pack() -> None:
    found, source = plan(), manifest()
    assert found.contrast_ids == source.contrast_ids and found.row_set_hash == source.row_set_hash
    assert found.estimator_id == PACK.estimator_id and found.confidence_level == 0.95
    assert found.required_diagnostics == PACK.severities()
    assert found.primary_mask_rule_id == "outcome_observed" and found.fold_count is None
    assert source.prepared_bundle in found.parents and found.context_manifest.artifact_id


def test_a_cross_fitted_pack_pins_both_fold_fields() -> None:
    aipw = REGISTRY.get("aipw", "aipw-pack.v1")
    found = pc.compile_plan(manifest(), aipw, ref("context_manifest"))
    assert found.fold_count == aipw.fold_count_default
    assert found.fold_assignment_rule_id == "stratified_by_treatment_and_cluster"
    assert found.nuisance_profile_id == "regularized_glm"


def test_a_pack_missing_a_planned_field_fails_closed() -> None:
    stripped = PACK.model_copy(update={"parameter_defaults": {}})
    with pytest.raises(EstimationError) as caught:
        pc.compile_plan(manifest(), stripped, ref("context_manifest"))
    assert caught.value.code == pc.MISSING_PLAN_FIELD


def test_the_capacity_recheck_passes_the_approved_structure() -> None:
    assert pc.recheck_capacity(plan(), structure()) is None


def test_an_oversized_result_set_returns_a_design_conflict() -> None:
    over = structure(cardinalities=dict(CARDINALITIES) | {"contrasts": 400})
    conflict = pc.recheck_capacity(plan(), over)
    assert conflict is not None and conflict.conflict_code == pc.CAPACITY_EXCEEDED
    assert conflict.recommended_action == "revise_design"
    assert conflict.affected_dimension_counts["contrasts"] == 400
    assert conflict.evidence_artifact_ids[0] == "capacity_check"


def test_unrenderable_evidence_and_version_drift_both_conflict() -> None:
    missing = pc.recheck_capacity(plan(), structure(required_visual_evidence=("nowhere",)))
    assert missing is not None and missing.failed_rule_id == "no_template:nowhere"
    drifted = pc.compile_plan(
        pc.compile_context_manifest(
            inputs(), policy(registry_versions=dict(VERSIONS) | {"capacity": "capacity.v2"})),
        PACK, ref("context_manifest"))
    conflict = pc.recheck_capacity(drifted, structure())
    assert conflict is not None and conflict.failed_rule_id == pc.CAPACITY_VERSION_DRIFT


def test_the_capacity_recheck_fails_closed_on_bad_input(tmp_path: Path) -> None:
    with pytest.raises(EstimationError) as unreadable:
        pc.recheck_capacity(plan(), structure(registry_path=tmp_path / "absent.json"))
    assert unreadable.value.code == pc.INVALID_CAPACITY_REGISTRY
    with pytest.raises(EstimationError) as unknown:
        pc.recheck_capacity(plan(), structure(cardinalities={"invented": 1}))
    assert unknown.value.code == pc.INVALID_CAPACITY_REGISTRY
