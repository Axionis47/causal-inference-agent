# The estimation entry gate, deterministic plan compiler, and exact capacity recheck
# (T-023 §2; PRD-004 §4, §6.1, §6.5).

from __future__ import annotations

import pytest

from causal.analysis.integration import plancompile as pc
from causal.analysis.integration.contracts import ESTIMATION_REGISTRY_KEYS, EstimationError
from causal.analysis.integration.tests.support.plans import (
    DECLARED,
    PACK,
    REGISTRY,
    ROW_HASH,
    digest,
    inputs,
    manifest,
    payloads,
    plan,
    policy,
    ref,
)
from causal.shared.canonical import content_hash

# A prepared V2 handoff plus the stabilization record.


def test_plan_preserves_approved_units_instead_of_estimand_or_pack_defaults() -> None:
    metadata = {"label": "Weight change", "units": "kilograms", "scale": "ratio",
                "source_card": ref("outcome_card").model_dump(mode="json"),
                "supporting_evidence_ids": ["ev:source/units"]}
    supplied = inputs(design=payloads()["design"] | {"column_measurements": {"finished": metadata}})
    context = pc.compile_context_manifest(supplied, policy())
    compiled = pc.compile_plan(context, PACK, ref("manifest"))
    assert compiled.outcome_scale == "kilograms"
    assert compiled.column_measurements["finished"].source_card == ref("outcome_card")
    assert pc.compile_plan(manifest(), PACK, ref("legacy_manifest")).outcome_scale == "outcome units"


BASE = payloads()
# One violation per §4 entry condition, in the PRD's order, with the code it must report.
VIOLATIONS: tuple[tuple[str, pc.EntryInputs, pc.EntryPolicy], ...] = (
    (pc.PREPARATION_NOT_PREPARED, inputs(outcome=BASE["outcome"] | {"status": "failed"}), policy()),
    (pc.ENTRY_HASH_MISMATCH,
     inputs(committed=dict(DECLARED) | {"capacity_report": ref("moved")}), policy()),
    (pc.BUNDLE_BINDING_MISMATCH,
     inputs(bundle=BASE["bundle"] | {"compiled_design": ref("other").model_dump()}), policy()),
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
)


def test_the_golden_handoff_compiles_a_closed_context_manifest() -> None:
    assert pc.entry_codes(inputs(), policy()) == ()
    found = manifest()
    assert found.method_id == PACK.method_id and found.row_set_hash == ROW_HASH
    assert found.contrast_ids == ("arm_b_vs_control",)
    assert found.capacity_report == DECLARED["capacity_report"] and "group" not in found.role_columns
    assert set(found.registry_versions) == set(ESTIMATION_REGISTRY_KEYS)
    assert not hasattr(found, "required_diagnostic_ids") and not hasattr(found, "seed")


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










def test_display_capacity_cannot_block_a_valid_numerical_request() -> None:
    supplied = inputs(capacity=BASE["capacity"] | {"status": "fail"})
    assert pc.entry_codes(supplied, policy()) == ()
    assert pc.compile_context_manifest(supplied, policy()).compiled_design == DECLARED["compiled_design"]
