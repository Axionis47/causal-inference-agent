"""Preparation payload contracts: happy paths and boundary rules (T-015 §1.1; PRD-003 §5–§16)."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from causal.preparation.contracts import (
    PREPARATION_REGISTRY_KEYS,
    ColumnSchemaFieldV1,
    ConflictAction,
    DesignConflictV1,
    DiagnosticStatus,
    DimensionImpactV1,
    DispositionCountV1,
    DispositionLedgerSummaryV1,
    EligibilityEvaluationSummaryV1,
    FrameStage,
    ObjectRefV1,
    PreparationContextManifestV1,
    PreparationDiagnosticV1,
    PreparationOutcomeStatus,
    PreparationOutcomeV1,
    PreparedFrameBundleV1,
    PreparedFrameV1,
    RowDisposition,
    RowSetFreezeV1,
    SourceRowIndexSummaryV1,
    StabilizationRecordV1,
    StabilizedFrameV1,
    _Payload,
)
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef

HASH = "a" * 64
OTHER_HASH = "b" * 64
REF = ArtifactRef(artifact_id="art-1", content_hash=HASH)
OBJECT = ObjectRefV1(object_locator="objects/" + HASH, content_hash=HASH)
VERSIONS = dict.fromkeys(PREPARATION_REGISTRY_KEYS, "v1")

MANIFEST: dict[str, Any] = {
    "selected_csv": REF, "source_object_locator": "objects/" + HASH, "parser_profile_id": "csv.v1",
    "question_id": "q-1", "population_id": "pop-1", "timeframe_id": "tf-1",
    "treatment_id": "tr-1", "outcome_id": "out-1", "comparator_id": "cmp-1",
    "estimand_id": "ate", "method_id": "aipw", "method_pack_version": "aipw-pack.v1",
    "column_roles": {"age": "confounder_candidate"}, "protected_columns": ("treat", "y"),
    "permitted_repair_columns": ("age",), "permitted_imputation_columns": ("age",),
    "approved_grain": "one_row_per_unit", "key_columns": ("unit_id",),
    "prepared_frame_schema_id": "aipw-prepared-frame.v1",
    "eligibility_rule_ids": ("target_population_filter",),
    "unusable_row_rule_ids": ("corrupt_record",),
    "structural_requirements": ("one_row_per_unit",),
    "permitted_operation_ids": ("type_conversion",), "permitted_diagnostic_ids": ("missingness",),
    "deletion_impact_dimensions": ("overall",), "registry_versions": VERSIONS,
    "recipient_map": {"table_wide": ("get_preparation_contract",)}, "manifest_hash": HASH,
}
DIAGNOSTIC: dict[str, Any] = {
    "diagnostic_id": "missingness", "diagnostic_version": "v1",
    "frame_stage": FrameStage.STABILIZED, "inputs": (REF,), "columns_read": ("age",),
    "total_rows": 100, "used_rows": 100, "unused_reason_counts": {}, "row_set_hash": HASH,
    "values": {"missing_fraction": 0.02}, "warnings": (), "status": DiagnosticStatus.PASS,
    "implementation_version": "v1",
}
LEDGER = DispositionLedgerSummaryV1(
    counts=(
        DispositionCountV1(disposition=RowDisposition.RETAINED, row_count=90),
        DispositionCountV1(disposition=RowDisposition.RETAINED_WITH_MISSINGNESS, row_count=10),
        DispositionCountV1(disposition=RowDisposition.NOT_ELIGIBLE_POPULATION, row_count=5),
    ),
    ledger_object=OBJECT,
)
FREEZE = RowSetFreezeV1(
    retained_row_object=OBJECT, retained_row_count=100, unique_unit_count=100, row_set_hash=HASH
)
RECORD: dict[str, Any] = {
    "context_manifest": REF,
    "source_row_index": SourceRowIndexSummaryV1(
        row_count=105, parse_warning_counts={"ragged_row": 1}, index_object=OBJECT
    ),
    "eligibility": EligibilityEvaluationSummaryV1(
        evaluated_row_count=105, rule_counts={"target_population_filter": 5}
    ),
    "dispositions": LEDGER, "impact": (
        DimensionImpactV1(
            dimension_id="overall", retained_by_level={"all": 100},
            excluded_by_level={"all": 5}, warning_codes=(),
        ),
    ),
    "method_structure_status": DiagnosticStatus.PASS, "method_structure_codes": (),
    "freeze": FREEZE, "pre_stabilization_diagnostics": (),
    "post_stabilization_diagnostics": (PreparationDiagnosticV1(**DIAGNOSTIC),),
    "versions": VERSIONS,
}
COLUMNS = (ColumnSchemaFieldV1(column_name="age", dtype="float64", prepared_from=("age_raw",)),)
FRAME: dict[str, Any] = {
    "columns": COLUMNS, "row_count": 100, "row_set_hash": HASH, "frame_object": OBJECT,
    "writer_version": "v1",
}
STABILIZED: dict[str, Any] = {**FRAME, "stabilization_record": REF, "source_csv": REF}
PREPARED: dict[str, Any] = {
    **FRAME, "stabilized_frame": REF, "execution_receipt_bundle": REF,
    "prepared_frame_schema_id": "aipw-prepared-frame.v1",
}
BUNDLE: dict[str, Any] = {
    "selected_table": REF, "compiled_design": REF, "capacity_report": REF,
    "design_approval": REF, "stabilization_record": REF, "stabilized_frame": REF,
    "prepared_frame": REF, "execution_receipt_bundle": REF, "row_set_hash": HASH,
    "stabilized_frame_row_set_hash": HASH, "prepared_frame_row_set_hash": HASH,
    "versions": VERSIONS,
}
OUTCOME: dict[str, Any] = {
    "status": PreparationOutcomeStatus.PREPARED, "context_manifest": REF, "prepared_bundle": REF,
    "design_conflict": None, "stage_run_id": "run-1", "graph_thread_id": "thread-1",
    "error_code": None,
}
CONFLICT: dict[str, Any] = {
    "conflict_code": "unique_grain_unestablished", "failed_rule_id": "one_row_per_unit",
    "affected_row_count": 12, "affected_unit_count": 6,
    "affected_dimension_counts": {"treatment_group": 12},
    "evidence_artifact_ids": ("art-1",), "why_no_permitted_operation": "no registered rule can "
    "pick the valid record without outcome-dependent judgment",
    "material_design_fields": ("approved_grain",),
    "recommended_action": ConflictAction.REVISE_DESIGN, "conflict_id": "conf-1",
    "context_manifest": REF, "evidence": (REF,), "preparation_revision": 1,
}

PAYLOADS: list[tuple[type[_Payload], dict[str, Any]]] = [
    (PreparationContextManifestV1, MANIFEST),
    (StabilizationRecordV1, RECORD),
    (StabilizedFrameV1, STABILIZED),
    (PreparedFrameV1, PREPARED),
    (PreparedFrameBundleV1, BUNDLE),
    (PreparationOutcomeV1, OUTCOME),
    (DesignConflictV1, CONFLICT),
]

BOUNDARIES: list[tuple[type[BaseModel], dict[str, Any], dict[str, Any], str]] = [
    (PreparationContextManifestV1, MANIFEST, {"registry_versions": {"schema": "v1"}},
     "registry_versions key mismatch"),
    (PreparationContextManifestV1, MANIFEST, {"permitted_imputation_columns": ("treat",)},
     "protected columns cannot be imputation targets"),
    (PreparedFrameBundleV1, BUNDLE, {"prepared_frame_row_set_hash": OTHER_HASH},
     "must share one row_set_hash"),
    (PreparedFrameBundleV1, BUNDLE, {"stabilized_frame_row_set_hash": OTHER_HASH},
     "must share one row_set_hash"),
    (PreparationOutcomeV1, OUTCOME, {"prepared_bundle": None},
     "prepared_bundle is present if and only if"),
    (PreparationOutcomeV1, OUTCOME,
     {"status": PreparationOutcomeStatus.DESIGN_CONFLICT, "prepared_bundle": None},
     "design_conflict is present if and only if"),
    (PreparationOutcomeV1, OUTCOME,
     {"status": PreparationOutcomeStatus.FAILED, "prepared_bundle": None,
      "design_conflict": REF}, "design_conflict is present if and only if"),
    (StabilizationRecordV1, RECORD, {"freeze": FREEZE.model_copy(
        update={"retained_row_count": 101, "unique_unit_count": 101})},
     "must equal the frozen retained row count"),
    (RowSetFreezeV1, FREEZE.model_dump(), {"unique_unit_count": 101},
     "unique_unit_count cannot exceed"),
    (PreparationDiagnosticV1, DIAGNOSTIC, {"used_rows": 101}, "used_rows cannot exceed"),
    (DesignConflictV1, CONFLICT, {"evidence_artifact_ids": ("art-2",)},
     "must stamp exactly the drafted evidence ids"),
]


def build(model: type[BaseModel], base: dict[str, Any], **overrides: Any) -> BaseModel:
    return model(**{**base, **overrides})


@pytest.mark.parametrize(("model", "base"), PAYLOADS)
class TestCommittedPayloads:
    def test_happy_path_roundtrip(self, model: type[_Payload], base: dict[str, Any]) -> None:
        built = model(**base)
        again = model.model_validate(built.model_dump())
        assert again == built
        assert content_hash(again.canonical_payload()) == content_hash(built.canonical_payload())

    def test_extra_field_rejected(self, model: type[_Payload], base: dict[str, Any]) -> None:
        with pytest.raises(ValidationError):
            build(model, base, surprise="x")


@pytest.mark.parametrize(("model", "base", "overrides", "match"), BOUNDARIES)
def test_boundary_rejected(
    model: type[BaseModel], base: dict[str, Any], overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValidationError, match=match):
        build(model, base, **overrides)


def test_the_nine_dispositions_are_closed_and_ordered() -> None:
    assert tuple(item.value for item in RowDisposition) == (
        "retained", "retained_with_missingness", "not_eligible_population",
        "not_eligible_timeframe", "unusable_corrupt_record", "unusable_required_identity",
        "unusable_required_role", "unusable_grain_violation", "unresolved_conflict",
    )


def test_an_unresolved_conflict_blocks_the_freeze() -> None:
    ledger = LEDGER.model_copy(update={"counts": (
        *LEDGER.counts,
        DispositionCountV1(disposition=RowDisposition.UNRESOLVED_CONFLICT, row_count=1),
    )})
    with pytest.raises(ValidationError, match="unresolved_conflict"):
        build(StabilizationRecordV1, RECORD, dispositions=ledger)


def test_a_disposition_cannot_be_counted_twice() -> None:
    with pytest.raises(ValidationError, match="at most once"):
        DispositionLedgerSummaryV1(counts=(LEDGER.counts[0], LEDGER.counts[0]),
                                   ledger_object=OBJECT)


def test_the_five_outcome_statuses_match_the_prd() -> None:
    assert tuple(status.value for status in PreparationOutcomeStatus) == (
        "prepared", "design_conflict", "not_runnable", "failed_observability", "failed",
    )
