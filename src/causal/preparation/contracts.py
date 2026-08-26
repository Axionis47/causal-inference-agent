"""Preparation manifest, stabilization, frame, outcome, and conflict payloads (PRD-003 §5–§7, §9, §15, §16, §24; D-057)."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from causal.shared.contracts import ArtifactRef, Identity, PayloadLocator, Sha256Hex

__all__ = [
    "PREPARATION_REGISTRY_KEYS", "RETAINING_DISPOSITIONS", "ColumnSchemaFieldV1",
    "ConflictAction", "DesignConflictDraftV1", "DesignConflictV1", "DiagnosticStatus",
    "DimensionImpactV1", "DispositionCountV1", "DispositionLedgerSummaryV1",
    "EligibilityEvaluationSummaryV1", "FrameStage", "ObjectRefV1",
    "PreparationContextManifestV1", "PreparationDiagnosticV1", "PreparationOutcomeStatus",
    "PreparationOutcomeV1", "PreparedFrameBundleV1", "PreparedFrameV1", "RowDisposition",
    "RowSetFreezeV1", "SourceRowIndexSummaryV1", "StabilizationRecordV1", "StabilizedFrameV1",
    "require_exact_keys",
]

# The closed version keys every preparation payload pins (§7.2 last bullet).
PREPARATION_REGISTRY_KEYS: Final = (
    "artifact_types", "method_packs", "preparation_packs", "operations", "diagnostics",
    "parser", "schema", "validators",
)

_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
_NonNegInt = Annotated[int, Field(ge=0)]
_PositiveInt = Annotated[int, Field(ge=1)]
_Ids = Annotated[tuple[Identity, ...], Field(min_length=1)]


def require_exact_keys(
    mapping: Mapping[str, object], expected: tuple[str, ...], label: str
) -> None:
    """Reject a mapping whose key set differs from a closed vocabulary."""
    if set(mapping) != set(expected):
        missing = sorted(set(expected) - set(mapping))
        extra = sorted(set(mapping) - set(expected))
        raise ValueError(f"{label} key mismatch: missing={missing} extra={extra}")


class _Row(BaseModel):
    """Frozen, strict base for preparation row and fragment models."""

    model_config = _MODEL_CONFIG


class _Payload(_Row):
    """Base for committed preparation payloads."""

    def canonical_payload(self) -> dict[str, Any]:
        """Canonical-ready dict; `content_hash(self.canonical_payload())` is replay-stable."""
        return self.model_dump(mode="json")


class ObjectRefV1(_Row):
    """One content-addressed object: where it lives and what it hashes to."""

    object_locator: PayloadLocator
    content_hash: Sha256Hex


class RowDisposition(StrEnum):
    """The nine terminal primary dispositions; every source row gets exactly one (§6.2)."""

    RETAINED = "retained"
    RETAINED_WITH_MISSINGNESS = "retained_with_missingness"
    NOT_ELIGIBLE_POPULATION = "not_eligible_population"
    NOT_ELIGIBLE_TIMEFRAME = "not_eligible_timeframe"
    UNUSABLE_CORRUPT_RECORD = "unusable_corrupt_record"
    UNUSABLE_REQUIRED_IDENTITY = "unusable_required_identity"
    UNUSABLE_REQUIRED_ROLE = "unusable_required_role"
    UNUSABLE_GRAIN_VIOLATION = "unusable_grain_violation"
    UNRESOLVED_CONFLICT = "unresolved_conflict"


# The two dispositions that keep a row in the stabilized frame (§6.2).
RETAINING_DISPOSITIONS: Final = (
    RowDisposition.RETAINED, RowDisposition.RETAINED_WITH_MISSINGNESS,
)


class FrameStage(StrEnum):
    SOURCE = "source"
    STABILIZED = "stabilized"
    PREPARED = "prepared"


class DiagnosticStatus(StrEnum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    NOT_COMPUTABLE = "not_computable"


class PreparationContextManifestV1(_Payload):
    """The closed approved-context surface every preparation task reads from (§7.2)."""

    schema_version: Literal["preparation-context-manifest.v1"] = "preparation-context-manifest.v1"
    selected_csv: ArtifactRef
    source_object_locator: PayloadLocator
    parser_profile_id: Identity
    question_id: Identity
    population_id: Identity
    timeframe_id: Identity
    treatment_id: Identity
    outcome_id: Identity
    comparator_id: Identity
    estimand_id: Identity
    method_id: Identity
    method_pack_version: Identity
    measurement_map: ArtifactRef
    role_ledger: ArtifactRef
    column_concepts: dict[str, Identity]
    column_roles: dict[str, Identity]
    protected_columns: tuple[Identity, ...]
    permitted_repair_columns: tuple[Identity, ...]
    permitted_imputation_columns: tuple[Identity, ...]
    approved_grain: Identity
    key_columns: _Ids
    prepared_frame_schema_id: Identity
    eligibility_rule_ids: tuple[Identity, ...]
    unusable_row_rule_ids: tuple[Identity, ...]
    structural_requirements: tuple[Identity, ...]
    permitted_operation_ids: tuple[Identity, ...]
    permitted_diagnostic_ids: tuple[Identity, ...]
    deletion_impact_dimensions: _Ids
    registry_versions: dict[str, Identity]
    recipient_map: dict[str, tuple[Identity, ...]]
    # SHA-256 of the approved upstream inputs this manifest was compiled from; the
    # manifest's own content hash lives on its envelope.
    manifest_hash: Sha256Hex

    @model_validator(mode="after")
    def _closed_versions_and_protected_columns(self) -> Self:
        require_exact_keys(self.registry_versions, PREPARATION_REGISTRY_KEYS, "registry_versions")
        overlap = sorted(set(self.protected_columns) & set(self.permitted_imputation_columns))
        if overlap:
            raise ValueError(f"protected columns cannot be imputation targets: {overlap}")
        return self


class DispositionCountV1(_Row):
    """How many source rows landed on one terminal disposition."""

    disposition: RowDisposition
    row_count: _NonNegInt


class DimensionImpactV1(_Row):
    """Retained and excluded counts per level of one deletion-impact dimension (§9.5)."""

    dimension_id: Identity
    retained_by_level: dict[str, int]
    excluded_by_level: dict[str, int]
    warning_codes: tuple[Identity, ...]


class SourceRowIndexSummaryV1(_Row):
    """The `SourceRowIndex` folded to counts plus its object pointer (§8, §24.2)."""

    row_count: _NonNegInt
    parse_warning_counts: dict[str, int]
    index_object: ObjectRefV1


class EligibilityEvaluationSummaryV1(_Row):
    """Approved eligibility rules evaluated, as rule id to affected-row count (§9.2)."""

    evaluated_row_count: _NonNegInt
    rule_counts: dict[str, int]


class DispositionLedgerSummaryV1(_Row):
    """The `RowDispositionLedger` folded to counts plus its object pointer (§24.2)."""

    counts: Annotated[tuple[DispositionCountV1, ...], Field(min_length=1)]
    ledger_object: ObjectRefV1

    @model_validator(mode="after")
    def _one_row_per_disposition(self) -> Self:
        seen = [count.disposition for count in self.counts]
        if len(set(seen)) != len(seen):
            raise ValueError("a disposition may appear at most once in the ledger summary")
        return self

    def total(self, *dispositions: RowDisposition) -> int:
        """Rows carrying any of `dispositions`."""
        return sum(row.row_count for row in self.counts if row.disposition in dispositions)


class RowSetFreezeV1(_Row):
    """The frozen retained row set and its `row_set_hash` (§9.6)."""

    retained_row_object: ObjectRefV1
    retained_row_count: _NonNegInt
    unique_unit_count: _NonNegInt
    row_set_hash: Sha256Hex

    @model_validator(mode="after")
    def _units_within_rows(self) -> Self:
        if self.unique_unit_count > self.retained_row_count:
            raise ValueError("unique_unit_count cannot exceed retained_row_count")
        return self


class PreparationDiagnosticV1(_Row):
    """One diagnostic computed over a declared frame stage (§15)."""

    diagnostic_id: Identity
    diagnostic_version: Identity
    frame_stage: FrameStage
    inputs: Annotated[tuple[ArtifactRef, ...], Field(min_length=1)]
    columns_read: tuple[Identity, ...]
    total_rows: _NonNegInt
    used_rows: _NonNegInt
    unused_reason_counts: dict[str, int]
    row_set_hash: Sha256Hex | None
    values: dict[str, float | int | str | bool | None]
    warnings: tuple[Identity, ...]
    status: DiagnosticStatus
    implementation_version: Identity

    @model_validator(mode="after")
    def _used_rows_within_total(self) -> Self:
        if self.used_rows > self.total_rows:
            raise ValueError("used_rows cannot exceed total_rows")
        return self


class StabilizationRecordV1(_Payload):
    """Row index, eligibility, dispositions, impact, and freeze in one record (§24.2)."""

    schema_version: Literal["stabilization-record.v1"] = "stabilization-record.v1"
    context_manifest: ArtifactRef
    source_row_index: SourceRowIndexSummaryV1
    eligibility: EligibilityEvaluationSummaryV1
    dispositions: DispositionLedgerSummaryV1
    impact: Annotated[tuple[DimensionImpactV1, ...], Field(min_length=1)]
    method_structure_status: DiagnosticStatus
    method_structure_codes: tuple[Identity, ...]
    freeze: RowSetFreezeV1
    pre_stabilization_diagnostics: tuple[PreparationDiagnosticV1, ...]
    post_stabilization_diagnostics: tuple[PreparationDiagnosticV1, ...]
    versions: dict[str, Identity]

    @model_validator(mode="after")
    def _dispositions_reconcile_with_the_freeze(self) -> Self:
        if self.dispositions.total(RowDisposition.UNRESOLVED_CONFLICT):
            raise ValueError("stabilization cannot finish while a row is unresolved_conflict")
        retained = self.dispositions.total(*RETAINING_DISPOSITIONS)
        if retained != self.freeze.retained_row_count:
            raise ValueError(
                f"retained dispositions ({retained}) must equal the frozen retained row count "
                f"({self.freeze.retained_row_count})"
            )
        return self


class ColumnSchemaFieldV1(_Row):
    """One column of a frame: its name, dtype, and the columns it was prepared from."""

    column_name: Identity
    dtype: Identity
    prepared_from: tuple[Identity, ...]


class _FrameV1(_Payload):
    """Metadata for one immutable frame; the rows live in a content-addressed object."""

    columns: Annotated[tuple[ColumnSchemaFieldV1, ...], Field(min_length=1)]
    row_count: _NonNegInt
    row_set_hash: Sha256Hex
    frame_object: ObjectRefV1
    writer_version: Identity


class StabilizedFrameV1(_FrameV1):
    """The frozen post-stabilization frame (§9.6)."""

    schema_version: Literal["stabilized-frame.v1"] = "stabilized-frame.v1"
    stabilization_record: ArtifactRef
    source_csv: ArtifactRef


class PreparedFrameV1(_FrameV1):
    """The final frame PRD-004 estimates on (§5.1)."""

    schema_version: Literal["prepared-frame.v1"] = "prepared-frame.v1"
    stabilized_frame: ArtifactRef
    execution_receipt_bundle: ArtifactRef
    prepared_frame_schema_id: Identity


class PreparedFrameBundleV1(_Payload):
    """The PRD-004 handoff; §5.1's list is satisfied through consolidated parents (§24.2)."""

    schema_version: Literal["prepared-frame-bundle.v1"] = "prepared-frame-bundle.v1"
    selected_table: ArtifactRef
    experiment_design: ArtifactRef
    runnable_frame_contract: ArtifactRef
    capacity_check: ArtifactRef
    stabilization_record: ArtifactRef
    stabilized_frame: ArtifactRef
    prepared_frame: ArtifactRef
    execution_receipt_bundle: ArtifactRef
    row_set_hash: Sha256Hex
    stabilized_frame_row_set_hash: Sha256Hex
    prepared_frame_row_set_hash: Sha256Hex
    versions: dict[str, Identity]

    @model_validator(mode="after")
    def _one_row_set_hash(self) -> Self:
        others = (self.stabilized_frame_row_set_hash, self.prepared_frame_row_set_hash)
        if any(other != self.row_set_hash for other in others):
            raise ValueError("the stabilized and prepared frames must share one row_set_hash")
        return self


class PreparationOutcomeStatus(StrEnum):
    PREPARED = "prepared"
    DESIGN_CONFLICT = "design_conflict"
    NOT_RUNNABLE = "not_runnable"
    FAILED_OBSERVABILITY = "failed_observability"
    FAILED = "failed"


class PreparationOutcomeV1(_Payload):
    """The preparation stage's single terminal record (§5.2)."""

    schema_version: Literal["preparation-outcome.v1"] = "preparation-outcome.v1"
    status: PreparationOutcomeStatus
    context_manifest: ArtifactRef
    prepared_bundle: ArtifactRef | None
    design_conflict: ArtifactRef | None
    stage_run_id: Identity
    graph_thread_id: Identity
    error_code: Identity | None

    @model_validator(mode="after")
    def _status_matches_refs(self) -> Self:
        prepared = self.status is PreparationOutcomeStatus.PREPARED
        conflict = self.status is PreparationOutcomeStatus.DESIGN_CONFLICT
        if (self.prepared_bundle is not None) is not prepared:
            raise ValueError("prepared_bundle is present if and only if status is prepared")
        if (self.design_conflict is not None) is not conflict:
            raise ValueError("design_conflict is present if and only if status is design_conflict")
        return self


class ConflictAction(StrEnum):
    """What PRD-002 should do with the returned conflict (§16)."""

    ASK_USER = "ask_user"
    REVISE_DESIGN = "revise_design"
    REFUSE = "refuse"


class DesignConflictDraftV1(_Row):
    """The agent's proposed conflict: no ids or hashes, those are stamped by the harness."""

    conflict_code: Identity
    failed_rule_id: Identity
    affected_row_count: _NonNegInt
    affected_unit_count: _NonNegInt
    affected_dimension_counts: dict[str, int]
    evidence_artifact_ids: tuple[Identity, ...]
    why_no_permitted_operation: str
    material_design_fields: _Ids
    recommended_action: ConflictAction


class DesignConflictV1(DesignConflictDraftV1, _Payload):
    """The committed conflict PRD-003 returns to PRD-002 (§16)."""

    schema_version: Literal["design-conflict.v1"] = "design-conflict.v1"
    conflict_id: Identity
    context_manifest: ArtifactRef
    evidence: tuple[ArtifactRef, ...]
    preparation_revision: _PositiveInt

    @model_validator(mode="after")
    def _evidence_matches_the_draft_ids(self) -> Self:
        stamped = tuple(ref.artifact_id for ref in self.evidence)
        if stamped != self.evidence_artifact_ids:
            raise ValueError("evidence refs must stamp exactly the drafted evidence ids, in order")
        return self
