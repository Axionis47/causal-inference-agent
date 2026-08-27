"""The nine preparation artifact-type registry rows (T-015 §1.5; PRD-003 §24.2; SC §3.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from causal.shared.registry import load_artifact_type_registry

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "registries" / "artifact-types.v1.json"
REGISTRY = load_artifact_type_registry(REGISTRY_PATH)

PREPARATION_TYPES = (
    "PreparationContextManifest", "StabilizationRecord", "StabilizedFrame", "PreparationPlan",
    "ExecutionReceiptBundle", "PreparedFrame", "PreparedFrameBundle", "PreparationOutcome",
    "DesignConflict",
)
# PRD-003 §5.2: the preparation stage's five terminal statuses, in the order the PRD lists them.
OUTCOME_STATUSES = (
    "prepared", "design_conflict", "not_runnable", "failed_observability", "failed",
)
# SC §3.1 amended row: the bundle names every consolidated parent PRD-004 reads.
BUNDLE_PARENTS = (
    "TableSelection", "ExperimentDesign", "RunnableFrameContract", "DeliveryCapacityCheck",
    "StabilizationRecord", "StabilizedFrame", "PreparedFrame", "ExecutionReceiptBundle",
)
ESTIMATION_READABLE = (
    "StabilizationRecord", "StabilizedFrame", "ExecutionReceiptBundle", "PreparedFrame",
    "PreparedFrameBundle", "PreparationOutcome",
)


def test_registry_loads_with_the_preparation_rows_appended() -> None:
    assert len(PREPARATION_TYPES) == 9
    assert REGISTRY.registry_version == "artifact-types.v1"
    assert len(REGISTRY) == 57


@pytest.mark.parametrize("artifact_type", PREPARATION_TYPES)
class TestPreparationRow:
    def test_resolves_and_is_produced_by_the_preparation_harness(self, artifact_type: str) -> None:
        assert REGISTRY.lookup(artifact_type).producer_component == "preparation-harness"

    def test_every_parent_type_resolves(self, artifact_type: str) -> None:
        row = REGISTRY.lookup(artifact_type)
        for parent in (*row.required_parent_types, *row.optional_parent_types):
            assert REGISTRY.lookup(parent).artifact_type == parent

    def test_validator_version_follows_the_schema_stem(self, artifact_type: str) -> None:
        row = REGISTRY.lookup(artifact_type)
        stem = row.schema_version.removesuffix(".v1")
        assert row.validator_version == f"{stem}-validator.v1"

    def test_the_preparation_harness_may_read_what_it_writes(self, artifact_type: str) -> None:
        assert "preparation-harness" in REGISTRY.lookup(artifact_type).allowed_reader_components

    def test_estimation_reads_exactly_the_handoff_types(self, artifact_type: str) -> None:
        row = REGISTRY.lookup(artifact_type)
        readable = artifact_type in ESTIMATION_READABLE
        assert ("estimation-harness" in row.allowed_reader_components) is readable
        assert ("estimation" in row.destinations) is readable


def test_the_context_manifest_requires_the_four_handoff_parents() -> None:
    row = REGISTRY.lookup("PreparationContextManifest")
    assert row.required_parent_types == (
        "TableSelection", "ExperimentDesign", "RunnableFrameContract", "DeliveryCapacityCheck",
    )
    assert row.destinations == ("preparation",)


def test_the_prepared_bundle_carries_the_consolidated_parents_and_one_status() -> None:
    row = REGISTRY.lookup("PreparedFrameBundle")
    assert row.required_parent_types == BUNDLE_PARENTS
    assert row.terminal_statuses == ("prepared",)
    assert row.destinations == ("estimation",)


def test_the_outcome_carries_the_five_terminal_statuses_and_returns_to_design() -> None:
    row = REGISTRY.lookup("PreparationOutcome")
    assert row.terminal_statuses == OUTCOME_STATUSES
    assert row.optional_parent_types == ("PreparedFrameBundle", "DesignConflict")
    assert row.destinations == ("preparation", "estimation", "design")


def test_a_design_conflict_goes_back_to_the_design_harness() -> None:
    row = REGISTRY.lookup("DesignConflict")
    assert row.allowed_reader_components == ("preparation-harness", "design-harness")
    assert row.terminal_statuses == ("open", "resolved", "refused")
    assert row.destinations == ("preparation", "design")
    # D-084/D-085: PRD-004 produces the same conflict type off its own context manifest, so
    # neither manifest can be required and both are registered as the permitted parent.
    assert row.required_parent_types == ()
    assert row.optional_parent_types == (
        "PreparationContextManifest", "EstimationContextManifest")


def test_the_two_frames_carry_row_data_and_are_restricted() -> None:
    for artifact_type in ("StabilizedFrame", "PreparedFrame"):
        assert REGISTRY.lookup(artifact_type).sensitivity_class == "restricted"


def test_the_plan_takes_the_stabilization_record_as_an_optional_parent() -> None:
    row = REGISTRY.lookup("PreparationPlan")
    assert row.required_parent_types == ("PreparationContextManifest",)
    assert row.optional_parent_types == ("StabilizationRecord",)
