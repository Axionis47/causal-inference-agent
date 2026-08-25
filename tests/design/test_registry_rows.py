"""The eighteen design artifact-type registry rows (T-009 §4, §6; PRD-002 §6)."""

from __future__ import annotations

from pathlib import Path

import pytest

from causal.shared.registry import load_artifact_type_registry

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "registries" / "artifact-types.v1.json"
REGISTRY = load_artifact_type_registry(REGISTRY_PATH)

DESIGN_TYPES = (
    "TableSelection", "DesignContextManifest", "DesignIntent", "ColumnSemanticCard",
    "MeasurementMap", "RoleEvidence", "CausalContext", "RoleLedger",
    "PreRepairFeasibilityReport", "ExperimentDesign", "CausalGraphView", "DeliveryCapacityCheck",
    "UserQuestionPacket", "UserContextAnswer", "TableSelectionDecision",
    "DesignApprovalDecision", "DesignApproval", "DesignOutcome",
)
# PRD-002 §6: the design stage's seven terminal statuses, in the order the PRD lists them.
OUTCOME_STATUSES = (
    "approved", "needs_context", "changes_requested", "declined", "refused",
    "failed_observability", "failed",
)


def test_registry_loads_with_the_design_rows_appended() -> None:
    assert len(DESIGN_TYPES) == 18
    assert len(REGISTRY) == 26


@pytest.mark.parametrize("artifact_type", DESIGN_TYPES)
class TestDesignRow:
    def test_resolves_and_is_produced_by_the_design_harness(self, artifact_type: str) -> None:
        assert REGISTRY.lookup(artifact_type).producer_component == "design-harness"

    def test_every_parent_type_resolves(self, artifact_type: str) -> None:
        row = REGISTRY.lookup(artifact_type)
        for parent in (*row.required_parent_types, *row.optional_parent_types):
            assert REGISTRY.lookup(parent).artifact_type == parent

    def test_validator_version_follows_the_schema_stem(self, artifact_type: str) -> None:
        row = REGISTRY.lookup(artifact_type)
        stem = row.schema_version.removesuffix(".v1")
        assert row.validator_version == f"{stem}-validator.v1"


def test_design_outcome_carries_the_seven_terminal_statuses() -> None:
    assert REGISTRY.lookup("DesignOutcome").terminal_statuses == OUTCOME_STATUSES
