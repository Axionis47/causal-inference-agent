"""The Design V2 artifact-type registry rows (T-036)."""

from __future__ import annotations

from pathlib import Path

import pytest

from causal.shared.registry import load_artifact_type_registry

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "registries" / "artifact-types.v1.json"
REGISTRY = load_artifact_type_registry(REGISTRY_PATH)

DESIGN_TYPES = (
    "TableSelection", "DesignContextManifest", "DesignIntent", "ColumnTriageRecord",
    "ColumnSemanticCard",
    "MeasurementMap", "RoleEvidence", "CausalContext", "RoleLedger",
    "AgentDesignProposal", "DiagnosticObservationSet", "DesignFactSet", "DiagnosticPlan",
    "DiagnosticReport",
    "CompiledDesign", "CausalGraphView", "GraphViewSet", "CapacityReport", "DesignReviewBundle",
    "UserQuestionPacket", "UserContextAnswer", "TableSelectionDecision",
    "DesignApprovalDecision", "DesignApproval", "DesignOutcome",
)
OUTCOME_STATUSES = (
    "approved", "needs_context", "needs_data", "unsupported", "changes_requested", "declined",
    "system_failure",
)


def test_registry_loads_with_the_design_rows_appended() -> None:
    assert len(DESIGN_TYPES) == 25
    assert len(REGISTRY) == 70


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
        stem, version = row.schema_version.rsplit(".", 1)
        assert row.validator_version == f"{stem}-validator.{version}"


def test_design_outcome_carries_the_explicit_terminal_statuses() -> None:
    assert REGISTRY.lookup("DesignOutcome").terminal_statuses == OUTCOME_STATUSES


def test_review_artifact_readers_are_least_privilege() -> None:
    assert REGISTRY.lookup("CausalGraphView").allowed_reader_components == (
        "design-harness", "presentation-coordinator", "post-analysis")
    assert REGISTRY.lookup("GraphViewSet").allowed_reader_components == ("design-harness", "post-analysis")
    assert REGISTRY.lookup("DesignReviewBundle").allowed_reader_components == (
        "design-harness", "preparation-harness", "post-analysis")
