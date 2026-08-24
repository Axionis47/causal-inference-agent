"""Tests for the artifact-type registry (T-004)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from causal.shared.contracts import SensitivityClass
from causal.shared.registry import (
    ArtifactTypeRegistrationV1,
    ArtifactTypeRegistry,
    RegistryError,
    load_artifact_type_registry,
)

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "registries" / "artifact-types.v1.json"


def registration_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "artifact_type": "IntakeOutcome",
        "schema_version": "intake-outcome.v1",
        "producer_component": "intake-coordinator",
        "allowed_reader_components": ("design-harness",),
        "required_parent_types": ("QuestionRecord",),
        "optional_parent_types": (),
        "sensitivity_class": SensitivityClass.INTERNAL,
        "terminal_statuses": ("usable",),
        "destinations": ("design",),
        "validator_version": "intake-outcome-validator.v1",
    }
    base.update(overrides)
    return base


def make_registration(**overrides: object) -> ArtifactTypeRegistrationV1:
    return ArtifactTypeRegistrationV1(**registration_kwargs(**overrides))  # type: ignore[arg-type]


class TestRegistrationModel:
    def test_extra_field_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_registration(surprise="x")

    def test_empty_readers_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_registration(allowed_reader_components=())

    def test_empty_terminal_statuses_rejected(self) -> None:
        with pytest.raises(ValidationError):
            make_registration(terminal_statuses=())


class TestRegistry:
    def test_lookup_missing_is_unsupported_schema(self) -> None:
        registry = ArtifactTypeRegistry((make_registration(),))
        with pytest.raises(RegistryError) as excinfo:
            registry.lookup("NoSuchType")
        assert excinfo.value.code == "unsupported_schema"

    def test_duplicate_registration_rejected_at_load(self) -> None:
        with pytest.raises(RegistryError) as excinfo:
            ArtifactTypeRegistry((make_registration(), make_registration()))
        assert excinfo.value.code == "duplicate_registration"


class TestRealRegistryFile:
    def test_loads_and_looks_up_seeded_rows(self) -> None:
        registry = load_artifact_type_registry(REGISTRY_PATH)
        assert len(registry) == 8
        intake = registry.lookup("IntakeOutcome")
        assert intake.producer_component == "intake-coordinator"
        assert intake.terminal_statuses == ("usable", "partial", "refused")
        frame = registry.lookup("RunnableFrameContract")
        assert frame.required_parent_types == ("ExperimentDesign",)
        assert frame.destinations == ("preparation", "estimation")

    def test_intake_lineage_chain(self) -> None:
        registry = load_artifact_type_registry(REGISTRY_PATH)
        assert registry.lookup("QuestionRecord").required_parent_types == ()
        capture = registry.lookup("KaggleCapture")
        assert capture.required_parent_types == ("QuestionRecord",)
        assert capture.sensitivity_class.value == "restricted"
        assert capture.allowed_reader_components == ("intake-coordinator",)
        assert registry.lookup("SourceManifest").required_parent_types == ("KaggleCapture",)
        assert registry.lookup("TableProfile").required_parent_types == ("SourceManifest",)
        assert registry.lookup("EvidenceBundle").required_parent_types == ("KaggleCapture",)
        assert registry.lookup("SemanticMap").required_parent_types == ("EvidenceBundle",)

    def test_malformed_json_is_invalid_registry_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "broken.json"
        bad.write_text("{not json")
        with pytest.raises(RegistryError) as excinfo:
            load_artifact_type_registry(bad)
        assert excinfo.value.code == "invalid_registry_file"

    def test_wrong_shape_is_invalid_registry_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "shape.json"
        bad.write_text(json.dumps({"registry_version": "artifact-types.v1"}))
        with pytest.raises(RegistryError) as excinfo:
            load_artifact_type_registry(bad)
        assert excinfo.value.code == "invalid_registry_file"

    def test_wrong_version_is_invalid_registry_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "version.json"
        bad.write_text(json.dumps({"registry_version": "artifact-types.v2", "registrations": []}))
        with pytest.raises(RegistryError) as excinfo:
            load_artifact_type_registry(bad)
        assert excinfo.value.code == "invalid_registry_file"
