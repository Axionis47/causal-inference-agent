"""Artifact-type registry (SYSTEM-CONTRACT §3.1; decisions D-009..D-012)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from causal.shared.contracts import Identity, SensitivityClass

__all__ = [
    "ArtifactTypeRegistrationV1",
    "ArtifactTypeRegistry",
    "RegistryError",
    "load_artifact_type_registry",
]

UNSUPPORTED_SCHEMA: Final = "unsupported_schema"
DUPLICATE_REGISTRATION: Final = "duplicate_registration"
INVALID_REGISTRY_FILE: Final = "invalid_registry_file"


class RegistryError(ValueError):
    """A registry operation failed. `code` is a stable contract value."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class ArtifactTypeRegistrationV1(BaseModel):
    """One immutable artifact-type registration (§3.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    artifact_type: Identity
    schema_version: Identity
    producer_component: Identity
    # Other components a registration permits to produce this type; the envelope stamps
    # whichever one actually produced the artifact, never the row's first name (D-086).
    also_produced_by: tuple[Identity, ...] = ()
    allowed_reader_components: Annotated[tuple[Identity, ...], Field(min_length=1)]
    required_parent_types: tuple[Identity, ...]
    optional_parent_types: tuple[Identity, ...]
    sensitivity_class: SensitivityClass
    terminal_statuses: Annotated[tuple[Identity, ...], Field(min_length=1)]
    destinations: tuple[Identity, ...]
    validator_version: Identity


class _RegistryFileV1(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    registry_version: Literal["artifact-types.v1"]
    registrations: tuple[ArtifactTypeRegistrationV1, ...]


class ArtifactTypeRegistry:
    """Immutable lookup over registrations; a missing or ambiguous type fails closed."""

    def __init__(
        self,
        registrations: tuple[ArtifactTypeRegistrationV1, ...],
        registry_version: str = "artifact-types.v1",
    ) -> None:
        self.registry_version = registry_version
        self._by_type: dict[str, ArtifactTypeRegistrationV1] = {}
        for registration in registrations:
            if registration.artifact_type in self._by_type:
                raise RegistryError(
                    f"duplicate registration for {registration.artifact_type!r}",
                    DUPLICATE_REGISTRATION,
                )
            self._by_type[registration.artifact_type] = registration

    def lookup(self, artifact_type: str) -> ArtifactTypeRegistrationV1:
        registration = self._by_type.get(artifact_type)
        if registration is None:
            raise RegistryError(
                f"no registration for artifact type {artifact_type!r}", UNSUPPORTED_SCHEMA
            )
        return registration

    def __len__(self) -> int:
        return len(self._by_type)


def load_artifact_type_registry(path: Path) -> ArtifactTypeRegistry:
    """Load and validate one append-only registry JSON file (D-010, D-012)."""
    try:
        parsed = _RegistryFileV1.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise RegistryError(
            f"invalid registry file {path}: {error}", INVALID_REGISTRY_FILE
        ) from error
    return ArtifactTypeRegistry(parsed.registrations, parsed.registry_version)
