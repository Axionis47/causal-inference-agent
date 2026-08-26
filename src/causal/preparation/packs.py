"""Method-pack preparation overlay: one sidecar row per design pack (PRD-003 §12, §13; D-057)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import Field, ValidationError, model_validator

from causal.preparation.contracts import _Row
from causal.preparation.plans import FitScope
from causal.shared.contracts import Identity
from causal.shared.registry import INVALID_REGISTRY_FILE, RegistryError

__all__ = [
    "UNKNOWN_METHOD_PACK", "ImputationTargetV1", "PreparationPackRegistry", "PreparationPackV1",
    "eligibility_vocabulary", "load_preparation_packs",
]

UNKNOWN_METHOD_PACK: Final = "unknown_method_pack"

_Ids = Annotated[tuple[Identity, ...], Field(min_length=1)]
_MinimumCount = Annotated[int, Field(ge=1)]


class ImputationTargetV1(_Row):
    """One permitted imputation target: the role, its fit scope, and the registered strategy."""

    role: Identity
    fit_scope: FitScope
    strategy_id: Identity
    requires_missingness_indicator: bool


class PreparationPackV1(_Row):
    """The §13 preparation contract one method pack adds, keyed by (method_id, pack_version)."""

    method_id: Identity
    pack_version: Identity
    permitted_disposition_rule_ids: _Ids
    protected_roles: _Ids
    required_observed_role_rules: tuple[Identity, ...]
    minimum_rows: _MinimumCount
    minimum_unique_units: _MinimumCount
    required_structure_gates: _Ids
    cell_support_gates: tuple[Identity, ...]
    dimension_impact_dimensions: _Ids
    invalidation_rule_ids: _Ids
    permitted_repair_operation_ids: _Ids
    permitted_imputation_targets: tuple[ImputationTargetV1, ...]
    required_missingness_indicators: tuple[Identity, ...]
    required_poststabilization_diagnostic_ids: _Ids
    required_postrepair_diagnostic_ids: _Ids
    prepared_frame_schema_id: Identity
    estimator_input_contract_id: Identity

    @model_validator(mode="after")
    def _protected_roles_are_never_imputed(self) -> Self:
        targets = {target.role for target in self.permitted_imputation_targets}
        if overlap := sorted(targets & set(self.protected_roles)):
            raise ValueError(f"{self.method_id} would impute protected roles: {overlap}")
        return self


class _OverlayFileV1(_Row):
    registry_version: Literal["method-pack-preparation.v1"]
    packs: tuple[PreparationPackV1, ...]


def _design_pack_keys(method_packs_path: Path) -> set[tuple[str, str]]:
    """The (method_id, pack_version) pairs of the frozen design registry, read as raw data."""
    try:
        document = json.loads(method_packs_path.read_text(encoding="utf-8"))
        return {(row["method_id"], row["pack_version"]) for row in document["packs"]}
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise RegistryError(
            f"invalid registry file {method_packs_path}: {error}", INVALID_REGISTRY_FILE
        ) from error


class PreparationPackRegistry:
    """The preparation overlay; a row that no design pack backs fails closed."""

    registry_version: Final = "method-pack-preparation.v1"

    def __init__(self, packs: tuple[PreparationPackV1, ...], known: set[tuple[str, str]]) -> None:
        self._by_key: dict[tuple[str, str], PreparationPackV1] = {}
        for pack in packs:
            key = (pack.method_id, pack.pack_version)
            if key in self._by_key:
                raise RegistryError(f"duplicate overlay row {key}", INVALID_REGISTRY_FILE)
            if key not in known:
                raise RegistryError(f"no method pack for {key}", UNKNOWN_METHOD_PACK)
            self._by_key[key] = pack
        if missing := sorted(known - set(self._by_key)):
            raise RegistryError(f"overlay rows missing for {missing}", UNKNOWN_METHOD_PACK)

    def get(self, method_id: str, pack_version: str) -> PreparationPackV1:
        pack = self._by_key.get((method_id, pack_version))
        if pack is None:
            raise RegistryError(
                f"no preparation overlay for {(method_id, pack_version)}", UNKNOWN_METHOD_PACK
            )
        return pack

    def all(self) -> tuple[PreparationPackV1, ...]:
        return tuple(self._by_key.values())

    def __len__(self) -> int:
        return len(self._by_key)


def eligibility_vocabulary(method_packs_path: Path, method_id: str) -> tuple[str, ...]:
    """The T-011 `eligibility_rule_vocabulary` of one design pack, read as raw registry data."""
    try:
        document = json.loads(method_packs_path.read_text(encoding="utf-8"))
        for pack in document["packs"]:
            if pack["method_id"] == method_id:
                return tuple(pack["eligibility_rule_vocabulary"])
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise RegistryError(f"invalid registry file {method_packs_path}: {error}",
                            INVALID_REGISTRY_FILE) from error
    raise RegistryError(f"no method pack for {method_id}", UNKNOWN_METHOD_PACK)


def load_preparation_packs(path: Path, method_packs_path: Path) -> PreparationPackRegistry:
    """Load the overlay and bind every row to the frozen design method packs (§13)."""
    try:
        parsed = _OverlayFileV1.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise RegistryError(
            f"invalid registry file {path}: {error}", INVALID_REGISTRY_FILE
        ) from error
    return PreparationPackRegistry(parsed.packs, _design_pack_keys(method_packs_path))
