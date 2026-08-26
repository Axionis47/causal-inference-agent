# The estimation method-pack registry: one row per (method, pack version) carrying the estimator,
# required diagnostics, prespecified sensitivity branches, and figure builders a run may execute
# (PRD-004 §8–§12, §14.2, §15).

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Final, Literal, Self

from pydantic import Field, ValidationError, model_validator

from causal.estimation.contracts import DiagnosticSeverity, EstimationError, ValueMap, _Ids, _Row
from causal.shared.contracts import Identity

INVALID_PACK_FILE: Final = "invalid_estimation_pack_file"
UNKNOWN_ESTIMATION_PACK: Final = "unknown_estimation_pack"
DUPLICATE_PACK_ID: Final = "duplicate_estimation_pack_id"
UNREGISTERED_PACK_REFERENCE: Final = "unregistered_pack_reference"
# Severities whose trigger must be decided by a stated threshold before execution (§14.2).
_GUARD_SEVERITIES: Final = ("invalidation_guard", "qualification_guard")


# One required diagnostic and the severity fixed before it runs (§14.1, §14.2).
class DiagnosticRequirementV1(_Row):
    diagnostic_id: Identity
    severity: DiagnosticSeverity
    threshold_params: ValueMap

    @model_validator(mode="after")
    def _a_guard_states_its_threshold(self) -> Self:
        if self.severity in _GUARD_SEVERITIES and not self.threshold_params:
            raise ValueError(f"{self.diagnostic_id} is a {self.severity} with no threshold")
        return self


# One prespecified branch: its exact difference from the primary plan and comparison rule (§15).
class SensitivityBranchV1(_Row):
    branch_id: Identity
    purpose: str
    parameter_delta: ValueMap
    comparison_rule_id: Identity

    @model_validator(mode="after")
    def _a_branch_differs_from_the_primary_plan(self) -> Self:
        if not self.parameter_delta:
            raise ValueError(f"branch {self.branch_id} states no parameter difference")
        return self


# One registered nuisance combination; exactly one is primary, the rest are branches (§10.3).
class NuisanceProfileV1(_Row):
    profile_id: Identity
    role: Literal["primary", "sensitivity"]
    propensity_learner: Identity
    outcome_learner: Identity
    hyperparameters: ValueMap


# One registered estimator variant, such as DiD simultaneous versus staggered adoption (§11.1).
class EstimatorProfileV1(_Row):
    profile_id: Identity
    estimator_id: Identity
    estimator_version: Identity
    parameter_defaults: ValueMap
    required_structure_ids: _Ids


# The §8 manifest one method pack adds for estimation, keyed by (method_id, pack_version).
class EstimationPackV1(_Row):
    method_id: Identity
    pack_version: Identity
    estimator_id: Identity
    estimator_version: Identity
    parameter_defaults: ValueMap
    # Estimator-input role to the dtypes its adapter accepts (§8 role-to-input mapping).
    estimator_input_schema: dict[str, _Ids]
    allowed_mask_rule_ids: _Ids
    uncertainty_method: Identity
    finite_sample_correction: Identity
    confidence_level: Annotated[float, Field(gt=0.0, lt=1.0)]
    estimator_profiles: tuple[EstimatorProfileV1, ...]
    nuisance_profiles: tuple[NuisanceProfileV1, ...]
    fold_count_default: Annotated[int, Field(ge=2)] | None
    required_diagnostics: Annotated[tuple[DiagnosticRequirementV1, ...], Field(min_length=1)]
    sensitivity_branches: tuple[SensitivityBranchV1, ...]
    figure_builder_ids: _Ids
    not_estimable_rule_ids: _Ids
    invalidation_rule_ids: _Ids

    # Every required diagnostic id to the severity the plan must copy (§6.1, §14.2).
    def severities(self) -> dict[str, DiagnosticSeverity]:
        return {row.diagnostic_id: row.severity for row in self.required_diagnostics}

    def branch(self, branch_id: str) -> SensitivityBranchV1:
        for row in self.sensitivity_branches:
            if row.branch_id == branch_id:
                return row
        raise EstimationError(f"{self.method_id} registers no branch {branch_id!r}",
                              UNREGISTERED_PACK_REFERENCE)

    # The registered estimator variant one plan binds, such as DiD staggered adoption (§11.1).
    def estimator_profile(self, profile_id: str) -> EstimatorProfileV1:
        for row in self.estimator_profiles:
            if row.profile_id == profile_id:
                return row
        raise EstimationError(f"{self.method_id} registers no profile {profile_id!r}",
                              UNREGISTERED_PACK_REFERENCE)

    # The one nuisance profile a primary run may fit; alternatives are branches (§10.3).
    def primary_nuisance_profile(self) -> NuisanceProfileV1 | None:
        return next((row for row in self.nuisance_profiles if row.role == "primary"), None)


# Ids declared twice, and parameter deltas naming a profile or mask rule this pack never
# registered, fail closed before any run can bind the pack (§8 last paragraph).
def _check_pack_references(pack: EstimationPackV1) -> None:
    registered = {
        "nuisance_profile_id": {row.profile_id for row in pack.nuisance_profiles},
        "adoption_profile_id": {row.profile_id for row in pack.estimator_profiles},
        "mask_rule_id": set(pack.allowed_mask_rule_ids)}
    declared = [row.diagnostic_id for row in pack.required_diagnostics]
    declared += [row.branch_id for row in pack.sensitivity_branches]
    declared += [row.profile_id for row in pack.nuisance_profiles]
    declared += [row.profile_id for row in pack.estimator_profiles]
    if len(set(declared)) != len(declared):
        raise EstimationError(f"{pack.method_id} declares one id twice", DUPLICATE_PACK_ID)
    deltas = (pack.parameter_defaults, *(row.parameter_delta for row in pack.sensitivity_branches))
    for delta in deltas:
        for key, known in registered.items():
            if (value := delta.get(key)) is not None and value not in known:
                raise EstimationError(f"{pack.method_id} names unregistered {key} {value!r}",
                                      UNREGISTERED_PACK_REFERENCE)
    if pack.nuisance_profiles and pack.primary_nuisance_profile() is None:
        raise EstimationError(f"{pack.method_id} registers no primary nuisance profile",
                              UNREGISTERED_PACK_REFERENCE)


class _PackFileV1(_Row):
    registry_version: Literal["method-pack-estimation.v1"]
    packs: tuple[EstimationPackV1, ...]


class EstimationPackRegistry:
    # The §8 estimation registry; a row no approved design method pack backs fails closed.

    registry_version: Final = "method-pack-estimation.v1"

    def __init__(self, packs: tuple[EstimationPackV1, ...], known: set[tuple[str, str]]) -> None:
        self._by_key: dict[tuple[str, str], EstimationPackV1] = {}
        for pack in packs:
            key = (pack.method_id, pack.pack_version)
            if key in self._by_key:
                raise EstimationError(f"duplicate estimation pack {key}", DUPLICATE_PACK_ID)
            if key not in known:
                raise EstimationError(f"no design method pack for {key}", UNKNOWN_ESTIMATION_PACK)
            _check_pack_references(pack)
            self._by_key[key] = pack
        if missing := sorted(known - set(self._by_key)):
            raise EstimationError(f"estimation packs missing for {missing}",
                                  UNKNOWN_ESTIMATION_PACK)

    def get(self, method_id: str, pack_version: str) -> EstimationPackV1:
        pack = self._by_key.get((method_id, pack_version))
        if pack is None:
            raise EstimationError(f"no estimation pack for {(method_id, pack_version)}",
                                  UNKNOWN_ESTIMATION_PACK)
        return pack

    def all(self) -> tuple[EstimationPackV1, ...]:
        return tuple(self._by_key.values())

    def __len__(self) -> int:
        return len(self._by_key)


# The (method_id, pack_version) pairs of the frozen design registry, read as raw data.
def _design_pack_keys(method_packs_path: Path) -> set[tuple[str, str]]:
    try:
        document = json.loads(method_packs_path.read_text(encoding="utf-8"))
        return {(row["method_id"], row["pack_version"]) for row in document["packs"]}
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise EstimationError(f"invalid registry file {method_packs_path}: {error}",
                              INVALID_PACK_FILE) from error


# Load the estimation packs and bind every row to the frozen design method packs (§8).
def load_estimation_packs(path: Path, method_packs_path: Path) -> EstimationPackRegistry:
    try:
        parsed = _PackFileV1.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise EstimationError(f"invalid registry file {path}: {error}",
                              INVALID_PACK_FILE) from error
    return EstimationPackRegistry(parsed.packs, _design_pack_keys(method_packs_path))
