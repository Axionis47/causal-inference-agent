"""Preparation entry gate: handoff acceptance, the §4 checks, manifest compilation (PRD-003 §4, §7.2)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from causal.preparation.contracts import PREPARATION_REGISTRY_KEYS, PreparationContextManifestV1
from causal.preparation.packs import PreparationPackV1
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactEnvelopeV1, ArtifactRef, HandoffManifestV1
from causal.shared.events import OperationalEventV1
from causal.shared.handoff import UNKNOWN_HANDOFF, EventFactory, HandoffGate, HandoffStore
from causal.shared.persistence import PersistenceError
from causal.shared.readers import ObjectReader, ProductsReader

__all__ = [
    "COMPONENT", "ENTRY_TYPES", "EntryInputs", "EntryPolicy", "HandoffAcceptance",
    "ManifestCommitter", "PreparationEntryError", "accept_handoff", "commit_manifest",
    "compile_context_manifest", "handoff_id", "read_entries", "validate_entry",
]

COMPONENT, APPROVED, COMPUTED = "preparation-harness", "approved", "computed"
ENTRY_VALIDATION_FAILED, ENTRY_UNREADABLE = "entry_validation_failed", "entry_unreadable"
HANDOFF_REJECTED, HANDOFF_ENTRIES_CHANGED = "handoff_rejected", "handoff_entries_changed"
# The nine §4 conditions, one stable code family each; every failing family is reported at once.
DESIGN_NOT_APPROVED, APPROVAL_NOT_BOUND = "design_not_approved", "approval_not_bound"
MISSING_ARTIFACT, ENTRY_HASH_MISMATCH = "missing_artifact", "entry_hash_mismatch"
UNEXPECTED_ENTRY_TYPE, BROKEN_HASH_CHAIN = "unexpected_entry_type", "broken_hash_chain"
FRAME_CONTRACT_NOT_BOUND, CSV_HASH_MISMATCH = "frame_contract_not_bound", "csv_hash_mismatch"
UNSUPPORTED_REGISTRY_VERSION, UNREGISTERED_RULE_ID = "unsupported_registry_version", "unregistered_rule_id"
PREREPAIR_DIAGNOSTIC_MISSING, UPSTREAM_MUTATION_MARKER = "prerepair_diagnostic_missing", "upstream_mutation_marker"
PREREPAIR_HANDLING_NOT_APPROVED = "prerepair_handling_not_approved"
CAPACITY_CHECK_NOT_PASS, CAPACITY_BINDING_MISMATCH = "capacity_check_not_pass", "capacity_binding_mismatch"
# The four handoff entries, in the order PRD-002 builds them (design's `open_design_handoff`).
ENTRY_TYPES: Final = ("TableSelection", "ExperimentDesign", "RunnableFrameContract",
                      "DeliveryCapacityCheck")
# Any upstream payload key that would declare the immutable source table mutated (§4 condition 8).
MUTATION_MARKERS: Final = ("source_table_mutated", "source_mutated", "table_mutated")


class PreparationEntryError(ValueError):
    """A preparation entry step failed. `code` is stable; `detail_codes` carries every family."""

    def __init__(self, message: str, code: str, detail_codes: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.code, self.detail_codes = code, detail_codes


# The one `ArtifactCommitter` method the entry gate needs (SC §8.2 flush-gated commit).
class ManifestCommitter(Protocol):
    def commit(self, envelope: ArtifactEnvelopeV1, payload: dict[str, object],
               event: OperationalEventV1) -> ArtifactEnvelopeV1: ...


# The opened handoff plus whether this run replayed an already accepted one.
@dataclass(frozen=True)
class HandoffAcceptance:
    manifest: HandoffManifestV1
    replayed: bool


# The DesignOutcome and the four entry payloads, read as data (no `causal.design` import).
@dataclass(frozen=True)
class EntryInputs:
    outcome: Mapping[str, Any]
    selection: Mapping[str, Any]
    design: Mapping[str, Any]
    contract: Mapping[str, Any]
    capacity: Mapping[str, Any]
    envelopes: tuple[ArtifactEnvelopeV1 | None, ...]


# The registered vocabularies and approved handling the §4 checks measure against.
@dataclass(frozen=True)
class EntryPolicy:
    pack: PreparationPackV1
    registry_versions: Mapping[str, str]
    eligibility_vocabulary: tuple[str, ...] = ()
    imputation_strategy_ids: tuple[str, ...] = ()
    prerepair_statuses: Mapping[str, str] = field(default_factory=dict)
    approved_handling: frozenset[str] = frozenset()


def handoff_id(analysis_id: str, outcome_content_hash: str) -> str:
    """The id design stamps: `ho:{analysis_id}:{outcome_hash16}` (recomputed, never imported)."""
    return f"ho:{analysis_id}:{outcome_content_hash[:16]}"


def accept_handoff(gate: HandoffGate, store: HandoffStore, manifest: HandoffManifestV1,
                   event_factory: EventFactory) -> HandoffAcceptance:
    """Load first, then check: a recorded handoff is never re-recorded (duplicate_handoff)."""
    try:
        recorded = store.load(manifest.handoff_id)
    except PersistenceError as error:
        if error.code != UNKNOWN_HANDOFF:
            raise
        result = gate.accept(manifest, COMPONENT, frozenset({APPROVED}), event_factory)
        if not result.accepted:
            raise PreparationEntryError(f"handoff {manifest.handoff_id} refused",
                                        HANDOFF_REJECTED, result.error_codes) from None
        return HandoffAcceptance(manifest, replayed=False)
    if recorded.receiver_validation_result != "accepted":
        raise PreparationEntryError(f"handoff {manifest.handoff_id} was refused before",
                                    HANDOFF_REJECTED, tuple(recorded.receiver_error_codes))
    if recorded.entries != manifest.entries:
        raise PreparationEntryError(f"handoff {manifest.handoff_id} entries changed",
                                    HANDOFF_ENTRIES_CHANGED)
    return HandoffAcceptance(recorded, replayed=True)


def _envelope(products: ProductsReader, artifact_id: str) -> ArtifactEnvelopeV1 | None:
    try:
        return products.load_envelope(artifact_id)
    except Exception:  # noqa: BLE001 -- any read failure is a missing artifact
        return None


def _body(objects: ObjectReader, envelope: ArtifactEnvelopeV1 | None) -> Mapping[str, Any]:
    if envelope is None:
        return {}
    try:
        parsed = json.loads(objects.get(envelope.payload_locator))
    except Exception:  # noqa: BLE001 -- an unreadable object is an unreadable entry
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def _ref_id(payload: Mapping[str, Any], key: str) -> str | None:
    held = payload.get(key)
    return str(held["artifact_id"]) if isinstance(held, Mapping) else None


def read_entries(manifest: HandoffManifestV1, outcome_artifact_id: str,
                 products: ProductsReader, objects: ObjectReader) -> EntryInputs:
    """Resolve the outcome and the four entries; an unreadable entry becomes an empty payload."""
    if len(manifest.entries) != len(ENTRY_TYPES):
        raise PreparationEntryError("a preparation handoff carries four entries", ENTRY_UNREADABLE)
    envelopes = tuple(_envelope(products, entry.artifact_id) for entry in manifest.entries)
    selection, design, contract, capacity = (_body(objects, found) for found in envelopes)
    return EntryInputs(_body(objects, _envelope(products, outcome_artifact_id)), selection,
                       design, contract, capacity, envelopes)


# §4 conditions 1 and 2: an approved outcome whose entries and parents still hash true.
def _lineage_codes(manifest: HandoffManifestV1, inputs: EntryInputs,
                   products: ProductsReader) -> set[str]:
    codes: set[str] = set()
    if inputs.outcome.get("status") != APPROVED or manifest.originating_outcome != APPROVED:
        codes.add(DESIGN_NOT_APPROVED)
    approval = _ref_id(inputs.outcome, "approval")
    if approval not in manifest.approval_ids or not _ref_id(inputs.outcome, "causal_graph_view"):
        codes.add(APPROVAL_NOT_BOUND)
    for ref, envelope, expected in zip(manifest.entries, inputs.envelopes, ENTRY_TYPES,
                                       strict=True):
        if envelope is None:
            codes.add(MISSING_ARTIFACT)
            continue
        codes |= {ENTRY_HASH_MISMATCH} if envelope.content_hash != ref.content_hash else set()
        codes |= {UNEXPECTED_ENTRY_TYPE} if envelope.artifact_type != expected else set()
        for parent in envelope.parent_artifacts:
            committed = _envelope(products, parent.artifact_id)
            if committed is None or committed.content_hash != parent.content_hash:
                codes.add(BROKEN_HASH_CHAIN)
    return codes


# §4 conditions 3 and 9: the contract, the outcome, and the capacity check name one design.
def _binding_codes(manifest: HandoffManifestV1, inputs: EntryInputs) -> set[str]:
    selection, design, contract, capacity = manifest.entries
    bound = ((_ref_id(inputs.outcome, "experiment_design"), design.artifact_id),
             (_ref_id(inputs.outcome, "runnable_frame_contract"), contract.artifact_id),
             (_ref_id(inputs.design, "selected_csv"), selection.artifact_id),
             (inputs.contract.get("experiment_design_hash"), design.content_hash))
    pinned = ((inputs.capacity.get("method_id"), inputs.design.get("method_id")),
              (inputs.capacity.get("visualization_catalog_version"),
               inputs.design.get("visualization_catalog_version")),
              (inputs.capacity.get("capacity_registry_version"),
               inputs.design.get("capacity_registry_version")),
              (_ref_id(inputs.design, "capacity_check"), capacity.artifact_id),
              (_ref_id(inputs.outcome, "capacity_check"), capacity.artifact_id))
    codes = {FRAME_CONTRACT_NOT_BOUND} if any(a != b for a, b in bound) else set()
    codes |= {CAPACITY_CHECK_NOT_PASS} if inputs.capacity.get("status") != "pass" else set()
    return codes | ({CAPACITY_BINDING_MISMATCH}
                    if any(a is None or a != b for a, b in pinned) else set())


# §4 conditions 4 and 8: the selected CSV still hashes true and nothing claims a mutation.
def _source_codes(inputs: EntryInputs, objects: ObjectReader) -> set[str]:
    digest = inputs.selection.get("resource_sha256")
    try:
        observed = hashlib.sha256(
            objects.get(str(inputs.selection.get("resource_object_locator")))).hexdigest()
    except Exception:  # noqa: BLE001 -- an unreadable source object cannot match its hash
        observed = ""
    codes = {CSV_HASH_MISMATCH} if observed != digest else set()
    payloads = (inputs.outcome, inputs.selection, inputs.design, inputs.contract, inputs.capacity)
    return codes | ({UPSTREAM_MUTATION_MARKER}
                    if any(row.get(key) for row in payloads for key in MUTATION_MARKERS) else set())


# §4 conditions 5, 6, and 7: supported versions, registered rule ids, handled diagnostics.
def _registry_codes(inputs: EntryInputs, policy: EntryPolicy) -> set[str]:
    versions, permitted = policy.registry_versions, set(policy.pack.permitted_disposition_rule_ids)
    codes: set[str] = set()
    if set(versions) != set(PREPARATION_REGISTRY_KEYS) or not all(versions.values()) or not (
        isinstance(inputs.design.get("registry_versions"), Mapping)
    ):
        codes.add(UNSUPPORTED_REGISTRY_VERSION)
    families = ((inputs.contract.get("eligibility_rules") or (),
                 permitted | set(policy.eligibility_vocabulary)),
                (inputs.contract.get("exclusion_reason_vocabulary") or (), permitted),
                (policy.imputation_strategy_ids,
                 {target.strategy_id for target in policy.pack.permitted_imputation_targets}))
    if any(set(found) - registered for found, registered in families):
        codes.add(UNREGISTERED_RULE_ID)
    for diagnostic_id in inputs.design.get("required_prerepair_diagnostics") or ():
        status = policy.prerepair_statuses.get(str(diagnostic_id))
        if status is None:
            codes.add(PREREPAIR_DIAGNOSTIC_MISSING)
        elif status != COMPUTED and status not in policy.approved_handling:
            codes.add(PREREPAIR_HANDLING_NOT_APPROVED)
    return codes


def validate_entry(manifest: HandoffManifestV1, inputs: EntryInputs, policy: EntryPolicy, *,
                   products: ProductsReader, objects: ObjectReader) -> None:
    """The §4 nine entry checks; every failing condition is reported at once (fail closed)."""
    codes = (_lineage_codes(manifest, inputs, products) | _binding_codes(manifest, inputs)
             | _source_codes(inputs, objects) | _registry_codes(inputs, policy))
    if codes:
        raise PreparationEntryError("the preparation entry gate refused the design handoff",
                                    ENTRY_VALIDATION_FAILED, tuple(sorted(codes)))


# Column-to-concept and column-to-role maps plus the protected set, from the design side.
def _columns(role_ledger: Mapping[str, Any], measurement_map: Mapping[str, Any],
             pack: PreparationPackV1) -> tuple[dict[str, str], dict[str, str], tuple[str, ...]]:
    concepts: dict[str, str] = {}
    roles: dict[str, str] = {}
    for link in measurement_map.get("links") or ():
        concepts.setdefault(str(link["column_name"]), str(link["concept_id"]))
    for claim in role_ledger.get("claims") or ():
        for column in claim.get("column_refs") or ():
            roles.setdefault(str(column), str(claim["role"]))
    return concepts, roles, tuple(sorted(
        column for column, role in roles.items() if role in pack.protected_roles))


def compile_context_manifest(
    inputs: EntryInputs, entries: tuple[ArtifactRef, ...], *, pack: PreparationPackV1,
    role_ledger: Mapping[str, Any], measurement_map: Mapping[str, Any],
    causal_context: ArtifactRef, question_id: str, parser_profile_id: str,
    registry_versions: Mapping[str, str], recipient_map: Mapping[str, Sequence[str]],
) -> PreparationContextManifestV1:
    """Hydrate the §7.2 manifest from the four entries and the design-side semantic artifacts."""
    design, contract = inputs.design, inputs.contract
    frame = design.get("frame") or {}
    concepts, roles, protected = _columns(role_ledger, measurement_map, pack)
    forbidden = set(contract.get("imputation_forbidden") or ()) | set(protected)
    strategies = {target.strategy_id for target in pack.permitted_imputation_targets}
    upstream = {"entries": [ref.model_dump(mode="json") for ref in entries],
                "measurement_map": design.get("measurement_map"),
                "role_ledger": design.get("role_ledger"),
                "causal_context": causal_context.model_dump(mode="json")}
    return PreparationContextManifestV1(
        selected_csv=entries[0], parser_profile_id=parser_profile_id, question_id=question_id,
        source_object_locator=str(inputs.selection["resource_object_locator"]),
        population_id=str(frame["population"]), timeframe_id=str(frame["timeframe"]),
        treatment_id=str(frame["treatment"]), outcome_id=str(frame["outcome"]),
        comparator_id=str(design["comparator"]), estimand_id=str(design["estimand"]),
        method_id=str(design["method_id"]),
        method_pack_version=str(design["method_pack_version"]),
        measurement_map=ArtifactRef.model_validate(design["measurement_map"]),
        role_ledger=ArtifactRef.model_validate(design["role_ledger"]),
        column_concepts=concepts, column_roles=roles, protected_columns=protected,
        permitted_repair_columns=tuple(sorted(set(roles) - forbidden)),
        permitted_imputation_columns=tuple(
            column for column in contract.get("imputation_permitted") or ()
            if column not in protected),
        approved_grain=str(contract["output_grain"]),
        key_columns=tuple(str(column) for column in contract["key_columns"]),
        prepared_frame_schema_id=pack.prepared_frame_schema_id,
        eligibility_rule_ids=tuple(contract.get("eligibility_rules") or ()),
        unusable_row_rule_ids=tuple(contract.get("exclusion_reason_vocabulary") or ()),
        structural_requirements=tuple(pack.required_structure_gates),
        permitted_operation_ids=tuple(sorted(
            set(pack.permitted_repair_operation_ids) | strategies)),
        permitted_diagnostic_ids=tuple(sorted(
            set(pack.required_poststabilization_diagnostic_ids)
            | set(pack.required_postrepair_diagnostic_ids)
            | set(contract.get("required_final_diagnostics") or ()))),
        deletion_impact_dimensions=tuple(contract.get("deletion_impact_dimensions") or ())
        or tuple(pack.dimension_impact_dimensions),
        registry_versions=dict(registry_versions),
        recipient_map={key: tuple(value) for key, value in recipient_map.items()},
        manifest_hash=content_hash(upstream))


def commit_manifest(committer: ManifestCommitter, envelope: ArtifactEnvelopeV1,
                    manifest: PreparationContextManifestV1,
                    event: OperationalEventV1) -> ArtifactEnvelopeV1:
    """Commit the frozen manifest through the caller's flush-gated committer (SC §8.2)."""
    return committer.commit(envelope, manifest.canonical_payload(), event)
