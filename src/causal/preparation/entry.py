"""Preparation entry gate: handoff acceptance, the §4 checks, manifest compilation (PRD-003 §4, §7.2)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from causal.preparation import contracts as pc
from causal.preparation.contracts import PREPARATION_REGISTRY_KEYS, PreparationError
from causal.preparation.plans import PreparationPackV1, strategy_operation
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactEnvelopeV1, ArtifactRef, HandoffManifestV1
from causal.shared.handoff import UNKNOWN_HANDOFF, EventFactory, HandoffGate, HandoffStore
from causal.shared.persistence import PersistenceError
from causal.shared.readers import ObjectReader, ProductsReader

COMPONENT, APPROVED, COMPUTED = "preparation-harness", "approved", "computed"
ENTRY_VALIDATION_FAILED, ENTRY_UNREADABLE = "entry_validation_failed", "entry_unreadable"
HANDOFF_REJECTED, HANDOFF_ENTRIES_CHANGED = "handoff_rejected", "handoff_entries_changed"
# The nine §4 conditions, one stable code family each; every failing family is reported at once.
DESIGN_NOT_APPROVED, APPROVAL_NOT_BOUND = "design_not_approved", "approval_not_bound"
MISSING_ARTIFACT, ENTRY_HASH_MISMATCH = "missing_artifact", "entry_hash_mismatch"
UNEXPECTED_ENTRY_TYPE, BROKEN_HASH_CHAIN = "unexpected_entry_type", "broken_hash_chain"
COMPILED_HANDOFF_NOT_BOUND, CSV_HASH_MISMATCH = "compiled_handoff_not_bound", "csv_hash_mismatch"
UNSUPPORTED_REGISTRY_VERSION, UNREGISTERED_RULE_ID = "unsupported_registry_version", "unregistered_rule_id"
DIAGNOSTIC_REPORT_NOT_COMPUTABLE = "diagnostic_report_not_computable"
UPSTREAM_MUTATION_MARKER = "upstream_mutation_marker"
CAPACITY_REPORT_NOT_PASS = "capacity_report_not_pass"
# The exact Design V2 handoff. Unfinished V1 runs restart; no compatibility reader exists.
ENTRY_TYPES: Final = ("TableSelection", "CompiledDesign", "DiagnosticReport", "CapacityReport",
                      "DesignReviewBundle", "DesignApproval")
# Any upstream payload key that would declare the immutable source table mutated (§4 condition 8).
MUTATION_MARKERS: Final = ("source_table_mutated", "source_mutated", "table_mutated")


class PreparationEntryError(PreparationError):
    """A preparation entry step failed. `code` is stable; `detail_codes` carries every family."""


# The opened handoff plus whether this run replayed an already accepted one.
@dataclass(frozen=True)
class HandoffAcceptance:
    manifest: HandoffManifestV1
    replayed: bool


# The DesignOutcome and six entry payloads, read as data (no `causal.design` import).
@dataclass(frozen=True)
class EntryInputs:
    outcome: Mapping[str, Any]
    selection: Mapping[str, Any]
    design: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    capacity: Mapping[str, Any]
    review: Mapping[str, Any]
    approval: Mapping[str, Any]
    envelopes: tuple[ArtifactEnvelopeV1 | None, ...]


# The registered vocabularies and approved handling the §4 checks measure against.
@dataclass(frozen=True)
class EntryPolicy:
    pack: PreparationPackV1
    registry_versions: Mapping[str, str]
    eligibility_vocabulary: tuple[str, ...] = ()
    imputation_strategy_ids: tuple[str, ...] = ()


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


# The artifact id one payload reference names, or None when the reference is absent.
def _ref_id(payload: Mapping[str, Any], key: str) -> str | None:
    held = payload.get(key)
    return str(held["artifact_id"]) if isinstance(held, Mapping) else None


def read_entries(manifest: HandoffManifestV1, outcome_artifact_id: str,
                 products: ProductsReader, objects: ObjectReader) -> EntryInputs:
    """Resolve the outcome and six entries; an unreadable entry becomes an empty payload."""
    if len(manifest.entries) != len(ENTRY_TYPES):
        raise PreparationEntryError("a preparation handoff carries six entries", ENTRY_UNREADABLE)
    envelopes = tuple(_envelope(products, entry.artifact_id) for entry in manifest.entries)
    selection, design, diagnostics, capacity, review, approval = (
        _body(objects, found) for found in envelopes)
    return EntryInputs(_body(objects, _envelope(products, outcome_artifact_id)), selection,
                       design, diagnostics, capacity, review, approval, envelopes)


# §4 conditions 1 and 2: an approved outcome whose entries and parents still hash true.
def _lineage_codes(manifest: HandoffManifestV1, inputs: EntryInputs,
                   products: ProductsReader) -> set[str]:
    codes: set[str] = set()
    if inputs.outcome.get("status") != APPROVED or manifest.originating_outcome != APPROVED:
        codes.add(DESIGN_NOT_APPROVED)
    approval = _ref_id(inputs.outcome, "approval")
    if approval not in manifest.approval_ids or not _ref_id(inputs.outcome, "review_bundle"):
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


# §4 conditions 3 and 9: every compiled reference and the exact approval hash agree.
def _binding_codes(manifest: HandoffManifestV1, inputs: EntryInputs) -> set[str]:
    selection, design, diagnostics, capacity, review, approval = manifest.entries
    bound = ((_ref_id(inputs.outcome, "compiled_design"), design.artifact_id),
             (_ref_id(inputs.outcome, "diagnostic_report"), diagnostics.artifact_id),
             (_ref_id(inputs.outcome, "capacity_report"), capacity.artifact_id),
             (_ref_id(inputs.outcome, "review_bundle"), review.artifact_id),
             (_ref_id(inputs.outcome, "approval"), approval.artifact_id),
             (_ref_id(inputs.design, "selected_csv"), selection.artifact_id),
             (_ref_id(inputs.capacity, "compiled_design"), design.artifact_id),
             (_ref_id(inputs.review, "compiled_design"), design.artifact_id),
             (_ref_id(inputs.review, "diagnostic_report"), diagnostics.artifact_id),
             (_ref_id(inputs.review, "capacity_report"), capacity.artifact_id),
             (_ref_id(inputs.approval, "review_bundle"), review.artifact_id),
             (inputs.approval.get("approved_bundle_hash"), review.content_hash))
    codes = {COMPILED_HANDOFF_NOT_BOUND} if any(a != b for a, b in bound) else set()
    codes |= {CAPACITY_REPORT_NOT_PASS} if inputs.capacity.get("status") != "pass" else set()
    return codes


# §4 conditions 4 and 8: the selected CSV still hashes true and nothing claims a mutation.
def _source_codes(inputs: EntryInputs, objects: ObjectReader) -> set[str]:
    digest = inputs.selection.get("resource_sha256")
    try:
        observed = hashlib.sha256(
            objects.get(str(inputs.selection.get("resource_object_locator")))).hexdigest()
    except Exception:  # noqa: BLE001 -- an unreadable source object cannot match its hash
        observed = ""
    codes = {CSV_HASH_MISMATCH} if observed != digest else set()
    payloads = (inputs.outcome, inputs.selection, inputs.design, inputs.diagnostics,
                inputs.capacity, inputs.review, inputs.approval)
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
    preparation = inputs.design.get("preparation") or {}
    families = ((preparation.get("eligibility_rule_ids") or (),
                 permitted | set(policy.eligibility_vocabulary)),
                (preparation.get("unusable_row_rule_ids") or (), permitted),
                (policy.imputation_strategy_ids,
                 {target.strategy_id for target in policy.pack.permitted_imputation_targets}))
    if any(set(found) - registered for found, registered in families):
        codes.add(UNREGISTERED_RULE_ID)
    if inputs.diagnostics.get("computable") is not True or inputs.diagnostics.get("issues"):
        codes.add(DIAGNOSTIC_REPORT_NOT_COMPUTABLE)
    return codes


def validate_entry(manifest: HandoffManifestV1, inputs: EntryInputs, policy: EntryPolicy, *,
                   products: ProductsReader, objects: ObjectReader) -> None:
    """The §4 nine entry checks; every failing condition is reported at once (fail closed)."""
    codes = (_lineage_codes(manifest, inputs, products) | _binding_codes(manifest, inputs)
             | _source_codes(inputs, objects) | _registry_codes(inputs, policy))
    if codes:
        raise PreparationEntryError("the preparation entry gate refused the design handoff",
                                    ENTRY_VALIDATION_FAILED, tuple(sorted(codes)))


# The compiler-owned role bindings flattened to the preparation stage's per-column policy map.
def _columns(design: Mapping[str, Any], pack: PreparationPackV1) -> dict[str, str]:
    roles: dict[str, str] = {}
    priority = {role: index for index, role in enumerate(dict.fromkeys((*tuple((design.get("preparation") or {}).get("required_roles") or ()), *pack.protected_roles)))}
    for binding in sorted(design.get("role_bindings") or (), key=lambda row: priority.get(str(row["role"]), len(priority))):
        for column in binding.get("columns") or ():
            roles.setdefault(str(column), str(binding["role"]))
    return roles


def compile_context_manifest(
    inputs: EntryInputs, entries: tuple[ArtifactRef, ...], *, pack: PreparationPackV1,
    question_id: str, parser_profile_id: str, registry_versions: Mapping[str, str],
) -> pc.PreparationContextManifestV1:
    """Hydrate the preparation manifest directly from the compiled V2 design."""
    design = inputs.design
    preparation = design.get("preparation") or {}
    frame = design.get("frame") or {}
    roles = _columns(design, pack)
    protected = tuple(str(column) for column in preparation.get("protected_columns") or ())
    forbidden = set(protected)
    # §7.2 carries OPERATION ids: a permitted strategy is named here by the operation that runs it.
    strategies = {found for target in pack.permitted_imputation_targets
                  for found in (strategy_operation(target.strategy_id, target.fit_scope),)
                  if found is not None}
    return pc.PreparationContextManifestV1(
        selected_csv=entries[0], parser_profile_id=parser_profile_id, question_id=question_id,
        source_object_locator=str(inputs.selection["resource_object_locator"]),
        population_id=str(frame["population"]), timeframe_id=str(frame["timeframe"]),
        treatment_id=str(frame["treatment"]), outcome_id=str(frame["outcome"]),
        comparator_id=str(design["comparator"]), estimand_id=str(design["estimand"]),
        method_id=str(design["method_id"]),
        method_pack_version=str(design["method_pack_version"]),
        column_roles=roles, protected_columns=protected,
        permitted_repair_columns=tuple(sorted(set(roles) - forbidden)),
        permitted_imputation_columns=tuple(
            column for column in preparation.get("imputation_permitted") or ()
            if column not in protected),
        approved_grain=str(preparation["output_grain"]),
        key_columns=tuple(str(column) for column in preparation["key_columns"]),
        prepared_frame_schema_id=pack.prepared_frame_schema_id,
        eligibility_rule_ids=tuple(preparation.get("eligibility_rule_ids") or ()),
        unusable_row_rule_ids=tuple(preparation.get("unusable_row_rule_ids") or ()),
        structural_requirements=tuple(pack.required_structure_gates),
        permitted_operation_ids=tuple(sorted(
            set(pack.permitted_repair_operation_ids) | strategies)),
        permitted_diagnostic_ids=tuple(sorted(
            set(pack.required_poststabilization_diagnostic_ids)
            | set(pack.required_postrepair_diagnostic_ids)
            | set(preparation.get("required_final_diagnostic_ids") or ()))),
        deletion_impact_dimensions=tuple(preparation.get("deletion_impact_dimensions") or ())
        or tuple(pack.dimension_impact_dimensions),
        registry_versions=dict(registry_versions),
        recipient_map={},
        manifest_hash=content_hash({"entries": [ref.model_dump(mode="json") for ref in entries]}))
