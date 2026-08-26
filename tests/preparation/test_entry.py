"""Preparation entry gate: replay, the nine §4 checks, manifest compilation (T-018 §1.1)."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from causal.preparation import entry
from causal.preparation.contracts import PREPARATION_REGISTRY_KEYS
from causal.preparation.packs import eligibility_vocabulary, load_preparation_packs
from causal.shared import contracts as sc
from causal.shared import handoff as sh
from causal.shared.persistence import PersistenceError

NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
REGISTRIES = Path(__file__).resolve().parents[2] / "registries"
PACKS = REGISTRIES / "method-packs.v1.json"
AIPW = load_preparation_packs(REGISTRIES / "method-pack-preparation.v1.json", PACKS).get(
    "aipw", "aipw-pack.v1")
CSV = b"unit_id,treat\n1,0\n"
IDS = ("sel-1", "des-1", "rfc-1", "cap-1")
DIGESTS = {name: str(index) * 64 for index, name in enumerate((*IDS, "out-1", "app-1"), start=1)}
ENTRIES = tuple(sc.ArtifactRef(artifact_id=name, content_hash=DIGESTS[name]) for name in IDS)
CATALOG, CAPACITY = "visualization-catalog.v1", "delivery-capacity.v1"
VERSIONS = dict.fromkeys(PREPARATION_REGISTRY_KEYS, "v1")
LEDGER: dict[str, Any] = {"claims": [{"role": "treatment", "column_refs": ["treat"]},
                                     {"role": "confounder_candidate", "column_refs": ["age"]}]}
MEASUREMENT: dict[str, Any] = {"links": [{"column_name": "age", "concept_id": "c-age"}]}


def ref(name: str) -> dict[str, str]:
    return {"artifact_id": name, "content_hash": DIGESTS[name]}


# The DesignOutcome and the four entry payloads a clean approved handoff carries.
PAYLOADS: dict[str, dict[str, Any]] = {
    "out-1": {"status": "approved", "approval": ref("app-1"), "causal_graph_view": ref("sel-1"),
              "experiment_design": ref("des-1"), "runnable_frame_contract": ref("rfc-1"),
              "capacity_check": ref("cap-1")},
    "sel-1": {"resource_object_locator": "objects/csv",
              "resource_sha256": hashlib.sha256(CSV).hexdigest()},
    "des-1": {"selected_csv": ref("sel-1"), "method_id": "aipw", "comparator": "cmp",
              "method_pack_version": "aipw-pack.v1", "estimand": "ate",
              "frame": {"treatment": "tr", "outcome": "out", "population": "pop",
                        "timeframe": "tf"},
              "measurement_map": ref("sel-1"), "role_ledger": ref("des-1"),
              "capacity_check": ref("cap-1"), "visualization_catalog_version": CATALOG,
              "capacity_registry_version": CAPACITY, "registry_versions": {"schema": "v1"},
              "required_prerepair_diagnostics": ["missingness"]},
    "rfc-1": {"experiment_design_hash": DIGESTS["des-1"], "output_grain": "one_row_per_unit",
              "key_columns": ["unit_id"], "eligibility_rules": ["target_population_filter"],
              "exclusion_reason_vocabulary": ["corrupt_record"], "imputation_permitted": ["age"],
              "imputation_forbidden": [], "deletion_impact_dimensions": ["overall"],
              "required_final_diagnostics": ["missingness"]},
    "cap-1": {"status": "pass", "method_id": "aipw", "visualization_catalog_version": CATALOG,
              "capacity_registry_version": CAPACITY},
}


def envelope(artifact_id: str, artifact_type: str,
             parents: tuple[sc.ArtifactRef, ...] = ()) -> sc.ArtifactEnvelopeV1:
    return sc.ArtifactEnvelopeV1(
        artifact_id=artifact_id, artifact_type=artifact_type, schema_version="v1",
        content_hash=DIGESTS[artifact_id], analysis_id="an-1", stage_run_id="run-1",
        producer_component="design-harness", producer_version="0.1.0", parent_artifacts=parents,
        sensitivity_class=sc.SensitivityClass.INTERNAL, created_at_utc=NOW,
        payload_locator=f"objects/{artifact_id}")


class Fakes:
    """One stand-in for the products and object readers, the handoff store, and the gate."""

    def __init__(self, envelopes: tuple[sc.ArtifactEnvelopeV1, ...] = (),
                 objects: Mapping[str, bytes] | None = None,
                 recorded: sc.HandoffManifestV1 | None = None,
                 result: sh.HandoffResult | None = None) -> None:
        self.by_id = {found.artifact_id: found for found in envelopes}
        self.objects, self.recorded, self.accepts = dict(objects or {}), recorded, 0
        self.result = result or sh.HandoffResult(True, ())

    def load_envelope(self, artifact_id: str) -> sc.ArtifactEnvelopeV1:
        if artifact_id not in self.by_id:
            raise PersistenceError(artifact_id, "missing_artifact")
        return self.by_id[artifact_id]

    def get(self, locator: str) -> bytes:
        return self.objects[locator]

    def load(self, handoff_id: str) -> sc.HandoffManifestV1:
        if self.recorded is None:
            raise PersistenceError(handoff_id, "unknown_handoff")
        return self.recorded

    def record(self, manifest: sc.HandoffManifestV1) -> None:
        raise AssertionError("the gate owns the single `record` call (duplicate_handoff)")

    def accept(self, manifest: sc.HandoffManifestV1, component: str, outcomes: frozenset[str],
               factory: Any) -> sh.HandoffResult:
        assert (component, outcomes) == ("preparation-harness", frozenset({"approved"}))
        self.accepts += 1
        return self.result


def handoff(**overrides: Any) -> sc.HandoffManifestV1:
    fields: dict[str, Any] = {
        "handoff_id": entry.handoff_id("an-1", DIGESTS["out-1"]), "schema_version": "handoff.v1",
        "analysis_id": "an-1", "producing_stage_run_id": "run-1", "entries": ENTRIES,
        "receiving_stage_run_id": "run-2", "originating_outcome": "approved",
        "approval_ids": ("app-1",), "registry_version": "artifact-types.v1",
        "compatibility_version": "handoff.v1", "receiver_validation_result": None,
        "receiver_error_codes": (), "created_at_utc": NOW, "accepted_at_utc": None}
    return sc.HandoffManifestV1(**{**fields, **overrides})


def build(patch: Mapping[str, Mapping[str, Any]] | None = None,
          envelopes: tuple[sc.ArtifactEnvelopeV1, ...] | None = None
          ) -> tuple[sc.HandoffManifestV1, entry.EntryInputs, Fakes]:
    """Assemble the fakes and resolve the four entries through `read_entries`."""
    bodies = {name: {**body, **(patch or {}).get(name, {})} for name, body in PAYLOADS.items()}
    committed = envelopes if envelopes is not None else tuple(
        envelope(name, kind) for name, kind in zip(IDS, entry.ENTRY_TYPES, strict=True))
    fakes = Fakes((*committed, envelope("out-1", "DesignOutcome")), {
        "objects/csv": CSV,
        **{f"objects/{name}": json.dumps(body).encode() for name, body in bodies.items()}})
    manifest = handoff()
    return manifest, entry.read_entries(manifest, "out-1", fakes, fakes), fakes


def policy(**overrides: Any) -> entry.EntryPolicy:
    fields: dict[str, Any] = {
        "pack": AIPW, "registry_versions": VERSIONS, "approved_handling": frozenset({"partial"}),
        "eligibility_vocabulary": eligibility_vocabulary(PACKS, "aipw"),
        "prerepair_statuses": {"missingness": "computed"}}
    return entry.EntryPolicy(**{**fields, **overrides})


def refused(patch: Mapping[str, Mapping[str, Any]] | None = None,
            envelopes: tuple[sc.ArtifactEnvelopeV1, ...] | None = None,
            **policy_overrides: Any) -> tuple[str, ...]:
    manifest, inputs, fakes = build(patch, envelopes)
    with pytest.raises(entry.PreparationEntryError) as error:
        entry.validate_entry(manifest, inputs, policy(**policy_overrides), products=fakes,
                             objects=fakes)
    assert error.value.code == entry.ENTRY_VALIDATION_FAILED
    return error.value.detail_codes


def opened(recorded: sc.HandoffManifestV1 | None, accepted: bool = True,
           gate_codes: tuple[str, ...] = ()) -> tuple[entry.HandoffAcceptance, Fakes]:
    fakes = Fakes(recorded=recorded,
                  result=sh.HandoffResult(accepted=accepted, error_codes=gate_codes))
    return entry.accept_handoff(cast(sh.HandoffGate, fakes), cast(sh.HandoffStore, fakes),
                                handoff(), cast(Any, None)), fakes


class TestHandoffAcceptance:
    def test_an_absent_handoff_is_accepted_through_the_gate(self) -> None:
        assert entry.handoff_id("an-1", DIGESTS["out-1"]) == f"ho:an-1:{'5' * 16}"
        acceptance, fakes = opened(None)
        assert (acceptance.replayed, fakes.accepts) == (False, 1)

    def test_a_recorded_acceptance_replays_without_touching_the_gate(self) -> None:
        acceptance, fakes = opened(handoff(receiver_validation_result="accepted"), accepted=False)
        assert (acceptance.replayed, fakes.accepts) == (True, 0)
        assert acceptance.manifest.receiver_validation_result == "accepted"

    @pytest.mark.parametrize(("recorded", "accepted", "gate_codes", "expected", "detail"), [
        (handoff(receiver_validation_result="rejected", receiver_error_codes=("wrong_outcome",)),
         True, (), entry.HANDOFF_REJECTED, ("wrong_outcome",)),
        (handoff(entries=ENTRIES[:3], receiver_validation_result="accepted"), True, (),
         entry.HANDOFF_ENTRIES_CHANGED, ()),
        (None, False, ("missing_object",), entry.HANDOFF_REJECTED, ("missing_object",)),
    ])
    def test_a_refused_or_changed_handoff_is_a_typed_terminal_failure(
        self, recorded: sc.HandoffManifestV1 | None, accepted: bool, gate_codes: tuple[str, ...],
        expected: str, detail: tuple[str, ...]
    ) -> None:
        with pytest.raises(entry.PreparationEntryError) as error:
            opened(recorded, accepted, gate_codes)
        assert (error.value.code, error.value.detail_codes) == (expected, detail)


class TestEntryChecks:
    def test_a_clean_approved_handoff_passes_every_check(self) -> None:
        manifest, inputs, fakes = build()
        entry.validate_entry(manifest, inputs, policy(), products=fakes, objects=fakes)

    @pytest.mark.parametrize(("patch", "expected"), [
        ({"out-1": {"status": "changes_requested"}}, entry.DESIGN_NOT_APPROVED),
        ({"out-1": {"approval": {"artifact_id": "x", "content_hash": DIGESTS["app-1"]}}},
         entry.APPROVAL_NOT_BOUND),
        ({"rfc-1": {"experiment_design_hash": "9" * 64}}, entry.FRAME_CONTRACT_NOT_BOUND),
        ({"sel-1": {"resource_sha256": "9" * 64}}, entry.CSV_HASH_MISMATCH),
        ({"rfc-1": {"eligibility_rules": ["invented_rule"]}}, entry.UNREGISTERED_RULE_ID),
        ({"des-1": {"required_prerepair_diagnostics": ["absent"]}},
         entry.PREREPAIR_DIAGNOSTIC_MISSING),
        ({"des-1": {"source_table_mutated": True}}, entry.UPSTREAM_MUTATION_MARKER),
        ({"cap-1": {"status": "fail"}}, entry.CAPACITY_CHECK_NOT_PASS),
        ({"cap-1": {"method_id": "did"}}, entry.CAPACITY_BINDING_MISMATCH),
    ])
    def test_each_failing_condition_reports_its_stable_code(
        self, patch: Mapping[str, Mapping[str, Any]], expected: str
    ) -> None:
        assert expected in refused(patch)

    def test_registry_versions_and_diagnostic_handling_are_checked_together(self) -> None:
        assert refused(registry_versions={"schema": "v1"},
                       prerepair_statuses={"missingness": "not_computable"}) == (
            entry.PREREPAIR_HANDLING_NOT_APPROVED, entry.UNSUPPORTED_REGISTRY_VERSION)

    def test_a_missing_entry_a_wrong_type_and_a_broken_parent_chain_are_reported(self) -> None:
        found = refused(envelopes=(
            envelope("sel-1", "TableSelection"), envelope("des-1", "TableSelection"),
            envelope("rfc-1", "RunnableFrameContract",
                     (sc.ArtifactRef(artifact_id="gone-1", content_hash="7" * 64),))))
        assert {entry.MISSING_ARTIFACT, entry.UNEXPECTED_ENTRY_TYPE,
                entry.BROKEN_HASH_CHAIN} <= set(found)


def test_the_manifest_hydrates_from_the_entries_and_the_design_side() -> None:
    _, inputs, _ = build({"rfc-1": {"imputation_permitted": ["age", "treat"]}})
    found = entry.compile_context_manifest(
        inputs, ENTRIES, pack=AIPW, role_ledger=LEDGER, measurement_map=MEASUREMENT,
        causal_context=ENTRIES[1], question_id="q-1", parser_profile_id="csv.v1",
        registry_versions=VERSIONS, recipient_map={"table_wide": ("get_contract",)})
    assert (found.selected_csv, found.source_object_locator, found.method_id, found.estimand_id,
            found.population_id, found.treatment_id, found.approved_grain, found.key_columns) == (
        ENTRIES[0], "objects/csv", "aipw", "ate", "pop", "tr", "one_row_per_unit", ("unit_id",))
    assert (found.column_concepts, found.column_roles) == (
        {"age": "c-age"}, {"treat": "treatment", "age": "confounder_candidate"})
    # A protected role is never a repair or an imputation target, whatever the contract lists.
    assert (found.protected_columns, found.permitted_repair_columns,
            found.permitted_imputation_columns) == (("treat",), ("age",), ("age",))
    assert (found.eligibility_rule_ids, found.unusable_row_rule_ids,
            found.prepared_frame_schema_id, len(found.manifest_hash)) == (
        ("target_population_filter",), ("corrupt_record",), "aipw-prepared-frame.v1", 64)
    assert "numeric_median_with_indicator" in found.permitted_operation_ids
    assert "rough_overlap" in found.permitted_diagnostic_ids
