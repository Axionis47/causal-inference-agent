"""Preparation V2 entry gate, replay, and manifest compilation."""

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
from causal.preparation.plans import eligibility_vocabulary, load_preparation_packs
from causal.shared import contracts as sc
from causal.shared import handoff as sh
from causal.shared.persistence import PersistenceError

NOW = datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
REGISTRIES = Path(__file__).resolve().parents[2] / "registries"
PACKS = REGISTRIES / "method-packs.v1.json"
AIPW = load_preparation_packs(REGISTRIES / "method-pack-preparation.v1.json", PACKS).get(
    "aipw", "aipw-pack.v1")
CSV = b"unit_id,treat\n1,0\n"
IDS = ("sel-1", "des-1", "dia-1", "cap-1", "rev-1", "app-1")
DIGESTS = {name: str(index) * 64
           for index, name in enumerate((*IDS, "out-1"), start=1)}
ENTRIES = tuple(sc.ArtifactRef(artifact_id=name, content_hash=DIGESTS[name]) for name in IDS)
VERSIONS = dict.fromkeys(PREPARATION_REGISTRY_KEYS, "v1")
def ref(name: str) -> dict[str, str]:
    return {"artifact_id": name, "content_hash": DIGESTS[name]}


PREPARATION = {
    "output_grain": "one_row_per_unit", "key_columns": ["unit_id"],
    "eligibility_rule_ids": ["target_population_filter"],
    "unusable_row_rule_ids": ["corrupt_record"], "protected_columns": ["treat"],
    "imputation_permitted": ["age"], "deletion_impact_dimensions": ["overall"],
    "required_final_diagnostic_ids": ["missingness"],
    "estimator_input_schema_id": "aipw-prepared-frame.v1",
}
PAYLOADS: dict[str, dict[str, Any]] = {
    "out-1": {"status": "approved", "approval": ref("app-1"),
              "compiled_design": ref("des-1"), "diagnostic_report": ref("dia-1"),
              "capacity_report": ref("cap-1"), "review_bundle": ref("rev-1")},
    "sel-1": {"resource_object_locator": "objects/csv",
              "resource_sha256": hashlib.sha256(CSV).hexdigest()},
    "des-1": {"selected_csv": ref("sel-1"), "method_id": "aipw", "comparator": "cmp",
              "method_pack_version": "aipw-pack.v1", "estimand": "ate",
              "frame": {"treatment": "tr", "outcome": "out", "population": "pop",
                        "timeframe": "tf"}, "measurement_map": ref("sel-1"),
              "role_ledger": ref("des-1"), "diagnostic_report": ref("dia-1"),
              "role_bindings": [
                  {"role": "group", "columns": ["treat"]}, {"role": "treatment", "columns": ["treat"]},
                  {"role": "confounder_candidate", "columns": ["age"]}],
              "registry_versions": {"schema": "v2"}, "preparation": PREPARATION},
    "dia-1": {"computable": True, "issues": [],
              "results": [{"diagnostic_id": "missingness", "status": "computed"}]},
    "cap-1": {"status": "pass", "compiled_design": ref("des-1")},
    "rev-1": {"compiled_design": ref("des-1"), "diagnostic_report": ref("dia-1"),
              "capacity_report": ref("cap-1")},
    "app-1": {"review_bundle": ref("rev-1"),
              "approved_bundle_hash": DIGESTS["rev-1"]},
}


def envelope(artifact_id: str, artifact_type: str,
             parents: tuple[sc.ArtifactRef, ...] = ()) -> sc.ArtifactEnvelopeV1:
    return sc.ArtifactEnvelopeV1(
        artifact_id=artifact_id, artifact_type=artifact_type, schema_version="v1",
        content_hash=DIGESTS[artifact_id], analysis_id="an-1", stage_run_id="run-1",
        producer_component="design-harness", producer_version="0.1.0",
        parent_artifacts=parents, sensitivity_class=sc.SensitivityClass.INTERNAL,
        created_at_utc=NOW, payload_locator=f"objects/{artifact_id}")


class Fakes:
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
        raise AssertionError("the handoff gate owns record")

    def accept(self, manifest: sc.HandoffManifestV1, component: str, outcomes: frozenset[str],
               factory: Any) -> sh.HandoffResult:
        assert (component, outcomes) == ("preparation-harness", frozenset({"approved"}))
        self.accepts += 1
        return self.result


def handoff(**overrides: Any) -> sc.HandoffManifestV1:
    fields: dict[str, Any] = {
        "handoff_id": entry.handoff_id("an-1", DIGESTS["out-1"]),
        "schema_version": "handoff.v1", "analysis_id": "an-1",
        "producing_stage_run_id": "run-1", "entries": ENTRIES,
        "receiving_stage_run_id": "run-2", "originating_outcome": "approved",
        "approval_ids": ("app-1",), "registry_version": "artifact-types.v1",
        "compatibility_version": "handoff.v1", "receiver_validation_result": None,
        "receiver_error_codes": (), "created_at_utc": NOW, "accepted_at_utc": None}
    return sc.HandoffManifestV1(**(fields | overrides))


def build(patch: Mapping[str, Mapping[str, Any]] | None = None,
          envelopes: tuple[sc.ArtifactEnvelopeV1, ...] | None = None
          ) -> tuple[sc.HandoffManifestV1, entry.EntryInputs, Fakes]:
    bodies = {name: body | dict((patch or {}).get(name, {}))
              for name, body in PAYLOADS.items()}
    committed = envelopes if envelopes is not None else tuple(
        envelope(name, kind) for name, kind in zip(IDS, entry.ENTRY_TYPES, strict=True))
    fakes = Fakes((*committed, envelope("out-1", "DesignOutcome")), {
        "objects/csv": CSV,
        **{f"objects/{name}": json.dumps(body).encode() for name, body in bodies.items()}})
    manifest = handoff()
    return manifest, entry.read_entries(manifest, "out-1", fakes, fakes), fakes


def policy(**overrides: Any) -> entry.EntryPolicy:
    fields: dict[str, Any] = {
        "pack": AIPW, "registry_versions": VERSIONS,
        "eligibility_vocabulary": eligibility_vocabulary(PACKS, "aipw")}
    return entry.EntryPolicy(**(fields | overrides))


def refused(patch: Mapping[str, Mapping[str, Any]] | None = None,
            envelopes: tuple[sc.ArtifactEnvelopeV1, ...] | None = None,
            **policy_overrides: Any) -> tuple[str, ...]:
    manifest, inputs, fakes = build(patch, envelopes)
    with pytest.raises(entry.PreparationEntryError) as error:
        entry.validate_entry(manifest, inputs, policy(**policy_overrides), products=fakes,
                             objects=fakes)
    return error.value.detail_codes


def opened(recorded: sc.HandoffManifestV1 | None, accepted: bool = True,
           gate_codes: tuple[str, ...] = ()) -> tuple[entry.HandoffAcceptance, Fakes]:
    fakes = Fakes(recorded=recorded, result=sh.HandoffResult(accepted, gate_codes))
    return entry.accept_handoff(cast(sh.HandoffGate, fakes), cast(sh.HandoffStore, fakes),
                                handoff(), cast(Any, None)), fakes


def test_handoff_accepts_once_and_replays_an_accepted_record() -> None:
    acceptance, fakes = opened(None)
    assert entry.handoff_id("an-1", DIGESTS["out-1"]) == f"ho:an-1:{'7' * 16}"
    assert not acceptance.replayed and fakes.accepts == 1
    replay, fakes = opened(handoff(receiver_validation_result="accepted"))
    assert replay.replayed and fakes.accepts == 0


@pytest.mark.parametrize(("recorded", "accepted", "expected"), [
    (handoff(receiver_validation_result="rejected"), True, entry.HANDOFF_REJECTED),
    (handoff(entries=ENTRIES[:-1], receiver_validation_result="accepted"), True,
     entry.HANDOFF_ENTRIES_CHANGED),
    (None, False, entry.HANDOFF_REJECTED),
])
def test_refused_or_changed_handoff_is_typed(recorded: sc.HandoffManifestV1 | None,
                                             accepted: bool, expected: str) -> None:
    with pytest.raises(entry.PreparationEntryError) as error:
        opened(recorded, accepted)
    assert error.value.code == expected


def test_clean_v2_handoff_passes() -> None:
    manifest, inputs, fakes = build()
    entry.validate_entry(manifest, inputs, policy(), products=fakes, objects=fakes)


@pytest.mark.parametrize(("patch", "expected"), [
    ({"out-1": {"status": "changes_requested"}}, entry.DESIGN_NOT_APPROVED),
    ({"out-1": {"approval": {"artifact_id": "x", "content_hash": DIGESTS["app-1"]}}},
     entry.APPROVAL_NOT_BOUND),
    ({"rev-1": {"compiled_design": ref("sel-1")}}, entry.COMPILED_HANDOFF_NOT_BOUND),
    ({"sel-1": {"resource_sha256": "9" * 64}}, entry.CSV_HASH_MISMATCH),
    ({"des-1": {"preparation": {"eligibility_rule_ids": ["invented_rule"]}}},
     entry.UNREGISTERED_RULE_ID),
    ({"dia-1": {"computable": False}}, entry.DIAGNOSTIC_REPORT_NOT_COMPUTABLE),
    ({"des-1": {"source_table_mutated": True}}, entry.UPSTREAM_MUTATION_MARKER),
    ({"cap-1": {"status": "fail"}}, entry.CAPACITY_REPORT_NOT_PASS),
    ({"cap-1": {"compiled_design": ref("sel-1")}}, entry.COMPILED_HANDOFF_NOT_BOUND),
])
def test_each_failing_condition_reports_its_stable_code(
        patch: Mapping[str, Mapping[str, Any]], expected: str) -> None:
    assert expected in refused(patch)


def test_registry_versions_are_checked_without_reinterpreting_the_compiler_report() -> None:
    assert refused(registry_versions={"schema": "v1"}) == (
        entry.UNSUPPORTED_REGISTRY_VERSION,)


def test_missing_wrong_type_and_broken_parent_are_reported() -> None:
    found = refused(envelopes=(
        envelope("sel-1", "TableSelection"), envelope("des-1", "TableSelection"),
        envelope("dia-1", "DiagnosticReport",
                 (sc.ArtifactRef(artifact_id="gone-1", content_hash="8" * 64),))))
    assert {entry.MISSING_ARTIFACT, entry.UNEXPECTED_ENTRY_TYPE,
            entry.BROKEN_HASH_CHAIN} <= set(found)


def test_manifest_hydrates_directly_from_compiled_design() -> None:
    preparation = dict(PREPARATION) | {"imputation_permitted": ["age", "treat"]}
    _, inputs, _ = build({"des-1": {"preparation": preparation}})
    found = entry.compile_context_manifest(
        inputs, ENTRIES, pack=AIPW, question_id="q-1", parser_profile_id="csv.v1",
        registry_versions=VERSIONS)
    assert (found.selected_csv, found.method_id, found.estimand_id, found.approved_grain,
            found.key_columns) == (ENTRIES[0], "aipw", "ate", "one_row_per_unit", ("unit_id",))
    assert found.column_roles == {"treat": "treatment", "age": "confounder_candidate"}
    assert (found.protected_columns, found.permitted_imputation_columns) == (("treat",), ("age",))
    assert found.eligibility_rule_ids == ("target_population_filter",)
    assert found.unusable_row_rule_ids == ("corrupt_record",)
