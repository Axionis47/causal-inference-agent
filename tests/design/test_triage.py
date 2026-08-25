"""Deterministic triage `triage.v1`: golden tiers, batching, and record rules (T-011 §3, §8)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from causal.design.contracts import (
    REGISTRY_VERSION_KEYS,
    AvailabilityRowV1,
    ConceptProposalV1,
    DesignContextManifestV1,
    DesignIntentV1,
    QuestionKind,
    StructuralFieldV1,
)
from causal.design.triage import (
    MAX_BATCHES,
    ColumnTriageRecordV1,
    TriageBatchV1,
    build_batches,
    normalize_column_name,
    triage,
)
from causal.shared.contracts import ArtifactRef
from causal.shared.registry import load_artifact_type_registry

REGISTRY_PATH = Path(__file__).resolve().parents[2] / "registries" / "artifact-types.v1.json"
REF = ArtifactRef(artifact_id="art-1", content_hash="a" * 64)
TABLE = "nsw.csv"

# Golden fixture: thirteen columns in inventory order, one per rule path.
GOLDEN_COLUMNS = (
    "treat", "Re 78", "unit_id", "education", "black", "sample_row_id", "age", "married",
    "nodegree", "re74", "hispanic", "notes_blob", "stray_flag",
)
GOLDEN_RULES = (
    "intent_candidate", "intent_candidate", "intent_candidate", "evidenced_meaning",
    "flagged_by_profile", "flagged_by_profile", "profiled_only", "profiled_only", "profiled_only",
    "profiled_only", "no_signal", "no_signal", "no_signal",
)
GOLDEN_MEASURED = GOLDEN_COLUMNS[:10]
GOLDEN_HYPOTHESES = {
    "black": ("missing_sentinel",),
    "sample_row_id": ("identifier",),
    "notes_blob": ("constant",),  # D-025: not a tier-2 kind, and absent from the measured surface
}


def structural(columns: tuple[str, ...], table: str = TABLE) -> tuple[StructuralFieldV1, ...]:
    return tuple(
        StructuralFieldV1(table_name=table, column_name=name, dtype="float64", ordinal=ordinal)
        for ordinal, name in enumerate(columns)
    )


def availability(column: str, slot: str, status: str) -> AvailabilityRowV1:
    return AvailabilityRowV1(
        scope_kind="column", table_name=TABLE, column_name=column, field_or_slot_name=slot,
        status=status, evidence_count=1, json_pointer=f"/columns/{column}",
    )


def manifest(
    columns: tuple[str, ...],
    semantic: tuple[AvailabilityRowV1, ...] = (),
    measured: tuple[str, ...] = (),
) -> DesignContextManifestV1:
    return DesignContextManifestV1(
        design_revision=1, question_artifact=REF, intake_outcome_artifact=REF,
        table_selection_artifact=REF, selected_table=TABLE,
        structural_inventory=structural(columns), semantic_available=semantic,
        semantic_missing=(),
        measured_surface=tuple(availability(c, "profile", "measured") for c in measured),
        provenance_surface=(), retrieval_surfaces=("catalog.structural_manifest",),
        registry_versions=dict.fromkeys(REGISTRY_VERSION_KEYS, "artifact-types.v1"),
        recipient_map={"design_intent": ("list_intake_inventory",)},
    )


def proposal(name: str, *columns: str) -> ConceptProposalV1:
    return ConceptProposalV1(name=name, description=f"the {name}", candidate_columns=columns)


def intent(
    treatment: tuple[str, ...] = ("treat",), unit: tuple[str, ...] = ("unit_id",)
) -> DesignIntentV1:
    return DesignIntentV1(
        question_kind=QuestionKind.CAUSAL, causal_claim="training raises earnings",
        intended_decision="fund the program", treatment=proposal("treatment", *treatment),
        outcome=proposal("outcome", "re_78"), population=proposal("population"),
        comparator=proposal("comparator"), unit=proposal("unit", *unit),
        timeframe=proposal("timeframe"), candidate_grain="one row per applicant",
        mandatory_concepts=(), claims=(),
    )


def golden_record() -> ColumnTriageRecordV1:
    """Triage the golden fixture: `Re 78` matches intent `re_78` only after normalization."""
    return triage(
        intent(),
        manifest(
            GOLDEN_COLUMNS,
            semantic=(
                availability("education", "meaning", "evidenced"),
                availability("re74", "meaning", "hypothesis"),  # not evidenced: stays supporting
                availability("stray_flag", "encoding", "evidenced"),  # wrong slot: stays unused
            ),
            measured=GOLDEN_MEASURED,
        ),
        GOLDEN_HYPOTHESES,
    )


class TestGoldenFixture:
    def test_tiers_are_sorted_and_exhaustive(self) -> None:
        record = golden_record()
        assert record.table_name == TABLE
        assert record.schema_version == "column-triage.v1"
        assert record.triage_rule_version == "triage.v1"
        assert record.tiers == {
            "critical": ("Re 78", "treat", "unit_id"),
            "plausible_adjustment": ("black", "education", "sample_row_id"),
            "supporting": ("age", "married", "nodegree", "re74"),
            "unused": ("hispanic", "notes_blob", "stray_flag"),
        }

    def test_match_trace_names_the_rule_that_placed_every_column(self) -> None:
        assert golden_record().match_trace == dict(zip(GOLDEN_COLUMNS, GOLDEN_RULES, strict=True))

    def test_deferred_is_supporting_plus_unused(self) -> None:
        record = golden_record()
        assert record.deferred == ("age", "hispanic", "married", "nodegree", "notes_blob",
                                   "re74", "stray_flag")
        assert {name for tier in record.tiers.values() for name in tier} == set(GOLDEN_COLUMNS)

    def test_batches_take_critical_first_in_inventory_order(self) -> None:
        assert [batch.column_names for batch in golden_record().batches] == [
            ("treat", "Re 78", "unit_id", "education"), ("black", "sample_row_id"),
        ]

    def test_intent_beats_every_later_rule(self) -> None:
        record = triage(intent(treatment=("black",)), golden_record_manifest(), GOLDEN_HYPOTHESES)
        assert record.match_trace["black"] == "intent_candidate"
        assert "black" in record.tiers["critical"]

    def test_without_hypotheses_flagged_columns_fall_to_supporting(self) -> None:
        record = triage(intent(), golden_record_manifest())
        assert record.tiers["plausible_adjustment"] == ("education",)
        assert {"black", "sample_row_id"} <= set(record.tiers["supporting"])


def golden_record_manifest() -> DesignContextManifestV1:
    return manifest(
        GOLDEN_COLUMNS,
        semantic=(availability("education", "meaning", "evidenced"),),
        measured=GOLDEN_MEASURED,
    )


class TestDeterminism:
    def test_normalization_folds_case_spaces_and_hyphens(self) -> None:
        assert normalize_column_name("  Re-78 Value ") == "re_78_value"

    def test_same_input_gives_an_identical_record(self) -> None:
        first, second = golden_record(), golden_record()
        assert first == second
        assert first.canonical_payload() == second.canonical_payload()
        ids = [batch.batch_id for batch in first.batches]
        assert ids == [batch.batch_id for batch in second.batches]
        assert all(bid.startswith("tb:") and len(bid) == 19 for bid in ids)


class TestBatchCap:
    def test_thirty_batchable_columns_fill_exactly_eight_batches(self) -> None:
        columns = tuple(f"c{index:02d}" for index in range(30))
        record = triage(
            intent(treatment=columns[:15], unit=()),
            manifest(
                columns,
                semantic=tuple(availability(c, "meaning", "evidenced") for c in columns[15:]),
            ),
        )
        assert len(record.batches) == MAX_BATCHES
        batched = [name for batch in record.batches for name in batch.column_names]
        assert sorted(batched) == sorted(columns)
        assert len(batched) == len(set(batched))

    def test_batch_count_stays_capped_and_empty_input_makes_no_batch(self) -> None:
        columns = tuple(f"x{index:03d}" for index in range(97))
        batches = build_batches(TABLE, columns, ())
        assert len(batches) <= MAX_BATCHES
        assert sum(len(batch.column_names) for batch in batches) == len(columns)
        assert build_batches(TABLE, (), ()) == ()


TIERS: dict[str, tuple[str, ...]] = {
    "critical": ("a",), "plausible_adjustment": ("b",), "supporting": ("c",), "unused": ("d",),
}
RECORD_KWARGS: dict[str, object] = {
    "table_name": TABLE,
    "tiers": TIERS,
    "batches": (TriageBatchV1(batch_id="tb:0123456789abcdef", column_names=("a", "b")),),
    "deferred": ("c", "d"),
    "match_trace": {
        "a": "intent_candidate", "b": "evidenced_meaning", "c": "profiled_only", "d": "no_signal",
    },
}


NINE = tuple(f"c{index:02d}" for index in range(9))
NINE_BATCHES: dict[str, object] = {
    "tiers": {"critical": NINE, "plausible_adjustment": (), "supporting": (), "unused": ()},
    "batches": tuple(
        TriageBatchV1(batch_id=f"tb:{index}", column_names=(name,))
        for index, name in enumerate(NINE)
    ),
    "deferred": (),
    "match_trace": dict.fromkeys(NINE, "intent_candidate"),
}
REJECTED: tuple[tuple[str, dict[str, object]], ...] = (
    ("overlaps", {"tiers": {**TIERS, "supporting": ("a", "c")}, "deferred": ("a", "c", "d")}),
    ("sorted", {"tiers": {**TIERS, "supporting": ("e", "c")}, "deferred": ("c", "d", "e")}),
    ("tiers key mismatch", {"tiers": {**TIERS, "extra": ()}}),
    ("match_trace", {"match_trace": {"a": "intent_candidate"}}),
    ("deferred", {"deferred": ("c",)}),
    ("non-batchable", {"batches": (TriageBatchV1(batch_id="tb:0", column_names=("a", "c")),)}),
    ("repeats", {"batches": (
        TriageBatchV1(batch_id="tb:1", column_names=("a", "b")),
        TriageBatchV1(batch_id="tb:2", column_names=("b",)),
    )}),
    ("at most 8 batches", NINE_BATCHES),
)


def record(**overrides: object) -> BaseModel:
    """Build the minimal valid record with per-test overrides (kwargs typed loosely)."""
    model: type[BaseModel] = ColumnTriageRecordV1
    return model(**{**RECORD_KWARGS, **overrides})


class TestRecordValidators:
    def test_minimal_record_is_valid(self) -> None:
        built = record()
        assert isinstance(built, ColumnTriageRecordV1)
        assert built.deferred == ("c", "d")

    @pytest.mark.parametrize(("message", "overrides"), REJECTED)
    def test_cross_field_rules_reject(self, message: str, overrides: dict[str, object]) -> None:
        with pytest.raises(ValidationError, match=message):
            record(**overrides)


def test_registry_row_resolves_with_its_design_intent_parent() -> None:
    registry = load_artifact_type_registry(REGISTRY_PATH)
    row = registry.lookup("ColumnTriageRecord")
    assert (row.schema_version, row.producer_component, row.validator_version) == (
        "column-triage.v1", "design-harness", "column-triage-validator.v1")
    assert row.allowed_reader_components == ("design-harness",)
    assert (row.required_parent_types, row.optional_parent_types) == (("DesignIntent",), ())
    assert (row.terminal_statuses, row.destinations) == (("committed",), ("design",))
    assert registry.lookup(row.required_parent_types[0]).artifact_type == "DesignIntent"
