"""Tests for evidence/semantic-map builders (T-008; PRD-001 §7, §8; D-032)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from causal.intake.contracts import AVAILABLE_STATUSES, SemanticStatus
from causal.intake.fields import load_field_classes
from causal.intake.kaggle import capture
from causal.intake.profiler import profile_table
from causal.intake.semantic import (
    MAX_DOCUMENT_BYTES,
    build_evidence_bundle,
    build_semantic_map,
    measured_index_rows,
    slot_index_rows,
)
from tests.intake.conftest import CSV, FrozenKaggleClient

CLASSES = load_field_classes(
    Path(__file__).resolve().parents[2] / "registries" / "kaggle-field-classes.v1.json"
)


@pytest.fixture(scope="module")
def payload() -> dict[str, object]:
    return capture(FrozenKaggleClient(), "lalonde/nsw").payload


@pytest.fixture(scope="module")
def profiles() -> dict[str, dict[str, object]]:
    return {"nsw.csv": profile_table(CSV, "csv", "profiler.v1")}


def items_of(bundle: dict[str, object]) -> list[dict[str, Any]]:
    items = bundle["items"]
    assert isinstance(items, list)
    return items


def column_slots(semantic_map: dict[str, object]) -> dict[str, dict[str, Any]]:
    columns = semantic_map["columns"]
    assert isinstance(columns, list)
    return {entry["column_name"]: entry["slots"] for entry in columns}


class TestEvidenceBundle:
    def test_semantic_texts_carry_evidence_ids(self, payload: dict[str, object]) -> None:
        bundle = build_evidence_bundle(payload, {"readme.md": "# NSW"}, CLASSES)
        ids = {item["evidence_id"] for item in items_of(bundle)}
        assert {
            "ev:kaggle/dataset/title", "ev:kaggle/dataset/description",
            "ev:kaggle/file/nsw.csv/description",
            "ev:kaggle/column/nsw.csv/unit_id/description", "ev:doc/readme.md",
        } <= ids

    def test_blank_description_yields_no_evidence(self, payload: dict[str, object]) -> None:
        bundle = build_evidence_bundle(payload, {}, CLASSES)
        ids = {item["evidence_id"] for item in items_of(bundle)}
        assert "ev:kaggle/column/nsw.csv/earnings/description" not in ids

    def test_oversized_document_truncated(self, payload: dict[str, object]) -> None:
        bundle = build_evidence_bundle(payload, {"big.txt": "x" * (2 * 1024 * 1024)}, CLASSES)
        doc = next(
            item for item in items_of(bundle) if item["evidence_id"] == "ev:doc/big.txt"
        )
        assert len(str(doc["value"]).encode()) <= MAX_DOCUMENT_BYTES


class TestSemanticMap:
    def test_column_meaning_statuses(
        self, payload: dict[str, object], profiles: dict[str, dict[str, object]]
    ) -> None:
        slots = column_slots(build_semantic_map(payload, profiles))
        assert slots["unit_id"]["meaning"]["status"] == "evidenced"
        assert slots["earnings"]["meaning"]["status"] == "empty"
        assert slots["group"]["meaning"]["status"] == "not_offered"

    def test_profiler_hypotheses_fill_slots(
        self, payload: dict[str, object], profiles: dict[str, dict[str, object]]
    ) -> None:
        slots = column_slots(build_semantic_map(payload, profiles))
        assert slots["earnings"]["missing_sentinel"]["status"] == "hypothesis"
        assert slots["earnings"]["missing_sentinel"]["evidence_ids"] == []
        assert slots["unit_id"]["kind"]["status"] == "hypothesis"

    def test_dataset_slots_all_present_and_not_offered(
        self, payload: dict[str, object], profiles: dict[str, dict[str, object]]
    ) -> None:
        dataset = build_semantic_map(payload, profiles)["dataset"]
        assert isinstance(dataset, dict)
        dataset_slots = dataset["slots"]
        assert isinstance(dataset_slots, dict)
        assert len(dataset_slots) == 7
        assert all(slot["status"] == "not_offered" for slot in dataset_slots.values())


class TestIndexRows:
    def test_every_slot_emitted(
        self, payload: dict[str, object], profiles: dict[str, dict[str, object]]
    ) -> None:
        rows = slot_index_rows(build_semantic_map(payload, profiles))
        assert len(rows) == 7 + 3 * 8  # 7 dataset slots + 8 slots x 3 columns
        available = [row for row in rows if row.status in AVAILABLE_STATUSES]
        assert len(available) == 3  # meaning + missing_sentinel + kind

    def test_measured_rows_point_at_profile_artifacts(
        self, profiles: dict[str, dict[str, object]]
    ) -> None:
        rows = measured_index_rows(profiles, {"nsw.csv": "tableprofile:an-x:abc"})
        assert len(rows) == 3
        assert all(row.status is SemanticStatus.EVIDENCED for row in rows)
        assert rows[0].json_pointer.startswith("tableprofile:an-x:abc#/columns/")
