"""Tests for provider-field classification (T-008; PRD-001 §5.7; D-030)."""

from __future__ import annotations

from pathlib import Path

from causal.intake.contracts import ContextClass, SemanticStatus
from causal.intake.fields import classify_capture, escape_pointer, load_field_classes
from causal.intake.kaggle import capture
from tests.intake.conftest import FrozenKaggleClient

REGISTRY_PATH = (
    Path(__file__).resolve().parents[2] / "registries" / "kaggle-field-classes.v1.json"
)


def rows_by_key() -> dict[tuple[str, str | None, str | None, str], object]:
    classes = load_field_classes(REGISTRY_PATH)
    payload = capture(FrozenKaggleClient(), "lalonde/nsw").payload
    return {
        (row.scope_kind, row.table_name, row.column_name, row.field_or_slot_name): row
        for row in classify_capture(payload, classes)
    }


class TestRegistry:
    def test_registry_loads(self) -> None:
        classes = load_field_classes(REGISTRY_PATH)
        assert classes[("dataset", "title")] is ContextClass.SEMANTIC
        assert classes[("column", "description")] is ContextClass.SEMANTIC
        assert classes[("dataset", "usabilityRating")] is ContextClass.POPULARITY

    def test_pointer_escaping(self) -> None:
        assert escape_pointer("a/b~c") == "a~1b~0c"


class TestClassification:
    def test_semantic_field_with_value_is_evidenced(self) -> None:
        row = rows_by_key()[("dataset", None, None, "title")]
        assert row.context_class is ContextClass.SEMANTIC  # type: ignore[attr-defined]
        assert row.status is SemanticStatus.EVIDENCED  # type: ignore[attr-defined]
        assert row.evidence_count == 1  # type: ignore[attr-defined]

    def test_semantic_field_with_blank_value_is_empty(self) -> None:
        for key in (("dataset", None, None, "subtitle"),
                    ("column", "nsw.csv", "earnings", "description")):
            row = rows_by_key()[key]
            assert row.status is SemanticStatus.EMPTY  # type: ignore[attr-defined]
            assert row.evidence_count == 0  # type: ignore[attr-defined]

    def test_popularity_and_operational_are_not_applicable(self) -> None:
        rows = rows_by_key()
        rating = rows[("dataset", None, None, "usabilityRating")]
        assert rating.context_class is ContextClass.POPULARITY  # type: ignore[attr-defined]
        assert rating.status is SemanticStatus.NOT_APPLICABLE  # type: ignore[attr-defined]

    def test_unlisted_field_defaults_to_operational(self) -> None:
        row = rows_by_key()[("dataset", None, None, "mysteryField")]
        assert row.context_class is ContextClass.OPERATIONAL  # type: ignore[attr-defined]

    def test_every_indexed_field_has_exactly_one_class(self) -> None:
        rows = rows_by_key()
        assert rows  # acceptance 15: total classification
        for row in rows.values():
            assert isinstance(row.context_class, ContextClass)  # type: ignore[attr-defined]

    def test_structural_column_facts_indexed_with_pointers(self) -> None:
        row = rows_by_key()[("column", "nsw.csv", "unit_id", "type")]
        assert row.context_class is ContextClass.STRUCTURAL  # type: ignore[attr-defined]
        assert row.json_pointer == "/files_response/files/0/columns/0/type"  # type: ignore[attr-defined]
