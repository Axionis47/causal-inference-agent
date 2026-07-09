"""Scoped handles retain their meaning when accepted prose travels between tasks."""

from copy import deepcopy
from uuid import UUID

import pytest

from backend.analysis import packets, references
from backend.context.contracts import SourceReference, Statement, empty_content


def context():
    result = empty_content(UUID("00000000-0000-0000-0000-000000000001"), ["cohort", "unit", "outcome"])
    source = SourceReference(kind="message", id=UUID(int=2), author_id=UUID(int=3))
    for index, column in enumerate(result.columns.values()):
        column.definition[f"meaning_{index}"] = Statement(value=f"Meaning of {column.identity.name}.", source=source)
    return result


def packet(content, selected, *, prior=None, available=()):
    scoped = content.model_copy(deep=True)
    scoped.columns = {cid: scoped.columns[cid] for cid in selected}
    return {
        "column_identities": [column.identity.model_dump() for column in scoped.columns.values()],
        "task_columns": list(selected),
        "evidence_records": packets.evidence_records(scoped),
        "available_context": list(available),
        "decisions": prior or {},
    }


def test_saved_claim_prose_keeps_its_column_and_source_meaning_in_larger_role_packet():
    content = context()
    cohort, unit, outcome = content.columns
    initial = packet(content, [cohort])
    first, first_aliases, _ = references.selections(initial)
    cohort_handle = first["task_columns"][0]
    source_handle = first["evidence_records"][0]["reference"]
    # Exact identifiers are decoded at acceptance; prose retains handles verbatim.
    claim = references.translate({
        "column_ids": [cohort_handle],
        "statement": f"{cohort_handle} records adoption timing, as described by {source_handle}.",
        "references": [source_handle],
    }, first_aliases)
    assert claim["column_ids"] == [cohort]
    assert claim["references"] == [initial["evidence_records"][0]["key"]]
    later_packet = packet(content, [outcome, unit, cohort], prior={"treatments": claim})
    later, later_aliases, _ = references.selections(later_packet)
    later_cohort = next(item for item in later["column_identities"] if item["name"] == "cohort")
    assert later_cohort["column_id"] == cohort_handle
    assert later_aliases[cohort_handle] == cohort
    assert later_aliases[source_handle] == first_aliases[source_handle]
    assert later["decisions"]["treatments"]["statement"] == claim["statement"]
    assert later["decisions"]["treatments"]["column_ids"] == [cohort_handle]
    assert later["decisions"]["treatments"]["references"] == [source_handle]
    assert initial == packet(content, [cohort]), "Encoding must not mutate the immutable source packet."


def test_column_and_evidence_handles_survive_reordering_renaming_and_subsetting():
    content = context()
    ids = list(content.columns)
    full, full_aliases, _ = references.selections(packet(content, ids))
    renamed = content.model_copy(deep=True)
    renamed.columns[ids[1]].identity.name = "opaque_742"
    renamed.columns[ids[1]].identity.position = 99
    subset, subset_aliases, _ = references.selections(packet(renamed, [ids[1]]))
    reordered, reordered_aliases, _ = references.selections(packet(content, list(reversed(ids))))
    assert subset_aliases.items() <= full_aliases.items()
    assert reordered_aliases == full_aliases
    assert subset["task_columns"][0] == full["task_columns"][1]
    assert reordered["task_columns"] == list(reversed(full["task_columns"]))
    for alias in full_aliases:
        assert len(alias) == 13 and all(char in "0123456789abcdef" for char in alias[1:])
    assert set(subset["reference_handles"]) == set(subset_aliases)
    assert len(subset_aliases) == 2, "Only the scoped column and its statement are exposed."


@pytest.mark.parametrize("key", [
    "relationships.adoption_order", "concepts.enrollment",
    "dataset.study_process.assignment", "columns.durable_column.definition.meaning",
])
def test_catalogued_statement_handle_is_identical_after_widening(key):
    before, before_aliases, before_choices = references.selections({"available_context": [key]})
    after, after_aliases, after_choices = references.selections({
        "evidence_records": [{"key": key, "reference": "obsolete_packet_position", "statement": {"basis": "source_statement"}}],
        "available_context": [key],
    })
    alias = before["available_context"][0]
    assert alias.startswith("e")
    assert after["evidence_records"][0]["reference"] == alias
    assert after["evidence_records"][0]["key"] == alias
    assert before_aliases == after_aliases == {alias: key}
    assert before_choices["requested_keys"] == after_choices["requested_keys"] == [alias]
    assert alias not in before_choices["citations"]
    assert alias in after_choices["citations"]


def test_collection_handles_are_stable_and_do_not_grant_evidence_authority():
    collection = "columns.durable_column"
    key = collection + ".definition.meaning"
    first, aliases, choices = references.selections({"available_context": [collection]})
    later, later_aliases, _ = references.selections({
        "available_context": ["columns.other", collection],
        "evidence_records": [{"key": key, "reference": "unused", "statement": {}}],
    })
    alias = first["available_context"][0]
    assert alias.startswith("k") and later["available_context"][1] == alias
    assert later_aliases[alias] == aliases[alias] == collection
    assert first["reference_handles"][alias]["kind"] == "context_key"
    assert alias not in choices["citations"]
    assert references.handle("c", collection) != references.handle("k", collection)
    assert references.handle("e", collection) != references.handle("k", collection)


@pytest.mark.parametrize("payload", [
    {"column_identities": [{"column_id": "one"}, {"column_id": "two"}]},
    {"available_context": ["columns.one", "columns.two"]},
    {"available_context": ["relationships.one"], "evidence_records": [{"key": "relationships.two", "statement": {}}]},
])
def test_colliding_handles_fail_closed_without_mutating_packet(monkeypatch, payload):
    monkeypatch.setattr(references, "handle", lambda kind, identity: kind + "0" * 12)
    original = deepcopy(payload)
    with pytest.raises(ValueError, match="Reference handle collision"):
        references.selections(payload)
    assert payload == original


def test_evidence_packet_itself_rejects_colliding_handles(monkeypatch):
    monkeypatch.setattr(references, "handle", lambda kind, identity: kind + "0" * 12)
    with pytest.raises(ValueError, match="Reference handle collision"):
        packets.evidence_records(context())


def test_repeated_identity_is_safe_but_a_missing_handle_is_never_fuzzily_resolved():
    payload, aliases, _ = references.selections({"available_context": ["relationships.same"] * 2})
    assert len(aliases) == 1
    assert payload["available_context"][0] == payload["available_context"][1]
    assert references.translate("e000000000000", aliases) == "e000000000000"
