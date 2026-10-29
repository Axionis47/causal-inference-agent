"""Source fidelity, all-column coverage, bounded corrections and refusal to invent context."""

import pytest

from causal_agent.common.llm import set_llm
from causal_agent.memory.context import extract as E
from causal_agent.memory.context.contracts import Assessment, ColumnIdentity, Concept, Correction, Participant
from causal_agent.memory.context.tests.builders import COLUMNS, SOURCE, Scripted, citation, draft, happy


@pytest.fixture(autouse=True)
def model_guard():
    set_llm(Scripted([]))
    yield
    set_llm(None)


def test_cited_unicode_extraction_preserves_source():
    model = happy()
    set_llm(model)
    result = E.extract(SOURCE, COLUMNS)
    assert result.content.columns[1].statements[0].value == "dollars"
    assert result.content.relationships[0].kind == "association"
    assert result.model_calls == len(model.calls) == 2
    assert E.validate(result.content, SOURCE, COLUMNS) == []


@pytest.mark.parametrize("source", ["", " \n\t "])
def test_empty_source_has_every_column_and_no_model(source):
    columns = [*COLUMNS, ColumnIdentity(column_id="c3", position=2, name="apparently_treatment", key="apparently_treatment")]
    result = E.extract(source, columns)
    assert [c.column_id for c in result.content.columns] == ["c1", "c2", "c3"]
    assert all(not c.statements for c in result.content.columns)
    assert not result.content.dataset and not result.content.relationships and result.model_calls == 0


def test_invalid_citation_is_rejected():
    record = draft()
    record.columns[1].statements[0].citations[0].start += 1
    assert any("citation does not match" in e for e in E.validate(record, SOURCE, COLUMNS)), "Invalid citations must be rejected"
    record.columns[1].statements[0].citations[0].quote = "invented quote"
    set_llm(Scripted([record, record]))
    with pytest.raises(E.InvalidExtraction):
        E.extract(SOURCE, COLUMNS)


def test_repeated_passage_offsets_and_whitespace():
    source = SOURCE + "\n" + SOURCE
    record = draft()
    cite = record.columns[1].statements[0].citations[0]
    cite.start += len(SOURCE) + 1
    cite.end += len(SOURCE) + 1
    cite.source_line += SOURCE.count("\n") + 1
    assert E.validate(record, source, COLUMNS) == []
    cite.quote = "amount  is measured in dollars."
    assert E.validate(record, source, COLUMNS)


def test_participant_validation():
    record = draft()
    record.relationships[0].participants[0].id = "another-version-column"
    assert any("unknown column participant" in e for e in E.validate(record, SOURCE, COLUMNS)), "Cross-version participants must be rejected"


def test_revision_after_semantic_assessment_is_bounded():
    wrong = draft()
    wrong.columns[1].statements[0].value = "euros"
    model = Scripted(
        [wrong, Assessment(corrections=[Correction(statement_ids=["s1"], source_lines=[2], reason="The source says dollars.")]), draft(), Assessment()]
    )
    set_llm(model)
    result = E.extract(SOURCE, COLUMNS)
    assert result.content.columns[1].statements[0].value == "dollars"
    assert result.model_calls == len(model.calls) == 4
    assert "The source says dollars." in model.calls[2][1][1]


def test_unsupported_interpretation_never_publishes():
    model = Scripted(
        [
            draft(),
            Assessment(corrections=[Correction(statement_ids=["s1"], source_lines=[2], reason="Wrong mapping")]),
            draft(),
            Assessment(corrections=[Correction(statement_ids=["s1"], source_lines=[2], reason="Still wrong")]),
        ]
    )
    set_llm(model)
    with pytest.raises(E.InvalidExtraction):
        E.extract(SOURCE, COLUMNS)
    assert len(model.calls) == 4


def test_all_columns_required_even_when_unmentioned():
    record = draft()
    record.columns.pop()
    assert E.validate(record, SOURCE, COLUMNS)
    record = draft()
    record.columns.append(record.columns[0])
    assert E.validate(record, SOURCE, COLUMNS)


def test_qualified_compound_relationship_and_unmeasured_concept():
    record = draft()
    record.concepts.append(
        Concept(id="event1", kind="event", value="Joint recording", basis="interpretation", citations=[citation(SOURCE, "recorded together")])
    )
    relation = record.relationships[0]
    relation.participants.append(Participant(kind="event", id="event1"))
    relation.qualifications = "This does not establish causation."
    relation.conditions = "During joint recording"
    assert E.validate(record, SOURCE, COLUMNS) == []
    restored = type(record).model_validate_json(record.model_dump_json())
    assert restored.relationships[0].directness == "unspecified"
    assert restored.relationships[0].qualifications == relation.qualifications
    assert len(restored.relationships) == 1 and len(restored.relationships[0].participants) == 3


def test_duplicate_identifiers_and_report_bounds(monkeypatch):
    record = draft()
    record.relationships[0].id = "s1"
    assert any("unique" in e for e in E.validate(record, SOURCE, COLUMNS))
    monkeypatch.setattr(E, "MAX_REPORT_BYTES", 10)
    assert any("bounded report" in e for e in E.validate(draft(), SOURCE, COLUMNS))


def test_transport_failure_is_not_retried_as_semantic_correction():
    model = Scripted([OSError("provider unavailable")])
    set_llm(model)
    with pytest.raises(OSError):
        E.extract(SOURCE, COLUMNS)
    assert len(model.calls) == 1


def test_unique_quotes_are_located_without_changing_source_or_words():
    record = draft()
    cite = record.columns[1].statements[0].citations[0]
    quote = cite.quote
    cite.start, cite.end = 0, 1
    E.resolve_citations(record, SOURCE)
    assert cite.start == SOURCE.index(quote) and SOURCE[cite.start : cite.end] == quote
    assert E.validate(record, SOURCE, COLUMNS) == []


def test_ambiguous_quote_is_not_silently_assigned_to_first_occurrence():
    record = draft()
    cite = record.columns[1].statements[0].citations[0]
    cite.start, cite.end = 0, 1
    source = SOURCE.replace("amount is measured in dollars.", "amount is measured in dollars. amount is measured in dollars.")
    E.resolve_citations(record, source)
    assert cite.start >= len(source)
    assert E.validate(record, source, COLUMNS)


def test_provider_schema_simplification_keeps_local_limits():
    from pydantic import ValidationError

    from causal_agent.memory.context.contracts import DraftContext

    record = draft().model_dump()
    record["columns"][1]["statements"][0]["value"] = "x" * 10001
    with pytest.raises(ValidationError):
        DraftContext.model_validate(record)


def test_full_source_is_sent_without_silent_truncation():
    import json

    source = "z" * 99000 + "\n" + SOURCE
    model = happy(source=source)
    set_llm(model)
    result = E.extract(source, COLUMNS)
    payload = json.JSONDecoder().raw_decode(model.calls[0][1][1])[0]
    assert "".join(line["text"] for line in payload["source_lines"]) == source
    assert result.content.columns[1].statements[0].citations[0].start > 99000


@pytest.mark.parametrize("source", ["a a", "a\r\na a", "a\r a a", "a\u2028a a"])
def test_failed_locator_cannot_accidentally_validate_first_character(source):
    from causal_agent.memory.context.contracts import Citation, ColumnContext, ColumnStatement, DraftContext

    record = DraftContext(columns=[ColumnContext(column_id=c.column_id) for c in COLUMNS])
    record.columns[0].statements.append(
        ColumnStatement(
            id="s",
            field="definition",
            value="a",
            basis="source_statement",
            citations=[Citation(source_line=len(source.splitlines(keepends=True)), quote="a")],
        )
    )
    E.resolve_citations(record, source)
    assert E.validate(record, source, COLUMNS), "Ambiguous citations cannot pass through placeholder offsets"


@pytest.mark.parametrize("statement_ids,lines", [(["unknown"], [2]), (["s1"], [0]), (["s1"], [1000])])
def test_assessor_references_must_resolve(statement_ids, lines):
    assessment = Assessment(corrections=[Correction(statement_ids=statement_ids, source_lines=lines, reason="Correct this mapping")])
    assert E.validate_assessment(assessment, draft(), SOURCE)


def test_omission_correction_references_source_without_invented_statement():
    assessment = Assessment(corrections=[Correction(source_lines=[2], reason="The meaning of code is supplied but omitted")])
    assert E.validate_assessment(assessment, draft(), SOURCE) == []
    model = Scripted([draft(), assessment, draft(), assessment])
    set_llm(model)
    with pytest.raises(E.InvalidExtraction):
        E.extract(SOURCE, COLUMNS)
    assert len(model.calls) == 4


def test_invalid_assessment_output_cannot_publish():
    model = Scripted([draft(), ValueError("Malformed assessment"), draft(), ValueError("Still malformed")])
    set_llm(model)
    with pytest.raises(E.InvalidExtraction):
        E.extract(SOURCE, COLUMNS)
    assert len(model.calls) == 4


def test_assessor_provider_schema_retains_shape_and_local_validation():
    from pydantic import ValidationError

    schema = Assessment.model_json_schema()
    assert schema["$defs"]["Correction"]["required"] == ["source_lines", "reason"]
    assert schema["$defs"]["Correction"]["properties"]["source_lines"]["items"]["type"] == "integer"
    assert "maxItems" not in schema["properties"]["corrections"]
    assert schema["$defs"]["Correction"]["properties"]["source_lines"]["minItems"] == 1
    from causal_agent.memory.context.contracts import DraftContext

    assert DraftContext.model_json_schema()["$defs"]["Relationship"]["properties"]["participants"]["minItems"] == 2
    with pytest.raises(ValidationError):
        Assessment.model_validate({"corrections": [{"source_lines": [], "reason": "Missing passage"}]})
    with pytest.raises(ValidationError):
        Assessment.model_validate({"corrections": [], "unexpected": True})


def test_participant_kind_is_derived_without_changing_claim_or_identity():
    record = draft()
    record.concepts.append(
        Concept(id="visit", kind="event", value="Joint recording", basis="interpretation", citations=[citation(SOURCE, "recorded together")])
    )
    record.relationships[0].participants.append(Participant(id="visit", kind="concept"))
    model = Scripted([record, Assessment()])
    set_llm(model)
    result = E.extract(SOURCE, COLUMNS)
    participant = result.content.relationships[0].participants[-1]
    assert (participant.id, participant.kind) == ("visit", "event")
    assert result.content.relationships[0].value == "Recorded together, without an established causal relationship."
    assert result.model_calls == 2


def test_participant_resolution_cannot_invent_or_disambiguate_entities():
    record = draft()
    record.relationships[0].participants[0].id = "undeclared"
    E.resolve_participants(record, COLUMNS)
    assert any("unknown column participant" in e for e in E.validate(record, SOURCE, COLUMNS))
    record = draft()
    record.concepts.append(Concept(id="c1", kind="event", value="Joint recording", basis="interpretation", citations=[citation(SOURCE, "recorded together")]))
    E.resolve_participants(record, COLUMNS)
    assert record.relationships[0].participants[0].kind == "column"
    assert any("unique" in e for e in E.validate(record, SOURCE, COLUMNS))


def test_nonblocking_notes_do_not_trigger_revision_or_claim_readiness():
    assessment = Assessment(notes=["Missingness is not supplied; the timing fact appears in both a column attribute and a relationship."])
    model = Scripted([draft(), assessment])
    set_llm(model)
    result = E.extract(SOURCE, COLUMNS)
    assert result.model_calls == len(model.calls) == 2
    assert result.assessment.notes == assessment.notes
    assert not result.assessment.corrections
