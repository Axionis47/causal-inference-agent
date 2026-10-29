"""Authority, shared relationship identity and immutable review transitions."""

import pytest

from causal_agent.memory.context.contracts import ColumnIdentity, Participant
from causal_agent.memory.context.review import ColumnDisposition, ContextEdit, Knowledge, KnowledgeStatement, SourceReference, apply_edits, complete_review


def knowledge():
    return Knowledge(
        columns=[ColumnIdentity(column_id=x, name=x, key=x, position=i) for i, x in enumerate(("a", "b"))],
        statements=[
            KnowledgeStatement(
                id="relation",
                scope="relationship",
                field="causal_influence",
                value="A changes B",
                participants=[Participant(id="a", kind="column", role="from"), Participant(id="b", kind="column", role="to")],
                source=SourceReference(kind="extraction", id="pack"),
            )
        ],
    )


def test_column_review_does_not_confirm_relationship():
    original = knowledge()
    result = complete_review(original, [ColumnDisposition(column_id="a", status="reviewed"), ColumnDisposition(column_id="b", status="unresolved")])
    assert result.statements[0].status == "proposed"
    assert not original.dispositions
    assert result.for_column("a")[0] == result.for_column("b")[0]


def test_all_columns_need_explicit_disposition():
    with pytest.raises(ValueError, match="every column"):
        complete_review(knowledge(), [ColumnDisposition(column_id="a", status="reviewed")])


def test_withdrawal_does_not_assert_opposite():
    original = knowledge()
    result = apply_edits(original, [ContextEdit(action="retire", target_id="relation", reason="Not supported")], SourceReference(kind="review", id="r"))
    assert result.statements[0].value == "A changes B"
    assert result.statements[0].status == "withdrawn"
    assert not result.for_column("a")
    assert original.statements[0].status == "proposed"


def test_replacement_visible_under_both_participants():
    original = knowledge()
    replacement = original.statements[0].model_copy(update={"id": "corrected", "value": "A changes B before enrollment", "timing": "before enrollment"})
    result = apply_edits(
        original,
        [ContextEdit(action="replace", target_id="relation", statement=replacement, reason="Clarified timing")],
        SourceReference(kind="message", id="m", text="A changes B before enrollment"),
    )
    for column_id in ("a", "b"):
        assert result.for_column(column_id)[0].id == "corrected"
        assert result.for_column(column_id)[0].replaces == "relation"
    assert original.statements[0].timing == ""


def test_atomic_edit_rejects_conflicting_operations():
    original = knowledge()
    with pytest.raises(ValueError, match="Conflicting"):
        apply_edits(
            original,
            [ContextEdit(action="review", target_id="relation"), ContextEdit(action="unresolved", target_id="relation")],
            SourceReference(kind="review", id="r"),
        )
    assert original.statements[0].status == "proposed"


def test_extraction_has_no_confirmation_authority():
    with pytest.raises(ValueError, match="Extraction"):
        apply_edits(knowledge(), [ContextEdit(action="review", target_id="relation")], SourceReference(kind="extraction", id="model"))


def test_unknown_participant_cannot_enter_knowledge():
    original = knowledge()
    bad = original.statements[0].model_copy(
        update={"id": "new", "participants": [Participant(kind="column", id="foreign"), Participant(kind="column", id="a")]}
    )
    with pytest.raises(ValueError, match="unknown participant"):
        apply_edits(original, [ContextEdit(action="add", statement=bad)], SourceReference(kind="review", id="r"))


def test_changed_assignment_reopens_canonical_dependents_without_reversing_them():
    from causal_agent.memory.context.projection import validate_bindings
    from causal_agent.memory.context.review import KnowledgeStatement, RequirementBinding

    source = SourceReference(kind="review", id="review")
    original = knowledge()
    original.statements += [
        KnowledgeStatement(
            id=field,
            scope="requirement",
            field=field,
            value=str(value),
            status="reviewed",
            source=source,
            binding=RequirementBinding(address="claim:assignment." + field, value=value),
        )
        for field, value in [("kind", "own_choice"), ("rule", "Participant choice")]
    ]
    replacement = original.statements[-2].model_copy(
        update={"id": "new-kind", "value": "lottery", "binding": RequirementBinding(address="claim:assignment.kind", value="lottery")}
    )
    changed = apply_edits(original, [ContextEdit(action="replace", target_id="kind", statement=replacement, reason="Correct assignment")], source)
    dependent = next(s for s in changed.statements if s.id == "rule")
    assert dependent.status == "proposed" and dependent.binding.value == "Participant choice"
    assert original.statements[-1].status == "reviewed"
    validate_bindings(changed)
