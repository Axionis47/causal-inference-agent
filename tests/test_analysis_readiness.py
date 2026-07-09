"""Applied source references, not model confidence or column names, govern preflight."""

import json
from copy import deepcopy
from uuid import UUID, uuid5

import pytest
from pydantic import ValidationError

from backend.analysis.readiness import assess
from backend.context.contracts import (
    BriefClaim,
    BriefDraft,
    ContextContent,
    ContextEdit,
    DatasetStatement,
    EditStatement,
    SaveInput,
    SourceReference,
    Statement,
    empty_content,
)
from backend.context.domain import apply_brief, apply_edits, project
from backend.context.service import fingerprint


@pytest.fixture(autouse=True)
def isolated():
    yield


VERSION, REVISION, OWNER = UUID(int=101), UUID(int=102), UUID(int=103)


def fixture():
    content = empty_content(VERSION, ["x1", "x2", "x3", "x4", "optional"])
    columns = list(content.columns)
    source = SourceReference(kind="review", id=REVISION, author_id=OWNER)
    for index, cid in enumerate(columns[:4]):
        content.columns[cid].definition[f"definition_{index}"] = Statement(value=f"Recorded measurement {index}.", source=source, status="reviewed")
    for field, identity, value in [
        ("question", "comparison", "Compare the two specified intervention variables jointly."),
        ("population", "population", "Enrolled participants."),
        ("observations", "unit", "One participant per observation."),
        ("measurement", "timing", "Outcome follows the baseline interventions."),
        ("study_process", "change", "The two interventions are assigned at baseline."),
    ]:
        getattr(content.dataset, field)[identity] = DatasetStatement(value=value, source=source, status="reviewed")
    draft = BriefDraft.model_validate({
        "question": "Estimate the two prespecified effects.",
        "target": {"column_ids": columns[:2], "meaning": {"source_keys": [f"columns.{cid}.definition.definition_{index}" for index, cid in enumerate(columns[:2])]}},
        "treatments": [{"id": f"factor_{index}", "column_ids": [columns[index]], "meaning": {"source_keys": [f"columns.{columns[index]}.definition.definition_{index}"]}, "change": {"source_keys": ["dataset.study_process.change"]}} for index in (2, 3)],
        **{field: {"source_keys": [key]} for field, key in {"comparison": "dataset.question.comparison", "population": "dataset.population.population", "unit": "dataset.observations.unit", "timing": "dataset.measurement.timing"}.items()},
    })
    return apply_brief(content, draft, [], OWNER, REVISION), draft, columns


def view(content, **kwargs):
    return assess(content, VERSION, REVISION, source_sha256="a" * 64, **kwargs)


def test_historical_context_and_request_serialization_hashes_remain_exact():
    original = empty_content(VERSION, ["source"]).model_dump(mode="json")
    original["columns"][next(iter(original["columns"]))]["definition"]["old"] = {
        "value": "Historical source meaning.", "basis": "source_statement", "citations": [], "qualifications": "",
        "status": "reviewed", "source": {"kind": "review", "id": str(REVISION), "author_id": str(OWNER), "text": ""},
        "review_source": None, "replaces": None, "reason": "",
    }
    restored = ContextContent.model_validate(original)
    assert restored.model_dump(mode="json") == original
    assert json.loads(restored.model_dump_json()) == original
    assert fingerprint(restored.model_dump(mode="json")) == fingerprint(original)
    assert "analysis_brief" not in original
    assert "analysis_brief" in ContextContent.model_json_schema(mode="serialization")["properties"]
    old_request = {"submission_id": str(OWNER), "expected_revision_id": str(REVISION), "state": "draft", "edits": [], "leave_remaining_unresolved": False}
    assert SaveInput.model_validate(old_request).model_dump(mode="json") == old_request


def test_multiple_treatments_and_multicolumn_target_are_ready_without_optional_role_questions():
    content, _, columns = fixture()
    content.analysis_brief.optional_unknowns = ["The optional column's causal role is unknown."]
    content.columns[columns[-1]].definition["optional_unresolved"] = Statement(value="Unresolved optional meaning.", source=content.analysis_brief.source, status="unresolved")
    result = view(content)
    assert result.status == "ready" and result.missing == []
    assert result.question == content.analysis_brief.question
    assert result.optional_unknowns == content.analysis_brief.optional_unknowns


@pytest.mark.parametrize("field", ["comparison", "population", "unit", "timing"])
def test_missing_essential_reference_never_falls_back_to_names_or_other_dataset_text(field):
    content, _, _ = fixture()
    getattr(content.analysis_brief, field).source_keys = []
    result = view(content)
    assert result.status == "needs_context" and field in [item.field for item in result.missing]


@pytest.mark.parametrize("field", ["comparison", "population", "unit", "timing"])
def test_valid_but_unrelated_definition_cannot_fill_an_essential_field(field):
    content, _, _ = fixture()
    getattr(content.analysis_brief, field).source_keys = content.analysis_brief.target.meaning.source_keys[:1]
    assert view(content).status == "needs_context"


@pytest.mark.parametrize("status,basis", [("proposed", "source_statement"), ("unresolved", "source_statement"), ("withdrawn", "source_statement"), ("reviewed", "interpretation")])
def test_essential_source_authority_must_be_reviewed_and_not_an_interpretation(status, basis):
    content, _, _ = fixture()
    content.dataset.population["population"].status = status
    content.dataset.population["population"].basis = basis
    assert view(content).status == "needs_context"


@pytest.mark.parametrize("defect", ["unknown_column", "overlap", "duplicate_factor", "uncovered_column"])
def test_ambiguous_bindings_block(defect):
    content, _, columns = fixture()
    brief = content.analysis_brief
    if defect == "unknown_column":
        brief.target.column_ids.append("unknown")
    elif defect == "overlap":
        brief.treatments[0].column_ids = [columns[0]]
    elif defect == "duplicate_factor":
        brief.treatments[1].id = brief.treatments[0].id
    else:
        brief.target.column_ids.append(columns[-1])
    assert view(content).status == "needs_context"


def test_reviewed_conditional_records_are_visible_and_never_fill_essential_meaning():
    content, _, _ = fixture()
    content.dataset.qualifications["conditional"] = DatasetStatement(value="No interference is a working assumption.", basis="interpretation", source=content.analysis_brief.source, status="reviewed")
    content.analysis_brief.assumptions = [BriefClaim(source_keys=["dataset.qualifications.conditional"])]
    assert view(content).status == "ready_with_assumptions"
    content.dataset.qualifications["conditional"].status = "proposed"
    assert view(content).status == "needs_context"
    content.dataset.qualifications["conditional"].status = "reviewed"
    content.analysis_brief.population.source_keys = ["dataset.qualifications.conditional"]
    assert view(content).status == "needs_context"


def test_selected_qualifications_remain_visible_in_conditional_readiness():
    content, _, _ = fixture()
    content.dataset.population["population"].qualifications = "Inference is limited to the enrolled population."
    result = view(content)
    assert result.status == "ready_with_assumptions"
    assert result.uncertainties == [BriefClaim(source_keys=["dataset.population.population"])]


@pytest.mark.parametrize("change", ["version", "revision", "checksum", "policy", "brief_question", "source_record"])
def test_fingerprint_binds_exact_revision_source_question_and_policy(change):
    content, _, _ = fixture()
    original = view(content)
    kwargs = {"dataset_version_id": VERSION, "context_revision_id": REVISION, "source_sha256": "a" * 64}
    if change in {"version", "revision"}:
        kwargs["dataset_version_id" if change == "version" else "context_revision_id"] = UUID(int=999)
    elif change == "checksum":
        kwargs["source_sha256"] = "b" * 64
    elif change == "policy":
        kwargs["policy_version"] = "next-policy"
    elif change == "brief_question":
        content.analysis_brief.question += " Different request."
    else:
        content.dataset.population["population"].value += " Updated supplied restriction."
    assert assess(content, **kwargs).fingerprint != original.fingerprint


def test_separate_launch_question_is_rejected_and_absent_brief_is_not_inferred():
    content, _, _ = fixture()
    assert view(content, question="A different request.").status == "needs_context"
    content.analysis_brief = None
    assert view(content).status == "needs_context"
    assert view(content).question == ""


def test_same_apply_edit_pointer_resolves_to_exact_revision_and_does_not_mutate_proposal():
    content, draft, _ = fixture()
    before = deepcopy(content.model_dump(mode="json"))
    draft.population.source_keys = ["edit:0"]
    edits = [ContextEdit(action="add", key="dataset.population", statement=EditStatement(value="A clarified population."))]
    new_revision = UUID(int=104)
    result = apply_brief(apply_edits(content, edits, OWNER, new_revision), draft, edits, OWNER, new_revision, source_text="Applied current user clarification.")
    expected = "dataset.population." + uuid5(new_revision, "statement:0").hex
    assert result.analysis_brief.population.source_keys == [expected]
    assert result.analysis_brief.source.id == new_revision
    assert result.analysis_brief.source.text == "Applied current user clarification."
    assert result.dataset.population[expected.rsplit(".", 1)[-1]].status == "reviewed"
    assert content.model_dump(mode="json") == before and draft.population.source_keys == ["edit:0"]


@pytest.mark.parametrize("reference", ["edit:-1", "edit:01", "edit:1", "edit:x", "dataset.population.unknown"])
def test_invalid_brief_pointer_fails_without_mutating_applied_context(reference):
    content, draft, _ = fixture()
    before = content.model_dump(mode="json")
    draft.population.source_keys = [reference]
    with pytest.raises(ValueError):
        apply_brief(content, draft, [], OWNER, REVISION)
    assert content.model_dump(mode="json") == before


def test_brief_pointer_cannot_target_review_action_and_context_projection_retains_exact_dependencies():
    content, draft, _ = fixture()
    draft.population.source_keys = ["edit:0"]
    with pytest.raises(ValueError, match="newly added"):
        apply_brief(content, draft, [ContextEdit(action="review", key="dataset.population.population")], OWNER, REVISION)
    projected, included, _ = project(content, ["analysis_brief"])
    assert projected.analysis_brief == content.analysis_brief
    assert "analysis_brief" in included and "dataset.population.population" in included
    scoped, _, _ = project(content, ["dataset.population"])
    assert scoped.analysis_brief is None


@pytest.mark.parametrize("incoming", [False, True])
def test_live_declared_conflicts_block_even_when_conflicting_record_not_selected(incoming):
    content, _, _ = fixture()
    content.dataset.qualifications["conflict"] = DatasetStatement(value="A different source disputes the population.", source=content.analysis_brief.source, status="reviewed")
    if incoming:
        content.dataset.qualifications["conflict"].conflicts_with = ["dataset.population.population"]
    else:
        content.dataset.population["population"].conflicts_with = ["dataset.qualifications.conflict"]
    assert any(c.field == "population.conflicts" for c in view(content).missing)
    projected, included, _ = project(content, ["analysis_brief"])
    assert "dataset.qualifications.conflict" in included
    assert projected.dataset.qualifications["conflict"].value == content.dataset.qualifications["conflict"].value
    content.dataset.qualifications["conflict"].status = "withdrawn"
    assert view(content).status == "ready"


def test_unresolved_essential_competitor_blocks_but_reviewed_prose_is_not_automatically_classified():
    content, _, _ = fixture()
    content.dataset.population["alternative"] = DatasetStatement(value="A conflicting population description.", source=content.analysis_brief.source, status="unresolved")
    assert any(c.field == "population.unresolved" for c in view(content).missing)
    content.dataset.population["alternative"].status = "reviewed"
    assert view(content).status == "ready", "This structural policy does not claim to detect contradictions from prose."


def test_replacement_preserves_declared_conflict_unless_explicitly_resolved():
    content, _, _ = fixture()
    content.dataset.qualifications["dispute"] = DatasetStatement(value="Disputed population account.", source=content.analysis_brief.source, status="reviewed", conflicts_with=["dataset.population.population"])
    edits = [ContextEdit(action="replace", key="dataset.population.population", reason="Clarify the source account.", statement=EditStatement(value="Clarified account."))]
    revision = UUID(int=105)
    result = apply_edits(content, edits, OWNER, revision)
    new = result.dataset.population[uuid5(revision, "statement:0").hex]
    assert new.conflicts_with == ["dataset.qualifications.dispute"]
    edits[0].statement.conflicts_with = []
    assert apply_edits(content, edits, OWNER, revision).dataset.population[uuid5(revision, "statement:0").hex].conflicts_with == []


def test_conflict_pointers_resolve_atomically_and_invalid_references_fail():
    content, _, _ = fixture()
    edits = [ContextEdit(action="add", key="dataset.population", statement=EditStatement(value="Alternative account.", conflicts_with=["edit:1"])),
             ContextEdit(action="add", key="dataset.population", statement=EditStatement(value="Other account."))]
    result = apply_edits(content, edits, OWNER, REVISION)
    assert result.dataset.population[uuid5(REVISION, "statement:0").hex].conflicts_with == ["dataset.population." + uuid5(REVISION, "statement:1").hex]
    edits[0].statement.conflicts_with = ["dataset.population.forged"]
    with pytest.raises(ValidationError, match="Declared conflicts"):
        apply_edits(content, edits, OWNER, REVISION)


def test_draft_cannot_author_readiness_or_duplicate_factual_authority():
    _, draft, _ = fixture()
    with pytest.raises(ValidationError, match="Extra inputs"):
        BriefDraft.model_validate({**draft.model_dump(), "ready": True})
    with pytest.raises(ValidationError, match="Extra inputs"):
        BriefClaim.model_validate({"source_keys": ["dataset.population.population"], "text": "Invented meaning", "basis": "supplied"})
