"""AgentBrief / AgentCapability / Flag: the orchestrator-worker handoff shape."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.domain.briefs import (
    AgentBrief,
    AgentCapability,
    Criterion,
    Flag,
    refusal_brief,
)


# --- Flag enum --------------------------------------------------------------


def test_flag_enum_includes_refusal_kinds():
    assert Flag.NEEDS_NOT_MET.value == "needs_not_met"
    assert Flag.PRECONDITION_FAILED.value == "precondition_failed"
    assert Flag.STATE_INCONSISTENT.value == "state_inconsistent"
    assert Flag.CAPABILITY_MISMATCH.value == "capability_mismatch"


def test_flag_enum_includes_per_stage_kinds():
    # spot-check one from each category to guard against accidental removal
    assert Flag.ENCODING_REQUIRED
    assert Flag.OUTCOME_SKEWED
    assert Flag.DAG_CONFLICT
    assert Flag.POOR_OVERLAP
    assert Flag.ATE_OUTLIER
    assert Flag.NOT_ROBUST


def test_flag_values_are_lowercase_snake_case():
    for f in Flag:
        assert f.value == f.value.lower()
        assert " " not in f.value


# --- Criterion --------------------------------------------------------------


def test_criterion_is_frozen():
    c = Criterion(id="x.y", description="some condition")
    with pytest.raises(ValidationError):
        c.description = "changed"


def test_criterion_rejects_empty_id_or_description():
    with pytest.raises(ValidationError):
        Criterion(id="", description="x")
    with pytest.raises(ValidationError):
        Criterion(id="x", description="")


def test_criterion_can_link_to_a_flag():
    c = Criterion(
        id="ps.overlap.flagged",
        description="raises poor_overlap when PS leaves [0.05, 0.95]",
        raises_flag=Flag.POOR_OVERLAP,
    )
    assert c.raises_flag is Flag.POOR_OVERLAP


# --- AgentCapability --------------------------------------------------------


def test_capability_one_line_format():
    cap = AgentCapability(
        name="ps_diagnostics",
        answers="Are propensity scores well-overlapped and balanced?",
    )
    assert cap.one_line() == (
        "ps_diagnostics: Are propensity scores well-overlapped and balanced?"
    )


def test_capability_is_frozen():
    cap = AgentCapability(name="x", answers="y")
    with pytest.raises(ValidationError):
        cap.name = "z"


def test_capability_rejects_extra_fields():
    with pytest.raises(ValidationError):
        AgentCapability(name="x", answers="y", extra_field="oops")


def test_capability_carries_typed_criteria():
    cap = AgentCapability(
        name="ps_diagnostics",
        answers="check overlap",
        needs=("treatment", "fitted_propensity"),
        delivers=("overlap_report",),
        refuses_when=("fitted_propensity missing",),
        success_criteria=(
            Criterion(id="ps.delivers", description="writes overlap_report"),
            Criterion(
                id="ps.overlap.flagged",
                description="poor_overlap when PS outside [0.05, 0.95]",
                raises_flag=Flag.POOR_OVERLAP,
            ),
        ),
    )
    assert len(cap.success_criteria) == 2
    assert cap.success_criteria[1].raises_flag is Flag.POOR_OVERLAP


# --- AgentBrief -------------------------------------------------------------


def test_brief_round_trip():
    brief = AgentBrief(
        agent="ps_diagnostics",
        status="done",
        headline="overlap acceptable, SMD within 0.1, calibration ok",
        flags=[],
        raised_issues=[],
        artifact_keys=["ps_diagnostics_result"],
    )
    payload = brief.model_dump()
    restored = AgentBrief.model_validate(payload)
    assert restored == brief


def test_brief_rejects_extra_fields():
    with pytest.raises(ValidationError):
        AgentBrief(
            agent="x",
            status="done",
            headline="ok",
            extra="should not be here",
        )


def test_brief_headline_cap_200_chars():
    with pytest.raises(ValidationError):
        AgentBrief(agent="x", status="done", headline="a" * 201)


def test_brief_headline_must_be_non_empty():
    with pytest.raises(ValidationError):
        AgentBrief(agent="x", status="done", headline="")


def test_brief_flags_reject_duplicates():
    with pytest.raises(ValidationError):
        AgentBrief(
            agent="x",
            status="done",
            headline="ok",
            flags=[Flag.POOR_OVERLAP, Flag.POOR_OVERLAP],
        )


def test_brief_raised_issue_cap_280_chars():
    with pytest.raises(ValidationError):
        AgentBrief(
            agent="x",
            status="done",
            headline="ok",
            raised_issues=["a" * 281],
        )


def test_brief_raised_issue_rejects_empty_string():
    with pytest.raises(ValidationError):
        AgentBrief(
            agent="x",
            status="done",
            headline="ok",
            raised_issues=[""],
        )


@pytest.mark.parametrize("status", ["done", "failed", "refused"])
def test_brief_accepts_each_status(status):
    AgentBrief(agent="x", status=status, headline="ok")


def test_brief_rejects_unknown_status():
    with pytest.raises(ValidationError):
        AgentBrief(agent="x", status="halfway", headline="ok")


# --- refusal_brief helper ---------------------------------------------------


def test_refusal_brief_constructs_refused_status():
    b = refusal_brief(
        agent="ps_diagnostics",
        flag=Flag.NEEDS_NOT_MET,
        headline="missing fitted_propensity",
    )
    assert b.status == "refused"
    assert b.flags == [Flag.NEEDS_NOT_MET]
    assert b.artifact_keys == []
    assert b.headline == "missing fitted_propensity"


def test_refusal_brief_carries_issue_detail():
    b = refusal_brief(
        agent="ps_diagnostics",
        flag=Flag.PRECONDITION_FAILED,
        headline="treatment is not binary",
        issues=["treatment column has 5 unique values, expected 2"],
    )
    assert b.raised_issues == [
        "treatment column has 5 unique values, expected 2",
    ]
