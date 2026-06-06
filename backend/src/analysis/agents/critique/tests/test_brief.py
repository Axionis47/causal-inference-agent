"""Contract tests for the critique brief.

Pin the capability shape, preflight refusal, and status / headline /
issues derivation. Critique deliberately raises no closed-enum flags;
those tests verify the absence.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.analysis.agents.base import CritiqueDecision, CritiqueFeedback, TreatmentEffectResult
from src.analysis.agents.base.state import AnalysisState, DatasetInfo
from src.analysis.agents.critique.brief import CAPABILITY, build_brief, preflight
from src.domain.briefs import Flag


# --- fixtures ---------------------------------------------------------------


def _effect(estimate: float = 1.0) -> TreatmentEffectResult:
    return TreatmentEffectResult(
        method="OLS",
        estimand="ATE",
        estimate=estimate,
        std_error=0.1,
        ci_lower=estimate - 0.2,
        ci_upper=estimate + 0.2,
        p_value=0.01,
        treatment_variable="t",
        outcome_variable="y",
    )


def _feedback(
    *,
    decision: CritiqueDecision = CritiqueDecision.APPROVE,
    iteration: int = 1,
    issues: list[str] | None = None,
    avg_score: float = 4.0,
) -> CritiqueFeedback:
    score = int(avg_score)
    return CritiqueFeedback(
        decision=decision,
        iteration=iteration,
        scores={
            "statistical_validity": score,
            "assumption_checking": score,
            "method_selection": score,
            "completeness": score,
            "reproducibility": score,
            "interpretation": score,
        },
        issues=issues or [],
        improvements=[],
        reasoning="test",
    )


def _make_state(
    *,
    treatment_effects: list[TreatmentEffectResult] | None = None,
    critique_history: list[CritiqueFeedback] | None = None,
) -> AnalysisState:
    state = AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(url="https://example.com/data"),
        treatment_variable="t",
        outcome_variable="y",
    )
    if treatment_effects is not None:
        state.treatment_effects = treatment_effects
    if critique_history is not None:
        state.critique_history = critique_history
    return state


# --- CAPABILITY -------------------------------------------------------------


class TestCapability:
    def test_name_is_critique(self):
        assert CAPABILITY.name == "critique"

    def test_needs_treatment_effects(self):
        assert CAPABILITY.needs == ("treatment_effects",)

    def test_delivers_critique_history(self):
        assert "critique_history" in CAPABILITY.delivers

    def test_is_frozen(self):
        with pytest.raises(ValidationError):
            CAPABILITY.name = "other"

    def test_success_criteria_pin_needs_not_met(self):
        flags_with_criteria = {
            c.raises_flag for c in CAPABILITY.success_criteria if c.raises_flag
        }
        assert Flag.NEEDS_NOT_MET in flags_with_criteria

    def test_capability_does_not_advertise_closed_enum_issue_flags(self):
        # Critique deliberately raises no primary issue flags; upstream
        # agents own that vocabulary. Only NEEDS_NOT_MET should appear.
        flags = {
            c.raises_flag for c in CAPABILITY.success_criteria if c.raises_flag
        }
        assert flags == {Flag.NEEDS_NOT_MET}


# --- preflight --------------------------------------------------------------


class TestPreflight:
    def test_refuses_when_treatment_effects_empty(self):
        state = _make_state(treatment_effects=[])
        brief = preflight(state)
        assert brief is not None
        assert brief.flags == [Flag.NEEDS_NOT_MET]
        assert "treatment_effects" in brief.headline

    def test_passes_when_treatment_effects_present(self):
        state = _make_state(treatment_effects=[_effect()])
        assert preflight(state) is None


# --- build_brief: status ----------------------------------------------------


class TestBuildBriefStatus:
    def test_failed_when_no_critique_history(self):
        state = _make_state(treatment_effects=[_effect()], critique_history=[])
        brief = build_brief(state)
        assert brief.status == "failed"
        assert brief.artifact_keys == []
        assert "no feedback" in brief.headline.lower()

    def test_done_when_feedback_present(self):
        state = _make_state(
            treatment_effects=[_effect()],
            critique_history=[_feedback()],
        )
        brief = build_brief(state)
        assert brief.status == "done"
        assert brief.artifact_keys == ["critique_history"]


# --- build_brief: headline + issues ----------------------------------------


class TestBuildBriefHeadline:
    def test_headline_includes_decision(self):
        state = _make_state(
            treatment_effects=[_effect()],
            critique_history=[_feedback(decision=CritiqueDecision.REJECT)],
        )
        brief = build_brief(state)
        assert "REJECT" in brief.headline

    def test_headline_includes_iteration(self):
        state = _make_state(
            treatment_effects=[_effect()],
            critique_history=[_feedback(iteration=3)],
        )
        brief = build_brief(state)
        assert "iteration 3" in brief.headline

    def test_uses_latest_feedback_when_history_has_many(self):
        state = _make_state(
            treatment_effects=[_effect()],
            critique_history=[
                _feedback(decision=CritiqueDecision.ITERATE, iteration=1),
                _feedback(decision=CritiqueDecision.APPROVE, iteration=2),
            ],
        )
        brief = build_brief(state)
        assert "APPROVE" in brief.headline
        assert "iteration 2" in brief.headline

    def test_raised_issues_lift_from_feedback(self):
        state = _make_state(
            treatment_effects=[_effect()],
            critique_history=[
                _feedback(issues=["estimates inconsistent", "outliers detected"]),
            ],
        )
        brief = build_brief(state)
        assert "estimates inconsistent" in brief.raised_issues
        assert "outliers detected" in brief.raised_issues

    def test_raised_issues_truncated_to_280_chars(self):
        long_issue = "x" * 500
        state = _make_state(
            treatment_effects=[_effect()],
            critique_history=[_feedback(issues=[long_issue])],
        )
        brief = build_brief(state)
        assert len(brief.raised_issues[0]) == 280
        assert brief.raised_issues[0].endswith("...")


# --- build_brief: flag absence ---------------------------------------------


class TestBuildBriefNoFlags:
    def test_emits_no_closed_enum_flags_on_reject(self):
        # Critique synthesizes; primary issue flags belong to upstream agents.
        state = _make_state(
            treatment_effects=[_effect()],
            critique_history=[_feedback(decision=CritiqueDecision.REJECT)],
        )
        brief = build_brief(state)
        assert brief.flags == []

    def test_emits_no_flags_on_approve(self):
        state = _make_state(
            treatment_effects=[_effect()],
            critique_history=[_feedback(decision=CritiqueDecision.APPROVE)],
        )
        brief = build_brief(state)
        assert brief.flags == []
