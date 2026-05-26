"""The standard orchestrator's context prompt stays bounded as state grows.

Before this PR, _build_context_prompt embedded several unbounded lists
(EDA quality issues, multicollinearity warnings, treatment effects,
critique issues, critique improvements). On long jobs the prompt grew
linearly with state and started to dwarf the actual decision the LLM
was being asked to make. The fix caps each list at a small max-show
count, with a "(+N more)" tail.

These tests pin the cap so a future regression that re-introduces
unbounded dumping fails loudly.
"""

from __future__ import annotations

import pytest

from src.analysis.agents import (
    AnalysisState,
    CritiqueDecision,
    CritiqueFeedback,
    DatasetInfo,
    EDAResult,
    TreatmentEffectResult,
)
from src.analysis.orchestrator.standard import StandardOrchestrator


def _state_with_long_lists(n: int) -> AnalysisState:
    state = AnalysisState(
        job_id="big",
        dataset_info=DatasetInfo(url="u", name="n"),
        treatment_variable="t",
        outcome_variable="y",
    )
    state.eda_result = EDAResult(
        data_quality_score=80.0,
        data_quality_issues=[f"issue_{i}_with_some_text" for i in range(n)],
        multicollinearity_warnings=[f"warn_{i}_about_collinearity" for i in range(n)],
        balance_summary="ok",
    )
    state.treatment_effects = [
        TreatmentEffectResult(
            method=f"M{i}", estimand="ATE", estimate=0.0, std_error=0.1,
            ci_lower=-0.2, ci_upper=0.2, p_value=0.5,
        )
        for i in range(n)
    ]
    state.critique_history = [
        CritiqueFeedback(
            decision=CritiqueDecision.ITERATE,
            iteration=1,
            scores={
                "statistical_validity": 3,
                "assumption_checking": 3,
                "method_selection": 3,
                "completeness": 3,
                "reproducibility": 3,
                "interpretation": 3,
            },
            issues=[f"issue_{i}_in_critique_with_padding" for i in range(n)],
            improvements=[f"improvement_{i}_with_padding_text" for i in range(n)],
            reasoning="r",
        )
    ]
    return state


class TestPromptBounded:
    """Prompt size grows sublinearly as list state grows."""

    @pytest.mark.parametrize("n", [10, 50, 200])
    def test_prompt_does_not_grow_unboundedly(self, n: int):
        orchestrator = StandardOrchestrator()
        state = _state_with_long_lists(n)
        prompt = orchestrator._build_context_prompt(state)
        # The prompt must stay well under what an unbounded dump would
        # produce. With n=200 issues, the unbounded version grew to many
        # thousands of characters; the capped version stays under 4 KB.
        assert len(prompt) < 4000, (
            f"Prompt grew to {len(prompt)} characters for n={n}; "
            "an unbounded dump regressed."
        )

    def test_truncate_list_static_cases(self):
        truncate = StandardOrchestrator._truncate_list
        assert truncate([]) == "[]"
        assert truncate(["a"]) == "['a']"
        assert truncate(["a", "b", "c", "d", "e"]) == "['a', 'b', 'c', 'd', 'e']"
        # Six items -> first five plus "(+1 more)"
        truncated = truncate(["a", "b", "c", "d", "e", "f"])
        assert "+1 more" in truncated
        assert "f" not in truncated

    def test_more_marker_present_when_capped(self):
        """Critique issues are the only list inlined in the lean prompt; cap at 3.

        After the lean-context refactor, the orchestrator no longer
        inlines treatment effects, EDA issues, or improvements. The one
        remaining inlined list is critique issues (when decision is
        ITERATE), capped at 3. Everything else is summarized via
        progress / focus or recoverable by dispatching a specialist.
        """
        orchestrator = StandardOrchestrator()
        state = _state_with_long_lists(20)
        prompt = orchestrator._build_context_prompt(state)
        # Critique issues: capped at 3, so "+17 more" should appear.
        assert "+17 more" in prompt
        # Treatment effects and other lists no longer dump; their counts
        # appear via the progress summary instead.
        assert "20 effects" in prompt
