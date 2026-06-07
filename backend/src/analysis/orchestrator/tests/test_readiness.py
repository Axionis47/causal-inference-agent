"""Tests for the readiness gate (orchestrator/common/readiness.py).

classify_brief is the orchestrator's verification step: it reads one sealed
AgentBrief and decides whether the run may advance. These cases pin the three
verdicts and, importantly, that a clean or absent brief never halts a run (the
slice-1 contract: a clean run is unchanged).
"""
from __future__ import annotations

from src.analysis.orchestrator.common import classify_brief
from src.domain.briefs import AgentBrief, Flag


def _brief(status: str, flags: list[Flag] | None = None) -> AgentBrief:
    return AgentBrief(
        agent="dag_expert",
        status=status,
        headline="2 nodes, 1 edge, adjustment set of 0",
        flags=flags or [],
    )


def test_refused_brief_halts_with_the_agent_and_headline_in_the_reason() -> None:
    verdict, reason = classify_brief(_brief("refused"))
    assert verdict == "halt"
    assert "dag_expert" in reason
    assert "refused" in reason


def test_failed_brief_halts() -> None:
    verdict, reason = classify_brief(_brief("failed"))
    assert verdict == "halt"
    assert "failed" in reason


def test_cycle_detected_halts_even_when_status_is_done() -> None:
    # A directed cycle means the effect is not identified, so estimation must
    # not run on it regardless of the agent reporting status=done.
    verdict, reason = classify_brief(_brief("done", [Flag.CYCLE_DETECTED]))
    assert verdict == "halt"
    assert "cycle_detected" in reason


def test_soft_quality_flag_does_not_halt() -> None:
    # NO_ADJUSTMENT_SET has a documented fallback (use the profiler's
    # confounders), so it is surfaced, not fatal.
    verdict, reason = classify_brief(_brief("done", [Flag.NO_ADJUSTMENT_SET]))
    assert verdict == "soft"
    assert "no_adjustment_set" in reason


def test_a_fatal_flag_wins_over_a_soft_flag() -> None:
    verdict, _ = classify_brief(
        _brief("done", [Flag.NO_ADJUSTMENT_SET, Flag.CYCLE_DETECTED])
    )
    assert verdict == "halt"


def test_clean_done_brief_is_ok_with_no_reason() -> None:
    verdict, reason = classify_brief(_brief("done"))
    assert verdict == "ok"
    assert reason == ""


def test_missing_brief_is_ok_so_a_clean_run_is_unchanged() -> None:
    verdict, reason = classify_brief(None)
    assert verdict == "ok"
    assert reason == ""
