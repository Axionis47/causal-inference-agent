"""The spine: ordering, walking, and the stage->agent map."""
from __future__ import annotations

from src.analysis_v2.core import (
    SPINE,
    STAGE_AGENT,
    TERMINAL_STAGE,
    AnalysisStage,
    is_earlier,
    next_stage,
    stage_index,
)


def test_spine_runs_s0_through_s12_in_order():
    assert SPINE[0] == AnalysisStage.S0_DATASET_SAVED
    assert SPINE[-1] == AnalysisStage.S12_JOB_COMPLETE
    assert len(SPINE) == 13
    indices = [stage_index(s) for s in SPINE]
    assert indices == sorted(indices)


def test_next_stage_walks_the_whole_spine_and_stops_at_terminal():
    walked = [AnalysisStage.S0_DATASET_SAVED]
    while (nxt := next_stage(walked[-1])) is not None:
        walked.append(nxt)
    assert walked == list(SPINE)
    assert next_stage(TERMINAL_STAGE) is None


def test_is_earlier_orders_stages_strictly():
    assert is_earlier(AnalysisStage.S1_INTAKE_PARSED, AnalysisStage.S7_METHOD_EXECUTED)
    assert not is_earlier(AnalysisStage.S7_METHOD_EXECUTED, AnalysisStage.S1_INTAKE_PARSED)
    assert not is_earlier(AnalysisStage.S5_PLAN_CRITIQUED, AnalysisStage.S5_PLAN_CRITIQUED)


def test_every_stage_has_an_agent_map_entry_and_bounds_have_none():
    assert set(STAGE_AGENT) == set(AnalysisStage)
    assert STAGE_AGENT[AnalysisStage.S0_DATASET_SAVED] is None
    assert STAGE_AGENT[AnalysisStage.S6_USER_CONFIRMED_OR_AUTO_APPROVED] is None
    assert STAGE_AGENT[AnalysisStage.S12_JOB_COMPLETE] is None
    assert STAGE_AGENT[AnalysisStage.S7_METHOD_EXECUTED] == "method_lane"
