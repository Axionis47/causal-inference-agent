"""Tests for the pipeline_briefs notebook section.

Pure transcription, so the tests check that what goes into
state.agent_briefs comes out verbatim in the rendered markdown
table, with correct ordering and pipe escaping.
"""
from __future__ import annotations

from src.analysis.agents.base.state import AnalysisState, DatasetInfo
from src.analysis.agents.notebook.sections import render_pipeline_briefs
from src.domain.briefs import AgentBrief, Flag, refusal_brief


def _state(briefs: dict[str, AgentBrief] | None = None) -> AnalysisState:
    state = AnalysisState(
        job_id="test-job",
        dataset_info=DatasetInfo(url="https://example.com/data"),
    )
    if briefs:
        state.agent_briefs = dict(briefs)
    return state


def _markdown(cells: list) -> str:
    """Concatenate all markdown-cell sources for substring assertions."""
    return "\n".join(c["source"] for c in cells if c["cell_type"] == "markdown")


# --- empty case -------------------------------------------------------------


def test_renders_skip_placeholder_when_no_briefs():
    cells = render_pipeline_briefs(_state())
    md = _markdown(cells)
    assert "Pipeline Issues & Flags" in md
    assert "Section skipped" in md


# --- single done brief, no flags -------------------------------------------


def test_done_with_no_flags_renders_em_dash():
    brief = AgentBrief(
        agent="data_profiler",
        status="done",
        headline="200 samples, 5 features",
        flags=[],
        raised_issues=[],
        artifact_keys=["data_profile"],
    )
    cells = render_pipeline_briefs(_state({brief.agent: brief}))
    md = _markdown(cells)
    assert "data_profiler" in md
    assert "200 samples, 5 features" in md
    # Em-dash in both flag and issue columns when nothing fires.
    assert md.count("—") >= 2


# --- single done brief, with flags + issues ---------------------------------


def test_done_with_flags_renders_flag_values_and_bulleted_issues():
    brief = AgentBrief(
        agent="ps_diagnostics",
        status="done",
        headline="overlap 72%",
        flags=[Flag.POOR_OVERLAP, Flag.SMD_HIGH],
        raised_issues=[
            "common support 72% (< 90%)",
            "max weighted SMD is 0.18 (> 0.1)",
        ],
        artifact_keys=["ps_diagnostics"],
    )
    cells = render_pipeline_briefs(_state({brief.agent: brief}))
    md = _markdown(cells)
    assert "poor_overlap" in md
    assert "smd_high" in md
    assert "common support 72% (< 90%)" in md
    assert "max weighted SMD is 0.18 (> 0.1)" in md
    # Issues separated by <br> so each appears on its own line in the cell.
    assert "<br>" in md


def test_escapes_pipes_in_headline_and_issues():
    brief = AgentBrief(
        agent="effect_estimator",
        status="done",
        headline="estimate | with | pipes",
        flags=[Flag.METHOD_UNSTABLE],
        raised_issues=["a | b | c"],
        artifact_keys=["treatment_effects"],
    )
    cells = render_pipeline_briefs(_state({brief.agent: brief}))
    md = _markdown(cells)
    # Pipes escaped so the markdown table stays valid.
    assert r"estimate \| with \| pipes" in md
    assert r"a \| b \| c" in md


# --- refused brief ----------------------------------------------------------


def test_refused_brief_renders_with_needs_not_met():
    brief = refusal_brief(
        agent="data_repair",
        flag=Flag.NEEDS_NOT_MET,
        headline="data_profile missing; profiler has not run",
        issues=["state.data_profile is None"],
    )
    cells = render_pipeline_briefs(_state({brief.agent: brief}))
    md = _markdown(cells)
    assert "data_repair" in md
    assert "refused" in md
    assert "needs_not_met" in md
    assert "state.data_profile is None" in md


# --- ordering: refused first, then flagged-done, then clean-done ------------


def test_multiple_briefs_sort_refused_first_then_flagged_then_clean():
    clean = AgentBrief(
        agent="data_profiler",
        status="done",
        headline="profile written",
        flags=[],
        raised_issues=[],
        artifact_keys=["data_profile"],
    )
    flagged = AgentBrief(
        agent="effect_estimator",
        status="done",
        headline="3 methods ran",
        flags=[Flag.METHOD_UNSTABLE],
        raised_issues=["CV 0.62"],
        artifact_keys=["treatment_effects"],
    )
    refused = refusal_brief(
        agent="sensitivity_analyst",
        flag=Flag.NEEDS_NOT_MET,
        headline="no treatment_effects to stress-test",
        issues=["state.treatment_effects is empty"],
    )

    # Insert in a non-sorted order on purpose; renderer should reorder.
    state = _state({
        "data_profiler": clean,
        "effect_estimator": flagged,
        "sensitivity_analyst": refused,
    })
    md = _markdown(render_pipeline_briefs(state))

    pos_refused = md.find("sensitivity_analyst")
    pos_flagged = md.find("effect_estimator")
    pos_clean = md.find("data_profiler")

    assert pos_refused != -1 and pos_flagged != -1 and pos_clean != -1
    assert pos_refused < pos_flagged < pos_clean
