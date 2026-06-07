"""The base ReAct agent pushes the analyst gate note into every observation.

Human notes from the data and DAG gates land on
dataset_info.user_provided_context. Before this, only domain_knowledge read
them, so a re-run agent (dag_expert after a REVISE) never saw the guidance.
_build_initial_observation now prepends the note for every agent.
"""
from __future__ import annotations

from src.analysis.agents.base.react_agent import ReActAgent, analyst_note_block
from src.analysis.agents.base.state import AnalysisState, DatasetInfo


class _StubAgent(ReActAgent):
    """Minimal agent: skips the heavy base __init__ (LLM client, tools)."""

    def __init__(self) -> None:
        pass

    def _get_initial_observation(self, state: AnalysisState) -> str:
        return "BASE_OBS"


def _state(note: str | None = None) -> AnalysisState:
    return AnalysisState(
        job_id="j", dataset_info=DatasetInfo(url="kaggle.com/x", user_provided_context=note)
    )


def test_analyst_note_block_is_empty_without_a_note():
    assert analyst_note_block(_state()) == ""
    assert analyst_note_block(_state(note="   ")) == ""


def test_analyst_note_block_formats_a_present_note():
    block = analyst_note_block(_state(note="age is a mediator, not a confounder"))
    assert "Analyst-provided context" in block
    assert "age is a mediator, not a confounder" in block


def test_build_initial_observation_prepends_the_note_before_the_agent_obs():
    obs = _StubAgent()._build_initial_observation(_state(note="drop the x1->y edge"))
    assert obs.startswith("Analyst-provided context")
    assert "drop the x1->y edge" in obs
    assert obs.endswith("BASE_OBS")


def test_build_initial_observation_is_just_the_agent_obs_without_a_note():
    assert _StubAgent()._build_initial_observation(_state()) == "BASE_OBS"
