"""DatasetInspectorAgent.execute() — fan-out / fan-in / pick / mutate.

The inner data_profiler invocation is hidden behind `_profile_one_file`
so this test substitutes a stub that returns canned DataProfiles. The
real wiring lands in a follow-up commit; behaviour the orchestrator
depends on (state mutation shape, decision push, SSE events, brief
attachment) is pinned here.
"""
from __future__ import annotations

from typing import Awaitable, Callable

import pytest

from src.analysis.agents.base.state import (
    AnalysisState,
    DatasetInfo,
    FileEntry,
)
from src.analysis.agents.data_profiler import DataProfile
from src.analysis.agents.dataset_inspector.agent import DatasetInspectorAgent


def _profile(
    *,
    n_samples: int = 1000,
    treatment: list[str] | None = None,
    outcome: list[str] | None = None,
    confounders: list[str] | None = None,
) -> DataProfile:
    return DataProfile(
        n_samples=n_samples,
        n_features=4,
        feature_names=["t", "y", "x1", "x2"],
        feature_types={"t": "binary", "y": "numeric", "x1": "numeric", "x2": "numeric"},
        missing_values={"t": 0, "y": 0, "x1": 0, "x2": 0},
        numeric_stats={"y": {"mean": 0, "std": 1, "min": -1, "max": 1}},
        categorical_stats={},
        treatment_candidates=treatment or [],
        outcome_candidates=outcome or [],
        potential_confounders=confounders or [],
    )


def _state(files: list[FileEntry]) -> AnalysisState:
    return AnalysisState(
        job_id="job-1",
        dataset_info=DatasetInfo(url="kaggle.com/x", files=files),
    )


def _files(*names_and_formats: tuple[str, str]) -> list[FileEntry]:
    return [FileEntry(name=n, size_bytes=1000, format=fmt) for n, fmt in names_and_formats]


def _stub_profiler(
    canned: dict[str, DataProfile | None]
) -> Callable[[AnalysisState, str], Awaitable[DataProfile | None]]:
    async def _stub(state: AnalysisState, filename: str) -> DataProfile | None:
        return canned.get(filename)
    return _stub


# --- refusal path ----------------------------------------------------------


@pytest.mark.asyncio
async def test_refuses_when_no_candidate_data_files():
    agent = DatasetInspectorAgent()
    state = _state(_files(("README.md", "md"), ("metadata.json", "json")))
    result = await agent.execute(state)
    brief = result.agent_briefs["dataset_inspector"]
    assert brief.status == "refused"
    assert result.file_profiles == {}
    assert result.data_profile is None


# --- happy path: multi-file pick + state mutation -------------------------


@pytest.mark.asyncio
async def test_happy_path_picks_winner_mutates_state_and_emits_events():
    agent = DatasetInspectorAgent()
    agent._profile_one_file = _stub_profiler({
        "train.csv": _profile(n_samples=950, treatment=["treated"], outcome=["y"], confounders=["x1", "x2"]),
        "test.csv": _profile(n_samples=200),
    })
    state = _state(_files(("train.csv", "csv"), ("test.csv", "csv")))
    result = await agent.execute(state)

    # Both files profiled
    assert set(result.file_profiles) == {"train.csv", "test.csv"}

    # Winner copied into the canonical slot
    assert result.data_profile is not None
    assert result.data_profile.treatment_candidates == ["treated"]

    # used flag flipped on the winner only
    used = [f for f in result.dataset_info.files if f.used]
    assert len(used) == 1 and used[0].name == "train.csv"

    # Decision recorded
    decisions = [d for d in result.decisions if d.decision_type == "file_selected"]
    assert len(decisions) == 1
    assert decisions[0].choice == "train.csv"
    assert "train.csv" in decisions[0].reason

    # SSE events bookend the work
    types = [e["event_type"] for e in result.sse_events]
    assert "dataset_inspection_started" in types
    assert "dataset_inspection_complete" in types
    completed = next(e for e in result.sse_events if e["event_type"] == "dataset_inspection_complete")
    assert completed["data"]["selected_file"] == "train.csv"
    assert "train.csv" in completed["data"]["file_profiles"]
    assert "test.csv" in completed["data"]["file_profiles"]

    # Brief attached and points at the artifacts the rest of the pipeline reads
    brief = result.agent_briefs["dataset_inspector"]
    assert brief.status == "done"
    assert "train.csv" in brief.headline


# --- single-candidate bundle: still works, no alternatives in audit -------


@pytest.mark.asyncio
async def test_single_csv_bundle_picks_it_without_runner_up():
    agent = DatasetInspectorAgent()
    agent._profile_one_file = _stub_profiler({
        "data.csv": _profile(treatment=["treated"], outcome=["y"]),
    })
    state = _state(_files(("data.csv", "csv")))
    result = await agent.execute(state)

    decision = next(d for d in result.decisions if d.decision_type == "file_selected")
    assert decision.choice == "data.csv"
    assert decision.alternatives == []


# --- inner profile failure: scorer skips it ------------------------------


@pytest.mark.asyncio
async def test_inner_profile_failure_is_skipped_not_fatal():
    """If one candidate's inner data_profiler raises, the inspector
    must still pick from the remaining successful profiles instead of
    propagating the exception out of execute()."""
    agent = DatasetInspectorAgent()

    async def _selective(state: AnalysisState, filename: str) -> DataProfile | None:
        if filename == "broken.csv":
            raise RuntimeError("simulated profile failure")
        return _profile(treatment=["treated"], outcome=["y"])

    agent._profile_one_file = _selective
    state = _state(_files(("train.csv", "csv"), ("broken.csv", "csv")))
    result = await agent.execute(state)

    # train.csv profiled and chosen; broken.csv is absent from file_profiles
    assert "train.csv" in result.file_profiles
    assert "broken.csv" not in result.file_profiles
    decision = next(d for d in result.decisions if d.decision_type == "file_selected")
    assert decision.choice == "train.csv"


# --- registry: agent is discoverable -------------------------------------


def test_dataset_inspector_is_registered():
    from src.analysis.agents.registry import _REGISTRY

    assert "dataset_inspector" in _REGISTRY
    assert _REGISTRY["dataset_inspector"] is DatasetInspectorAgent
