"""dataset_inspector: pick the right file in a multi-file Kaggle bundle.

Runs once after download. Fans out a full data_profiler invocation per
candidate file (csv / parquet), collects the resulting DataProfile
objects, scores them via a deterministic rubric, and writes the winner
to state.data_profile while exposing all candidates' profiles in
state.file_profiles for the UI Data panel to render.

See helpers.py for the rubric, brief.py for the contract, agent.py for
the fan-out / fan-in execution.
"""
from __future__ import annotations

from src.analysis.agents.dataset_inspector.brief import (
    AGENT_NAME,
    CAPABILITY,
    build_brief,
    preflight,
)
from src.analysis.agents.dataset_inspector.helpers import (
    FileScore,
    explain_choice,
    pick_winner,
    score_profile,
)
from src.analysis.agents.dataset_inspector.output import DataProfile

__all__ = [
    "AGENT_NAME",
    "CAPABILITY",
    "DataProfile",
    "FileScore",
    "build_brief",
    "explain_choice",
    "pick_winner",
    "preflight",
    "score_profile",
]
