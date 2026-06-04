"""Dataset Inspector - pick the right file in a multi-file Kaggle bundle.

Runs once after download, before the rest of the analysis pipeline. Fans
out a profile pass per candidate data file (csv / parquet), scores each
result via the deterministic rubric in helpers.py, and writes the winner
into the canonical state.data_profile / state.dataframe_path slots so
the rest of the pipeline reads exactly one source of truth.

The per-file profiles all stay on state.file_profiles for the UI Data
panel to render — that is the user-visible piece. The decision (winner
name + reason + ranked alternatives) is also pushed onto state.decisions
so it survives into the notebook audit trail.

The inner profile dispatch is exposed as `_profile_one_file` so tests
can substitute behaviour without spinning up a real LLM-driven
data_profiler run. The real wiring lands in a follow-up commit.
"""
from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable

from src.analysis.agents.base import AnalysisState, BaseAgent, JobStatus
from src.analysis.agents.data_profiler import DataProfile
from src.analysis.agents.dataset_inspector.brief import (
    AGENT_NAME,
    CAPABILITY as DI_CAPABILITY,
    build_brief,
    preflight,
)
from src.analysis.agents.dataset_inspector.helpers import (
    FileScore,
    explain_choice,
    pick_winner,
    score_profile,
)
from src.analysis.agents.dataset_inspector.prompt import SYSTEM_PROMPT
from src.analysis.agents.registry import register_agent
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

# Cap on parallel inner data_profiler invocations. Each inner run is
# LLM-heavy; running too many concurrently risks per-minute token rate
# limits at the provider. 2 is conservative; can be raised once the
# rate-limit headroom is observed in production.
DEFAULT_PARALLELISM = 2


# Each inner profile produces (filename, DataProfile or None on failure).
ProfileFn = Callable[[AnalysisState, str], Awaitable[DataProfile | None]]


@register_agent("dataset_inspector")
class DatasetInspectorAgent(BaseAgent):
    """Picks the analysis target from a multi-file Kaggle bundle.

    Procedural agent (no ReAct loop on its own behalf). The LLM work
    happens inside the per-file profile invocations it fans out to.
    """

    AGENT_NAME = "dataset_inspector"
    SYSTEM_PROMPT = SYSTEM_PROMPT
    TOOLS = []
    REQUIRED_STATE_FIELDS = ["dataset_info"]
    WRITES_STATE_FIELDS = ["file_profiles", "data_profile", "dataframe_path"]
    JOB_STATUS = JobStatus.PROFILING
    PROGRESS_WEIGHT = 0.05
    CAPABILITY = DI_CAPABILITY

    # Class-level hook the test suite substitutes. Default is None;
    # production wiring (next commit) provides a real implementation that
    # materialises the candidate file and runs data_profiler on it.
    _profile_one_file: ProfileFn | None = None

    def __init__(self, parallelism: int = DEFAULT_PARALLELISM) -> None:
        super().__init__()
        self._parallelism = parallelism

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Fan out profile-per-file, score, pick winner, mutate state, brief."""
        refusal = preflight(state)
        if refusal is not None:
            state.agent_briefs[AGENT_NAME] = refusal
            return state

        candidates = [
            f.name for f in state.dataset_info.files
            if (f.format or "").lower() in {"csv", "parquet"}
        ]

        state.push_sse_event(
            "dataset_inspection_started",
            {"n_candidates": len(candidates)},
        )

        if self._profile_one_file is None:
            # Defensive: production wiring must set this before dispatch.
            logger.error("dataset_inspector_unwired", job_id=state.job_id)
            state.agent_briefs[AGENT_NAME] = build_brief(state)
            return state

        start = time.time()
        profiles = await self._run_inner_profiles(state, candidates)
        elapsed_ms = int((time.time() - start) * 1000)

        # Record every successful profile so the UI shows the multi-file
        # reality even for files that failed scoring.
        for name, profile in profiles.items():
            if profile is not None:
                state.file_profiles[name] = profile

        scored = self._score_all(state.file_profiles)
        if not scored:
            logger.warning(
                "dataset_inspector_no_profiles",
                job_id=state.job_id,
                attempted=len(candidates),
            )
            state.agent_briefs[AGENT_NAME] = build_brief(state)
            return state

        winner = pick_winner(scored)
        reason = explain_choice(winner, scored)

        self._commit_winner(state, winner.name, reason, scored, elapsed_ms)
        state.agent_briefs[AGENT_NAME] = build_brief(state)
        return state

    # ── inner orchestration ────────────────────────────────────────────

    async def _run_inner_profiles(
        self, state: AnalysisState, candidates: list[str]
    ) -> dict[str, DataProfile | None]:
        """Run `_profile_one_file` per candidate under a concurrency cap.

        Exceptions in one file's profile do not abort the others — the
        scorer simply skips files with no DataProfile. The error is
        logged so an operator can see which file failed and why.
        """
        sem = asyncio.Semaphore(self._parallelism)

        async def _bounded(name: str) -> tuple[str, DataProfile | None]:
            async with sem:
                try:
                    return name, await self._profile_one_file(state, name)
                except Exception as exc:
                    logger.warning(
                        "dataset_inspector_profile_failed",
                        job_id=state.job_id,
                        file=name,
                        error=str(exc),
                    )
                    return name, None

        results = await asyncio.gather(*(_bounded(name) for name in candidates))
        return dict(results)

    def _score_all(
        self, profiles: dict[str, DataProfile]
    ) -> list[FileScore]:
        """Score every file that has a profile. Description lookup is a
        no-op for now; the Kaggle resource description plumb-through is
        a follow-up so the rubric falls back to the filename pattern."""
        return [
            score_profile(name, profile, resource_description=None)
            for name, profile in profiles.items()
        ]

    def _commit_winner(
        self,
        state: AnalysisState,
        winner_name: str,
        reason: str,
        scored: list[FileScore],
        elapsed_ms: int,
    ) -> None:
        """Mutate state with the winner: used flag, data_profile,
        decision audit, SSE inspection_complete event."""
        # Flip the used flag so the existing UI affordance highlights
        # the chosen file. Defensive against missing entries.
        for f in state.dataset_info.files:
            f.used = (f.name == winner_name)

        state.data_profile = state.file_profiles[winner_name]
        # dataframe_path is set by the inner profile run (data_profiler
        # writes it as a side-effect of save_dataframe). We trust that
        # write for the canonical winner; if it is empty the next-stage
        # agents will refuse and the orchestrator will reroute.

        ranked_alternatives = [
            {"option": s.name, "reason": "; ".join(s.reasons) or "no rubric matches",
             "score": f"{s.score:g}"}
            for s in sorted(scored, key=lambda s: (-s.score, -s.n_samples, s.name))
            if s.name != winner_name
        ]
        state.push_decision(
            agent=AGENT_NAME,
            decision_type="file_selected",
            choice=winner_name,
            reason=reason,
            alternatives=ranked_alternatives,
        )

        state.push_sse_event(
            "dataset_inspection_complete",
            {
                "selected_file": winner_name,
                "reason": reason,
                "elapsed_ms": elapsed_ms,
                "file_profiles": {
                    name: {
                        "n_samples": p.n_samples,
                        "n_features": p.n_features,
                        "treatment_candidates": list(p.treatment_candidates),
                        "outcome_candidates": list(p.outcome_candidates),
                    }
                    for name, p in state.file_profiles.items()
                },
            },
        )
