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
import shutil
import time
from pathlib import Path
from typing import Awaitable, Callable

import pandas as pd

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
from src.analysis.agents.dataset_inspector.relational import (
    build_relational_profile,
    candidate_keys,
)
from src.analysis.agents.registry import register_agent
from src.logging_config.structured import get_logger
from src.storage.cleanup import CAUSAL_TEMP_DIR

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

    def __init__(self, parallelism: int = DEFAULT_PARALLELISM) -> None:
        super().__init__()
        self._parallelism = parallelism
        # filename -> on-disk parquet path, populated by _materialize_candidates
        # before the fan-out begins. Cleared between execute() runs.
        self._candidate_paths: dict[str, str] = {}

    async def _profile_one_file(
        self, state: AnalysisState, filename: str
    ) -> DataProfile | None:
        """Run data_profiler against the per-file candidate parquet.

        Deep-copies state, points dataframe_path at the candidate's
        parquet, and uniqueifies job_id with a per-file suffix so the
        inner save_dataframe writes don't collide across parallel runs.
        Returns the inner DataProfile (or None if profiling failed).

        Tests override this hook at the instance level — see
        tests/test_agent.py for the substitution pattern.
        """
        parquet_path = self._candidate_paths.get(filename)
        if parquet_path is None:
            logger.warning(
                "dataset_inspector_no_candidate_path",
                job_id=state.job_id,
                file=filename,
            )
            return None

        from src.analysis.agents.data_profiler.agent import DataProfilerAgent

        state_copy = state.model_copy(deep=True)
        state_copy.dataframe_path = parquet_path
        # Suffix the job_id so the inner save_dataframe writes go to a
        # per-file parquet (the existing helper uses job_id verbatim in
        # the filename). Without this, N parallel inner runs would race
        # on the same canonical path.
        state_copy.job_id = f"{state.job_id}__inspect__{filename}"

        profiler = DataProfilerAgent()
        try:
            result_state = await profiler.execute_with_tracing(state_copy)
            return result_state.data_profile
        except Exception as exc:
            logger.warning(
                "inner_profiler_failed",
                job_id=state.job_id,
                file=filename,
                error=str(exc),
            )
            return None

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

        # Pre-fetch Kaggle metadata once on the outer state so each inner
        # data_profiler invocation doesn't re-fetch (it short-circuits when
        # state.raw_metadata is already populated).
        await self._prefetch_kaggle_metadata(state)

        # Materialise every candidate on disk as a per-file parquet. The
        # inner _profile_one_file hook reads from self._candidate_paths.
        try:
            self._candidate_paths = await self._materialize_candidates(
                state, candidates
            )
        except Exception as exc:
            logger.error(
                "dataset_inspector_materialize_failed",
                job_id=state.job_id,
                error=str(exc),
            )
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

        # Describe how the files relate (read-only; nothing is assembled).
        self._build_relational(state)

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

    # ── workdir + candidate materialisation ──────────────────────────

    async def _prefetch_kaggle_metadata(self, state: AnalysisState) -> None:
        """Populate state.raw_metadata once before fan-out so inner runs
        don't each hit the Kaggle metadata API."""
        if state.raw_metadata or "kaggle.com" not in state.dataset_info.url:
            return
        from src.analysis.agents.data_profiler.loading import fetch_kaggle_metadata

        try:
            await fetch_kaggle_metadata(state)
        except Exception as exc:
            logger.warning(
                "dataset_inspector_metadata_prefetch_failed",
                job_id=state.job_id,
                error=str(exc),
            )

    async def _get_or_download_workdir(
        self, state: AnalysisState
    ) -> Path | None:
        """Resolve a directory containing every candidate file on disk.

        Three sources, in priority order: local_path on dataset_info
        (the file is already extracted somewhere on disk), Kaggle URL
        (download via download_to_workdir into CAUSAL_TEMP_DIR), or
        nothing (refused upstream).
        """
        if state.dataset_info.local_path:
            p = Path(state.dataset_info.local_path)
            if p.is_dir():
                return p
            if p.is_file():
                return p.parent

        if "kaggle.com" in state.dataset_info.url:
            from src.analysis.agents.data_profiler.loading import download_to_workdir

            workdir, files, error = await download_to_workdir(
                state, state.dataset_info.url
            )
            if error or workdir is None:
                logger.error(
                    "dataset_inspector_workdir_download_failed",
                    job_id=state.job_id,
                    error=error,
                )
                return None
            # Carry the file list onto the outer state so the UI Data panel
            # has something to render before per-file profiling completes.
            if not state.dataset_info.files:
                state.dataset_info.files = files
            return workdir

        return None

    async def _materialize_candidates(
        self, state: AnalysisState, candidates: list[str]
    ) -> dict[str, str]:
        """Read each candidate file and save it as a job+file-scoped parquet.

        The inner _profile_one_file hook reads back from these parquets
        instead of re-loading the original csv/parquet each time. A
        failure on one file does not abort the others.
        """
        workdir = await self._get_or_download_workdir(state)
        if workdir is None:
            return {}

        CAUSAL_TEMP_DIR.mkdir(parents=True, exist_ok=True)
        paths: dict[str, str] = {}
        for name in candidates:
            src = workdir / name
            if not src.is_file():
                logger.warning(
                    "dataset_inspector_candidate_missing",
                    job_id=state.job_id,
                    file=name,
                    workdir=str(workdir),
                )
                continue
            try:
                if src.suffix.lower() == ".csv":
                    df = pd.read_csv(src)
                else:
                    df = pd.read_parquet(src)
                # Use the bare stem to avoid path-traversal-ish characters
                # ("/" in nested archive layouts) leaking into filenames.
                safe = src.stem.replace("/", "_")
                dest = CAUSAL_TEMP_DIR / f"{state.job_id}_candidate_{safe}.parquet"
                df.to_parquet(dest)
                paths[name] = str(dest)
            except Exception as exc:
                logger.warning(
                    "dataset_inspector_candidate_load_failed",
                    job_id=state.job_id,
                    file=name,
                    error=str(exc),
                )
        return paths

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

    def _build_relational(self, state: AnalysisState) -> None:
        """Describe how the candidate files relate, from profiles + parquets.

        Candidate keys need the FULL frame, so each candidate parquet is read
        once (one at a time, then discarded) to avoid a false key from a sample.
        When the inner profiler was stubbed (tests) the parquet is absent and
        the keys default to empty; shape and counts still come from the profiles.
        """
        if not state.file_profiles:
            return
        key_candidates_by_file: dict[str, list[str]] = {}
        for name in state.file_profiles:
            path = self._candidate_paths.get(name)
            if not path or not Path(path).is_file():
                continue
            try:
                df = pd.read_parquet(path)
                key_candidates_by_file[name] = candidate_keys(df)
            except Exception as exc:
                logger.warning(
                    "dataset_inspector_key_scan_failed",
                    job_id=state.job_id,
                    file=name,
                    error=str(exc),
                )
        state.relational_profile = build_relational_profile(
            state.file_profiles, key_candidates_by_file
        )

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

        # Copy the winner's candidate parquet to the canonical
        # `{job_id}_data.parquet` path so downstream agents reading
        # state.dataframe_path see the conventional artifact (matching
        # save_dataframe's naming). In test paths where the inner hook
        # was stubbed, _candidate_paths is empty and this is skipped —
        # tests don't assert on dataframe_path then.
        winner_parquet = self._candidate_paths.get(winner_name)
        if winner_parquet and Path(winner_parquet).is_file():
            CAUSAL_TEMP_DIR.mkdir(parents=True, exist_ok=True)
            canonical = CAUSAL_TEMP_DIR / f"{state.job_id}_data.parquet"
            if str(canonical) != winner_parquet:
                shutil.copyfile(winner_parquet, canonical)
            state.dataframe_path = str(canonical)

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
                "relational_profile": (
                    state.relational_profile.model_dump()
                    if state.relational_profile is not None
                    else None
                ),
            },
        )
