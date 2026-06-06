"""Local JSON-based storage client for development."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock

from src.analysis.agents.base import AnalysisState, JobStatus
from src.config import get_settings
from src.logging_config.structured import get_logger
from src.storage.job_data import read_manifest
from src.storage.serialize import dump_state_jsonable

logger = get_logger(__name__)


def _utcnow() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class LocalStorageClient:
    """Local file-based storage for development (no GCP required).

    All synchronous file I/O and FileLock operations are wrapped in
    asyncio.to_thread() to avoid blocking the event loop.
    """

    def __init__(self) -> None:
        """Initialize the local storage client."""
        settings = get_settings()
        self.storage_path = Path(settings.local_storage_path)
        self.jobs_file = self.storage_path / "jobs.json"
        self.results_file = self.storage_path / "results.json"
        self.traces_file = self.storage_path / "traces.json"
        # Parked states: full AnalysisState dumps for jobs waiting at the
        # human-approval gate. Keyed by job_id. Cleared on approve/reject.
        self.parked_states_file = self.storage_path / "parked_states.json"

        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize files if they don't exist
        for file in [
            self.jobs_file,
            self.results_file,
            self.traces_file,
            self.parked_states_file,
        ]:
            if not file.exists():
                file.write_text("{}")

        logger.info("local_storage_initialized", path=str(self.storage_path))

    def _get_lock(self, file: Path) -> FileLock:
        """Get a file lock for the given file."""
        return FileLock(str(file) + ".lock", timeout=10)

    def _load_json(self, file: Path) -> dict[str, Any]:
        """Load JSON from file."""
        try:
            return json.loads(file.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return {}

    def _save_json(self, file: Path, data: dict[str, Any]) -> None:
        """Save JSON to file."""
        file.write_text(json.dumps(data, default=str, indent=2))

    def _job_data_from_state(self, state: AnalysisState) -> dict[str, Any]:
        """Build a job data dict from state, including instance_id."""
        settings = get_settings()
        return {
            "id": state.job_id,
            "kaggle_url": state.dataset_info.url,
            "dataset_name": state.dataset_info.name,
            "status": state.status.value,
            "treatment_variable": state.treatment_variable,
            "outcome_variable": state.outcome_variable,
            "iteration_count": state.iteration_count,
            "max_iterations": state.max_iterations,
            "created_at": state.created_at.isoformat() if state.created_at else None,
            "updated_at": state.updated_at.isoformat() if state.updated_at else None,
            "error_message": state.error_message,
            "error_agent": state.error_agent,
            "instance_id": settings.instance_id,
        }

    async def create_job(self, state: AnalysisState) -> str:
        """Create a new job document."""
        job_data = self._job_data_from_state(state)

        def _sync():
            lock = self._get_lock(self.jobs_file)
            with lock:
                jobs = self._load_json(self.jobs_file)
                jobs[state.job_id] = job_data
                self._save_json(self.jobs_file, jobs)

        await asyncio.to_thread(_sync)

        logger.info("job_created", job_id=state.job_id)
        return state.job_id

    async def create_job_if_capacity(
        self, state: AnalysisState, max_concurrent: int
    ) -> bool:
        """Atomically create a job only if under the concurrency limit.

        Args:
            state: Initial analysis state
            max_concurrent: Maximum allowed concurrent jobs

        Returns:
            True if job was created, False if at capacity
        """
        job_data = self._job_data_from_state(state)
        terminal = {"completed", "failed", "cancelled"}

        def _sync():
            lock = self._get_lock(self.jobs_file)
            with lock:
                jobs = self._load_json(self.jobs_file)
                active = sum(
                    1 for j in jobs.values() if j.get("status") not in terminal
                )
                if active >= max_concurrent:
                    return False
                jobs[state.job_id] = job_data
                self._save_json(self.jobs_file, jobs)
                return True

        return await asyncio.to_thread(_sync)

    async def update_job(
        self,
        state: AnalysisState,
        expected_status: JobStatus | None = None,
    ) -> bool:
        """Update a job document with optional status guard.

        Args:
            state: Updated analysis state
            expected_status: If set, only update if current status matches.

        Returns:
            True if updated, False if rejected or not found.
        """
        settings = get_settings()

        def _sync():
            lock = self._get_lock(self.jobs_file)
            with lock:
                jobs = self._load_json(self.jobs_file)

                if state.job_id not in jobs:
                    logger.warning("job_not_found_for_update", job_id=state.job_id)
                    return False

                # Status guard
                if expected_status is not None:
                    current = jobs[state.job_id].get("status")
                    if current != expected_status.value:
                        logger.warning(
                            "state_transition_rejected",
                            job_id=state.job_id,
                            expected=expected_status.value,
                            actual=current,
                            attempted=state.status.value,
                        )
                        return False

                jobs[state.job_id].update({
                    "status": state.status.value,
                    "treatment_variable": state.treatment_variable,
                    "outcome_variable": state.outcome_variable,
                    "iteration_count": state.iteration_count,
                    "updated_at": _utcnow().isoformat(),
                    "error_message": state.error_message,
                    "error_agent": state.error_agent,
                    "notebook_path": state.notebook_path,
                    "instance_id": settings.instance_id,
                })

                if state.completed_at:
                    jobs[state.job_id]["completed_at"] = state.completed_at.isoformat()

                self._save_json(self.jobs_file, jobs)
                return True

        result = await asyncio.to_thread(_sync)
        logger.debug("job_updated", job_id=state.job_id, status=state.status.value)
        return result

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get a job by ID."""
        def _sync():
            jobs = self._load_json(self.jobs_file)
            return jobs.get(job_id)

        return await asyncio.to_thread(_sync)

    async def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List jobs with optional filtering."""
        def _sync():
            jobs = self._load_json(self.jobs_file)
            job_list = list(jobs.values())

            if status:
                job_list = [j for j in job_list if j.get("status") == status.value]

            # Sort by created_at descending
            job_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)

            total = len(job_list)
            paginated = job_list[offset : offset + limit]

            return paginated, total

        return await asyncio.to_thread(_sync)

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: str | None = None,
    ) -> bool:
        """Update just the status of a job."""
        def _sync():
            lock = self._get_lock(self.jobs_file)
            with lock:
                jobs = self._load_json(self.jobs_file)

                if job_id not in jobs:
                    return False

                jobs[job_id]["status"] = status.value
                jobs[job_id]["updated_at"] = _utcnow().isoformat()
                if error_message is not None:
                    jobs[job_id]["error_message"] = error_message

                self._save_json(self.jobs_file, jobs)
            return True

        result = await asyncio.to_thread(_sync)
        if result:
            logger.info("job_status_updated", job_id=job_id, status=status.value)
        return result

    async def delete_job(self, job_id: str, cascade: bool = True) -> dict[str, Any]:
        """Delete a job and optionally cascade to related data."""
        def _sync():
            result = {"job": False, "results": False, "traces": False}

            lock = self._get_lock(self.jobs_file)
            with lock:
                jobs = self._load_json(self.jobs_file)
                if job_id not in jobs:
                    return result

                del jobs[job_id]
                self._save_json(self.jobs_file, jobs)
                result["job"] = True

            if cascade:
                # Delete results
                results_lock = self._get_lock(self.results_file)
                with results_lock:
                    results = self._load_json(self.results_file)
                    if job_id in results:
                        del results[job_id]
                        self._save_json(self.results_file, results)
                        result["results"] = True

                # Delete traces
                traces_lock = self._get_lock(self.traces_file)
                with traces_lock:
                    traces = self._load_json(self.traces_file)
                    if job_id in traces:
                        del traces[job_id]
                        self._save_json(self.traces_file, traces)
                        result["traces"] = True

            return result

        result = await asyncio.to_thread(_sync)
        logger.info("job_deleted", job_id=job_id, cascade=cascade, result=result)
        return result

    async def save_results(self, state: AnalysisState) -> None:
        """Save analysis results."""
        def _sync():
            lock = self._get_lock(self.results_file)
            with lock:
                results = self._load_json(self.results_file)

                results_data: dict[str, Any] = {
                    "job_id": state.job_id,
                    "created_at": _utcnow().isoformat(),
                    "treatment_variable": state.treatment_variable,
                    "outcome_variable": state.outcome_variable,
                    "recommendations": state.recommendations,
                    "notebook_path": state.notebook_path,
                }

                # Add data profile
                if state.data_profile:
                    results_data["data_profile"] = {
                        "n_samples": state.data_profile.n_samples,
                        "n_features": state.data_profile.n_features,
                        "treatment_candidates": state.data_profile.treatment_candidates,
                        "outcome_candidates": state.data_profile.outcome_candidates,
                        "has_time_dimension": state.data_profile.has_time_dimension,
                    }

                # Persist file list captured during download so the
                # /jobs/{id}/dataset endpoint can hydrate after the
                # active state is evicted.
                if state.dataset_info.files:
                    results_data["dataset_files"] = [
                        f.model_dump() for f in state.dataset_info.files
                    ]

                # Persist the typed dataset manifest (per-file records + full
                # Kaggle metadata) so a revisited completed job can read its
                # dataset record without the live state.
                manifest = read_manifest(state.job_id)
                if manifest is not None:
                    results_data["dataset_manifest"] = manifest.model_dump()

                # Persist Kaggle metadata for the same reason.
                if (
                    state.dataset_info.kaggle_description
                    or state.dataset_info.kaggle_subtitle
                    or state.dataset_info.kaggle_column_descriptions
                    or state.dataset_info.kaggle_tags
                    or state.dataset_info.kaggle_keywords
                    or state.dataset_info.kaggle_domain
                ):
                    results_data["kaggle_meta"] = {
                        "description": state.dataset_info.kaggle_description,
                        "subtitle": state.dataset_info.kaggle_subtitle,
                        "column_descriptions": state.dataset_info.kaggle_column_descriptions,
                        "tags": state.dataset_info.kaggle_tags,
                        "keywords": state.dataset_info.kaggle_keywords,
                        "domain": state.dataset_info.kaggle_domain,
                        "metadata_quality": state.dataset_info.metadata_quality,
                    }

                # Analyst-provided prose context, when given on submission.
                if state.dataset_info.user_provided_context:
                    results_data["user_provided_context"] = (
                        state.dataset_info.user_provided_context
                    )

                # Add causal graph
                if state.proposed_dag:
                    results_data["causal_graph"] = {
                        "nodes": state.proposed_dag.nodes,
                        "edges": [
                            {"source": e.source, "target": e.target, "type": e.edge_type}
                            for e in state.proposed_dag.edges
                        ],
                        "discovery_method": state.proposed_dag.discovery_method,
                        "interpretation": state.proposed_dag.interpretation,
                    }

                # Add treatment effects
                if state.treatment_effects:
                    results_data["treatment_effects"] = [
                        {
                            "method": e.method,
                            "estimand": e.estimand,
                            "estimate": e.estimate,
                            "std_error": e.std_error,
                            "ci_lower": e.ci_lower,
                            "ci_upper": e.ci_upper,
                            "p_value": e.p_value,
                            "assumptions_tested": e.assumptions_tested,
                        }
                        for e in state.treatment_effects
                    ]

                # Add sensitivity results
                if state.sensitivity_results:
                    results_data["sensitivity_results"] = [
                        {
                            "method": s.method,
                            "robustness_value": s.robustness_value,
                            "interpretation": s.interpretation,
                        }
                        for s in state.sensitivity_results
                    ]

                # Methodology decisions audit trail. Surfaced via
                # /jobs/{id}/results.decision_log; rendered by the notebook
                # generator's decisions section. AnalysisDecision.timestamp
                # is a `str`, not a datetime — pass it through verbatim.
                if state.decisions:
                    results_data["decisions"] = [
                        {
                            "agent": d.agent,
                            "decision_type": d.decision_type,
                            "choice": d.choice,
                            "reason": d.reason,
                            "alternatives": d.alternatives,
                            "timestamp": d.timestamp,
                        }
                        for d in state.decisions
                    ]

                results[state.job_id] = results_data
                self._save_json(self.results_file, results)

        await asyncio.to_thread(_sync)
        logger.info("results_saved", job_id=state.job_id)

    async def get_results(self, job_id: str) -> dict[str, Any] | None:
        """Get analysis results."""
        def _sync():
            results = self._load_json(self.results_file)
            return results.get(job_id)

        return await asyncio.to_thread(_sync)

    # ── Parked-state store (human-approval gate) ─────────────────────────

    async def save_parked_state(self, state: AnalysisState) -> None:
        """Persist the full AnalysisState while the job waits at the gate.

        Round-trips via `state.model_dump(mode="json")`; reload happens
        via `AnalysisState.model_validate` in `load_parked_state`. Keyed
        by `state.job_id`; last write wins.
        """
        payload = dump_state_jsonable(state)

        def _sync():
            lock = self._get_lock(self.parked_states_file)
            with lock:
                store = self._load_json(self.parked_states_file)
                store[state.job_id] = payload
                self._save_json(self.parked_states_file, store)

        await asyncio.to_thread(_sync)
        logger.info("parked_state_saved", job_id=state.job_id)

    async def load_parked_state(self, job_id: str) -> AnalysisState | None:
        """Return the parked AnalysisState for `job_id`, or None if absent."""
        def _sync() -> AnalysisState | None:
            store = self._load_json(self.parked_states_file)
            payload = store.get(job_id)
            if payload is None:
                return None
            return AnalysisState.model_validate(payload)

        return await asyncio.to_thread(_sync)

    async def delete_parked_state(self, job_id: str) -> bool:
        """Remove the parked state for `job_id`. Returns True if removed."""
        def _sync() -> bool:
            lock = self._get_lock(self.parked_states_file)
            with lock:
                store = self._load_json(self.parked_states_file)
                if job_id not in store:
                    return False
                store.pop(job_id)
                self._save_json(self.parked_states_file, store)
                return True

        removed = await asyncio.to_thread(_sync)
        if removed:
            logger.info("parked_state_deleted", job_id=job_id)
        return removed

    async def save_traces(self, state: AnalysisState) -> None:
        """Save agent traces."""
        def _sync():
            lock = self._get_lock(self.traces_file)
            with lock:
                all_traces = self._load_json(self.traces_file)

                trace_list = []
                for trace in state.agent_traces:
                    trace_data = {
                        "agent_name": trace.agent_name,
                        "timestamp": trace.timestamp.isoformat() if trace.timestamp else None,
                        "action": trace.action,
                        "reasoning": trace.reasoning,
                        "inputs": trace.inputs,
                        "outputs": trace.outputs,
                        "tools_called": trace.tools_called,
                        "duration_ms": trace.duration_ms,
                        "token_usage": trace.token_usage,
                    }
                    trace_list.append(trace_data)

                all_traces[state.job_id] = trace_list
                self._save_json(self.traces_file, all_traces)

        await asyncio.to_thread(_sync)
        logger.debug("traces_saved", job_id=state.job_id, n_traces=len(state.agent_traces))

    async def get_traces(self, job_id: str) -> list[dict[str, Any]]:
        """Get agent traces for a job."""
        def _sync():
            all_traces = self._load_json(self.traces_file)
            traces = all_traces.get(job_id, [])
            # Sort by timestamp
            traces.sort(key=lambda x: x.get("timestamp", ""))
            return traces

        return await asyncio.to_thread(_sync)

    # ── Distributed coordination ───────────────────────────────

    async def get_orphaned_jobs(self, exclude_instance: str) -> list[dict[str, Any]]:
        """Find jobs stuck in non-terminal status from dead instances."""
        terminal = {"completed", "failed", "cancelled"}

        def _sync():
            jobs = self._load_json(self.jobs_file)
            return [
                j
                for j in jobs.values()
                if j.get("status") not in terminal
                and j.get("instance_id") != exclude_instance
            ]

        return await asyncio.to_thread(_sync)


# Singleton instance
_local_client: LocalStorageClient | None = None


def get_local_storage_client() -> LocalStorageClient:
    """Get the singleton local storage client."""
    global _local_client
    if _local_client is None:
        _local_client = LocalStorageClient()
    return _local_client


def reset_local_storage_client() -> None:
    """Reset the cached client. Useful for testing."""
    global _local_client
    _local_client = None
