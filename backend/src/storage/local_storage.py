"""Local JSON-based storage client for development."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from filelock import FileLock

from src.agents.base import AnalysisState, JobStatus
from src.config import get_settings
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


class LocalStorageClient:
    """Local file-based storage for development (no GCP required)."""

    def __init__(self) -> None:
        """Initialize the local storage client."""
        settings = get_settings()
        self.storage_path = Path(settings.local_storage_path)
        self.jobs_file = self.storage_path / "jobs.json"
        self.results_file = self.storage_path / "results.json"
        self.traces_file = self.storage_path / "traces.json"

        # Create storage directory
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Initialize files if they don't exist
        for file in [self.jobs_file, self.results_file, self.traces_file]:
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

    async def create_job(self, state: AnalysisState) -> str:
        """Create a new job document."""
        lock = self._get_lock(self.jobs_file)
        with lock:
            jobs = self._load_json(self.jobs_file)

            job_data = {
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
            }

            jobs[state.job_id] = job_data
            self._save_json(self.jobs_file, jobs)

        logger.info("job_created", job_id=state.job_id)
        return state.job_id

    async def update_job(self, state: AnalysisState) -> None:
        """Update a job document."""
        lock = self._get_lock(self.jobs_file)
        with lock:
            jobs = self._load_json(self.jobs_file)

            if state.job_id not in jobs:
                logger.warning("job_not_found_for_update", job_id=state.job_id)
                return

            jobs[state.job_id].update({
                "status": state.status.value,
                "treatment_variable": state.treatment_variable,
                "outcome_variable": state.outcome_variable,
                "iteration_count": state.iteration_count,
                "updated_at": datetime.utcnow().isoformat(),
                "error_message": state.error_message,
                "error_agent": state.error_agent,
                "notebook_path": state.notebook_path,
            })

            if state.completed_at:
                jobs[state.job_id]["completed_at"] = state.completed_at.isoformat()

            self._save_json(self.jobs_file, jobs)
        logger.debug("job_updated", job_id=state.job_id, status=state.status.value)

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get a job by ID."""
        jobs = self._load_json(self.jobs_file)
        return jobs.get(job_id)

    async def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List jobs with optional filtering."""
        jobs = self._load_json(self.jobs_file)
        job_list = list(jobs.values())

        if status:
            job_list = [j for j in job_list if j.get("status") == status.value]

        # Sort by created_at descending
        job_list.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        total = len(job_list)
        paginated = job_list[offset : offset + limit]

        return paginated, total

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: str | None = None,
    ) -> bool:
        """Update just the status of a job."""
        lock = self._get_lock(self.jobs_file)
        with lock:
            jobs = self._load_json(self.jobs_file)

            if job_id not in jobs:
                return False

            jobs[job_id]["status"] = status.value
            jobs[job_id]["updated_at"] = datetime.utcnow().isoformat()
            if error_message is not None:
                jobs[job_id]["error_message"] = error_message

            self._save_json(self.jobs_file, jobs)
        logger.info("job_status_updated", job_id=job_id, status=status.value)
        return True

    async def delete_job(self, job_id: str, cascade: bool = True) -> dict[str, Any]:
        """Delete a job and optionally cascade to related data."""
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

        logger.info("job_deleted", job_id=job_id, cascade=cascade, result=result)
        return result

    async def save_results(self, state: AnalysisState) -> None:
        """Save analysis results."""
        lock = self._get_lock(self.results_file)
        with lock:
            results = self._load_json(self.results_file)

            results_data: dict[str, Any] = {
                "job_id": state.job_id,
                "created_at": datetime.utcnow().isoformat(),
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

            results[state.job_id] = results_data
            self._save_json(self.results_file, results)
        logger.info("results_saved", job_id=state.job_id)

    async def get_results(self, job_id: str) -> dict[str, Any] | None:
        """Get analysis results."""
        results = self._load_json(self.results_file)
        return results.get(job_id)

    async def save_traces(self, state: AnalysisState) -> None:
        """Save agent traces."""
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
                }
                trace_list.append(trace_data)

            all_traces[state.job_id] = trace_list
            self._save_json(self.traces_file, all_traces)
        logger.debug("traces_saved", job_id=state.job_id, n_traces=len(state.agent_traces))

    async def get_traces(self, job_id: str) -> list[dict[str, Any]]:
        """Get agent traces for a job."""
        all_traces = self._load_json(self.traces_file)
        traces = all_traces.get(job_id, [])
        # Sort by timestamp
        traces.sort(key=lambda x: x.get("timestamp", ""))
        return traces


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
