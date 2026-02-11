"""Job manager for handling analysis jobs."""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Literal

from src.agents import (
    AnalysisState,
    DatasetInfo,
    EffectEstimatorReActAgent,
    JobStatus,
    OrchestratorAgent,
    ReActOrchestrator,
)
from src.agents.registry import create_all_agents
from src.logging_config.structured import get_logger
from src.storage.cleanup import cleanup_local_artifacts
from src.storage.firestore import get_storage_client

logger = get_logger(__name__)

# Orchestrator mode type
OrchestratorMode = Literal["standard", "react"]


class JobManager:
    """Manager for causal analysis jobs."""

    def __init__(self, orchestrator_mode: OrchestratorMode = "standard") -> None:
        """Initialize the job manager.

        Args:
            orchestrator_mode: Which orchestrator to use:
                - "standard": Original orchestrator with fixed workflow
                - "react": Fully autonomous ReAct orchestrator (experimental)
        """
        self.firestore = get_storage_client()
        self._running_jobs: dict[str, asyncio.Task] = {}
        self._jobs_lock = asyncio.Lock()  # Protects _running_jobs from race conditions
        self._orchestrator_mode = orchestrator_mode

        # Initialize orchestrator based on mode
        if orchestrator_mode == "react":
            self.orchestrator = ReActOrchestrator()
            logger.info("using_react_orchestrator")
        else:
            self.orchestrator = OrchestratorAgent()
            logger.info("using_standard_orchestrator")

        self._setup_agents()

    def _setup_agents(self) -> None:
        """Set up and register all agents with the orchestrator via the registry."""
        agents = create_all_agents()

        # Override effect_estimator with ReAct variant when in react mode
        if self._orchestrator_mode == "react":
            agents["effect_estimator"] = EffectEstimatorReActAgent()
            logger.info("using_react_effect_estimator")

        for name, agent in agents.items():
            self.orchestrator.register_specialist(name, agent)

        logger.info("agents_initialized", mode=self._orchestrator_mode, count=len(agents))

    async def create_job(
        self,
        kaggle_url: str,
        treatment_variable: str | None = None,
        outcome_variable: str | None = None,
        preferences: dict[str, Any] | None = None,
    ) -> str:
        """Create a new analysis job.

        Args:
            kaggle_url: Kaggle dataset URL
            treatment_variable: Optional treatment variable hint
            outcome_variable: Optional outcome variable hint
            preferences: Optional analysis preferences

        Returns:
            Job ID
        """
        # Generate job ID
        job_id = str(uuid.uuid4())[:8]

        # Create initial state
        state = AnalysisState(
            job_id=job_id,
            dataset_info=DatasetInfo(url=kaggle_url),
            treatment_variable=treatment_variable,
            outcome_variable=outcome_variable,
            status=JobStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        # Save to Firestore
        await self.firestore.create_job(state)

        # Start async processing
        task = asyncio.create_task(self._run_job(state))
        async with self._jobs_lock:
            self._running_jobs[job_id] = task

        logger.info(
            "job_created",
            job_id=job_id,
            kaggle_url=kaggle_url,
        )

        return job_id

    async def _run_job(self, state: AnalysisState) -> None:
        """Run the analysis job with timeout enforcement.

        Args:
            state: Initial analysis state
        """
        from src.config import get_settings

        job_id = state.job_id
        settings = get_settings()
        timeout_seconds = settings.agent_timeout_seconds * 10  # Total job timeout (10x agent timeout)

        logger.info("job_started", job_id=job_id, timeout_seconds=timeout_seconds)

        try:
            # Update status to fetching
            state.status = JobStatus.FETCHING_DATA
            await self.firestore.update_job(state)

            # Set up status callback so the orchestrator can persist
            # intermediate status updates to Firestore during the pipeline
            async def _persist_status(s: AnalysisState) -> None:
                try:
                    await self.firestore.update_job(s)
                except Exception:
                    pass  # Non-critical

            self.orchestrator.set_status_callback(_persist_status)

            # Run the orchestrator with timeout
            try:
                final_state = await asyncio.wait_for(
                    self.orchestrator.execute_with_tracing(state),
                    timeout=timeout_seconds,
                )
            except TimeoutError:
                logger.error(
                    "job_timeout",
                    job_id=job_id,
                    timeout_seconds=timeout_seconds,
                )
                state.mark_failed(
                    f"Job timed out after {timeout_seconds} seconds",
                    "job_manager"
                )
                await self.firestore.update_job(state)
                return

            # Save results if completed or has treatment effects
            if final_state.status == JobStatus.COMPLETED or final_state.treatment_effects:
                await self.firestore.save_results(final_state)

            # Save traces
            await self.firestore.save_traces(final_state)

            # Final update
            await self.firestore.update_job(final_state)

            logger.info(
                "job_completed",
                job_id=job_id,
                status=final_state.status.value,
                n_effects=len(final_state.treatment_effects),
            )

        except asyncio.CancelledError:
            logger.info("job_cancelled", job_id=job_id)
            state.mark_cancelled("Job was cancelled by user")
            await self.firestore.update_job(state)

            # Best effort: save partial traces if any exist
            if state.agent_traces:
                try:
                    await self.firestore.save_traces(state)
                except Exception:
                    pass  # Don't fail cancellation due to trace save failure

            # Re-raise to complete cancellation
            raise

        except Exception as e:
            logger.error(
                "job_failed",
                job_id=job_id,
                error=str(e),
            )

            state.mark_failed(str(e), "job_manager")
            await self.firestore.update_job(state)

        finally:
            # Remove from running jobs
            self._running_jobs.pop(job_id, None)

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get a job by ID.

        Args:
            job_id: Job ID

        Returns:
            Job data or None
        """
        return await self.firestore.get_job(job_id)

    async def get_job_status(self, job_id: str) -> dict[str, Any] | None:
        """Get lightweight job status.

        Args:
            job_id: Job ID

        Returns:
            Status data or None
        """
        job = await self.firestore.get_job(job_id)
        if job is None:
            return None

        # Calculate progress
        progress = self._calculate_progress(job.get("status", "pending"))

        return {
            "id": job_id,
            "status": job.get("status"),
            "progress_percentage": progress,
            "current_agent": self._get_current_agent(job.get("status")),
        }

    async def list_jobs(
        self,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List jobs.

        Args:
            status: Filter by status
            limit: Max results
            offset: Skip results

        Returns:
            Tuple of (jobs, total)
        """
        job_status = JobStatus(status) if status else None
        return await self.firestore.list_jobs(job_status, limit, offset)

    async def cancel_job(self, job_id: str, graceful_timeout: float = 5.0) -> dict[str, Any]:
        """Cancel a running job with graceful shutdown.

        Args:
            job_id: Job ID to cancel
            graceful_timeout: Seconds to wait for graceful shutdown

        Returns:
            Dict with cancellation result
        """
        result = {
            "job_id": job_id,
            "was_running": False,
            "cancelled": False,
            "status": None,
        }

        # Check current job status in Firestore
        job = await self.firestore.get_job(job_id)
        if job is None:
            return result

        current_status = job.get("status")
        result["status"] = current_status

        # If job is already in a terminal state, nothing to cancel
        if current_status in ["completed", "failed", "cancelled"]:
            result["cancelled"] = False
            return result

        # Check if task is in memory (protected by lock to avoid race conditions)
        async with self._jobs_lock:
            task = self._running_jobs.get(job_id)
            task_running = task and not task.done()

        if task_running:
            result["was_running"] = True

            # Mark as cancelling in Firestore first
            await self.firestore.update_job_status(
                job_id,
                JobStatus.CANCELLING,
                "Cancellation initiated by user",
            )

            # Request cancellation
            task.cancel()

            # Wait for graceful shutdown
            try:
                await asyncio.wait_for(
                    asyncio.shield(task),
                    timeout=graceful_timeout,
                )
            except (asyncio.CancelledError, TimeoutError):
                pass  # Expected

            # Remove from running jobs (protected by lock)
            async with self._jobs_lock:
                self._running_jobs.pop(job_id, None)
            result["cancelled"] = True
            result["status"] = "cancelled"

        else:
            # Job not in memory but status indicates running
            # This happens after server restart
            if current_status not in ["completed", "failed", "cancelled"]:
                # Mark as cancelled directly
                await self.firestore.update_job_status(
                    job_id,
                    JobStatus.CANCELLED,
                    "Job cancelled (was not actively running)",
                )
                result["cancelled"] = True
                result["status"] = "cancelled"

        logger.info("job_cancel_result", **result)
        return result

    async def delete_job(self, job_id: str, force: bool = False) -> dict[str, Any]:
        """Delete a job and all its artifacts.

        Args:
            job_id: Job ID to delete
            force: If True, delete even if job is running (will cancel first)

        Returns:
            Dict with deletion results

        Raises:
            ValueError: If job is running and force=False
        """
        result = {
            "job_id": job_id,
            "found": False,
            "cancelled": False,
            "firestore_deleted": False,
            "local_artifacts_deleted": {},
        }

        # Check if job exists
        job = await self.firestore.get_job(job_id)
        if job is None:
            return result

        result["found"] = True
        current_status = job.get("status")

        # If job is running, cancel it first (if force=True) or reject
        if current_status not in ["completed", "failed", "cancelled"]:
            if not force:
                raise ValueError(
                    f"Job {job_id} is still running (status: {current_status}). "
                    "Use force=True to cancel and delete."
                )
            # Cancel first
            cancel_result = await self.cancel_job(job_id)
            result["cancelled"] = cancel_result.get("cancelled", False)

        # Delete local artifacts
        result["local_artifacts_deleted"] = cleanup_local_artifacts(job_id)

        # Delete from Firestore (cascades to results and traces)
        firestore_result = await self.firestore.delete_job(job_id, cascade=True)
        result["firestore_deleted"] = firestore_result.get("job", False)

        logger.info("job_deleted", **result)
        return result

    async def get_results(self, job_id: str) -> dict[str, Any] | None:
        """Get analysis results.

        Args:
            job_id: Job ID

        Returns:
            Results data or None
        """
        return await self.firestore.get_results(job_id)

    async def get_traces(self, job_id: str) -> list[dict[str, Any]]:
        """Get agent traces.

        Args:
            job_id: Job ID

        Returns:
            List of traces
        """
        return await self.firestore.get_traces(job_id)

    def _calculate_progress(self, status: str) -> int:
        """Calculate progress percentage from status.

        Progress is distributed more evenly across stages (~10-12% each)
        to provide smoother visual feedback during analysis.
        """
        progress_map = {
            "pending": 0,
            "fetching_data": 8,
            "profiling": 20,
            "exploratory_analysis": 32,
            "discovering_causal": 44,
            "estimating_effects": 56,
            "sensitivity_analysis": 68,
            "critique_review": 78,
            "iterating": 84,
            "generating_notebook": 92,
            "completed": 100,
            "failed": 0,
            "cancelling": 0,
            "cancelled": 0,
        }
        return progress_map.get(status, 0)

    def _get_current_agent(self, status: str) -> str | None:
        """Get current agent name from status."""
        agent_map = {
            "profiling": "data_profiler",
            "exploratory_analysis": "eda_agent",
            "discovering_causal": "causal_discovery",
            "estimating_effects": "effect_estimator",
            "sensitivity_analysis": "sensitivity_analyst",
            "critique_review": "critique",
            "generating_notebook": "notebook_generator",
        }
        return agent_map.get(status)


# Singleton instance
_manager: JobManager | None = None


def get_job_manager(orchestrator_mode: OrchestratorMode = "standard") -> JobManager:
    """Get the singleton job manager.

    Args:
        orchestrator_mode: Which orchestrator to use:
            - "standard": Original orchestrator with fixed workflow
            - "react": Fully autonomous ReAct orchestrator (experimental)

    Note: The mode is only used on first initialization. Subsequent calls
    return the existing instance regardless of the mode parameter.
    """
    global _manager
    if _manager is None:
        _manager = JobManager(orchestrator_mode=orchestrator_mode)
    return _manager


def reset_job_manager() -> None:
    """Reset the job manager singleton (mainly for testing)."""
    global _manager
    _manager = None
