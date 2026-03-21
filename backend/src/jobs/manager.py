"""Job manager for handling analysis jobs.

Supports multi-instance deployment via:
- Distributed job semaphore (Firestore-backed capacity checks)
- Instance identity tracking (each process gets a unique instance_id)
- Orphaned job recovery (on startup, marks dead instances' jobs as FAILED)
- Heartbeat loop (proactive dead-instance detection every 30s)
"""

import asyncio
import uuid
from datetime import datetime, timezone
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
from src.config import get_settings
from src.logging_config.structured import get_logger
from src.storage.cleanup import cleanup_local_artifacts
from src.storage.firestore import get_storage_client

logger = get_logger(__name__)

# Orchestrator mode type
OrchestratorMode = Literal["standard", "react"]


async def recover_orphaned_jobs() -> int:
    """Recover jobs left in non-terminal status by dead instances.

    Called on server startup. Any non-terminal job whose instance_id
    differs from ours was running on a now-dead process.

    Returns:
        Number of orphaned jobs recovered.
    """
    settings = get_settings()
    storage = get_storage_client()
    instance_id = settings.instance_id

    try:
        orphaned = await storage.get_orphaned_jobs(exclude_instance=instance_id)
    except Exception:
        logger.warning("orphan_recovery_skipped", reason="storage unavailable")
        return 0

    recovered = 0
    for job in orphaned:
        job_id = job.get("id", "unknown")
        old_status = job.get("status", "unknown")
        old_instance = job.get("instance_id", "unknown")
        try:
            await storage.update_job_status(
                job_id,
                JobStatus.FAILED,
                f"Server restarted; job interrupted (was {old_status} on instance {old_instance})",
            )
            logger.info(
                "orphaned_job_recovered",
                job_id=job_id,
                old_status=old_status,
                old_instance=old_instance,
            )
            recovered += 1
        except Exception:
            logger.warning("orphan_recovery_failed", job_id=job_id, exc_info=True)

    if recovered:
        logger.info("orphan_recovery_complete", recovered=recovered)
    return recovered


class JobManager:
    """Manager for causal analysis jobs."""

    def __init__(self, orchestrator_mode: OrchestratorMode = "standard") -> None:
        """Initialize the job manager.

        Args:
            orchestrator_mode: Which orchestrator to use:
                - "standard": Original orchestrator with fixed workflow
                - "react": Fully autonomous ReAct orchestrator (experimental)
        """
        settings = get_settings()
        self.firestore = get_storage_client()
        self._running_jobs: dict[str, asyncio.Task] = {}
        self._active_states: dict[str, AnalysisState] = {}  # Live state refs for SSE
        self._jobs_lock = asyncio.Lock()  # Protects _running_jobs from race conditions
        self._job_semaphore = asyncio.Semaphore(settings.max_concurrent_jobs)
        self._orchestrator_mode = orchestrator_mode
        self._heartbeat_task: asyncio.Task | None = None

        logger.info(
            "job_manager_initialized",
            mode=orchestrator_mode,
            instance_id=settings.instance_id,
        )

    def _create_orchestrator(self):
        """Create a fresh orchestrator with fresh agent instances per job.

        Each job gets its own agent instances to prevent concurrent jobs from
        cross-contaminating mutable instance state (e.g., _df, _profile).
        """
        agents = create_all_agents()
        if self._orchestrator_mode == "react":
            agents["effect_estimator"] = EffectEstimatorReActAgent()
            orchestrator = ReActOrchestrator()
        else:
            orchestrator = OrchestratorAgent()

        for name, agent in agents.items():
            orchestrator.register_specialist(name, agent)

        return orchestrator

    async def create_job(
        self,
        kaggle_url: str,
        treatment_variable: str | None = None,
        outcome_variable: str | None = None,
    ) -> str:
        """Create a new analysis job.

        Uses a distributed capacity check (Firestore transaction) to enforce
        max_concurrent_jobs across all instances, plus a local semaphore as
        a per-instance guard.

        Args:
            kaggle_url: Kaggle dataset URL
            treatment_variable: Optional treatment variable hint
            outcome_variable: Optional outcome variable hint

        Returns:
            Job ID

        Raises:
            RuntimeError: If at capacity (global or per-instance).
        """
        settings = get_settings()

        # Per-instance guard (fast path, avoids Firestore round-trip)
        if self._job_semaphore.locked():
            n_running = len(self._running_jobs)
            raise RuntimeError(
                f"Server at capacity ({n_running} jobs running). Try again later."
            )

        # Generate job ID
        job_id = str(uuid.uuid4())[:8]

        # Create initial state
        now = datetime.now(timezone.utc)
        state = AnalysisState(
            job_id=job_id,
            dataset_info=DatasetInfo(url=kaggle_url),
            treatment_variable=treatment_variable,
            outcome_variable=outcome_variable,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
        )

        # Distributed capacity check + atomic job creation
        created = await self.firestore.create_job_if_capacity(
            state, settings.max_concurrent_jobs
        )
        if not created:
            raise RuntimeError(
                f"Server at capacity (global limit: {settings.max_concurrent_jobs} concurrent jobs). "
                "Try again later."
            )

        # Start async processing
        task = asyncio.create_task(self._run_job(state))
        async with self._jobs_lock:
            self._running_jobs[job_id] = task

        logger.info(
            "job_created",
            job_id=job_id,
            kaggle_url=kaggle_url,
            instance_id=settings.instance_id,
        )

        return job_id

    async def _run_job(self, state: AnalysisState) -> None:
        """Run the analysis job with timeout enforcement.

        Args:
            state: Initial analysis state
        """
        async with self._job_semaphore:
            await self._run_job_inner(state)

    async def _run_job_inner(self, state: AnalysisState) -> None:
        """Inner job execution (runs under semaphore)."""
        job_id = state.job_id
        settings = get_settings()
        timeout_seconds = settings.agent_timeout_seconds * settings.job_timeout_multiplier  # Total job timeout

        logger.info("job_started", job_id=job_id, timeout_seconds=timeout_seconds)

        # Store live state reference for SSE streaming
        self._active_states[job_id] = state

        try:
            # Update status to fetching (with status guard)
            state.status = JobStatus.FETCHING_DATA
            await self.firestore.update_job(state, expected_status=JobStatus.PENDING)

            # Set up status callback so the orchestrator can persist
            # intermediate status updates to Firestore during the pipeline
            async def _persist_status(s: AnalysisState) -> None:
                try:
                    await self.firestore.update_job(s)
                except Exception:
                    logger.debug("status_persist_failed", job_id=s.job_id, exc_info=True)

            orchestrator = self._create_orchestrator()
            orchestrator.set_status_callback(_persist_status)

            # Run the orchestrator with timeout
            try:
                final_state = await asyncio.wait_for(
                    orchestrator.execute_with_tracing(state),
                    timeout=timeout_seconds,
                )
                # Update live state reference (orchestrator may return a new object)
                self._active_states[job_id] = final_state
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
                    logger.warning("trace_save_on_cancel_failed", job_id=state.job_id, exc_info=True)

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
            # Remove from running jobs (under lock to avoid race with cancel)
            async with self._jobs_lock:
                self._running_jobs.pop(job_id, None)
            # Clean up live state reference
            self._active_states.pop(job_id, None)

    # ── Heartbeat loop (proactive dead-instance detection) ─────

    async def start_heartbeat(self) -> None:
        """Start the background heartbeat loop.

        Periodically writes a heartbeat to storage and checks for
        stale instances whose orphaned jobs need recovery.
        """
        if self._heartbeat_task is not None:
            return  # Already running

        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("heartbeat_started")

    async def stop_heartbeat(self) -> None:
        """Stop the background heartbeat loop."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
            logger.info("heartbeat_stopped")

    async def _heartbeat_loop(self) -> None:
        """Background loop: heartbeat every 30s, check for dead instances."""
        settings = get_settings()

        while True:
            try:
                # Write our heartbeat
                n_active = len(self._running_jobs)
                if hasattr(self.firestore, "upsert_heartbeat"):
                    await self.firestore.upsert_heartbeat(
                        settings.instance_id, n_active
                    )

                # Check for stale instances
                if hasattr(self.firestore, "get_stale_instances"):
                    stale = await self.firestore.get_stale_instances(
                        threshold_seconds=settings.heartbeat_stale_threshold_seconds
                    )
                    for dead_instance in stale:
                        # Recover orphaned jobs from this dead instance
                        orphaned = await self.firestore.get_orphaned_jobs(
                            exclude_instance=settings.instance_id
                        )
                        for job in orphaned:
                            if job.get("instance_id") == dead_instance:
                                job_id = job.get("id", "unknown")
                                await self.firestore.update_job_status(
                                    job_id,
                                    JobStatus.FAILED,
                                    f"Instance {dead_instance} died; job interrupted",
                                )
                                logger.info(
                                    "heartbeat_recovered_orphan",
                                    job_id=job_id,
                                    dead_instance=dead_instance,
                                )
                        # Clean up dead instance heartbeat
                        if hasattr(self.firestore, "delete_heartbeat"):
                            await self.firestore.delete_heartbeat(dead_instance)

            except asyncio.CancelledError:
                raise
            except Exception:
                logger.debug("heartbeat_error", exc_info=True)

            await asyncio.sleep(settings.heartbeat_interval_seconds)

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

        # Delete local artifacts (synchronous I/O, offload to thread)
        result["local_artifacts_deleted"] = await asyncio.to_thread(
            cleanup_local_artifacts, job_id
        )

        # Delete from Firestore (cascades to results and traces)
        firestore_result = await self.firestore.delete_job(job_id, cascade=True)
        result["firestore_deleted"] = firestore_result.get("job", False)

        logger.info("job_deleted", **result)
        return result

    def get_sse_events(self, job_id: str, after_index: int = 0) -> list[dict]:
        """Get SSE events for a running job, starting after the given index.

        Args:
            job_id: Job ID
            after_index: Return events after this index (0-based)

        Returns:
            List of new SSE events (may be empty)
        """
        state = self._active_states.get(job_id)
        if state is None:
            return []
        return state.sse_events[after_index:]

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
