"""Job manager for handling analysis jobs."""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Literal

from src.agents import (
    AnalysisState,
    CausalDiscoveryAgent,
    CritiqueAgent,
    DataProfilerAgent,
    DatasetInfo,
    EDAAgent,
    EffectEstimatorAgent,
    EffectEstimatorReActAgent,
    JobStatus,
    NotebookGeneratorAgent,
    OrchestratorAgent,
    ReActOrchestrator,
    SensitivityAnalystAgent,
)
from src.logging_config.structured import get_logger
from src.storage.firestore import get_firestore_client

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
        self.firestore = get_firestore_client()
        self._running_jobs: dict[str, asyncio.Task] = {}
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
        """Set up and register all agents with the orchestrator."""
        # Create specialist agents
        # Use ReAct variants when in react mode for supported agents
        data_profiler = DataProfilerAgent()
        eda_agent = EDAAgent()
        causal_discovery = CausalDiscoveryAgent()

        if self._orchestrator_mode == "react":
            effect_estimator = EffectEstimatorReActAgent()
            logger.info("using_react_effect_estimator")
        else:
            effect_estimator = EffectEstimatorAgent()

        sensitivity_analyst = SensitivityAnalystAgent()
        notebook_generator = NotebookGeneratorAgent()
        critique = CritiqueAgent()

        # Register with orchestrator
        self.orchestrator.register_specialist("data_profiler", data_profiler)
        self.orchestrator.register_specialist("eda_agent", eda_agent)
        self.orchestrator.register_specialist("causal_discovery", causal_discovery)
        self.orchestrator.register_specialist("effect_estimator", effect_estimator)
        self.orchestrator.register_specialist("sensitivity_analyst", sensitivity_analyst)
        self.orchestrator.register_specialist("notebook_generator", notebook_generator)
        self.orchestrator.register_specialist("critique", critique)

        logger.info("agents_initialized", mode=self._orchestrator_mode)

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

            # Run the orchestrator with timeout
            try:
                final_state = await asyncio.wait_for(
                    self.orchestrator.execute_with_tracing(state),
                    timeout=timeout_seconds,
                )
            except asyncio.TimeoutError:
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

            # Save results
            if final_state.status == JobStatus.COMPLETED:
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
            state.mark_failed("Job was cancelled", "job_manager")
            await self.firestore.update_job(state)

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

    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job.

        Args:
            job_id: Job ID

        Returns:
            True if cancelled
        """
        task = self._running_jobs.get(job_id)
        if task:
            task.cancel()
            self._running_jobs.pop(job_id, None)

            # Update status
            job = await self.firestore.get_job(job_id)
            if job:
                job["status"] = JobStatus.FAILED.value
                job["error_message"] = "Job cancelled by user"
                # Update would need the full state - simplified here

            return True
        return False

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
        """Calculate progress percentage from status."""
        progress_map = {
            "pending": 0,
            "fetching_data": 10,
            "profiling": 15,
            "exploratory_analysis": 25,
            "discovering_causal": 35,
            "estimating_effects": 50,
            "sensitivity_analysis": 70,
            "critique_review": 80,
            "iterating": 85,
            "generating_notebook": 90,
            "completed": 100,
            "failed": 0,
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
