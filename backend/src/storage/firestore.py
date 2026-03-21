"""Firestore storage client for jobs and results."""

import asyncio
from datetime import datetime, timezone
from typing import Any

from google.cloud import firestore

from src.agents.base import AnalysisState, JobStatus
from src.config import get_settings
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


def _utcnow() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class FirestoreClient:
    """Client for Firestore database operations.

    All synchronous Firestore SDK calls are wrapped in asyncio.to_thread()
    to avoid blocking the event loop.
    """

    def __init__(self) -> None:
        """Initialize the Firestore client."""
        settings = get_settings()
        self.db = firestore.Client(project=settings.gcp_project_id)
        self.jobs_collection = "jobs"
        self.results_collection = "results"
        self.traces_collection = "agent_traces"

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
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "error_message": state.error_message,
            "error_agent": state.error_agent,
            "instance_id": settings.instance_id,
        }

    async def create_job(self, state: AnalysisState) -> str:
        """Create a new job document.

        Args:
            state: Initial analysis state

        Returns:
            Job ID
        """
        job_data = self._job_data_from_state(state)

        def _sync():
            doc_ref = self.db.collection(self.jobs_collection).document(state.job_id)
            doc_ref.set(job_data)

        await asyncio.to_thread(_sync)

        logger.info("job_created", job_id=state.job_id)
        return state.job_id

    async def create_job_if_capacity(
        self, state: AnalysisState, max_concurrent: int
    ) -> bool:
        """Atomically create a job only if under the global concurrency limit.

        Uses a Firestore transaction to count active (non-terminal) jobs and
        create the new job in a single atomic operation. This enforces the
        concurrency limit across all Cloud Run instances.

        Args:
            state: Initial analysis state
            max_concurrent: Maximum allowed concurrent jobs

        Returns:
            True if job was created, False if at capacity
        """
        job_data = self._job_data_from_state(state)
        terminal = {"completed", "failed", "cancelled"}

        def _sync():
            @firestore.transactional
            def _txn(transaction):
                # Count non-terminal jobs
                docs = list(
                    self.db.collection(self.jobs_collection).stream(
                        transaction=transaction
                    )
                )
                active = sum(
                    1 for d in docs if d.to_dict().get("status") not in terminal
                )
                if active >= max_concurrent:
                    return False

                doc_ref = self.db.collection(self.jobs_collection).document(
                    state.job_id
                )
                transaction.set(doc_ref, job_data)
                return True

            transaction = self.db.transaction()
            return _txn(transaction)

        return await asyncio.to_thread(_sync)

    async def update_job(
        self,
        state: AnalysisState,
        expected_status: JobStatus | None = None,
    ) -> bool:
        """Update a job document using a Firestore transaction for safety.

        Args:
            state: Updated analysis state
            expected_status: If set, only update if current status matches.
                Prevents race conditions between concurrent transitions.

        Returns:
            True if updated, False if rejected (status mismatch or not found).
        """
        settings = get_settings()

        def _sync():
            doc_ref = self.db.collection(self.jobs_collection).document(state.job_id)

            @firestore.transactional
            def _update_in_transaction(transaction, doc_ref, state):
                snapshot = doc_ref.get(transaction=transaction)
                if not snapshot.exists:
                    logger.warning("job_not_found_for_update", job_id=state.job_id)
                    return False

                # Status guard: reject stale transitions
                if expected_status is not None:
                    current = snapshot.to_dict().get("status")
                    if current != expected_status.value:
                        logger.warning(
                            "state_transition_rejected",
                            job_id=state.job_id,
                            expected=expected_status.value,
                            actual=current,
                            attempted=state.status.value,
                        )
                        return False

                job_data = {
                    "status": state.status.value,
                    "treatment_variable": state.treatment_variable,
                    "outcome_variable": state.outcome_variable,
                    "iteration_count": state.iteration_count,
                    "updated_at": _utcnow(),
                    "error_message": state.error_message,
                    "error_agent": state.error_agent,
                    "notebook_path": state.notebook_path,
                    "instance_id": settings.instance_id,
                }

                if state.completed_at:
                    job_data["completed_at"] = state.completed_at

                transaction.update(doc_ref, job_data)
                return True

            transaction = self.db.transaction()
            return _update_in_transaction(transaction, doc_ref, state)

        result = await asyncio.to_thread(_sync)

        logger.debug("job_updated", job_id=state.job_id, status=state.status.value)
        return result

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get a job by ID.

        Args:
            job_id: Job ID

        Returns:
            Job data or None if not found
        """
        def _sync():
            doc_ref = self.db.collection(self.jobs_collection).document(job_id)
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict()
            return None

        return await asyncio.to_thread(_sync)

    async def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """List jobs with optional filtering.

        Args:
            status: Filter by status
            limit: Maximum number of results
            offset: Number of results to skip

        Returns:
            Tuple of (jobs, total_count)
        """
        def _sync():
            query = self.db.collection(self.jobs_collection)

            if status:
                query = query.where("status", "==", status.value)

            # Use count aggregation instead of materializing all documents
            count_query = query.count()
            count_results = count_query.get()
            total = count_results[0][0].value if count_results else 0

            # Apply pagination
            query = query.order_by("created_at", direction=firestore.Query.DESCENDING)
            query = query.limit(limit).offset(offset)

            jobs = [doc.to_dict() for doc in query.stream()]
            return jobs, total

        return await asyncio.to_thread(_sync)

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        error_message: str | None = None,
    ) -> bool:
        """Update just the status of a job (for cancel operations).

        Args:
            job_id: Job ID
            status: New status
            error_message: Optional error message

        Returns:
            True if updated, False if job not found
        """
        def _sync():
            doc_ref = self.db.collection(self.jobs_collection).document(job_id)

            @firestore.transactional
            def _update_in_transaction(transaction, doc_ref):
                snapshot = doc_ref.get(transaction=transaction)
                if not snapshot.exists:
                    return False

                update_data = {
                    "status": status.value,
                    "updated_at": _utcnow(),
                }
                if error_message is not None:
                    update_data["error_message"] = error_message

                transaction.update(doc_ref, update_data)
                return True

            transaction = self.db.transaction()
            return _update_in_transaction(transaction, doc_ref)

        result = await asyncio.to_thread(_sync)
        if result:
            logger.info("job_status_updated", job_id=job_id, status=status.value)
        return result

    async def delete_job(self, job_id: str, cascade: bool = True) -> dict[str, Any]:
        """Delete a job and optionally cascade to related data.

        Args:
            job_id: Job ID
            cascade: If True, also delete results and traces

        Returns:
            Dict with deletion results
        """
        def _sync():
            result = {"job": False, "results": False, "traces": False}

            doc_ref = self.db.collection(self.jobs_collection).document(job_id)
            doc = doc_ref.get()

            if not doc.exists:
                return result

            # Delete job document
            doc_ref.delete()
            result["job"] = True

            if cascade:
                # Delete results document
                results_ref = self.db.collection(self.results_collection).document(job_id)
                if results_ref.get().exists:
                    results_ref.delete()
                    result["results"] = True

                # Delete traces subcollection
                traces_deleted = self._delete_traces_collection(job_id)
                result["traces"] = traces_deleted > 0

            return result

        result = await asyncio.to_thread(_sync)
        logger.info("job_deleted", job_id=job_id, cascade=cascade, result=result)
        return result

    def _delete_traces_collection(self, job_id: str, batch_size: int = 100) -> int:
        """Delete all traces for a job.

        Args:
            job_id: Job ID
            batch_size: Number of docs to delete per batch

        Returns:
            Number of traces deleted
        """
        total_deleted = 0
        collection_ref = (
            self.db.collection(self.traces_collection)
            .document(job_id)
            .collection("traces")
        )

        while True:
            docs = list(collection_ref.limit(batch_size).stream())
            if not docs:
                break

            for doc in docs:
                doc.reference.delete()
                total_deleted += 1

        return total_deleted

    async def save_results(self, state: AnalysisState) -> None:
        """Save analysis results.

        Args:
            state: Analysis state with results
        """
        results_data = {
            "job_id": state.job_id,
            "created_at": _utcnow(),
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

        def _sync():
            doc_ref = self.db.collection(self.results_collection).document(state.job_id)
            doc_ref.set(results_data)

        await asyncio.to_thread(_sync)

        logger.info("results_saved", job_id=state.job_id)

    async def get_results(self, job_id: str) -> dict[str, Any] | None:
        """Get analysis results.

        Args:
            job_id: Job ID

        Returns:
            Results data or None
        """
        def _sync():
            doc_ref = self.db.collection(self.results_collection).document(job_id)
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict()
            return None

        return await asyncio.to_thread(_sync)

    async def save_traces(self, state: AnalysisState) -> None:
        """Save agent traces.

        Args:
            state: Analysis state with traces
        """
        trace_data_list = []
        for trace in state.agent_traces:
            trace_data_list.append({
                "agent_name": trace.agent_name,
                "timestamp": trace.timestamp,
                "action": trace.action,
                "reasoning": trace.reasoning,
                "inputs": trace.inputs,
                "outputs": trace.outputs,
                "tools_called": trace.tools_called,
                "duration_ms": trace.duration_ms,
                "token_usage": trace.token_usage,
            })

        def _sync():
            collection_ref = self.db.collection(self.traces_collection).document(state.job_id)
            traces_ref = collection_ref.collection("traces")
            for trace_data in trace_data_list:
                traces_ref.add(trace_data)

        await asyncio.to_thread(_sync)

        logger.debug("traces_saved", job_id=state.job_id, n_traces=len(state.agent_traces))

    async def get_traces(self, job_id: str) -> list[dict[str, Any]]:
        """Get agent traces for a job.

        Args:
            job_id: Job ID

        Returns:
            List of trace data
        """
        def _sync():
            collection_ref = self.db.collection(self.traces_collection).document(job_id)
            traces_ref = collection_ref.collection("traces")
            return [doc.to_dict() for doc in traces_ref.order_by("timestamp").stream()]

        return await asyncio.to_thread(_sync)

    def _delete_collection(self, path: str, batch_size: int = 100) -> None:
        """Delete a collection (helper for cascading deletes)."""
        try:
            collection_ref = self.db.collection(path)
            docs = collection_ref.limit(batch_size).stream()
            deleted = 0

            for doc in docs:
                doc.reference.delete()
                deleted += 1

            if deleted >= batch_size:
                self._delete_collection(path, batch_size)
        except Exception:
            logger.debug("subcollection_delete_skipped", exc_info=True)

    # ── Distributed coordination ───────────────────────────────

    async def get_orphaned_jobs(self, exclude_instance: str) -> list[dict[str, Any]]:
        """Find jobs stuck in non-terminal status from dead instances.

        Args:
            exclude_instance: The current instance's ID (its jobs are NOT orphaned).

        Returns:
            List of job dicts whose owning instance is gone.
        """
        terminal = {"completed", "failed", "cancelled"}

        def _sync():
            docs = list(self.db.collection(self.jobs_collection).stream())
            orphaned = []
            for d in docs:
                data = d.to_dict()
                status = data.get("status")
                inst = data.get("instance_id")
                # Non-terminal + belongs to a different (or missing) instance
                if status not in terminal and inst != exclude_instance:
                    orphaned.append(data)
            return orphaned

        return await asyncio.to_thread(_sync)

    async def upsert_heartbeat(self, instance_id: str, active_jobs: int) -> None:
        """Write a heartbeat for this instance.

        Args:
            instance_id: The current instance's ID
            active_jobs: Number of jobs currently running on this instance
        """
        def _sync():
            doc_ref = self.db.collection("instances").document(instance_id)
            doc_ref.set({
                "instance_id": instance_id,
                "last_heartbeat": _utcnow(),
                "active_jobs": active_jobs,
            })

        await asyncio.to_thread(_sync)

    async def get_stale_instances(self, threshold_seconds: int = 90) -> list[str]:
        """Find instances whose heartbeat is older than threshold.

        Args:
            threshold_seconds: Seconds after which an instance is considered dead.

        Returns:
            List of stale instance IDs.
        """
        from datetime import timedelta

        cutoff = _utcnow() - timedelta(seconds=threshold_seconds)

        def _sync():
            docs = list(self.db.collection("instances").stream())
            stale = []
            for d in docs:
                data = d.to_dict()
                hb = data.get("last_heartbeat")
                if hb is not None and hb < cutoff:
                    stale.append(data.get("instance_id", d.id))
            return stale

        return await asyncio.to_thread(_sync)

    async def delete_heartbeat(self, instance_id: str) -> None:
        """Remove a heartbeat document for a dead instance."""
        def _sync():
            self.db.collection("instances").document(instance_id).delete()

        await asyncio.to_thread(_sync)


# Singleton instance
_client: FirestoreClient | None = None


def get_firestore_client() -> FirestoreClient:
    """Get the singleton Firestore client (for production use)."""
    global _client
    if _client is None:
        _client = FirestoreClient()
    return _client


# Type alias for storage client (both have same interface)
StorageClient = FirestoreClient  # LocalStorageClient has the same interface


def get_storage_client() -> StorageClient:
    """Get the appropriate storage client based on settings.

    Returns FirestoreClient for production, LocalStorageClient for development.
    """
    settings = get_settings()

    if settings.use_firestore:
        logger.info("using_firestore_storage")
        return get_firestore_client()
    else:
        logger.info("using_local_storage", path=settings.local_storage_path)
        from .local_storage import get_local_storage_client
        return get_local_storage_client()  # type: ignore
