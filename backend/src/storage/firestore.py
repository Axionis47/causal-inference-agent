"""Firestore storage client for jobs and results."""

from datetime import datetime
from typing import Any

from google.cloud import firestore

from src.agents.base import AnalysisState, JobStatus
from src.config import get_settings
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


class FirestoreClient:
    """Client for Firestore database operations."""

    def __init__(self) -> None:
        """Initialize the Firestore client."""
        settings = get_settings()
        self.db = firestore.Client(project=settings.gcp_project_id)
        self.jobs_collection = "jobs"
        self.results_collection = "results"
        self.traces_collection = "agent_traces"

    async def create_job(self, state: AnalysisState) -> str:
        """Create a new job document.

        Args:
            state: Initial analysis state

        Returns:
            Job ID
        """
        job_data = {
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
        }

        doc_ref = self.db.collection(self.jobs_collection).document(state.job_id)
        doc_ref.set(job_data)

        logger.info("job_created", job_id=state.job_id)
        return state.job_id

    async def update_job(self, state: AnalysisState) -> None:
        """Update a job document.

        Args:
            state: Updated analysis state
        """
        job_data = {
            "status": state.status.value,
            "treatment_variable": state.treatment_variable,
            "outcome_variable": state.outcome_variable,
            "iteration_count": state.iteration_count,
            "updated_at": datetime.utcnow(),
            "error_message": state.error_message,
            "error_agent": state.error_agent,
            "notebook_path": state.notebook_path,
        }

        if state.completed_at:
            job_data["completed_at"] = state.completed_at

        doc_ref = self.db.collection(self.jobs_collection).document(state.job_id)
        doc_ref.update(job_data)

        logger.debug("job_updated", job_id=state.job_id, status=state.status.value)

    async def get_job(self, job_id: str) -> dict[str, Any] | None:
        """Get a job by ID.

        Args:
            job_id: Job ID

        Returns:
            Job data or None if not found
        """
        doc_ref = self.db.collection(self.jobs_collection).document(job_id)
        doc = doc_ref.get()

        if doc.exists:
            return doc.to_dict()
        return None

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
        query = self.db.collection(self.jobs_collection)

        if status:
            query = query.where("status", "==", status.value)

        # Get total count (simplified - in production use aggregation)
        total_docs = list(query.stream())
        total = len(total_docs)

        # Apply pagination
        query = query.order_by("created_at", direction=firestore.Query.DESCENDING)
        query = query.limit(limit).offset(offset)

        jobs = [doc.to_dict() for doc in query.stream()]
        return jobs, total

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
        doc_ref = self.db.collection(self.jobs_collection).document(job_id)
        doc = doc_ref.get()

        if not doc.exists:
            return False

        update_data = {
            "status": status.value,
            "updated_at": datetime.utcnow(),
        }
        if error_message is not None:
            update_data["error_message"] = error_message

        doc_ref.update(update_data)
        logger.info("job_status_updated", job_id=job_id, status=status.value)
        return True

    async def delete_job(self, job_id: str, cascade: bool = True) -> dict[str, Any]:
        """Delete a job and optionally cascade to related data.

        Args:
            job_id: Job ID
            cascade: If True, also delete results and traces

        Returns:
            Dict with deletion results
        """
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
            "created_at": datetime.utcnow(),
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

        doc_ref = self.db.collection(self.results_collection).document(state.job_id)
        doc_ref.set(results_data)

        logger.info("results_saved", job_id=state.job_id)

    async def get_results(self, job_id: str) -> dict[str, Any] | None:
        """Get analysis results.

        Args:
            job_id: Job ID

        Returns:
            Results data or None
        """
        doc_ref = self.db.collection(self.results_collection).document(job_id)
        doc = doc_ref.get()

        if doc.exists:
            return doc.to_dict()
        return None

    async def save_traces(self, state: AnalysisState) -> None:
        """Save agent traces.

        Args:
            state: Analysis state with traces
        """
        collection_ref = self.db.collection(self.traces_collection).document(state.job_id)
        traces_ref = collection_ref.collection("traces")

        for trace in state.agent_traces:
            trace_data = {
                "agent_name": trace.agent_name,
                "timestamp": trace.timestamp,
                "action": trace.action,
                "reasoning": trace.reasoning,
                "inputs": trace.inputs,
                "outputs": trace.outputs,
                "tools_called": trace.tools_called,
                "duration_ms": trace.duration_ms,
            }
            traces_ref.add(trace_data)

        logger.debug("traces_saved", job_id=state.job_id, n_traces=len(state.agent_traces))

    async def get_traces(self, job_id: str) -> list[dict[str, Any]]:
        """Get agent traces for a job.

        Args:
            job_id: Job ID

        Returns:
            List of trace data
        """
        collection_ref = self.db.collection(self.traces_collection).document(job_id)
        traces_ref = collection_ref.collection("traces")

        traces = []
        for doc in traces_ref.order_by("timestamp").stream():
            traces.append(doc.to_dict())

        return traces

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
            pass  # Collection may not exist


# Singleton instance
_client: FirestoreClient | None = None


def get_firestore_client() -> FirestoreClient:
    """Get the singleton Firestore client."""
    global _client
    if _client is None:
        _client = FirestoreClient()
    return _client
