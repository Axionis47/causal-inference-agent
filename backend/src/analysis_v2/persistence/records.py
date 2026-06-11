"""Run-state record persistence through the existing storage layer.

Thin seam over the storage client's analysis-run methods so analysis code
never constructs clients. Both backends (local JSON, Firestore) implement
save/load/delete with the same signatures.
"""
from __future__ import annotations

from src.analysis_v2.core import AnalysisRunState
from src.storage.firestore import get_storage_client


async def save_run(run: AnalysisRunState) -> None:
    await get_storage_client().save_analysis_run(run)


async def load_run(job_id: str) -> AnalysisRunState | None:
    return await get_storage_client().load_analysis_run(job_id)


async def delete_run(job_id: str) -> bool:
    return await get_storage_client().delete_analysis_run(job_id)
