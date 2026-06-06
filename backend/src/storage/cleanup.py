"""Local filesystem cleanup utilities for job artifacts."""

import shutil
import tempfile
from pathlib import Path

from src.logging_config.structured import get_logger
from src.storage.job_data import job_data_dir

logger = get_logger(__name__)

CAUSAL_TEMP_DIR = Path(tempfile.gettempdir()) / "causal_orchestrator"


def cleanup_local_artifacts(job_id: str) -> dict[str, bool]:
    """Clean up local temp files for a job.

    Removes:
    - Parquet DataFrame: {job_id}_data.parquet
    - Generated notebook: causal_analysis_{job_id}.ipynb
    - Durable per-job dataset dir: {local_storage_path}/{job_id}/ (raw + manifest)

    Args:
        job_id: Job ID to clean up

    Returns:
        Dict with cleanup status for each artifact type
    """
    cleaned = {"dataframe": False, "notebook": False, "job_data": False}

    # Clean up parquet DataFrame
    df_path = CAUSAL_TEMP_DIR / f"{job_id}_data.parquet"
    if df_path.exists():
        try:
            df_path.unlink()
            cleaned["dataframe"] = True
            logger.info("local_dataframe_deleted", job_id=job_id, path=str(df_path))
        except Exception as e:
            logger.warning(
                "local_dataframe_delete_failed",
                job_id=job_id,
                path=str(df_path),
                error=str(e),
            )

    # Clean up notebook
    notebook_dir = CAUSAL_TEMP_DIR / "notebooks"
    notebook_path = notebook_dir / f"causal_analysis_{job_id}.ipynb"
    if notebook_path.exists():
        try:
            notebook_path.unlink()
            cleaned["notebook"] = True
            logger.info("local_notebook_deleted", job_id=job_id, path=str(notebook_path))
        except Exception as e:
            logger.warning(
                "local_notebook_delete_failed",
                job_id=job_id,
                path=str(notebook_path),
                error=str(e),
            )

    # Clean up the durable per-job dataset directory (raw bundle + manifest)
    job_dir = job_data_dir(job_id)
    if job_dir.exists():
        try:
            shutil.rmtree(job_dir)
            cleaned["job_data"] = True
            logger.info("local_job_data_deleted", job_id=job_id, path=str(job_dir))
        except Exception as e:
            logger.warning(
                "local_job_data_delete_failed",
                job_id=job_id,
                path=str(job_dir),
                error=str(e),
            )

    return cleaned
