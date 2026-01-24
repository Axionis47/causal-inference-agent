"""Local filesystem cleanup utilities for job artifacts."""

import tempfile
from pathlib import Path

from src.logging_config.structured import get_logger

logger = get_logger(__name__)

CAUSAL_TEMP_DIR = Path(tempfile.gettempdir()) / "causal_orchestrator"


def cleanup_local_artifacts(job_id: str) -> dict[str, bool]:
    """Clean up local temp files for a job.

    Removes:
    - Pickled DataFrame: {job_id}_data.pkl
    - Generated notebook: causal_analysis_{job_id}.ipynb

    Args:
        job_id: Job ID to clean up

    Returns:
        Dict with cleanup status for each artifact type
    """
    cleaned = {"dataframe": False, "notebook": False}

    # Clean up pickled DataFrame
    df_path = CAUSAL_TEMP_DIR / f"{job_id}_data.pkl"
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

    return cleaned
