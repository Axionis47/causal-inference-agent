"""Dataset loading and Kaggle metadata fetching for the data_profiler agent.

Functions take state as a parameter and return results (with errors as part
of the return value) rather than mutating instance attributes. The Kaggle
import here is temporary; per the refactor plan the download seam (S21/S25)
will replace this with reads from the sealed download module.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis.agents.base import AnalysisState, FileEntry
from src.logging_config.structured import get_logger

logger = get_logger(__name__)


def _setup_kaggle_credentials() -> tuple[str | None, str | None]:
    """Apply Kaggle credentials from settings to env vars.

    Returns the prior (username, key) values so the caller can restore
    them in a finally block. Used by both load_from_kaggle and
    download_to_workdir so the env-var dance is not duplicated.
    """
    import json
    import os

    from src.config import get_settings

    settings = get_settings()

    original_username = os.environ.get("KAGGLE_USERNAME")
    original_key = os.environ.get("KAGGLE_KEY")

    if settings.kaggle_key_value:
        kaggle_key = settings.kaggle_key_value
        kaggle_username = settings.kaggle_username

        if kaggle_key.startswith("{"):
            try:
                kaggle_creds = json.loads(kaggle_key)
                kaggle_username = kaggle_creds.get("username", kaggle_username)
                kaggle_key = kaggle_creds.get("key", kaggle_key)
            except json.JSONDecodeError:
                logger.debug("kaggle_key_json_parse_failed", exc_info=True)

        if kaggle_username:
            os.environ["KAGGLE_USERNAME"] = kaggle_username
        if kaggle_key:
            os.environ["KAGGLE_KEY"] = kaggle_key

    return original_username, original_key


def _restore_kaggle_credentials(
    original_username: str | None, original_key: str | None
) -> None:
    """Restore env vars to their pre-setup values. Pairs with _setup."""
    import os

    if original_username is not None:
        os.environ["KAGGLE_USERNAME"] = original_username
    elif "KAGGLE_USERNAME" in os.environ:
        del os.environ["KAGGLE_USERNAME"]

    if original_key is not None:
        os.environ["KAGGLE_KEY"] = original_key
    elif "KAGGLE_KEY" in os.environ:
        del os.environ["KAGGLE_KEY"]


async def download_to_workdir(
    state: AnalysisState, url: str
) -> tuple[Path | None, list[FileEntry], str | None]:
    """Download a Kaggle archive into a persistent workdir.

    Unlike load_from_kaggle, this does not pick a single file and does
    not auto-cleanup the workdir. The workdir lives under CAUSAL_TEMP_DIR
    so the job-scoped cleanup pass tears it down. Used by
    dataset_inspector to materialise every candidate file once and reuse
    it across N inner data_profiler invocations.

    Returns (workdir, file_list, error). When error is non-None the
    workdir is not guaranteed to exist.
    """
    from src.storage.cleanup import CAUSAL_TEMP_DIR

    original_username, original_key = _setup_kaggle_credentials()
    workdir = CAUSAL_TEMP_DIR / f"{state.job_id}_workdir"

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        parts = url.rstrip("/").split("/")
        if "datasets" not in parts:
            return None, [], f"Invalid Kaggle URL: {url}"
        idx = parts.index("datasets")
        dataset_id = f"{parts[idx + 1]}/{parts[idx + 2]}"

        workdir.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(dataset_id, path=str(workdir), unzip=True)
        files = capture_file_list(workdir, used_path=None)
        return workdir, files, None
    except Exception as exc:
        error = f"Failed to download Kaggle archive to workdir: {str(exc)[:200]}"
        logger.error(
            "download_to_workdir_failed", error=error, url=url, job_id=state.job_id
        )
        return None, [], error
    finally:
        _restore_kaggle_credentials(original_username, original_key)


def pick_default_file(tmpdir: Path) -> Path | None:
    """Return the file `load_from_kaggle` would pick when no upstream
    selection has been made.

    Default heuristic: the largest CSV in the directory; falls back to
    the first parquet if no CSVs are present; returns None if neither.
    Kept as a named helper so the upcoming dataset_inspector agent can
    preempt it by populating `state.dataframe_path` before the loader
    runs, while this remains the well-defined fallback.
    """
    csv_files = list(tmpdir.glob("*.csv"))
    if csv_files:
        return max(csv_files, key=lambda p: p.stat().st_size)
    parquet_files = list(tmpdir.glob("*.parquet"))
    if parquet_files:
        return parquet_files[0]
    return None


async def load_dataset(
    state: AnalysisState,
) -> tuple[pd.DataFrame | None, str | None]:
    """Load dataset from the source. Returns (dataframe, error_message).

    Resolution order: explicit dataframe_path, local_path, Kaggle URL,
    direct URL. Each stage swallows its own exceptions and falls through
    to the next; only the Kaggle path surfaces a structured error string.
    """
    dataset_info = state.dataset_info

    if state.dataframe_path:
        try:
            if state.dataframe_path.endswith(".csv"):
                return pd.read_csv(state.dataframe_path), None
            return pd.read_parquet(state.dataframe_path), None
        except Exception as e:
            logger.error(
                "dataframe_path_load_failed",
                error=str(e),
                path=state.dataframe_path,
            )

    if dataset_info.local_path:
        try:
            if dataset_info.local_path.endswith(".csv"):
                return pd.read_csv(dataset_info.local_path), None
            if dataset_info.local_path.endswith(".parquet"):
                return pd.read_parquet(dataset_info.local_path), None
            return pd.read_csv(dataset_info.local_path), None
        except Exception as e:
            logger.error("local_load_failed", error=str(e))

    if "kaggle.com" in dataset_info.url:
        return await load_from_kaggle(state, dataset_info.url)

    try:
        return pd.read_csv(dataset_info.url), None
    except Exception as e:
        logger.error("url_load_failed", error=str(e))
        return None, None


def _read_existing_bundle(state: AnalysisState) -> pd.DataFrame | None:
    """Read a previously downloaded bundle for this job, or None if absent.

    Looks up the job's manifest and reads its winner file straight off disk,
    skipping the Kaggle pull. Returns None (so the caller downloads fresh) if
    there is no manifest, no recorded winner, or the file is missing.
    """
    from src.storage.job_data import job_raw_dir, read_manifest

    manifest = read_manifest(state.job_id)
    if manifest is None or not manifest.winner:
        return None

    path = job_raw_dir(state.job_id) / manifest.winner
    if not path.is_file():
        return None

    try:
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_parquet(path)
    except Exception as e:
        logger.warning("existing_bundle_read_failed", path=str(path), error=str(e))
        return None

    state.dataset_info.files = capture_file_list(path.parent, used_path=path)
    logger.info("dataset_bundle_reused", job_id=state.job_id, file=manifest.winner)
    return df


async def load_from_kaggle(
    state: AnalysisState, url: str
) -> tuple[pd.DataFrame | None, str | None]:
    """Load dataset from Kaggle. Returns (dataframe, error_message).

    Authenticates via env vars, downloads + unzips the archive, picks the
    largest CSV (or first parquet), and snapshots the file list onto
    state.dataset_info.files. Restores original env credentials in finally.

    If the job already has a bundle on disk (a manifest plus its winner file,
    as written by a prior download for the same job_id), the existing file is
    read instead of re-pulling from Kaggle. This is what lets profiling reuse
    the bundle the manager downloaded at the data-review gate, with no second
    network round trip.
    """
    import json
    import os

    from src.config import get_settings

    reused = _read_existing_bundle(state)
    if reused is not None:
        return reused, None

    settings = get_settings()

    original_username = os.environ.get("KAGGLE_USERNAME")
    original_key = os.environ.get("KAGGLE_KEY")

    if settings.kaggle_key_value:
        kaggle_key = settings.kaggle_key_value
        kaggle_username = settings.kaggle_username

        if kaggle_key.startswith("{"):
            try:
                kaggle_creds = json.loads(kaggle_key)
                kaggle_username = kaggle_creds.get("username", kaggle_username)
                kaggle_key = kaggle_creds.get("key", kaggle_key)
            except json.JSONDecodeError:
                logger.debug("kaggle_key_json_parse_failed", exc_info=True)

        if not kaggle_username:
            error = "KAGGLE_USERNAME is not configured. Cannot download datasets."
            logger.error("kaggle_missing_username")
            return None, error

        os.environ["KAGGLE_USERNAME"] = kaggle_username
        os.environ["KAGGLE_KEY"] = kaggle_key

    dataset_id = None
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        parts = url.rstrip("/").split("/")
        if "datasets" in parts:
            idx = parts.index("datasets")
            owner = parts[idx + 1]
            dataset_name = parts[idx + 2]
            dataset_id = f"{owner}/{dataset_name}"
        else:
            logger.error("invalid_kaggle_url", url=url)
            error = f"Invalid Kaggle URL: {url}"
            state.push_sse_event("dataset_load_failed", {"error": error})
            return None, error

        state.push_sse_event(
            "dataset_download_started",
            {"url": url, "dataset_id": dataset_id},
        )

        from src.storage.job_data import (
            build_manifest,
            reset_job_raw_dir,
            write_manifest,
        )

        raw_dir = reset_job_raw_dir(state.job_id)
        api.dataset_download_files(dataset_id, path=str(raw_dir), unzip=True)

        used_path: Path | None = pick_default_file(raw_dir)
        df: pd.DataFrame | None = None

        if used_path is not None:
            if used_path.suffix == ".csv":
                df = pd.read_csv(used_path)
            else:
                df = pd.read_parquet(used_path)

        if df is None or used_path is None:
            logger.error("no_data_files_found", path=str(raw_dir))
            error = "No CSV or parquet files found in dataset archive."
            state.dataset_info.files = capture_file_list(raw_dir, used_path=None)
            state.push_sse_event("dataset_load_failed", {"error": error})
            return None, error

        state.dataset_info.files = capture_file_list(raw_dir, used_path=used_path)
        # The bundle now persists durably under the job's storage dir; the
        # manifest is the typed record (per-file columns, row counts, hashes)
        # the Data panel and downstream read.
        write_manifest(state.job_id, build_manifest(state))
        state.push_sse_event(
            "dataset_download_complete",
            {
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "files": [f.model_dump() for f in state.dataset_info.files],
            },
        )
        return df, None

    except Exception as e:
        error_str = str(e)
        dataset_ref = dataset_id or url

        if "403" in error_str or "Forbidden" in error_str:
            error = f"Dataset '{dataset_ref}' is not accessible. It may be private, deleted, or the URL is incorrect."
        elif "401" in error_str or "Unauthorized" in error_str:
            error = f"Dataset '{dataset_ref}' not found or authentication failed."
        elif "404" in error_str or "Not Found" in error_str:
            error = f"Dataset '{dataset_ref}' not found. Please verify the URL."
        else:
            error = f"Failed to load dataset from Kaggle: {error_str[:200]}"

        logger.error("kaggle_load_failed", error=error)
        state.push_sse_event("dataset_load_failed", {"error": error})
        return None, error

    finally:
        if original_username is not None:
            os.environ["KAGGLE_USERNAME"] = original_username
        elif "KAGGLE_USERNAME" in os.environ:
            del os.environ["KAGGLE_USERNAME"]

        if original_key is not None:
            os.environ["KAGGLE_KEY"] = original_key
        elif "KAGGLE_KEY" in os.environ:
            del os.environ["KAGGLE_KEY"]


def capture_file_list(
    tmpdir: Path, used_path: Path | None
) -> list[FileEntry]:
    """Snapshot files in the unzipped dataset directory before it is wiped.

    used_path identifies which file we actually loaded into the DataFrame;
    the rest are reported as siblings so the user can see what else shipped
    in the archive.
    """
    entries: list[FileEntry] = []
    for path in sorted(tmpdir.iterdir()):
        if not path.is_file():
            continue
        ext = path.suffix.lstrip(".").lower() or "other"
        if ext not in {"csv", "parquet", "json", "txt", "tsv", "xlsx", "xls"}:
            ext = "other"
        entries.append(
            FileEntry(
                name=path.name,
                size_bytes=path.stat().st_size,
                format=ext,
                used=(used_path is not None and path == used_path),
            )
        )
    return entries


async def fetch_kaggle_metadata(
    state: AnalysisState, infer_domain_label: bool = True
) -> None:
    """Fetch Kaggle metadata for semantic variable analysis.

    Populates state.raw_metadata and several state.dataset_info.kaggle_*
    fields. Emits SSE events for the metadata lifecycle. Failures are
    logged and reported via SSE but never raise.

    `infer_domain_label` gates the one derived field: when False (the
    data-review gate calls it this way) the domain is left unset, so the
    pre-approval surface carries only facts Kaggle supplied, no inference.
    """
    state.push_sse_event("dataset_metadata_started", {})
    try:
        from src.kaggle.metadata_extractor import KaggleMetadataExtractor

        extractor = KaggleMetadataExtractor()
        metadata = await extractor.extract(state.dataset_info.url)

        if metadata.get("extraction_success"):
            state.raw_metadata = metadata

            state.dataset_info.kaggle_description = metadata.get("description") or None
            state.dataset_info.kaggle_subtitle = metadata.get("subtitle") or None
            state.dataset_info.kaggle_column_descriptions = metadata.get(
                "column_descriptions", {}
            )
            state.dataset_info.kaggle_tags = metadata.get("tags", []) or []
            state.dataset_info.kaggle_keywords = metadata.get("keywords", []) or []
            state.dataset_info.metadata_quality = metadata.get(
                "metadata_quality", "unknown"
            )
            if infer_domain_label:
                state.dataset_info.kaggle_domain = infer_domain(metadata)

            logger.info(
                "kaggle_metadata_fetched",
                quality=metadata.get("metadata_quality"),
                has_description=bool(state.dataset_info.kaggle_description),
                has_subtitle=bool(state.dataset_info.kaggle_subtitle),
                has_column_descs=bool(metadata.get("column_descriptions")),
                n_keywords=len(state.dataset_info.kaggle_keywords),
                domain=state.dataset_info.kaggle_domain,
            )
            state.push_sse_event(
                "dataset_metadata_ready",
                {
                    "description": state.dataset_info.kaggle_description,
                    "subtitle": state.dataset_info.kaggle_subtitle,
                    "column_descriptions": state.dataset_info.kaggle_column_descriptions,
                    "tags": state.dataset_info.kaggle_tags,
                    "keywords": state.dataset_info.kaggle_keywords,
                    "domain": state.dataset_info.kaggle_domain,
                    "metadata_quality": state.dataset_info.metadata_quality,
                },
            )
        else:
            state.push_sse_event(
                "dataset_metadata_failed",
                {"error": metadata.get("error", "extraction_unsuccessful")},
            )
    except ImportError:
        logger.warning("kaggle_metadata_extractor_not_available")
        state.push_sse_event(
            "dataset_metadata_failed",
            {"error": "kaggle_metadata_extractor_not_available"},
        )
    except Exception as e:
        logger.warning("kaggle_metadata_fetch_failed", error=str(e))
        state.push_sse_event("dataset_metadata_failed", {"error": str(e)[:200]})


def infer_domain(metadata: dict) -> str | None:
    """Infer a coarse domain string from Kaggle tags."""
    tags = [t.lower() for t in metadata.get("tags", [])]

    if any(t in tags for t in ["healthcare", "medical", "clinical", "health"]):
        return "healthcare"
    if any(t in tags for t in ["economics", "economic", "finance", "income", "wages"]):
        return "economics"
    if any(t in tags for t in ["education", "academic", "school", "students"]):
        return "education"
    if any(t in tags for t in ["social", "sociology", "demographics", "census"]):
        return "social_science"
    if any(t in tags for t in ["marketing", "business", "sales"]):
        return "business"

    return None
