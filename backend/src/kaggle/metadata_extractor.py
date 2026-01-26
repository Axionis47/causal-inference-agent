"""
Kaggle Metadata Extractor.

Extracts rich metadata from Kaggle datasets to provide domain knowledge
for causal inference analysis. Does NOT download the actual data.
"""

from __future__ import annotations

import json
import os
from typing import Any

from src.config.settings import get_settings
from src.logging_config.structured import get_logger

logger = get_logger(__name__)
settings = get_settings()


class KaggleMetadataExtractor:
    """
    Extracts metadata from Kaggle datasets.

    This class pulls all available metadata from Kaggle including:
    - Dataset description (often contains study context)
    - Column descriptions (if provided by uploader)
    - Tags (indicate domain - healthcare, economics, etc.)
    - Source information
    - License and usage info

    The metadata is used by the DomainKnowledgeAgent to extract
    causal constraints before statistical analysis.
    """

    def __init__(self):
        """Initialize the metadata extractor."""
        self._api = None
        self._setup_credentials()

    def _setup_credentials(self) -> None:
        """Set up Kaggle API credentials from settings."""
        if settings.kaggle_key_value:
            kaggle_key = settings.kaggle_key_value
            kaggle_username = settings.kaggle_username

            # Handle JSON format: {"username": "...", "key": "..."}
            if kaggle_key.startswith("{"):
                try:
                    kaggle_creds = json.loads(kaggle_key)
                    kaggle_username = kaggle_creds.get("username", kaggle_username)
                    kaggle_key = kaggle_creds.get("key", kaggle_key)
                except json.JSONDecodeError:
                    pass

            os.environ["KAGGLE_USERNAME"] = kaggle_username
            os.environ["KAGGLE_KEY"] = kaggle_key

    def _get_api(self):
        """Lazy initialization of Kaggle API."""
        if self._api is None:
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
                self._api = KaggleApi()
                self._api.authenticate()
            except Exception as e:
                logger.error("kaggle_api_init_failed", error=str(e))
                raise
        return self._api

    def parse_url(self, url: str) -> tuple[str, str] | None:
        """
        Parse a Kaggle URL to extract owner and dataset name.

        Args:
            url: Kaggle dataset URL

        Returns:
            Tuple of (owner, dataset_name) or None if parsing fails
        """
        # Handle various URL formats
        # https://www.kaggle.com/datasets/owner/dataset-name
        # https://kaggle.com/datasets/owner/dataset-name
        # kaggle.com/datasets/owner/dataset-name
        # owner/dataset-name

        url = url.strip().rstrip("/")

        # Check if it's already in owner/dataset format
        if "/" in url and "kaggle" not in url.lower():
            parts = url.split("/")
            if len(parts) == 2:
                return parts[0], parts[1]

        # Parse URL
        parts = url.split("/")
        if "datasets" in parts:
            idx = parts.index("datasets")
            if idx + 2 < len(parts):
                return parts[idx + 1], parts[idx + 2]

        logger.warning("invalid_kaggle_url", url=url)
        return None

    async def extract(self, dataset_url: str) -> dict[str, Any]:
        """
        Extract all available metadata from a Kaggle dataset.

        Args:
            dataset_url: Kaggle dataset URL

        Returns:
            Dictionary containing extracted metadata
        """
        parsed = self.parse_url(dataset_url)
        if not parsed:
            return self._empty_metadata(dataset_url, "Failed to parse URL")

        owner, dataset_name = parsed
        dataset_id = f"{owner}/{dataset_name}"

        logger.info("extracting_metadata", dataset_id=dataset_id)

        try:
            api = self._get_api()

            # Get dataset metadata
            metadata = self._fetch_dataset_metadata(api, owner, dataset_name)

            # Get file list to understand structure
            files_info = self._fetch_files_info(api, dataset_id)

            # Combine all metadata
            result = {
                "url": dataset_url,
                "dataset_id": dataset_id,
                "owner": owner,
                "dataset_name": dataset_name,

                # Core metadata
                "title": metadata.get("title", dataset_name),
                "description": metadata.get("description", ""),
                "subtitle": metadata.get("subtitle", ""),

                # Tags and categories
                "tags": metadata.get("tags", []),
                "keywords": metadata.get("keywords", []),

                # Source and license
                "source": metadata.get("source", ""),
                "license": metadata.get("license", ""),

                # Dataset stats
                "total_size": metadata.get("totalBytes", 0),
                "download_count": metadata.get("downloadCount", 0),
                "vote_count": metadata.get("voteCount", 0),
                "usability_rating": metadata.get("usabilityRating", 0),

                # Files info
                "files": files_info,

                # Column descriptions (if available)
                "column_descriptions": metadata.get("columns", {}),

                # Extraction metadata
                "extraction_success": True,
                "extraction_warnings": [],
            }

            # Assess metadata quality
            result["metadata_quality"] = self._assess_quality(result)

            logger.info(
                "metadata_extracted",
                dataset_id=dataset_id,
                quality=result["metadata_quality"],
                has_description=bool(result["description"]),
                num_tags=len(result["tags"]),
            )

            return result

        except Exception as e:
            logger.error("metadata_extraction_failed", error=str(e), dataset_id=dataset_id)
            return self._empty_metadata(dataset_url, str(e))

    def _fetch_dataset_metadata(self, api, owner: str, dataset_name: str) -> dict:
        """Fetch dataset metadata from Kaggle API."""
        try:
            # Get dataset info
            dataset_list = api.dataset_list(search=f"{owner}/{dataset_name}")

            for dataset in dataset_list:
                if dataset.ref == f"{owner}/{dataset_name}":
                    return {
                        "title": getattr(dataset, "title", dataset_name),
                        "description": getattr(dataset, "description", ""),
                        "subtitle": getattr(dataset, "subtitle", ""),
                        "source": getattr(dataset, "url", ""),
                        "license": getattr(dataset, "licenseName", ""),
                        "totalBytes": getattr(dataset, "totalBytes", 0),
                        "downloadCount": getattr(dataset, "downloadCount", 0),
                        "voteCount": getattr(dataset, "voteCount", 0),
                        "usabilityRating": getattr(dataset, "usabilityRating", 0),
                        "tags": self._extract_tags(dataset),
                        "keywords": getattr(dataset, "keywords", []) or [],
                    }

            # Dataset not found in search, try metadata endpoint
            try:
                metadata = api.dataset_metadata(owner, dataset_name)
                if metadata and hasattr(metadata, "info"):
                    info = metadata.info
                    return {
                        "title": getattr(info, "title", dataset_name),
                        "description": getattr(info, "description", ""),
                        "subtitle": getattr(info, "subtitle", ""),
                        "license": getattr(info, "licenseName", ""),
                        "tags": [],
                        "keywords": getattr(info, "keywords", []) or [],
                        "columns": self._extract_column_descriptions(metadata),
                    }
            except Exception:
                pass

            return {"title": dataset_name}

        except Exception as e:
            logger.warning("metadata_fetch_partial", error=str(e))
            return {"title": dataset_name}

    def _extract_tags(self, dataset) -> list[str]:
        """Extract tags from dataset object."""
        tags = []

        if hasattr(dataset, "tags"):
            dataset_tags = dataset.tags
            if dataset_tags:
                for tag in dataset_tags:
                    if hasattr(tag, "name"):
                        tags.append(tag.name)
                    elif isinstance(tag, str):
                        tags.append(tag)

        return tags

    def _extract_column_descriptions(self, metadata) -> dict[str, str]:
        """Extract column descriptions if available."""
        columns = {}

        try:
            if hasattr(metadata, "columns"):
                for col in metadata.columns:
                    if hasattr(col, "name") and hasattr(col, "description"):
                        if col.description:
                            columns[col.name] = col.description
        except Exception:
            pass

        return columns

    def _fetch_files_info(self, api, dataset_id: str) -> list[dict]:
        """Fetch information about files in the dataset."""
        try:
            files = api.dataset_list_files(dataset_id)

            files_info = []
            for f in files.files:
                files_info.append({
                    "name": f.name,
                    "size": getattr(f, "totalBytes", 0),
                    "type": self._infer_file_type(f.name),
                })

            return files_info

        except Exception as e:
            logger.warning("files_fetch_failed", error=str(e))
            return []

    def _infer_file_type(self, filename: str) -> str:
        """Infer file type from filename."""
        filename = filename.lower()
        if filename.endswith(".csv"):
            return "csv"
        elif filename.endswith(".parquet"):
            return "parquet"
        elif filename.endswith(".json"):
            return "json"
        elif filename.endswith((".xlsx", ".xls")):
            return "excel"
        elif filename.endswith(".zip"):
            return "archive"
        else:
            return "other"

    def _assess_quality(self, metadata: dict) -> str:
        """
        Assess the quality of extracted metadata.

        Returns:
            'high', 'medium', or 'low'
        """
        score = 0

        # Description is most important
        desc = metadata.get("description", "")
        if len(desc) > 500:
            score += 3
        elif len(desc) > 100:
            score += 2
        elif len(desc) > 20:
            score += 1

        # Tags help with domain identification
        if len(metadata.get("tags", [])) > 3:
            score += 2
        elif len(metadata.get("tags", [])) > 0:
            score += 1

        # Column descriptions are valuable
        if len(metadata.get("column_descriptions", {})) > 0:
            score += 2

        # Source information
        if metadata.get("source"):
            score += 1

        if score >= 6:
            return "high"
        elif score >= 3:
            return "medium"
        else:
            return "low"

    def _empty_metadata(self, url: str, error: str) -> dict[str, Any]:
        """Return empty metadata structure with error info."""
        return {
            "url": url,
            "dataset_id": None,
            "owner": None,
            "dataset_name": None,
            "title": "",
            "description": "",
            "subtitle": "",
            "tags": [],
            "keywords": [],
            "source": "",
            "license": "",
            "total_size": 0,
            "download_count": 0,
            "vote_count": 0,
            "usability_rating": 0,
            "files": [],
            "column_descriptions": {},
            "extraction_success": False,
            "extraction_warnings": [error],
            "metadata_quality": "low",
        }

    def extract_sync(self, dataset_url: str) -> dict[str, Any]:
        """
        Synchronous version of extract for non-async contexts.

        Args:
            dataset_url: Kaggle dataset URL

        Returns:
            Dictionary containing extracted metadata
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(self.extract(dataset_url))
