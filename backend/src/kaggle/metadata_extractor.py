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
                    logger.debug("kaggle_creds_json_parse_skipped", exc_info=True)

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
        """Combine authored metadata with search-result snippet.

        Two Kaggle surfaces carry different fields:
          * `api.dataset_metadata(slug, path)` downloads the
            uploader-authored `dataset-metadata.json` (rich:
            description, subtitle, keywords, per-column schema).
          * `api.dataset_list(search=slug)` returns the search-result
            snippet (rich stats: tags, usability, downloads, votes;
            description here is usually empty).

        Earlier versions of this extractor only checked the search
        endpoint and only fell through to dataset_metadata when the
        search failed — so for datasets that DO surface in search but
        with an empty snippet description, the rich authored content
        was silently skipped. This redesign pulls both unconditionally
        and merges, preferring authored fields where they're populated.
        """
        authored = self._fetch_authored_metadata(api, owner, dataset_name)
        search = self._fetch_search_metadata(api, owner, dataset_name)

        def _prefer(*candidates):
            for c in candidates:
                if c:
                    return c
            return candidates[-1] if candidates else None

        return {
            "title": _prefer(authored.get("title"), search.get("title"), dataset_name),
            "subtitle": _prefer(authored.get("subtitle"), search.get("subtitle"), ""),
            "description": _prefer(
                authored.get("description"), search.get("description"), ""
            ),
            "license": _prefer(authored.get("license"), search.get("license"), ""),
            "source": search.get("source", ""),
            "tags": search.get("tags", []) or [],
            "keywords": authored.get("keywords", []) or [],
            "columns": authored.get("columns", {}) or {},
            "totalBytes": search.get("totalBytes", 0) or 0,
            "downloadCount": search.get("downloadCount", 0) or 0,
            "voteCount": search.get("voteCount", 0) or 0,
            "usabilityRating": search.get("usabilityRating", 0) or 0,
        }

    def _fetch_authored_metadata(
        self,
        api,
        owner: str,
        dataset_name: str,
    ) -> dict:
        """Pull the uploader-authored `dataset-metadata.json`.

        This is the prize: it's the JSON the uploader edits at upload
        time. When filled in, it carries the dataset's prose
        description, subtitle, keywords, and optionally per-column
        schema via the `resources` array.
        """
        import tempfile
        from pathlib import Path

        slug = f"{owner}/{dataset_name}"
        try:
            with tempfile.TemporaryDirectory() as td:
                api.dataset_metadata(slug, td)
                jp = Path(td) / "dataset-metadata.json"
                if not jp.exists():
                    return {}
                data = json.loads(jp.read_text())
        except Exception as e:
            logger.debug("authored_metadata_failed", error=str(e))
            return {}

        licenses = data.get("licenses") or []
        license_name = licenses[0].get("name", "") if licenses else ""

        return {
            "title": data.get("title") or "",
            "subtitle": data.get("subtitle") or "",
            "description": data.get("description") or "",
            "keywords": data.get("keywords") or [],
            "license": license_name,
            "columns": self._extract_columns_from_resources(
                data.get("resources") or []
            ),
        }

    def _fetch_search_metadata(
        self,
        api,
        owner: str,
        dataset_name: str,
    ) -> dict:
        """Pull supplemental fields from the search-result snippet.

        Description here is usually empty; the value is the stats
        (totalBytes, downloadCount, voteCount, usabilityRating) and
        the tags list which `dataset_metadata` does not surface.
        """
        slug = f"{owner}/{dataset_name}"
        try:
            matches = [
                d for d in api.dataset_list(search=slug) if d.ref == slug
            ]
        except Exception as e:
            logger.debug("search_metadata_failed", error=str(e))
            return {}

        if not matches:
            return {}
        d = matches[0]
        return {
            "title": getattr(d, "title", "") or "",
            "subtitle": getattr(d, "subtitle", "") or "",
            "description": getattr(d, "description", "") or "",
            "source": getattr(d, "url", "") or "",
            "license": getattr(d, "licenseName", "") or "",
            "tags": self._extract_tags(d),
            "totalBytes": getattr(d, "totalBytes", 0) or 0,
            "downloadCount": getattr(d, "downloadCount", 0) or 0,
            "voteCount": getattr(d, "voteCount", 0) or 0,
            "usabilityRating": getattr(d, "usabilityRating", 0) or 0,
        }

    def _extract_columns_from_resources(
        self,
        resources: list[dict],
    ) -> dict[str, str]:
        """Walk `dataset-metadata.json::resources[*].schema.fields[*]` for column descs.

        Most uploaders don't fill this in, but when they do it's the
        single best source of per-column meaning.
        """
        cols: dict[str, str] = {}
        for resource in resources:
            schema = resource.get("schema") or {}
            for field in schema.get("fields") or []:
                name = field.get("name")
                desc = field.get("description") or ""
                if name and desc:
                    cols[name] = desc
        return cols

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
            logger.debug("column_description_extract_skipped", exc_info=True)

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

        created_loop = False
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            created_loop = True

        try:
            return loop.run_until_complete(self.extract(dataset_url))
        finally:
            if created_loop:
                loop.close()
