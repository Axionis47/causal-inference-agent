"""Unit tests for KaggleMetadataExtractor."""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from src.kaggle.metadata_extractor import KaggleMetadataExtractor


class TestURLParsing:
    """Test URL parsing functionality."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        with patch.object(KaggleMetadataExtractor, '_setup_credentials'):
            return KaggleMetadataExtractor()

    def test_parses_full_url(self, extractor):
        """Test parsing full Kaggle URL."""
        result = extractor.parse_url("https://www.kaggle.com/datasets/user/dataset-name")
        assert result == ("user", "dataset-name")

    def test_parses_url_without_www(self, extractor):
        """Test parsing URL without www."""
        result = extractor.parse_url("https://kaggle.com/datasets/user/dataset-name")
        assert result == ("user", "dataset-name")

    def test_parses_url_without_protocol(self, extractor):
        """Test parsing URL without protocol."""
        result = extractor.parse_url("kaggle.com/datasets/user/dataset-name")
        assert result == ("user", "dataset-name")

    def test_parses_short_format(self, extractor):
        """Test parsing owner/dataset format."""
        result = extractor.parse_url("user/dataset-name")
        assert result == ("user", "dataset-name")

    def test_parses_url_with_trailing_slash(self, extractor):
        """Test parsing URL with trailing slash."""
        result = extractor.parse_url("https://kaggle.com/datasets/user/dataset-name/")
        assert result == ("user", "dataset-name")

    def test_returns_none_for_invalid_url(self, extractor):
        """Test that invalid URLs return None."""
        result = extractor.parse_url("https://example.com/invalid")
        assert result is None

    def test_returns_none_for_incomplete_url(self, extractor):
        """Test that incomplete URLs return None."""
        result = extractor.parse_url("https://kaggle.com/datasets/user")
        assert result is None


class TestMetadataQuality:
    """Test metadata quality assessment."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        with patch.object(KaggleMetadataExtractor, '_setup_credentials'):
            return KaggleMetadataExtractor()

    def test_high_quality_metadata(self, extractor):
        """Test high quality assessment."""
        metadata = {
            "description": "A" * 600,  # Long description
            "tags": ["tag1", "tag2", "tag3", "tag4"],  # Many tags
            "column_descriptions": {"col1": "desc"},  # Has column descriptions
            "source": "Original research",
        }
        assert extractor._assess_quality(metadata) == "high"

    def test_medium_quality_metadata(self, extractor):
        """Test medium quality assessment."""
        metadata = {
            "description": "A" * 200,  # Medium description
            "tags": ["tag1", "tag2"],  # Some tags
            "column_descriptions": {},
            "source": "",
        }
        assert extractor._assess_quality(metadata) == "medium"

    def test_low_quality_metadata(self, extractor):
        """Test low quality assessment."""
        metadata = {
            "description": "Short",  # Short description
            "tags": [],
            "column_descriptions": {},
            "source": "",
        }
        assert extractor._assess_quality(metadata) == "low"


class TestEmptyMetadata:
    """Test empty metadata generation."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        with patch.object(KaggleMetadataExtractor, '_setup_credentials'):
            return KaggleMetadataExtractor()

    def test_empty_metadata_structure(self, extractor):
        """Test that empty metadata has correct structure."""
        result = extractor._empty_metadata("test_url", "Test error")

        assert result["url"] == "test_url"
        assert result["extraction_success"] is False
        assert "Test error" in result["extraction_warnings"]
        assert result["metadata_quality"] == "low"
        assert result["description"] == ""
        assert result["tags"] == []


class TestFileTypeInference:
    """Test file type inference."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        with patch.object(KaggleMetadataExtractor, '_setup_credentials'):
            return KaggleMetadataExtractor()

    def test_csv_detection(self, extractor):
        """Test CSV file detection."""
        assert extractor._infer_file_type("data.csv") == "csv"
        assert extractor._infer_file_type("DATA.CSV") == "csv"

    def test_parquet_detection(self, extractor):
        """Test parquet file detection."""
        assert extractor._infer_file_type("data.parquet") == "parquet"

    def test_json_detection(self, extractor):
        """Test JSON file detection."""
        assert extractor._infer_file_type("data.json") == "json"

    def test_excel_detection(self, extractor):
        """Test Excel file detection."""
        assert extractor._infer_file_type("data.xlsx") == "excel"
        assert extractor._infer_file_type("data.xls") == "excel"

    def test_archive_detection(self, extractor):
        """Test archive file detection."""
        assert extractor._infer_file_type("data.zip") == "archive"

    def test_other_detection(self, extractor):
        """Test other file type detection."""
        assert extractor._infer_file_type("readme.txt") == "other"


class TestExtraction:
    """Test metadata extraction with mocked API."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance with mocked API."""
        with patch.object(KaggleMetadataExtractor, '_setup_credentials'):
            ext = KaggleMetadataExtractor()
            return ext

    @pytest.fixture
    def mock_dataset(self):
        """Create mock dataset object."""
        dataset = MagicMock()
        dataset.ref = "test-user/test-dataset"
        dataset.title = "Test Dataset"
        dataset.description = "A comprehensive dataset for testing causal inference methods. Contains treatment and outcome data."
        dataset.subtitle = "Test subtitle"
        dataset.url = "https://kaggle.com/datasets/test-user/test-dataset"
        dataset.licenseName = "CC0: Public Domain"
        dataset.totalBytes = 1024000
        dataset.downloadCount = 100
        dataset.voteCount = 50
        dataset.usabilityRating = 8.5

        # Mock tags
        tag1 = MagicMock()
        tag1.name = "healthcare"
        tag2 = MagicMock()
        tag2.name = "causal-inference"
        dataset.tags = [tag1, tag2]
        dataset.keywords = ["treatment", "outcome"]

        return dataset

    @pytest.fixture
    def mock_file(self):
        """Create mock file object."""
        file = MagicMock()
        file.name = "data.csv"
        file.totalBytes = 512000
        return file

    @pytest.mark.asyncio
    async def test_successful_extraction(self, extractor, mock_dataset, mock_file):
        """Test successful metadata extraction."""
        # Mock the API
        mock_api = MagicMock()
        mock_api.dataset_list.return_value = [mock_dataset]

        mock_files = MagicMock()
        mock_files.files = [mock_file]
        mock_api.dataset_list_files.return_value = mock_files

        with patch.object(extractor, '_get_api', return_value=mock_api):
            result = await extractor.extract("https://kaggle.com/datasets/test-user/test-dataset")

        assert result["extraction_success"] is True
        assert result["title"] == "Test Dataset"
        assert "comprehensive dataset" in result["description"]
        assert "healthcare" in result["tags"]
        assert result["dataset_id"] == "test-user/test-dataset"
        assert len(result["files"]) == 1
        assert result["files"][0]["name"] == "data.csv"

    @pytest.mark.asyncio
    async def test_extraction_with_invalid_url(self, extractor):
        """Test extraction with invalid URL."""
        result = await extractor.extract("invalid-url")

        assert result["extraction_success"] is False
        assert "Failed to parse URL" in result["extraction_warnings"][0]

    @pytest.mark.asyncio
    async def test_extraction_handles_api_init_error(self, extractor):
        """Test extraction handles API initialization errors gracefully."""
        # When _get_api itself fails, it should return empty metadata
        with patch.object(extractor, '_get_api', side_effect=Exception("API Init Error")):
            result = await extractor.extract("https://kaggle.com/datasets/test-user/test-dataset")

        assert result["extraction_success"] is False
        assert "API Init Error" in result["extraction_warnings"][0]

    @pytest.mark.asyncio
    async def test_extraction_handles_partial_api_error(self, extractor):
        """Test extraction handles partial API errors gracefully."""
        # When dataset_list fails but API works, it should still return partial metadata
        mock_api = MagicMock()
        mock_api.dataset_list.side_effect = Exception("API Error")
        mock_api.dataset_list_files.return_value = MagicMock(files=[])

        with patch.object(extractor, '_get_api', return_value=mock_api):
            result = await extractor.extract("https://kaggle.com/datasets/test-user/test-dataset")

        # Should still succeed with minimal metadata
        assert result["extraction_success"] is True
        assert result["metadata_quality"] == "low"

    @pytest.mark.asyncio
    async def test_extraction_with_no_tags(self, extractor, mock_dataset, mock_file):
        """Test extraction when dataset has no tags."""
        mock_dataset.tags = None

        mock_api = MagicMock()
        mock_api.dataset_list.return_value = [mock_dataset]
        mock_files = MagicMock()
        mock_files.files = [mock_file]
        mock_api.dataset_list_files.return_value = mock_files

        with patch.object(extractor, '_get_api', return_value=mock_api):
            result = await extractor.extract("https://kaggle.com/datasets/test-user/test-dataset")

        assert result["extraction_success"] is True
        assert result["tags"] == []


class TestTagExtraction:
    """Test tag extraction from dataset objects."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        with patch.object(KaggleMetadataExtractor, '_setup_credentials'):
            return KaggleMetadataExtractor()

    def test_extract_tags_with_objects(self, extractor):
        """Test extracting tags from tag objects."""
        dataset = MagicMock()
        tag1 = MagicMock()
        tag1.name = "economics"
        tag2 = MagicMock()
        tag2.name = "healthcare"
        dataset.tags = [tag1, tag2]

        tags = extractor._extract_tags(dataset)
        assert tags == ["economics", "healthcare"]

    def test_extract_tags_with_strings(self, extractor):
        """Test extracting tags from string list."""
        dataset = MagicMock()
        dataset.tags = ["economics", "healthcare"]

        tags = extractor._extract_tags(dataset)
        assert tags == ["economics", "healthcare"]

    def test_extract_tags_none(self, extractor):
        """Test extracting tags when tags is None."""
        dataset = MagicMock()
        dataset.tags = None

        tags = extractor._extract_tags(dataset)
        assert tags == []

    def test_extract_tags_empty(self, extractor):
        """Test extracting tags when tags is empty."""
        dataset = MagicMock()
        dataset.tags = []

        tags = extractor._extract_tags(dataset)
        assert tags == []
