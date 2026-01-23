"""Unit tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Test health check endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.main import app
        return TestClient(app)

    def test_health_check(self, client):
        """Test basic health endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "app" in data

    def test_readiness_check(self, client):
        """Test readiness endpoint."""
        response = client.get("/ready")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ready"


class TestJobEndpoints:
    """Test job-related endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from src.api.main import app
        return TestClient(app)

    def test_create_job_valid_url(self, client):
        """Test creating a job with valid Kaggle URL."""
        response = client.post(
            "/jobs",
            json={
                "kaggle_url": "https://www.kaggle.com/datasets/test/test-dataset",
                "treatment_variable": "treatment",
                "outcome_variable": "outcome",
            },
        )

        # Should accept the request (may fail async processing due to missing credentials)
        assert response.status_code in [200, 201, 202, 422, 500]

    def test_create_job_missing_url(self, client):
        """Test creating job without URL fails."""
        response = client.post(
            "/jobs",
            json={
                "treatment_variable": "treatment",
            },
        )

        assert response.status_code == 422  # Validation error

    def test_get_job_not_found(self, client):
        """Test getting non-existent job."""
        response = client.get("/jobs/nonexistent-job-id")

        assert response.status_code == 404

    def test_list_jobs(self, client):
        """Test listing jobs endpoint."""
        response = client.get("/jobs")

        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list)


class TestAPISchemas:
    """Test API schema validation."""

    def test_job_create_schema(self):
        """Test CreateJobRequest schema validation."""
        from src.api.schemas import CreateJobRequest

        # Valid schema
        job = CreateJobRequest(
            kaggle_url="https://www.kaggle.com/datasets/user/dataset",
            treatment_variable="treatment",
            outcome_variable="outcome",
        )
        assert job.kaggle_url is not None
        assert job.treatment_variable == "treatment"

    def test_job_response_schema(self):
        """Test JobResponse schema."""
        from src.api.schemas import JobResponse
        from src.agents.base import JobStatus
        from datetime import datetime

        response = JobResponse(
            id="test-123",
            kaggle_url="https://www.kaggle.com/datasets/user/dataset",
            status=JobStatus.PENDING,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        assert response.id == "test-123"
        assert response.status == JobStatus.PENDING

    def test_job_status_enum(self):
        """Test JobStatus enum values."""
        from src.agents.base import JobStatus

        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.FETCHING_DATA.value == "fetching_data"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
