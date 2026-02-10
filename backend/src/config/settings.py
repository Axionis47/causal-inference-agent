"""Application configuration using Pydantic Settings."""

import logging
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Causal Inference Orchestrator"
    app_version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False
    log_level: str = "INFO"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # GCP Configuration
    gcp_project_id: str = Field(default="")
    gcp_region: str = "us-central1"

    # Storage
    use_firestore: bool = False  # Default False for local dev; set True in .env for production
    local_storage_path: str = "./data"  # Path for local JSON storage when use_firestore=False
    firestore_database: str = "(default)"

    # Cloud Storage
    gcs_bucket_datasets: str = Field(default="causal-datasets")
    gcs_bucket_notebooks: str = Field(default="causal-notebooks")

    # LLM Configuration
    llm_provider: Literal["gemini", "vertex", "claude"] = "claude"  # Which LLM provider to use

    # Gemini API (direct API key access)
    gemini_api_key: SecretStr | None = Field(default=None)
    gemini_model: str = "gemini-1.5-pro"
    gemini_temperature: float = 0.1
    gemini_max_tokens: int = 8192

    # Vertex AI (GCP-managed, uses service account or ADC)
    vertex_model: str = "gemini-2.0-flash-exp"  # Experimental model (has rate limits)

    # Claude API (Anthropic)
    claude_api_key: SecretStr | None = Field(
        default=None,
        # Accept both CLAUDE_API_KEY and ANTHROPIC_API_KEY from environment
        validation_alias="claude_api_key",
    )
    anthropic_api_key: SecretStr | None = Field(default=None)  # Alias for claude_api_key
    claude_model: str = "claude-sonnet-4-20250514"  # Latest Claude model
    claude_temperature: float = 0.1
    claude_max_tokens: int = 8192

    # Kaggle — no defaults, must be set via environment
    kaggle_username: str = Field(default="")
    kaggle_key: SecretStr | None = Field(default=None)

    # API Authentication
    api_key: SecretStr | None = Field(default=None)  # Set to enable API key auth

    # Agent Configuration
    max_agent_iterations: int = 3
    agent_timeout_seconds: int = 300

    # SSE (Server-Sent Events)
    sse_enabled: bool = True
    sse_heartbeat_seconds: int = 15

    # Redis (opt-in for future job queue scaling)
    redis_url: str = "redis://localhost:6379"
    redis_enabled: bool = False

    # CORS - Set via CORS_ORIGINS env var in production with actual Cloud Run URLs
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:80",
        "http://localhost",
    ]

    # API Documentation
    enable_api_docs: bool = True  # Enable Swagger/ReDoc in all environments

    @model_validator(mode="after")
    def _resolve_api_key_aliases(self) -> "Settings":
        """Resolve ANTHROPIC_API_KEY as alias for claude_api_key."""
        if self.claude_api_key is None and self.anthropic_api_key is not None:
            self.claude_api_key = self.anthropic_api_key
        return self

    @model_validator(mode="after")
    def _validate_required_config(self) -> "Settings":
        """Warn about missing configuration in non-test environments."""
        if self.environment == "production":
            if self.use_firestore and not self.gcp_project_id:
                raise ValueError(
                    "GCP_PROJECT_ID is required when USE_FIRESTORE=true in production"
                )
            if self.api_key is None:
                logger.warning(
                    "API_KEY not set in production — API endpoints are unauthenticated!"
                )
        if self.llm_provider == "claude" and self.claude_api_key is None:
            logger.warning(
                "LLM_PROVIDER=claude but no CLAUDE_API_KEY or ANTHROPIC_API_KEY set"
            )
        if self.llm_provider == "gemini" and self.gemini_api_key is None:
            logger.warning(
                "LLM_PROVIDER=gemini but no GEMINI_API_KEY set"
            )
        return self

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment == "production"

    @property
    def gemini_api_key_value(self) -> str | None:
        """Get the actual Gemini API key value."""
        if self.gemini_api_key is None:
            return None
        return self.gemini_api_key.get_secret_value()

    @property
    def claude_api_key_value(self) -> str | None:
        """Get the actual Claude API key value."""
        if self.claude_api_key is None:
            return None
        return self.claude_api_key.get_secret_value()

    @property
    def kaggle_key_value(self) -> str | None:
        """Get the actual Kaggle key value."""
        if self.kaggle_key is None:
            return None
        return self.kaggle_key.get_secret_value()

    @property
    def api_key_value(self) -> str | None:
        """Get the actual API key value for authentication."""
        if self.api_key is None:
            return None
        return self.api_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
