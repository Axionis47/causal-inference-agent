"""Application configuration using Pydantic Settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    gcp_project_id: str = Field(default="plotpointe")
    gcp_region: str = "us-central1"

    # Firestore
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
    claude_api_key: SecretStr | None = Field(default=None)
    claude_model: str = "claude-sonnet-4-20250514"  # Latest Claude model
    claude_temperature: float = 0.1
    claude_max_tokens: int = 8192

    # Kaggle
    kaggle_username: str = Field(default="siddharth47007")
    kaggle_key: SecretStr | None = Field(default=None)

    # Agent Configuration
    max_agent_iterations: int = 3
    agent_timeout_seconds: int = 300

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

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
    def kaggle_key_value(self) -> str | None:
        """Get the actual Kaggle key value."""
        if self.kaggle_key is None:
            return None
        return self.kaggle_key.get_secret_value()


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
