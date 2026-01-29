from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    database_url: str = "postgresql+psycopg://causal:causal@127.0.0.1:55478/causal"
    storage_dir: Path = Path(".runtime/files")
    cookie_secure: bool = False
    allowed_origins: str = "http://127.0.0.1:5178,http://localhost:5178,http://127.0.0.1:8008"
    session_hours: int = 24
    max_upload_bytes: int = 10 * 1024 * 1024
    max_rows: int = 100_000
    max_columns: int = 200
    context_source_bytes: int = 1024 * 1024
    context_max_sources: int = 10
    context_max_characters: int = 100_000
    vertex_project: str = ""
    vertex_location: str = "us-central1"
    gemini_model: str = "gemini-2.5-flash"
    langsmith_project: str = "causal-context"
    model_timeout_seconds: int = 180
    sandbox_queue: Path = Path(".runtime/sandbox")


@lru_cache
def settings() -> Settings:
    return Settings()
