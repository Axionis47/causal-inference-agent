"""Unified LLM client interface that supports multiple backends."""

from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from src.config import get_settings
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient(Protocol):
    """Protocol for LLM clients to enable backend switching."""

    async def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Generate content from the model."""
        ...

    async def generate_with_function_calling(
        self,
        prompt: str,
        system_instruction: str,
        tools: list[dict[str, Any]],
        max_iterations: int = 5,
    ) -> dict[str, Any]:
        """Generate content with function calling loop."""
        ...

    async def generate_structured(
        self,
        prompt: str,
        response_schema: type[T],
        system_instruction: str | None = None,
    ) -> T:
        """Generate structured output matching a Pydantic schema."""
        ...


# Cached client instance
_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Get the configured LLM client based on settings.

    This is the main entry point for getting an LLM client.
    It automatically selects between Gemini API and Vertex AI
    based on the llm_provider setting.

    Returns:
        An LLM client instance (either GeminiClient or VertexAIClient)
    """
    global _llm_client

    if _llm_client is not None:
        return _llm_client

    settings = get_settings()

    if settings.llm_provider == "claude":
        logger.info("using_claude_api_backend", model=settings.claude_model)
        from .claude_client import get_claude_client
        _llm_client = get_claude_client()
    elif settings.llm_provider == "vertex":
        logger.info("using_vertex_ai_backend", model=settings.vertex_model)
        from .vertex_client import get_vertex_client
        _llm_client = get_vertex_client()
    else:
        logger.info("using_gemini_api_backend", model=settings.gemini_model)
        from .gemini_client import get_gemini_client
        _llm_client = get_gemini_client()

    return _llm_client


def reset_llm_client() -> None:
    """Reset the cached LLM client. Useful for testing."""
    global _llm_client
    _llm_client = None
