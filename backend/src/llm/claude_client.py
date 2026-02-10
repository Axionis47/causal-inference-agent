"""Anthropic Claude client with function calling support."""

import json
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_settings
from src.logging_config.structured import get_logger

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)

# Simple circuit breaker state
_consecutive_failures = 0
_circuit_open_until = 0.0
MAX_CONSECUTIVE_FAILURES = 5
CIRCUIT_OPEN_DURATION = 60.0  # seconds


def _check_circuit_breaker() -> None:
    """Check if circuit breaker is open and raise if so."""
    global _circuit_open_until
    if _circuit_open_until > 0.0 and time.time() < _circuit_open_until:
        raise RuntimeError(
            f"Claude API circuit breaker is open. "
            f"Resumes at {time.strftime('%H:%M:%S', time.localtime(_circuit_open_until))}"
        )


def _record_success() -> None:
    """Record a successful call, resetting the circuit breaker."""
    global _consecutive_failures, _circuit_open_until
    _consecutive_failures = 0
    _circuit_open_until = 0.0


def _record_failure() -> None:
    """Record a failed call, potentially opening the circuit breaker."""
    global _consecutive_failures, _circuit_open_until
    _consecutive_failures += 1
    if _consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
        _circuit_open_until = time.time() + CIRCUIT_OPEN_DURATION
        logger.warning(
            "circuit_breaker_opened",
            failures=_consecutive_failures,
            open_until=_circuit_open_until,
        )


class ClaudeClient:
    """Client for interacting with Anthropic Claude models."""

    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self) -> None:
        """Initialize the Claude client."""
        settings = get_settings()
        self.api_key = settings.claude_api_key.get_secret_value() if settings.claude_api_key else None
        self.model_name = settings.claude_model
        self.temperature = settings.claude_temperature
        self.max_tokens = settings.claude_max_tokens

        if not self.api_key:
            raise ValueError("Claude API key is required. Set CLAUDE_API_KEY in environment.")

        self._client = httpx.AsyncClient(timeout=120.0)

    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, ConnectionError, TimeoutError, OSError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        """Generate content from the model.

        Args:
            prompt: The user prompt
            system_instruction: Optional system instruction
            tools: Optional list of function definitions for function calling

        Returns:
            The model response
        """
        _check_circuit_breaker()

        logger.debug(
            "generating_content_claude",
            prompt_length=len(prompt),
            has_system=system_instruction is not None,
            has_tools=tools is not None,
        )

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        payload: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        if system_instruction:
            payload["system"] = system_instruction

        if tools:
            payload["tools"] = [self._convert_tool(t) for t in tools]

        try:
            response = await self._client.post(self.API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            _record_success()
        except Exception:
            _record_failure()
            raise

        logger.debug(
            "content_generated_claude",
            stop_reason=result.get("stop_reason"),
        )

        return result

    @retry(
        retry=retry_if_exception_type((httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout, ConnectionError, TimeoutError, OSError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate_with_function_calling(
        self,
        prompt: str,
        system_instruction: str,
        tools: list[dict[str, Any]],
        max_iterations: int = 5,
    ) -> dict[str, Any]:
        """Generate content with automatic function calling loop.

        Args:
            prompt: The user prompt
            system_instruction: System instruction for the model
            tools: List of available tools
            max_iterations: Maximum function calling iterations

        Returns:
            Dict containing the final response and all tool calls made
        """
        _check_circuit_breaker()

        logger.info(
            "starting_function_calling_claude",
            tools=[t["name"] for t in tools],
            max_iterations=max_iterations,
        )

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }

        messages = [{"role": "user", "content": prompt}]
        tool_calls_made: list[dict[str, Any]] = []

        claude_tools = [self._convert_tool(t) for t in tools]

        for iteration in range(max_iterations):
            logger.debug(f"function_calling_iteration_claude_{iteration + 1}")

            payload: dict[str, Any] = {
                "model": self.model_name,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "system": system_instruction,
                "messages": messages,
                "tools": claude_tools,
            }

            response = await self._client.post(self.API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()

            # Extract content blocks
            content_blocks = result.get("content", [])
            stop_reason = result.get("stop_reason")

            # Check for tool use
            tool_uses = [block for block in content_blocks if block.get("type") == "tool_use"]
            text_blocks = [block for block in content_blocks if block.get("type") == "text"]

            if not tool_uses:
                # No more function calls, return final response
                final_text = ""
                for block in text_blocks:
                    final_text += block.get("text", "")

                return {
                    "response": final_text,
                    "tool_calls": tool_calls_made,
                    "iterations": iteration + 1,
                }

            # Process tool calls
            function_calls = []
            for tool_use in tool_uses:
                fc = {
                    "name": tool_use.get("name"),
                    "args": tool_use.get("input", {}),
                    "id": tool_use.get("id"),
                }
                function_calls.append(fc)
                tool_calls_made.append({
                    "name": fc["name"],
                    "args": fc["args"],
                    "iteration": iteration + 1,
                })

            # Return with pending function calls for the agent to execute
            return {
                "response": text_blocks[0].get("text") if text_blocks else None,
                "tool_calls": tool_calls_made,
                "pending_calls": function_calls,
                "iterations": iteration + 1,
            }

        logger.warning("max_iterations_reached_claude", max_iterations=max_iterations)
        return {
            "response": None,
            "tool_calls": tool_calls_made,
            "iterations": max_iterations,
            "error": "Max iterations reached",
        }

    async def generate_structured(
        self,
        prompt: str,
        response_schema: type[T],
        system_instruction: str | None = None,
    ) -> T:
        """Generate structured output matching a Pydantic schema.

        Args:
            prompt: The user prompt
            response_schema: Pydantic model class for the response
            system_instruction: Optional system instruction

        Returns:
            Parsed response matching the schema
        """
        # Add JSON formatting instruction
        schema_json = json.dumps(response_schema.model_json_schema(), indent=2)
        structured_prompt = f"""{prompt}

Respond with valid JSON matching this schema:
{schema_json}

Output only the JSON, no other text."""

        result = await self.generate(
            prompt=structured_prompt,
            system_instruction=system_instruction,
        )

        # Extract text from response
        content = result.get("content", [])
        response_text = ""
        for block in content:
            if block.get("type") == "text":
                response_text += block.get("text", "")

        response_text = response_text.strip()

        # Handle markdown code blocks
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        parsed = json.loads(response_text.strip())
        return response_schema.model_validate(parsed)

    def _convert_tool(self, tool_def: dict[str, Any]) -> dict[str, Any]:
        """Convert a tool definition to Claude's tool format.

        Args:
            tool_def: Tool definition in our internal format

        Returns:
            Tool definition in Claude's format
        """
        return {
            "name": tool_def["name"],
            "description": tool_def.get("description", ""),
            "input_schema": tool_def.get("parameters", {"type": "object", "properties": {}}),
        }


# Singleton instance
_client: ClaudeClient | None = None


def get_claude_client() -> ClaudeClient:
    """Get the singleton Claude client instance."""
    global _client
    if _client is None:
        _client = ClaudeClient()
    return _client


def reset_claude_client() -> None:
    """Reset the cached Claude client. Useful for testing."""
    global _client
    _client = None
