"""Anthropic Claude client with function calling support."""

import asyncio
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


class ClaudeClient:
    """Client for interacting with Anthropic Claude models."""

    API_URL = "https://api.anthropic.com/v1/messages"
    MAX_CONSECUTIVE_FAILURES = 5
    CIRCUIT_OPEN_DURATION = 60.0  # seconds

    def __init__(self) -> None:
        """Initialize the Claude client."""
        settings = get_settings()
        self._settings = settings  # Keep reference for observability settings
        self.api_key = settings.claude_api_key.get_secret_value() if settings.claude_api_key else None
        self.model_name = settings.claude_model
        self.temperature = settings.claude_temperature
        self.max_tokens = settings.claude_max_tokens

        if not self.api_key:
            raise ValueError("Claude API key is required. Set CLAUDE_API_KEY in environment.")

        self._client = httpx.AsyncClient(timeout=120.0)

        # Circuit breaker state (per-instance, not global)
        self._consecutive_failures = 0
        self._circuit_open_until = 0.0
        self._cb_lock = asyncio.Lock()  # Protects circuit breaker counters

    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker is open and raise if so."""
        if self._circuit_open_until > 0.0 and time.time() < self._circuit_open_until:
            raise RuntimeError(
                f"Claude API circuit breaker is open. "
                f"Resumes at {time.strftime('%H:%M:%S', time.localtime(self._circuit_open_until))}"
            )

    async def _record_success(self) -> None:
        """Record a successful call, resetting the circuit breaker."""
        async with self._cb_lock:
            self._consecutive_failures = 0
            self._circuit_open_until = 0.0

    async def _record_failure(self) -> None:
        """Record a failed call, potentially opening the circuit breaker."""
        async with self._cb_lock:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self.MAX_CONSECUTIVE_FAILURES:
                self._circuit_open_until = time.time() + self.CIRCUIT_OPEN_DURATION
                logger.warning(
                    "circuit_breaker_opened",
                    failures=self._consecutive_failures,
                    open_until=self._circuit_open_until,
                )

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
        self._check_circuit_breaker()

        logger.info(
            "llm_call_start",
            method="generate",
            prompt_length=len(prompt),
            prompt_preview=prompt[:200] + "..." if len(prompt) > 200 else prompt,
            has_system=system_instruction is not None,
            has_tools=tools is not None,
            n_tools=len(tools) if tools else 0,
        )

        if self._settings.log_llm_prompts:
            logger.debug("llm_full_prompt", prompt=prompt)

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
            await self._record_success()
        except Exception:
            await self._record_failure()
            raise

        # Extract token usage from Anthropic API response
        usage = result.get("usage", {})
        result["_token_usage"] = {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }

        # Extract response text for logging
        content_blocks = result.get("content", [])
        response_text = "".join(
            b.get("text", "") for b in content_blocks if b.get("type") == "text"
        )

        logger.info(
            "llm_call_complete",
            method="generate",
            stop_reason=result.get("stop_reason"),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            response_preview=response_text[:200] + "..." if len(response_text) > 200 else response_text,
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
        self._check_circuit_breaker()

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
        total_input_tokens = 0
        total_output_tokens = 0

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

            try:
                response = await self._client.post(self.API_URL, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
                await self._record_success()
            except Exception:
                await self._record_failure()
                raise

            # Accumulate token usage across iterations
            iter_usage = result.get("usage", {})
            total_input_tokens += iter_usage.get("input_tokens", 0)
            total_output_tokens += iter_usage.get("output_tokens", 0)

            # Extract content blocks
            content_blocks = result.get("content", [])
            stop_reason = result.get("stop_reason")

            tool_uses = [block for block in content_blocks if block.get("type") == "tool_use"]
            text_blocks = [block for block in content_blocks if block.get("type") == "text"]

            logger.info(
                "llm_fc_iteration",
                iteration=iteration + 1,
                max_iterations=max_iterations,
                input_tokens=iter_usage.get("input_tokens", 0),
                output_tokens=iter_usage.get("output_tokens", 0),
                n_tool_uses=len(tool_uses),
                tools_called=[t.get("name") for t in tool_uses],
                stop_reason=stop_reason,
            )

            if not tool_uses:
                # No more function calls, return final response
                final_text = ""
                for block in text_blocks:
                    final_text += block.get("text", "")

                return {
                    "response": final_text,
                    "tool_calls": tool_calls_made,
                    "iterations": iteration + 1,
                    "_token_usage": {
                        "input_tokens": total_input_tokens,
                        "output_tokens": total_output_tokens,
                    },
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
                "_token_usage": {
                    "input_tokens": total_input_tokens,
                    "output_tokens": total_output_tokens,
                },
            }

        logger.warning("max_iterations_reached_claude", max_iterations=max_iterations)
        return {
            "response": None,
            "tool_calls": tool_calls_made,
            "iterations": max_iterations,
            "error": "Max iterations reached",
            "_token_usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
            },
        }

    async def generate_structured(
        self,
        prompt: str,
        response_schema: type[T],
        system_instruction: str | None = None,
    ) -> T:
        """Generate structured output matching a Pydantic schema."""
        model, _ = await self.generate_structured_with_usage(
            prompt, response_schema, system_instruction
        )
        return model

    async def generate_structured_with_usage(
        self,
        prompt: str,
        response_schema: type[T],
        system_instruction: str | None = None,
    ) -> tuple[T, dict[str, int]]:
        """Structured output plus the call's token usage."""
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

        from .parsing import strip_fences

        parsed = json.loads(strip_fences(response_text))
        usage = result.get("_token_usage", {"input_tokens": 0, "output_tokens": 0})
        return response_schema.model_validate(parsed), usage

    async def generate_text_with_usage(
        self, prompt: str, system_instruction: str | None = None
    ) -> tuple[str, dict[str, int]]:
        """Plain text plus the call's token usage."""
        result = await self.generate(prompt=prompt, system_instruction=system_instruction)
        content = result.get("content", [])
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        usage = result.get("_token_usage", {"input_tokens": 0, "output_tokens": 0})
        return text, usage

    def user_message(self, text: str) -> dict[str, Any]:
        return {"role": "user", "content": text}

    def tool_result_message(self, call: dict[str, Any], output: str) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": call.get("id"),
                    "content": output[:8000],
                }
            ],
        }

    async def chat_with_tools(
        self,
        messages: list[Any],
        system_instruction: str | None,
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """One model turn over a running tool conversation.

        Returns {"text", "tool_calls": [{"id","name","args"}],
        "assistant_message", "usage"}; the caller executes the tools and
        appends assistant_message plus tool_result_message()s.
        """
        self._check_circuit_breaker()
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        }
        payload: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages,
            "tools": [self._convert_tool(t) for t in tools],
        }
        if system_instruction:
            payload["system"] = system_instruction
        try:
            response = await self._client.post(self.API_URL, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            await self._record_success()
        except Exception:
            await self._record_failure()
            raise

        usage_raw = result.get("usage", {})
        content = result.get("content", [])
        text = "".join(b.get("text", "") for b in content if b.get("type") == "text")
        calls = [
            {"id": b.get("id"), "name": b.get("name"), "args": b.get("input", {})}
            for b in content
            if b.get("type") == "tool_use"
        ]
        return {
            "text": text or None,
            "tool_calls": calls,
            "assistant_message": {"role": "assistant", "content": content},
            "usage": {
                "input_tokens": usage_raw.get("input_tokens", 0),
                "output_tokens": usage_raw.get("output_tokens", 0),
            },
        }

    async def close(self) -> None:
        """Close the underlying httpx client to release connections."""
        await self._client.aclose()

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


async def close_claude_client() -> None:
    """Close the Claude client's httpx connection pool (call on shutdown)."""
    global _client
    if _client is not None:
        await _client.close()


def reset_claude_client() -> None:
    """Reset the cached Claude client. Useful for testing."""
    global _client
    _client = None
