"""Google Gemini API client with function calling support."""

import json
from typing import Any, TypeVar

import google.generativeai as genai
from google.generativeai.types import (
    FunctionDeclaration,
    GenerateContentResponse,
    Tool,
)
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


class GeminiClient:
    """Client for interacting with Google Gemini API."""

    def __init__(self) -> None:
        """Initialize the Gemini client."""
        settings = get_settings()
        genai.configure(api_key=settings.gemini_api_key_value)
        self.model_name = settings.gemini_model
        self.temperature = settings.gemini_temperature
        self.max_tokens = settings.gemini_max_tokens
        self._model: genai.GenerativeModel | None = None

    @property
    def model(self) -> genai.GenerativeModel:
        """Get or create the generative model."""
        if self._model is None:
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            )
        return self._model

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def generate(
        self,
        prompt: str,
        system_instruction: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> GenerateContentResponse:
        """Generate content from the model.

        Args:
            prompt: The user prompt
            system_instruction: Optional system instruction
            tools: Optional list of function definitions for function calling

        Returns:
            The model response
        """
        logger.debug(
            "generating_content",
            prompt_length=len(prompt),
            has_system=system_instruction is not None,
            has_tools=tools is not None,
        )

        # Create model with system instruction if provided
        model = self.model
        if system_instruction:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
                system_instruction=system_instruction,
            )

        # Convert tools to Gemini format if provided
        gemini_tools = None
        if tools:
            gemini_tools = [self._convert_tool(t) for t in tools]

        # Generate content
        response = await model.generate_content_async(
            prompt,
            tools=gemini_tools,
        )

        logger.debug(
            "content_generated",
            finish_reason=str(response.candidates[0].finish_reason) if response.candidates else None,
        )

        return response

    async def generate_with_function_calling(
        self,
        prompt: str,
        system_instruction: str,
        tools: list[dict[str, Any]],
        max_iterations: int = 5,
    ) -> dict[str, Any]:
        """Generate content with automatic function calling loop.

        This method handles the function calling loop, executing tool calls
        and feeding results back to the model until it produces a final response.

        Args:
            prompt: The user prompt
            system_instruction: System instruction for the model
            tools: List of available tools
            max_iterations: Maximum function calling iterations

        Returns:
            Dict containing the final response and all tool calls made
        """
        logger.info(
            "starting_function_calling",
            tools=[t["name"] for t in tools],
            max_iterations=max_iterations,
        )

        # Build conversation history
        messages = [{"role": "user", "parts": [prompt]}]
        tool_calls_made: list[dict[str, Any]] = []

        # Create model with tools
        model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
            system_instruction=system_instruction,
            tools=[self._convert_tool(t) for t in tools],
        )

        for iteration in range(max_iterations):
            logger.debug(f"function_calling_iteration_{iteration + 1}")

            # Generate response
            response = await model.generate_content_async(messages)

            if not response.candidates:
                logger.warning("no_candidates_in_response")
                break

            candidate = response.candidates[0]

            # Check for function calls
            function_calls = self._extract_function_calls(candidate)

            if not function_calls:
                # No more function calls, return final response
                final_text = candidate.content.parts[0].text if candidate.content.parts else ""
                return {
                    "response": final_text,
                    "tool_calls": tool_calls_made,
                    "iterations": iteration + 1,
                }

            # Execute function calls and collect results
            for fc in function_calls:
                tool_calls_made.append({
                    "name": fc["name"],
                    "args": fc["args"],
                    "iteration": iteration + 1,
                })

            # Add assistant response and function results to messages
            messages.append({"role": "model", "parts": candidate.content.parts})

            # For now, we return with the function calls - the orchestrator will execute them
            # In a full implementation, we'd execute tools and feed results back
            return {
                "response": None,
                "tool_calls": tool_calls_made,
                "pending_calls": function_calls,
                "iterations": iteration + 1,
            }

        logger.warning("max_iterations_reached", max_iterations=max_iterations)
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

        response = await self.generate(
            prompt=structured_prompt,
            system_instruction=system_instruction,
        )

        # Parse JSON from response
        response_text = response.text.strip()

        # Handle markdown code blocks
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.startswith("```"):
            response_text = response_text[3:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]

        parsed = json.loads(response_text.strip())
        return response_schema.model_validate(parsed)

    def _convert_tool(self, tool_def: dict[str, Any]) -> Tool:
        """Convert a tool definition dict to Gemini Tool format."""
        function_declaration = FunctionDeclaration(
            name=tool_def["name"],
            description=tool_def.get("description", ""),
            parameters=tool_def.get("parameters", {}),
        )
        return Tool(function_declarations=[function_declaration])

    def _extract_function_calls(self, candidate: Any) -> list[dict[str, Any]]:
        """Extract function calls from a response candidate."""
        function_calls = []

        if not candidate.content or not candidate.content.parts:
            return function_calls

        for part in candidate.content.parts:
            if hasattr(part, "function_call") and part.function_call:
                fc = part.function_call
                function_calls.append({
                    "name": fc.name,
                    "args": dict(fc.args) if fc.args else {},
                })

        return function_calls


# Singleton instance
_client: GeminiClient | None = None


def get_gemini_client() -> GeminiClient:
    """Get the singleton Gemini client instance."""
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client
