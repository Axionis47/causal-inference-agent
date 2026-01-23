"""Google Vertex AI Gemini client with function calling support."""

import json
from typing import Any, TypeVar

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

# Lazy import to avoid requiring vertexai when using Gemini API
_vertexai_initialized = False


def _init_vertexai() -> None:
    """Initialize Vertex AI SDK."""
    global _vertexai_initialized
    if _vertexai_initialized:
        return

    import vertexai
    settings = get_settings()
    vertexai.init(
        project=settings.gcp_project_id,
        location=settings.gcp_region,
    )
    _vertexai_initialized = True


class VertexAIClient:
    """Client for interacting with Google Vertex AI Gemini models."""

    def __init__(self) -> None:
        """Initialize the Vertex AI client."""
        _init_vertexai()

        settings = get_settings()
        self.model_name = settings.vertex_model
        self.temperature = settings.gemini_temperature
        self.max_tokens = settings.gemini_max_tokens
        self._model = None

    @property
    def model(self):
        """Get or create the generative model."""
        if self._model is None:
            from vertexai.generative_models import GenerationConfig, GenerativeModel

            self._model = GenerativeModel(
                model_name=self.model_name,
                generation_config=GenerationConfig(
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
    ) -> Any:
        """Generate content from the model.

        Args:
            prompt: The user prompt
            system_instruction: Optional system instruction
            tools: Optional list of function definitions for function calling

        Returns:
            The model response
        """
        from vertexai.generative_models import (
            GenerationConfig,
            GenerativeModel,
        )

        logger.debug(
            "generating_content_vertex",
            prompt_length=len(prompt),
            has_system=system_instruction is not None,
            has_tools=tools is not None,
        )

        # Create model with system instruction if provided
        model = self.model
        if system_instruction:
            model = GenerativeModel(
                model_name=self.model_name,
                generation_config=GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
                system_instruction=system_instruction,
            )

        # Convert tools to Vertex AI format if provided
        vertex_tools = None
        if tools:
            vertex_tools = [self._convert_tool(t) for t in tools]

        # Generate content
        response = await model.generate_content_async(
            prompt,
            tools=vertex_tools,
        )

        logger.debug(
            "content_generated_vertex",
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

        For models that don't support multiple tools (like gemini-2.0-flash-exp),
        we combine all tools into a single unified tool with an action parameter.

        Args:
            prompt: The user prompt
            system_instruction: System instruction for the model
            tools: List of available tools
            max_iterations: Maximum function calling iterations

        Returns:
            Dict containing the final response and all tool calls made
        """
        from vertexai.generative_models import (
            Content,
            GenerationConfig,
            GenerativeModel,
            Part,
        )

        logger.info(
            "starting_function_calling_vertex",
            tools=[t["name"] for t in tools],
            max_iterations=max_iterations,
        )

        # Build conversation history
        messages = [Content(role="user", parts=[Part.from_text(prompt)])]
        tool_calls_made: list[dict[str, Any]] = []

        # For models that don't support multiple tools, combine into unified tool
        use_unified_tool = len(tools) > 1
        unified_tool = None
        original_tools = tools

        if use_unified_tool:
            unified_tool = self._create_unified_tool(tools)
            tools = [unified_tool]

        # Create model with tools
        model = GenerativeModel(
            model_name=self.model_name,
            generation_config=GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
            system_instruction=system_instruction,
            tools=[self._convert_tool(t) for t in tools],
        )

        for iteration in range(max_iterations):
            logger.debug(f"function_calling_iteration_vertex_{iteration + 1}")

            # Generate response
            response = await model.generate_content_async(messages)

            if not response.candidates:
                logger.warning("no_candidates_in_response_vertex")
                break

            candidate = response.candidates[0]

            # Check for function calls
            function_calls = self._extract_function_calls(candidate)

            if not function_calls:
                # No more function calls, return final response
                final_text = ""
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "text"):
                            final_text = part.text
                            break
                return {
                    "response": final_text,
                    "tool_calls": tool_calls_made,
                    "iterations": iteration + 1,
                }

            # Unpack unified tool calls if needed
            if use_unified_tool:
                function_calls = self._unpack_unified_calls(function_calls, original_tools)

            # Execute function calls and collect results
            for fc in function_calls:
                tool_calls_made.append({
                    "name": fc["name"],
                    "args": fc["args"],
                    "iteration": iteration + 1,
                })

            # Add assistant response to messages
            messages.append(Content(role="model", parts=candidate.content.parts))

            # Return with pending function calls for orchestrator to execute
            return {
                "response": None,
                "tool_calls": tool_calls_made,
                "pending_calls": function_calls,
                "iterations": iteration + 1,
            }

        logger.warning("max_iterations_reached_vertex", max_iterations=max_iterations)
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

    def _convert_tool(self, tool_def: dict[str, Any]):
        """Convert a tool definition dict to Vertex AI Tool format."""
        from vertexai.generative_models import FunctionDeclaration, Tool

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

    def _create_unified_tool(self, tools: list[dict[str, Any]]) -> dict[str, Any]:
        """Combine multiple tools into a single unified tool.

        This is used for models that don't support multiple tools.
        The unified tool has an 'action' parameter to select which tool to use.

        Args:
            tools: List of individual tool definitions

        Returns:
            A single unified tool definition
        """
        # Build action enum from tool names
        action_names = [t["name"] for t in tools]

        # Build descriptions for each action
        action_descriptions = "\n".join([
            f"- {t['name']}: {t.get('description', 'No description')}"
            for t in tools
        ])

        # Collect all possible parameters across all tools
        all_params = {}
        for tool in tools:
            params = tool.get("parameters", {}).get("properties", {})
            for param_name, param_def in params.items():
                if param_name not in all_params:
                    all_params[param_name] = param_def

        unified_tool = {
            "name": "execute_action",
            "description": f"Execute one of the following actions:\n{action_descriptions}\n\nChoose the appropriate action and provide its parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": action_names,
                        "description": "The action to execute",
                    },
                    **all_params,
                },
                "required": ["action"],
            },
        }

        return unified_tool

    def _unpack_unified_calls(
        self,
        function_calls: list[dict[str, Any]],
        original_tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Unpack unified tool calls back to original tool format.

        Args:
            function_calls: Calls to the unified execute_action tool
            original_tools: The original individual tools

        Returns:
            Function calls in the original tool format
        """
        unpacked = []

        for call in function_calls:
            if call["name"] == "execute_action":
                args = call.get("args", {})
                action = args.pop("action", None)
                if action:
                    unpacked.append({
                        "name": action,
                        "args": args,
                    })
            else:
                # Already in correct format
                unpacked.append(call)

        return unpacked


# Singleton instance
_client: VertexAIClient | None = None


def get_vertex_client() -> VertexAIClient:
    """Get the singleton Vertex AI client instance."""
    global _client
    if _client is None:
        _client = VertexAIClient()
    return _client
