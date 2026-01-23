"""Tool interface for agent capabilities."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ToolParameter(BaseModel):
    """Definition of a tool parameter."""

    name: str
    type: str  # string, number, boolean, object, array
    description: str
    required: bool = True
    enum: list[str] | None = None
    default: Any = None


class ToolDefinition(BaseModel):
    """Definition of a tool that can be called by agents."""

    name: str
    description: str
    parameters: list[ToolParameter]

    def to_gemini_format(self) -> dict[str, Any]:
        """Convert to Gemini function calling format."""
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }


class BaseTool(ABC):
    """Abstract base class for tools.

    Tools are capabilities that agents can use to interact with the world.
    Each tool has a definition (for the LLM) and an execute method.
    """

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Get the tool definition for the LLM."""
        pass

    @abstractmethod
    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        """Execute the tool with the given arguments.

        Args:
            **kwargs: Tool-specific arguments

        Returns:
            Result of the tool execution
        """
        pass

    def get_gemini_definition(self) -> dict[str, Any]:
        """Get the tool definition in Gemini format."""
        return self.definition.to_gemini_format()


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self._tools[tool.definition.name] = tool

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def get_all(self) -> list[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get all tool definitions in Gemini format."""
        return [tool.get_gemini_definition() for tool in self._tools.values()]

    async def execute(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Execute a tool by name.

        Args:
            name: Name of the tool to execute
            **kwargs: Arguments for the tool

        Returns:
            Result of the tool execution

        Raises:
            ValueError: If tool is not found
        """
        tool = self.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found")
        return await tool.execute(**kwargs)
