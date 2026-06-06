"""Structured error types for the agent system.

Provides a hierarchy of exceptions that carry context needed for
debugging, recovery decisions, and consistent error handling.
"""


class AgentError(Exception):
    """Base error for all agent-related failures."""

    def __init__(
        self,
        agent_name: str,
        message: str,
        recoverable: bool = False,
    ) -> None:
        self.agent_name = agent_name
        self.recoverable = recoverable
        super().__init__(f"[{agent_name}] {message}")


class ToolExecutionError(AgentError):
    """Raised when a tool execution fails within an agent."""

    def __init__(self, agent_name: str, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        super().__init__(agent_name, f"Tool '{tool_name}' failed: {message}", recoverable=True)
