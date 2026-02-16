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


class StateValidationError(AgentError):
    """Raised when required state fields are missing before agent execution."""

    def __init__(self, agent_name: str, field: str, message: str | None = None) -> None:
        self.field = field
        msg = message or f"Required state field '{field}' is None. Ensure it is populated before running {agent_name}."
        super().__init__(agent_name, msg, recoverable=False)


class ToolExecutionError(AgentError):
    """Raised when a tool execution fails within an agent."""

    def __init__(self, agent_name: str, tool_name: str, message: str) -> None:
        self.tool_name = tool_name
        super().__init__(agent_name, f"Tool '{tool_name}' failed: {message}", recoverable=True)


class LLMError(AgentError):
    """Raised when an LLM call fails (rate limit, timeout, parsing, etc.)."""

    def __init__(self, agent_name: str, message: str, recoverable: bool = True) -> None:
        super().__init__(agent_name, message, recoverable=recoverable)
