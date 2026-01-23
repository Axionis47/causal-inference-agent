"""Base agent class for all agents in the system."""

import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from src.llm import get_llm_client
from src.logging_config.structured import get_logger

from .state import AgentTrace, AnalysisState

logger = get_logger(__name__)


@dataclass
class DebugContext:
    """Context for self-debugging."""
    error_type: str
    error_message: str
    stack_trace: str
    attempt_number: int
    previous_strategies: list[str] = field(default_factory=list)

    def to_prompt(self) -> str:
        """Convert to LLM prompt."""
        return f"""Self-debugging context:
- Error Type: {self.error_type}
- Error Message: {self.error_message}
- Attempt: {self.attempt_number}
- Previous strategies tried: {', '.join(self.previous_strategies) if self.previous_strategies else 'None'}

Stack trace (last 5 lines):
{chr(10).join(self.stack_trace.split(chr(10))[-5:])}
"""


@dataclass
class RecoveryStrategy:
    """A recovery strategy for self-debugging."""
    name: str
    description: str
    action: Callable[["AnalysisState"], "AnalysisState"] | None = None
    fallback_params: dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all agents.

    All agents in the system inherit from this class. Each agent is responsible
    for a specific task in the causal inference pipeline and uses Gemini for
    reasoning - NO hardcoded if-else logic for decisions.

    The agent workflow is:
    1. Receive state from orchestrator
    2. Use Gemini to reason about the task
    3. Execute tools based on LLM decisions
    4. Return updated state
    """

    # Subclasses must define these
    AGENT_NAME: str = "base_agent"
    SYSTEM_PROMPT: str = ""
    TOOLS: list[dict[str, Any]] = []

    def __init__(self) -> None:
        """Initialize the agent."""
        self.llm = get_llm_client()
        self.logger = get_logger(self.AGENT_NAME)

    @abstractmethod
    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute the agent's task.

        Args:
            state: Current analysis state

        Returns:
            Updated analysis state
        """
        pass

    async def reason(
        self,
        prompt: str,
        context: dict[str, Any] | None = None,
        max_retries: int = 5,
    ) -> dict[str, Any]:
        """Use Gemini to reason about a task with retry logic for rate limiting.

        This is the core method for agentic reasoning. The agent describes
        the situation and asks the LLM to reason about what to do.

        Uses exponential backoff on rate limit (429) errors to maintain
        agentic behavior even under API throttling.

        Args:
            prompt: The reasoning prompt
            context: Additional context to include
            max_retries: Maximum number of retries on rate limit errors

        Returns:
            The LLM's reasoning and decisions
        """
        import asyncio

        # Build full prompt with context
        full_prompt = prompt
        if context:
            context_str = "\n".join(f"- {k}: {v}" for k, v in context.items())
            full_prompt = f"Context:\n{context_str}\n\n{prompt}"

        self.logger.debug("reasoning", prompt_preview=prompt[:200])

        # Retry with exponential backoff for rate limiting
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Use function calling if tools are defined
                if self.TOOLS:
                    result = await self.llm.generate_with_function_calling(
                        prompt=full_prompt,
                        system_instruction=self.SYSTEM_PROMPT,
                        tools=self.TOOLS,
                    )
                else:
                    response = await self.llm.generate(
                        prompt=full_prompt,
                        system_instruction=self.SYSTEM_PROMPT,
                    )
                    result = {"response": response.text, "tool_calls": []}

                self.logger.debug(
                    "reasoning_complete",
                    has_response=result.get("response") is not None,
                    tool_calls=len(result.get("tool_calls", [])),
                )

                return result

            except Exception as e:
                last_error = e
                error_str = str(e)
                # Check for rate limit errors
                if "429" in error_str or "Resource exhausted" in error_str or "Quota exceeded" in error_str:
                    if attempt < max_retries:
                        wait_time = 2 ** attempt * 5  # 5, 10, 20, 40 seconds
                        self.logger.info(
                            "rate_limit_retry",
                            attempt=attempt + 1,
                            wait_seconds=wait_time,
                        )
                        await asyncio.sleep(wait_time)
                        continue
                # Non-retryable error or max retries exceeded
                raise

        # If we get here, we've exhausted retries
        raise last_error

    def create_trace(
        self,
        action: str,
        reasoning: str,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        tools_called: list[str] | None = None,
        duration_ms: int = 0,
    ) -> AgentTrace:
        """Create an agent trace for observability.

        Args:
            action: Description of the action taken
            reasoning: The LLM's reasoning for this action
            inputs: Input data for this action
            outputs: Output data from this action
            tools_called: List of tools that were called
            duration_ms: Duration of the action in milliseconds

        Returns:
            AgentTrace instance
        """
        return AgentTrace(
            agent_name=self.AGENT_NAME,
            timestamp=datetime.utcnow(),
            action=action,
            reasoning=reasoning,
            inputs=inputs or {},
            outputs=outputs or {},
            tools_called=tools_called or [],
            duration_ms=duration_ms,
        )

    async def execute_with_tracing(self, state: AnalysisState) -> AnalysisState:
        """Execute the agent with automatic tracing and self-debugging.

        Args:
            state: Current analysis state

        Returns:
            Updated analysis state with traces added
        """
        start_time = time.time()
        self.logger.info(
            "agent_execution_start",
            agent=self.AGENT_NAME,
            job_id=state.job_id,
        )

        max_retries = getattr(self, 'MAX_RETRIES', 2)
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                # Execute the agent's main logic
                updated_state = await self.execute(state)

                duration_ms = int((time.time() - start_time) * 1000)
                self.logger.info(
                    "agent_execution_complete",
                    agent=self.AGENT_NAME,
                    job_id=state.job_id,
                    duration_ms=duration_ms,
                    attempt=attempt + 1,
                )

                return updated_state

            except Exception as e:
                last_error = e
                duration_ms = int((time.time() - start_time) * 1000)

                if attempt < max_retries:
                    # Attempt self-debugging recovery
                    debug_context = DebugContext(
                        error_type=type(e).__name__,
                        error_message=str(e),
                        stack_trace=traceback.format_exc(),
                        attempt_number=attempt + 1,
                        previous_strategies=getattr(self, '_recovery_attempts', []),
                    )

                    recovery = await self._attempt_recovery(state, debug_context)

                    if recovery:
                        self.logger.info(
                            "self_debugging_recovery",
                            agent=self.AGENT_NAME,
                            strategy=recovery.name,
                            attempt=attempt + 1,
                        )

                        # Track attempted strategies
                        if not hasattr(self, '_recovery_attempts'):
                            self._recovery_attempts = []
                        self._recovery_attempts.append(recovery.name)

                        # Apply recovery and retry
                        if recovery.action:
                            state = recovery.action(state)
                        continue
                    else:
                        self.logger.warning(
                            "no_recovery_strategy",
                            agent=self.AGENT_NAME,
                            error=str(e),
                        )

                # Final failure
                self.logger.error(
                    "agent_execution_failed",
                    agent=self.AGENT_NAME,
                    job_id=state.job_id,
                    error=str(e),
                    duration_ms=duration_ms,
                    attempts=attempt + 1,
                )

                # Add error trace
                error_trace = self.create_trace(
                    action="execution_failed",
                    reasoning=f"Agent failed after {attempt + 1} attempts: {str(e)}",
                    outputs={"error": str(e), "attempts": attempt + 1},
                    duration_ms=duration_ms,
                )
                state.add_trace(error_trace)
                state.mark_failed(str(e), self.AGENT_NAME)

                raise

        raise last_error

    async def _attempt_recovery(
        self,
        state: AnalysisState,
        debug_context: DebugContext,
    ) -> RecoveryStrategy | None:
        """Attempt to recover from an error using self-debugging.

        Args:
            state: Current analysis state
            debug_context: Context about the error

        Returns:
            RecoveryStrategy if one is found, None otherwise
        """
        # Get recovery strategies for this agent
        strategies = self._get_recovery_strategies()

        if not strategies:
            return None

        # Filter out already-tried strategies
        available = [s for s in strategies if s.name not in debug_context.previous_strategies]

        if not available:
            self.logger.warning("all_strategies_exhausted", agent=self.AGENT_NAME)
            return None

        # Try to use LLM to select best strategy
        try:
            strategy = await self._llm_select_recovery(debug_context, available)
            if strategy:
                return strategy
        except Exception:
            pass

        # Fallback: use heuristic matching
        return self._heuristic_select_recovery(debug_context, available)

    async def _llm_select_recovery(
        self,
        debug_context: DebugContext,
        strategies: list[RecoveryStrategy],
    ) -> RecoveryStrategy | None:
        """Use LLM to select best recovery strategy."""
        prompt = f"""Select the best recovery strategy for this error.

{debug_context.to_prompt()}

Available strategies:
"""
        for i, s in enumerate(strategies, 1):
            prompt += f"{i}. {s.name}: {s.description}\n"

        prompt += "\nRespond with just the strategy number (1, 2, etc.) that is most likely to fix this error."

        try:
            response = await self.llm.generate(prompt, system_instruction="You are a debugging expert.")
            # Parse response to get strategy number
            text = response.text.strip()
            for i, s in enumerate(strategies, 1):
                if str(i) in text or s.name.lower() in text.lower():
                    return s
        except Exception:
            pass

        return None

    def _heuristic_select_recovery(
        self,
        debug_context: DebugContext,
        strategies: list[RecoveryStrategy],
    ) -> RecoveryStrategy | None:
        """Select recovery strategy using heuristics."""
        error_lower = debug_context.error_message.lower()

        # Match error patterns to strategies
        for strategy in strategies:
            name_lower = strategy.name.lower()

            # Rate limit errors -> retry
            if "rate" in error_lower or "429" in error_lower or "quota" in error_lower:
                if "retry" in name_lower or "fallback" in name_lower:
                    return strategy

            # Memory errors -> reduce data
            if "memory" in error_lower or "oom" in error_lower:
                if "reduce" in name_lower or "sample" in name_lower:
                    return strategy

            # Numeric errors -> simplify
            if "singular" in error_lower or "convergence" in error_lower or "nan" in error_lower:
                if "simplify" in name_lower or "robust" in name_lower:
                    return strategy

        # Return first available as last resort
        return strategies[0] if strategies else None

    def _get_recovery_strategies(self) -> list[RecoveryStrategy]:
        """Get recovery strategies for this agent.

        Override in subclasses to provide agent-specific strategies.
        """
        return [
            RecoveryStrategy(
                name="retry_with_delay",
                description="Wait briefly and retry the operation",
            ),
            RecoveryStrategy(
                name="use_heuristic_fallback",
                description="Fall back to heuristic-based processing without LLM",
            ),
        ]


class ToolExecutionError(Exception):
    """Error raised when a tool execution fails."""

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        super().__init__(f"Tool '{tool_name}' failed: {message}")
