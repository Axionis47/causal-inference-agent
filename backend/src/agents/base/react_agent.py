"""ReAct Agent - True Reasoning and Acting agent with observe-reason-act loops.

This implements the ReAct paradigm where agents:
1. OBSERVE - Perceive the current state and results of previous actions
2. REASON - Think about what to do next based on observations
3. ACT - Execute a tool or action
4. Loop until task is complete

Reference: https://arxiv.org/abs/2210.03629
"""

import time
from abc import abstractmethod
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from src.llm import get_llm_client
from src.logging_config.structured import get_logger

from .state import AgentTrace, AnalysisState

logger = get_logger(__name__)


class ToolResultStatus(Enum):
    """Status of a tool execution."""
    SUCCESS = "success"
    ERROR = "error"
    NEEDS_RETRY = "needs_retry"


@dataclass
class ToolResult:
    """Result of a tool execution."""
    status: ToolResultStatus
    output: Any
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_observation(self) -> str:
        """Convert to an observation string for the LLM."""
        if self.status == ToolResultStatus.SUCCESS:
            if isinstance(self.output, dict):
                # Format dict nicely
                lines = ["Tool executed successfully:"]
                for k, v in self.output.items():
                    lines.append(f"  {k}: {v}")
                return "\n".join(lines)
            return f"Tool executed successfully: {self.output}"
        elif self.status == ToolResultStatus.ERROR:
            return f"Tool execution failed: {self.error}"
        else:
            return f"Tool needs retry: {self.error}"


@dataclass
class ReActStep:
    """A single step in the ReAct loop."""
    step_number: int
    observation: str
    thought: str
    action: str | None
    action_input: dict[str, Any] | None
    result: ToolResult | None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    duration_ms: int = 0


class ReActAgent:
    """Base class for ReAct agents that truly reason and act in a loop.

    Subclasses must:
    1. Define AGENT_NAME, SYSTEM_PROMPT
    2. Register tools using register_tool()
    3. Implement is_task_complete() to determine when to stop
    """

    AGENT_NAME: str = "react_agent"
    SYSTEM_PROMPT: str = ""
    MAX_STEPS: int = 10

    # Special tool names
    FINISH_TOOL = "finish"
    REFLECT_TOOL = "reflect"

    def __init__(self) -> None:
        """Initialize the ReAct agent."""
        self.llm = get_llm_client()
        self.logger = get_logger(self.AGENT_NAME)
        self._tools: dict[str, Callable[..., Coroutine[Any, Any, ToolResult]]] = {}
        self._tool_schemas: list[dict[str, Any]] = []
        self._steps: list[ReActStep] = []

        # Register built-in tools
        self._register_builtin_tools()

    def _register_builtin_tools(self) -> None:
        """Register built-in tools available to all ReAct agents."""
        # Finish tool - signals task completion
        self.register_tool(
            name=self.FINISH_TOOL,
            description="Call this when the task is complete. Provide a summary of what was accomplished.",
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Summary of what was accomplished",
                    },
                    "success": {
                        "type": "boolean",
                        "description": "Whether the task was completed successfully",
                    },
                },
                "required": ["summary", "success"],
            },
            handler=self._handle_finish,
        )

        # Reflect tool - self-reflection on current progress
        self.register_tool(
            name=self.REFLECT_TOOL,
            description="Pause to reflect on current progress and decide if approach needs adjustment.",
            parameters={
                "type": "object",
                "properties": {
                    "current_progress": {
                        "type": "string",
                        "description": "What has been accomplished so far",
                    },
                    "obstacles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Any obstacles or issues encountered",
                    },
                    "next_steps": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Proposed next steps",
                    },
                },
                "required": ["current_progress", "next_steps"],
            },
            handler=self._handle_reflect,
        )

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Coroutine[Any, Any, ToolResult]],
    ) -> None:
        """Register a tool that the agent can use.

        Args:
            name: Tool name (used by LLM to invoke)
            description: Description of what the tool does
            parameters: JSON Schema for the tool parameters
            handler: Async function that executes the tool
        """
        self._tools[name] = handler
        self._tool_schemas.append({
            "name": name,
            "description": description,
            "parameters": parameters,
        })
        self.logger.debug("tool_registered", tool=name)

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute the ReAct loop until task completion.

        Args:
            state: Current analysis state

        Returns:
            Updated analysis state
        """
        self._steps = []
        start_time = time.time()

        self.logger.info(
            "react_loop_start",
            agent=self.AGENT_NAME,
            job_id=state.job_id,
            max_steps=self.MAX_STEPS,
        )

        # Initial observation from state
        observation = self._get_initial_observation(state)

        for step_num in range(1, self.MAX_STEPS + 1):
            step_start = time.time()

            # REASON: Get thought and action from LLM
            thought, action, action_input = await self._reason(
                state=state,
                observation=observation,
                step_num=step_num,
            )

            self.logger.info(
                "react_step",
                step=step_num,
                thought=thought[:100] if thought else None,
                action=action,
            )

            # Check for finish action
            if action == self.FINISH_TOOL:
                result = await self._execute_tool(action, action_input, state)
                step = ReActStep(
                    step_number=step_num,
                    observation=observation,
                    thought=thought,
                    action=action,
                    action_input=action_input,
                    result=result,
                    duration_ms=int((time.time() - step_start) * 1000),
                )
                self._steps.append(step)

                # Finalize
                self._record_trace(state, step)
                break

            # ACT: Execute the tool
            if action and action in self._tools:
                result = await self._execute_tool(action, action_input, state)
                observation = result.to_observation()
            else:
                # No valid action, create error observation
                result = ToolResult(
                    status=ToolResultStatus.ERROR,
                    output=None,
                    error=f"Unknown action: {action}. Available: {list(self._tools.keys())}",
                )
                observation = result.to_observation()

            # Record step
            step = ReActStep(
                step_number=step_num,
                observation=observation,
                thought=thought,
                action=action,
                action_input=action_input,
                result=result,
                duration_ms=int((time.time() - step_start) * 1000),
            )
            self._steps.append(step)
            self._record_trace(state, step)

            # Check if task is complete based on state
            if await self.is_task_complete(state):
                self.logger.info("task_complete_detected", step=step_num)
                break

            # Error recovery: if we got an error, add recovery hint
            if result.status == ToolResultStatus.ERROR:
                observation = f"{observation}\n\nThe previous action failed. Consider an alternative approach."

        total_duration = int((time.time() - start_time) * 1000)
        self.logger.info(
            "react_loop_complete",
            agent=self.AGENT_NAME,
            steps=len(self._steps),
            duration_ms=total_duration,
        )

        return state

    async def _reason(
        self,
        state: AnalysisState,
        observation: str,
        step_num: int,
    ) -> tuple[str, str | None, dict[str, Any] | None]:
        """Use LLM to reason about next action.

        Args:
            state: Current state
            observation: Current observation
            step_num: Current step number

        Returns:
            Tuple of (thought, action_name, action_input)
        """
        # Build prompt with ReAct format
        prompt = self._build_react_prompt(state, observation, step_num)

        # Get LLM response with function calling
        result = await self.llm.generate_with_function_calling(
            prompt=prompt,
            system_instruction=self._build_system_prompt(),
            tools=self._tool_schemas,
            max_iterations=1,  # Single turn - we handle the loop ourselves
        )

        # Extract thought from response
        thought = result.get("response", "")

        # Extract action from pending calls
        action = None
        action_input = None

        if result.get("pending_calls"):
            call = result["pending_calls"][0]  # Take first action
            action = call["name"]
            action_input = call.get("args", {})
        elif result.get("tool_calls"):
            call = result["tool_calls"][-1]
            action = call["name"]
            action_input = call.get("args", {})

        return thought, action, action_input

    def _build_system_prompt(self) -> str:
        """Build the system prompt with ReAct instructions."""
        return f"""{self.SYSTEM_PROMPT}

You are operating in ReAct mode. For each step:
1. OBSERVE: Review the observation from your previous action
2. THINK: Reason about what to do next based on the observation
3. ACT: Call a tool to take action

Always explain your thinking before acting. If something isn't working, try a different approach.
When you have completed the task successfully, call the 'finish' tool.

Available tools: {[t['name'] for t in self._tool_schemas]}
"""

    def _build_react_prompt(
        self,
        state: AnalysisState,
        observation: str,
        step_num: int,
    ) -> str:
        """Build the prompt for a ReAct step."""
        prompt = f"""Step {step_num}/{self.MAX_STEPS}

OBSERVATION:
{observation}

"""
        # Add previous steps context (last 3 steps)
        if self._steps:
            prompt += "PREVIOUS STEPS:\n"
            for step in self._steps[-3:]:
                prompt += f"Step {step.step_number}: Action={step.action}, "
                prompt += f"Result={'Success' if step.result and step.result.status == ToolResultStatus.SUCCESS else 'Failed'}\n"
            prompt += "\n"

        prompt += """Based on the observation, think step by step:
1. What does this observation tell me?
2. What should I do next to make progress?
3. Which tool should I use?

Then call the appropriate tool."""

        return prompt

    async def _execute_tool(
        self,
        action: str,
        action_input: dict[str, Any] | None,
        state: AnalysisState,
    ) -> ToolResult:
        """Execute a tool and return the result.

        Args:
            action: Tool name
            action_input: Tool arguments
            state: Current state (passed to tool handlers)

        Returns:
            Tool result
        """
        handler = self._tools.get(action)
        if not handler:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=f"Tool '{action}' not found",
            )

        try:
            # Pass state and action_input to handler
            result = await handler(state, **(action_input or {}))
            return result
        except Exception as e:
            self.logger.error("tool_execution_error", tool=action, error=str(e))
            return ToolResult(
                status=ToolResultStatus.ERROR,
                output=None,
                error=str(e),
            )

    async def _handle_finish(
        self,
        state: AnalysisState,
        summary: str,
        success: bool = True,
    ) -> ToolResult:
        """Handle the finish tool call."""
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={"summary": summary, "success": success},
            metadata={"is_finish": True},
        )

    async def _handle_reflect(
        self,
        state: AnalysisState,
        current_progress: str,
        next_steps: list[str],
        obstacles: list[str] | None = None,
    ) -> ToolResult:
        """Handle the reflect tool call."""
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            output={
                "reflection": current_progress,
                "next_steps": next_steps,
                "obstacles": obstacles or [],
            },
            metadata={"is_reflection": True},
        )

    def _record_trace(self, state: AnalysisState, step: ReActStep) -> None:
        """Record a step as an agent trace."""
        trace = AgentTrace(
            agent_name=self.AGENT_NAME,
            timestamp=step.timestamp,
            action=f"step_{step.step_number}_{step.action or 'none'}",
            reasoning=step.thought,
            inputs=step.action_input or {},
            outputs={
                "status": step.result.status.value if step.result else "none",
                "output": str(step.result.output)[:500] if step.result else None,
            },
            tools_called=[step.action] if step.action else [],
            duration_ms=step.duration_ms,
        )
        state.add_trace(trace)

    @abstractmethod
    def _get_initial_observation(self, state: AnalysisState) -> str:
        """Get the initial observation from state.

        Subclasses must implement this to provide context-specific observations.

        Args:
            state: Current analysis state

        Returns:
            Initial observation string
        """
        pass

    @abstractmethod
    async def is_task_complete(self, state: AnalysisState) -> bool:
        """Check if the task is complete.

        Subclasses must implement this to define completion criteria.

        Args:
            state: Current analysis state

        Returns:
            True if task is complete
        """
        pass

    async def execute_with_tracing(self, state: AnalysisState) -> AnalysisState:
        """Execute with automatic tracing wrapper."""
        start_time = time.time()
        self.logger.info(
            "agent_execution_start",
            agent=self.AGENT_NAME,
            job_id=state.job_id,
        )

        try:
            updated_state = await self.execute(state)

            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.info(
                "agent_execution_complete",
                agent=self.AGENT_NAME,
                job_id=state.job_id,
                duration_ms=duration_ms,
            )

            return updated_state

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            self.logger.error(
                "agent_execution_failed",
                agent=self.AGENT_NAME,
                job_id=state.job_id,
                error=str(e),
                duration_ms=duration_ms,
            )

            error_trace = AgentTrace(
                agent_name=self.AGENT_NAME,
                timestamp=datetime.utcnow(),
                action="execution_failed",
                reasoning=f"Agent failed with error: {str(e)}",
                outputs={"error": str(e)},
                duration_ms=duration_ms,
            )
            state.add_trace(error_trace)
            state.mark_failed(str(e), self.AGENT_NAME)

            raise
