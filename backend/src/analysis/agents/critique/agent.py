"""Critique Agent - agentic analysis review with data investigation.

The critique agent reviews a finished analysis by INVESTIGATING the
actual data and results rather than reading summaries alone. It runs
a custom agentic loop (not ReActAgent) with seven sync investigation
tools and a finalize_critique tool that closes the loop.

Design note: Inherits BaseAgent (not ContextTools) intentionally. As
the final review stage, it has purpose-built investigation tools that
directly read state. ContextTools would bloat its tool schema with
seven generic context tools that overlap with its specialized ones.
"""

import time
from typing import Any

import pandas as pd

from src.analysis.agents.base import AnalysisState, BaseAgent
from src.analysis.agents.registry import register_agent
from src.logging_config.structured import get_logger

from . import tools
from .helpers import (
    auto_finalize,
    build_initial_prompt,
    create_feedback,
    heuristic_critique,
    load_dataframe,
)
from .prompt import SYSTEM_PROMPT

logger = get_logger(__name__)


@register_agent("critique")
class CritiqueAgent(BaseAgent):
    """Agentic critique with data-investigation tools."""

    AGENT_NAME = "critique"
    WRITES_STATE_FIELDS = ["critique_history"]

    SYSTEM_PROMPT = SYSTEM_PROMPT
    TOOLS = tools.SCHEMAS

    def __init__(self) -> None:
        super().__init__()
        self._df: pd.DataFrame | None = None
        self._state: AnalysisState | None = None
        self._investigation_evidence: list[str] = []
        self._treatment_var: str | None = None
        self._outcome_var: str | None = None
        self._tool_call_history: list[str] = []

    def _resolve_treatment_outcome(self) -> tuple[str | None, str | None]:
        """Cache the primary pair on first call so tools can read it cheaply."""
        if self._treatment_var and self._outcome_var:
            return self._treatment_var, self._outcome_var

        if self._state:
            t, o = self._state.get_primary_pair()
            self._treatment_var = t
            self._outcome_var = o
            return t, o

        return None, None

    async def execute(self, state: AnalysisState) -> AnalysisState:
        """Execute critique through investigation and debate."""
        self.logger.info(
            "critique_start",
            job_id=state.job_id,
            iteration=state.iteration_count,
            n_effects=len(state.treatment_effects),
        )
        start_time = time.time()

        try:
            self._df = load_dataframe(state.dataframe_path)
            self._state = state
            self._investigation_evidence = []
            self._treatment_var = None
            self._outcome_var = None
            self._tool_call_history = []

            treatment_var, outcome_var = self._resolve_treatment_outcome()
            initial_prompt = build_initial_prompt(state, treatment_var, outcome_var)

            final_result = await self._run_agentic_loop(initial_prompt, max_iterations=10)

            feedback = create_feedback(final_result, state.iteration_count + 1)
            state.critique_history.append(feedback)

            duration_ms = int((time.time() - start_time) * 1000)
            trace = self.create_trace(
                action="critique_complete",
                reasoning=feedback.reasoning,
                outputs={
                    "decision": feedback.decision.value,
                    "scores": feedback.scores,
                    "n_issues": len(feedback.issues),
                    "evidence_gathered": len(self._investigation_evidence),
                },
                duration_ms=duration_ms,
            )
            state.add_trace(trace)

            self.logger.info(
                "critique_complete",
                decision=feedback.decision.value,
                avg_score=sum(feedback.scores.values()) / len(feedback.scores),
            )

        except Exception as e:
            self.logger.exception("critique_failed", error=str(e))
            feedback = heuristic_critique(state, str(e))
            state.critique_history.append(feedback)

        return state

    async def _run_agentic_loop(
        self, initial_prompt: str, max_iterations: int = 10
    ) -> dict[str, Any]:
        """LLM-driven investigation loop with finalize / oscillation short-circuits."""
        messages = [{"role": "user", "content": initial_prompt}]

        for iteration in range(max_iterations):
            self.logger.info(
                "critique_iteration",
                iteration=iteration,
                max_iterations=max_iterations,
                evidence_gathered=len(self._investigation_evidence),
            )

            result = await self.reason(
                prompt=messages[-1]["content"],
                context={"iteration": iteration, "max_iterations": max_iterations},
            )

            response_text = result.get("response", "")
            if response_text:
                self.logger.info(
                    "critique_llm_reasoning",
                    reasoning=response_text[:500] + ("..." if len(response_text) > 500 else ""),
                )

            pending_calls = result.get("pending_calls", [])

            if not pending_calls:
                self.logger.info("critique_no_tool_calls", response_preview=response_text[:200])
                if "finalize" in response_text.lower() or iteration > 7:
                    return auto_finalize(self._state, self._investigation_evidence)
                messages.append({
                    "role": "user",
                    "content": "Please investigate using tools, then call finalize_critique.",
                })
                continue

            tool_results: list[str] = []
            for call in pending_calls:
                tool_name = call.get("name")
                tool_args = call.get("args", {})

                if tool_name == "finalize_critique":
                    scores = tool_args.get("scores", {})
                    decision = tool_args.get("decision")
                    self.logger.info(
                        "critique_finalizing",
                        decision=decision,
                        scores=scores,
                        avg_score=sum(scores.values()) / len(scores) if scores else 0,
                        reasoning=tool_args.get("reasoning", "")[:200],
                    )
                    return tool_args

                self.logger.info("critique_tool_decision", tool=tool_name, args=tool_args)

                # MED2: Detect tool oscillation — same 2 tools alternating 3+ times.
                self._tool_call_history.append(tool_name)
                if len(self._tool_call_history) >= 6:
                    recent = self._tool_call_history[-6:]
                    if (recent[0] == recent[2] == recent[4]
                            and recent[1] == recent[3] == recent[5]
                            and recent[0] != recent[1]):
                        self.logger.warning(
                            "critique_oscillation_detected",
                            tool_a=recent[0],
                            tool_b=recent[1],
                        )
                        return auto_finalize(self._state, self._investigation_evidence)

                tool_result = self._execute_tool(tool_name, tool_args)
                tool_results.append(f"## {tool_name}\n{tool_result}")

                self.logger.info(
                    "critique_tool_result",
                    tool=tool_name,
                    result_summary=tool_result[:200] + ("..." if len(tool_result) > 200 else ""),
                )

            results_text = "\n\n".join(tool_results)
            messages.append({
                "role": "user",
                "content": f"Tool results:\n\n{results_text}\n\nContinue investigation or call finalize_critique.",
            })

        self.logger.warning("critique_max_iterations")
        return auto_finalize(self._state, self._investigation_evidence)

    def _execute_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Dispatch a tool call to its handler via tools.BY_NAME."""
        module = tools.BY_NAME.get(tool_name)
        if module is None:
            return f"Unknown tool: {tool_name}"
        try:
            return module.handle(self, **args)
        except Exception as e:
            self.logger.error("tool_execution_error", tool=tool_name, error=str(e))
            return f"Error executing {tool_name}: {str(e)}"

    def _auto_finalize(self) -> dict[str, Any]:
        return auto_finalize(self._state, self._investigation_evidence)
