"""Base classes for agentic evaluation framework.

This module defines the core evaluation infrastructure for testing agent
reasoning capabilities with actual LLM calls (using Claude via .env config).
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.agents.base import AnalysisState, DatasetInfo
from src.logging_config.structured import get_logger

# Load environment variables for Claude API
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env")

logger = get_logger(__name__)


class EvalDifficulty(Enum):
    """Difficulty level of an evaluation case."""
    EASY = "easy"  # Clear-cut decisions
    MEDIUM = "medium"  # Some ambiguity
    HARD = "hard"  # Edge cases, conflicting signals


class EvalCategory(Enum):
    """Category of evaluation."""
    TOOL_SELECTION = "tool_selection"  # Did agent choose right tools?
    REASONING_QUALITY = "reasoning_quality"  # Was reasoning sound?
    TASK_COMPLETION = "task_completion"  # Did agent complete task correctly?
    ERROR_RECOVERY = "error_recovery"  # Did agent handle errors well?
    EFFICIENCY = "efficiency"  # How many steps to complete?
    CONTEXT_USE = "context_use"  # Did agent use available context?


@dataclass
class EvalCase:
    """A single evaluation case for an agent.

    An eval case defines:
    - Input data and state
    - Expected outcomes (ground truth)
    - Evaluation criteria
    """
    name: str
    description: str
    difficulty: EvalDifficulty
    category: EvalCategory

    # Input data
    dataframe: pd.DataFrame
    dataset_info: dict[str, Any]
    initial_state_overrides: dict[str, Any] = field(default_factory=dict)

    # Ground truth / expected outcomes
    expected_outputs: dict[str, Any] = field(default_factory=dict)
    expected_tool_calls: list[str] | None = None  # Expected tools to be called
    max_expected_steps: int | None = None  # Maximum acceptable steps

    # Evaluation settings
    timeout_seconds: int = 300

    def create_state(self, job_id: str | None = None) -> AnalysisState:
        """Create an AnalysisState from this eval case."""
        import pickle
        import tempfile

        # Save dataframe to temp file
        temp_dir = Path(tempfile.gettempdir()) / "agentic_evals"
        temp_dir.mkdir(exist_ok=True)

        job_id = job_id or f"eval-{self.name}-{int(time.time())}"
        df_path = temp_dir / f"{job_id}_data.pkl"

        with open(df_path, "wb") as f:
            pickle.dump(self.dataframe, f)

        state = AnalysisState(
            job_id=job_id,
            dataset_info=DatasetInfo(
                url=self.dataset_info.get("url", f"eval://{self.name}"),
                name=self.dataset_info.get("name", self.name),
                **{k: v for k, v in self.dataset_info.items() if k not in ["url", "name"]}
            ),
            dataframe_path=str(df_path),
        )

        # Apply overrides
        for key, value in self.initial_state_overrides.items():
            if hasattr(state, key):
                setattr(state, key, value)

        return state


@dataclass
class AgenticEvalMetrics:
    """Metrics for evaluating agent performance."""

    # Core metrics
    task_completed: bool = False
    task_correct: bool = False  # Did agent produce correct output?

    # Efficiency metrics
    steps_taken: int = 0
    steps_optimal: int | None = None  # Optimal number of steps
    efficiency_score: float = 0.0  # steps_optimal / steps_taken (if optimal known)

    # Tool usage metrics
    tools_called: list[str] = field(default_factory=list)
    expected_tools_called: float = 0.0  # % of expected tools that were called
    unnecessary_tool_calls: int = 0  # Tools called that weren't needed

    # Reasoning quality (LLM-judged)
    reasoning_clarity: float | None = None  # 1-5 score
    reasoning_soundness: float | None = None  # 1-5 score
    reasoning_completeness: float | None = None  # 1-5 score

    # Error handling
    errors_encountered: int = 0
    errors_recovered: int = 0
    recovery_rate: float = 0.0

    # Timing
    execution_time_seconds: float = 0.0

    # Output quality metrics (domain-specific)
    output_metrics: dict[str, float] = field(default_factory=dict)

    def overall_score(self) -> float:
        """Calculate overall evaluation score (0-100)."""
        score = 0.0

        # Task completion: 40 points
        if self.task_completed:
            score += 20
        if self.task_correct:
            score += 20

        # Efficiency: 20 points
        if self.efficiency_score > 0:
            score += 20 * min(1.0, self.efficiency_score)
        elif self.task_completed:
            score += 10  # Partial credit if completed but no optimal known

        # Tool usage: 20 points
        score += 10 * self.expected_tools_called
        score += 10 * max(0, 1 - self.unnecessary_tool_calls / 10)

        # Reasoning quality: 20 points
        reasoning_scores = [
            s for s in [
                self.reasoning_clarity,
                self.reasoning_soundness,
                self.reasoning_completeness
            ] if s is not None
        ]
        if reasoning_scores:
            avg_reasoning = np.mean(reasoning_scores)
            score += 20 * (avg_reasoning / 5.0)

        return min(100, max(0, score))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_completed": self.task_completed,
            "task_correct": self.task_correct,
            "steps_taken": self.steps_taken,
            "steps_optimal": self.steps_optimal,
            "efficiency_score": self.efficiency_score,
            "tools_called": self.tools_called,
            "expected_tools_called": self.expected_tools_called,
            "unnecessary_tool_calls": self.unnecessary_tool_calls,
            "reasoning_clarity": self.reasoning_clarity,
            "reasoning_soundness": self.reasoning_soundness,
            "reasoning_completeness": self.reasoning_completeness,
            "errors_encountered": self.errors_encountered,
            "errors_recovered": self.errors_recovered,
            "recovery_rate": self.recovery_rate,
            "execution_time_seconds": self.execution_time_seconds,
            "output_metrics": self.output_metrics,
            "overall_score": self.overall_score(),
        }


@dataclass
class AgenticEvalResult:
    """Result of running an evaluation case."""

    eval_case: EvalCase
    metrics: AgenticEvalMetrics
    agent_name: str

    # Raw outputs
    final_state: AnalysisState | None = None
    agent_traces: list[dict[str, Any]] = field(default_factory=list)
    tool_call_history: list[dict[str, Any]] = field(default_factory=list)

    # Evaluation details
    passed: bool = False
    failure_reason: str | None = None
    evaluator_notes: str | None = None

    # Metadata
    timestamp: datetime = field(default_factory=datetime.utcnow)
    llm_model: str = "claude"

    def summary(self) -> str:
        """Generate human-readable summary."""
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.eval_case.name} ({self.eval_case.difficulty.value})\n"
            f"  Score: {self.metrics.overall_score():.1f}/100\n"
            f"  Steps: {self.metrics.steps_taken}"
            f"{f' (optimal: {self.metrics.steps_optimal})' if self.metrics.steps_optimal else ''}\n"
            f"  Time: {self.metrics.execution_time_seconds:.1f}s\n"
            f"  Task Correct: {self.metrics.task_correct}\n"
            f"{f'  Failure: {self.failure_reason}' if self.failure_reason else ''}"
        )


@dataclass
class EvalSuite:
    """A collection of evaluation cases for an agent."""

    name: str
    agent_name: str
    description: str
    cases: list[EvalCase] = field(default_factory=list)

    def add_case(self, case: EvalCase) -> None:
        """Add an evaluation case."""
        self.cases.append(case)

    def get_cases_by_difficulty(self, difficulty: EvalDifficulty) -> list[EvalCase]:
        """Get cases by difficulty level."""
        return [c for c in self.cases if c.difficulty == difficulty]

    def get_cases_by_category(self, category: EvalCategory) -> list[EvalCase]:
        """Get cases by category."""
        return [c for c in self.cases if c.category == category]


class AgenticEvaluator(ABC):
    """Base class for agent evaluators.

    Subclasses implement agent-specific evaluation logic.
    """

    def __init__(self, use_llm_judge: bool = True):
        """Initialize evaluator.

        Args:
            use_llm_judge: Whether to use LLM to judge reasoning quality
        """
        self.use_llm_judge = use_llm_judge
        self.logger = get_logger(f"evaluator.{self.agent_name}")

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Name of the agent being evaluated."""
        pass

    @abstractmethod
    def create_agent(self) -> Any:
        """Create an instance of the agent to evaluate."""
        pass

    @abstractmethod
    def get_eval_suite(self) -> EvalSuite:
        """Get the evaluation suite for this agent."""
        pass

    @abstractmethod
    def check_task_correctness(
        self,
        case: EvalCase,
        final_state: AnalysisState,
    ) -> tuple[bool, dict[str, float]]:
        """Check if agent output is correct.

        Args:
            case: The evaluation case
            final_state: The final state after agent execution

        Returns:
            Tuple of (is_correct, output_metrics)
        """
        pass

    async def evaluate_case(self, case: EvalCase) -> AgenticEvalResult:
        """Run evaluation on a single case.

        Args:
            case: The evaluation case to run

        Returns:
            Evaluation result
        """
        self.logger.info(
            "evaluating_case",
            case=case.name,
            difficulty=case.difficulty.value,
            category=case.category.value,
        )

        metrics = AgenticEvalMetrics()
        start_time = time.time()

        try:
            # Create agent and state
            agent = self.create_agent()
            state = case.create_state()

            # Run agent with timeout
            try:
                final_state = await asyncio.wait_for(
                    agent.execute_with_tracing(state),
                    timeout=case.timeout_seconds,
                )
                metrics.task_completed = True
            except TimeoutError:
                self.logger.warning("case_timeout", case=case.name)
                final_state = state
                metrics.task_completed = False

            # Collect metrics
            metrics.execution_time_seconds = time.time() - start_time

            # Get tool calls from agent traces
            if hasattr(agent, '_step_traces'):
                metrics.tools_called = [
                    t.get('tool_name', t.get('action', 'unknown'))
                    for t in agent._step_traces
                    if t.get('tool_name') or t.get('action')
                ]
                metrics.steps_taken = len(agent._step_traces)

            # Check expected tools
            if case.expected_tool_calls:
                called_set = set(metrics.tools_called)
                expected_set = set(case.expected_tool_calls)
                overlap = len(called_set & expected_set)
                metrics.expected_tools_called = overlap / len(expected_set) if expected_set else 1.0
                metrics.unnecessary_tool_calls = len(called_set - expected_set)

            # Check efficiency
            if case.max_expected_steps:
                metrics.steps_optimal = case.max_expected_steps
                if metrics.steps_taken > 0:
                    metrics.efficiency_score = min(1.0, case.max_expected_steps / metrics.steps_taken)

            # Check correctness
            is_correct, output_metrics = self.check_task_correctness(case, final_state)
            metrics.task_correct = is_correct
            metrics.output_metrics = output_metrics

            # LLM-based reasoning evaluation
            if self.use_llm_judge and hasattr(agent, '_step_traces'):
                reasoning_scores = await self._evaluate_reasoning(agent._step_traces)
                metrics.reasoning_clarity = reasoning_scores.get('clarity')
                metrics.reasoning_soundness = reasoning_scores.get('soundness')
                metrics.reasoning_completeness = reasoning_scores.get('completeness')

            # Determine pass/fail
            passed = metrics.task_completed and metrics.task_correct
            if case.max_expected_steps and metrics.steps_taken > case.max_expected_steps * 2:
                passed = False  # Too inefficient

            failure_reason = None
            if not passed:
                if not metrics.task_completed:
                    failure_reason = "Task did not complete"
                elif not metrics.task_correct:
                    failure_reason = f"Incorrect output: {output_metrics}"
                else:
                    failure_reason = f"Too many steps: {metrics.steps_taken}"

            return AgenticEvalResult(
                eval_case=case,
                metrics=metrics,
                agent_name=self.agent_name,
                final_state=final_state,
                agent_traces=getattr(agent, '_step_traces', []),
                passed=passed,
                failure_reason=failure_reason,
            )

        except Exception as e:
            self.logger.error("case_error", case=case.name, error=str(e))
            metrics.execution_time_seconds = time.time() - start_time

            return AgenticEvalResult(
                eval_case=case,
                metrics=metrics,
                agent_name=self.agent_name,
                passed=False,
                failure_reason=f"Error: {str(e)}",
            )

    async def evaluate_suite(
        self,
        suite: EvalSuite | None = None,
        difficulty_filter: EvalDifficulty | None = None,
        category_filter: EvalCategory | None = None,
    ) -> list[AgenticEvalResult]:
        """Run evaluation on entire suite.

        Args:
            suite: Evaluation suite (uses default if None)
            difficulty_filter: Only run cases of this difficulty
            category_filter: Only run cases of this category

        Returns:
            List of evaluation results
        """
        suite = suite or self.get_eval_suite()
        cases = suite.cases

        if difficulty_filter:
            cases = [c for c in cases if c.difficulty == difficulty_filter]
        if category_filter:
            cases = [c for c in cases if c.category == category_filter]

        self.logger.info(
            "evaluating_suite",
            suite=suite.name,
            n_cases=len(cases),
        )

        results = []
        for case in cases:
            result = await self.evaluate_case(case)
            results.append(result)

            # Log progress
            self.logger.info(
                "case_complete",
                case=case.name,
                passed=result.passed,
                score=result.metrics.overall_score(),
            )

        return results

    async def _evaluate_reasoning(
        self,
        traces: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Use LLM to evaluate reasoning quality.

        Args:
            traces: Agent step traces containing reasoning

        Returns:
            Dict with reasoning scores
        """
        try:
            from src.llm.client import get_llm_client

            client = get_llm_client()

            # Format traces for evaluation
            trace_text = "\n".join([
                f"Step {i+1}: {t.get('reasoning', t.get('observation', 'N/A'))}"
                for i, t in enumerate(traces[:10])  # Limit to first 10 steps
            ])

            prompt = f"""Evaluate the following agent reasoning trace on three dimensions.
Rate each dimension from 1 (poor) to 5 (excellent).

REASONING TRACE:
{trace_text}

EVALUATION CRITERIA:
1. Clarity (1-5): Is the reasoning clear and easy to follow?
2. Soundness (1-5): Is the reasoning logically valid and well-justified?
3. Completeness (1-5): Does the reasoning cover all important aspects?

Respond with JSON only:
{{"clarity": <1-5>, "soundness": <1-5>, "completeness": <1-5>}}
"""

            result = await client.generate(
                prompt=prompt,
                system_instruction="You are an expert evaluator of AI agent reasoning. Evaluate objectively.",
            )

            # Parse response
            import json
            content = result.get("content", [])
            response_text = ""
            for block in content:
                if block.get("type") == "text":
                    response_text += block.get("text", "")

            # Clean up JSON
            response_text = response_text.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            scores = json.loads(response_text.strip())
            return {
                "clarity": float(scores.get("clarity", 3)),
                "soundness": float(scores.get("soundness", 3)),
                "completeness": float(scores.get("completeness", 3)),
            }

        except Exception as e:
            self.logger.warning("reasoning_eval_failed", error=str(e))
            return {}


def generate_summary_report(results: list[AgenticEvalResult]) -> str:
    """Generate a summary report from evaluation results.

    Args:
        results: List of evaluation results

    Returns:
        Formatted summary report
    """
    total = len(results)
    passed = sum(1 for r in results if r.passed)

    by_difficulty = {}
    by_category = {}

    for r in results:
        diff = r.eval_case.difficulty.value
        cat = r.eval_case.category.value

        if diff not in by_difficulty:
            by_difficulty[diff] = {"total": 0, "passed": 0}
        by_difficulty[diff]["total"] += 1
        if r.passed:
            by_difficulty[diff]["passed"] += 1

        if cat not in by_category:
            by_category[cat] = {"total": 0, "passed": 0}
        by_category[cat]["total"] += 1
        if r.passed:
            by_category[cat]["passed"] += 1

    scores = [r.metrics.overall_score() for r in results]

    report = f"""
{'=' * 70}
AGENTIC EVALUATION REPORT
{'=' * 70}

Agent: {results[0].agent_name if results else 'N/A'}
Total Cases: {total}
Passed: {passed} ({100*passed/total:.1f}%)
Average Score: {np.mean(scores):.1f}/100

BY DIFFICULTY:
"""

    for diff, stats in sorted(by_difficulty.items()):
        pct = 100 * stats["passed"] / stats["total"] if stats["total"] > 0 else 0
        report += f"  {diff:8s}: {stats['passed']}/{stats['total']} ({pct:.0f}%)\n"

    report += "\nBY CATEGORY:\n"

    for cat, stats in sorted(by_category.items()):
        pct = 100 * stats["passed"] / stats["total"] if stats["total"] > 0 else 0
        report += f"  {cat:20s}: {stats['passed']}/{stats['total']} ({pct:.0f}%)\n"

    report += f"\n{'=' * 70}\nDETAILED RESULTS:\n{'=' * 70}\n"

    for r in results:
        report += r.summary() + "\n"

    return report
