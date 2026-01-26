"""Runner for executing agentic evaluations.

This module provides CLI and programmatic interfaces for running
agent evaluations using Claude API (configured via .env file).
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.logging_config.structured import get_logger, setup_logging

from .base import (
    AgenticEvalResult,
    AgenticEvaluator,
    EvalCategory,
    EvalDifficulty,
    generate_summary_report,
)

logger = get_logger(__name__)


# Registry of evaluators
_evaluators: dict[str, type[AgenticEvaluator]] = {}


def register_evaluator(evaluator_class: type[AgenticEvaluator]) -> type[AgenticEvaluator]:
    """Decorator to register an evaluator class."""
    _evaluators[evaluator_class.__name__] = evaluator_class
    return evaluator_class


def get_evaluator(agent_name: str) -> AgenticEvaluator | None:
    """Get evaluator for a specific agent.

    Args:
        agent_name: Name of the agent (e.g., 'data_profiler', 'eda_agent')

    Returns:
        Evaluator instance or None if not found
    """
    # Import evaluators to ensure registration
    from . import (
        data_profiler_eval,
        eda_agent_eval,
        causal_discovery_eval,
        effect_estimator_eval,
        sensitivity_analyst_eval,
        critique_agent_eval,
        orchestrator_eval,
    )

    agent_to_evaluator = {
        "data_profiler": data_profiler_eval.DataProfilerEvaluator,
        "eda_agent": eda_agent_eval.EDAAgentEvaluator,
        "causal_discovery": causal_discovery_eval.CausalDiscoveryEvaluator,
        "effect_estimator": effect_estimator_eval.EffectEstimatorEvaluator,
        "sensitivity_analyst": sensitivity_analyst_eval.SensitivityAnalystEvaluator,
        "critique": critique_agent_eval.CritiqueAgentEvaluator,
        "orchestrator": orchestrator_eval.OrchestratorEvaluator,
    }

    evaluator_class = agent_to_evaluator.get(agent_name)
    if evaluator_class:
        return evaluator_class()
    return None


async def run_agent_evals(
    agent_name: str,
    difficulty: EvalDifficulty | str | None = None,
    category: EvalCategory | str | None = None,
    use_llm_judge: bool = True,
    save_results: bool = True,
    output_dir: str | None = None,
) -> list[AgenticEvalResult]:
    """Run evaluations for a specific agent.

    Args:
        agent_name: Name of agent to evaluate
        difficulty: Filter by difficulty level
        category: Filter by category
        use_llm_judge: Whether to use LLM for reasoning evaluation
        save_results: Whether to save results to file
        output_dir: Directory for output files

    Returns:
        List of evaluation results
    """
    logger.info("starting_agent_evals", agent=agent_name)

    evaluator = get_evaluator(agent_name)
    if not evaluator:
        raise ValueError(f"No evaluator found for agent: {agent_name}")

    evaluator.use_llm_judge = use_llm_judge

    # Convert string to enum if needed
    if isinstance(difficulty, str):
        difficulty = EvalDifficulty(difficulty)
    if isinstance(category, str):
        category = EvalCategory(category)

    results = await evaluator.evaluate_suite(
        difficulty_filter=difficulty,
        category_filter=category,
    )

    if save_results:
        _save_results(results, agent_name, output_dir)

    return results


async def run_all_evals(
    agents: list[str] | None = None,
    difficulty: EvalDifficulty | str | None = None,
    use_llm_judge: bool = True,
    save_results: bool = True,
    output_dir: str | None = None,
) -> dict[str, list[AgenticEvalResult]]:
    """Run evaluations for all agents.

    Args:
        agents: List of agent names (None for all)
        difficulty: Filter by difficulty level
        use_llm_judge: Whether to use LLM for reasoning evaluation
        save_results: Whether to save results to file
        output_dir: Directory for output files

    Returns:
        Dict mapping agent name to results
    """
    all_agents = agents or [
        "data_profiler",
        "eda_agent",
        "causal_discovery",
        "effect_estimator",
        "sensitivity_analyst",
        "critique",
        "orchestrator",
    ]

    all_results = {}

    for agent_name in all_agents:
        try:
            logger.info("evaluating_agent", agent=agent_name)
            results = await run_agent_evals(
                agent_name=agent_name,
                difficulty=difficulty,
                use_llm_judge=use_llm_judge,
                save_results=save_results,
                output_dir=output_dir,
            )
            all_results[agent_name] = results
        except Exception as e:
            logger.error("agent_eval_failed", agent=agent_name, error=str(e))
            all_results[agent_name] = []

    # Generate combined report
    if save_results:
        _save_combined_report(all_results, output_dir)

    return all_results


def _save_results(
    results: list[AgenticEvalResult],
    agent_name: str,
    output_dir: str | None = None,
) -> Path:
    """Save evaluation results to file."""
    output_dir = Path(output_dir or "eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{agent_name}_eval_{timestamp}.json"
    filepath = output_dir / filename

    # Convert results to serializable format
    data = {
        "agent": agent_name,
        "timestamp": timestamp,
        "total_cases": len(results),
        "passed": sum(1 for r in results if r.passed),
        "results": [
            {
                "case_name": r.eval_case.name,
                "difficulty": r.eval_case.difficulty.value,
                "category": r.eval_case.category.value,
                "passed": r.passed,
                "failure_reason": r.failure_reason,
                "metrics": r.metrics.to_dict(),
            }
            for r in results
        ],
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    # Also save human-readable report
    report_path = output_dir / f"{agent_name}_eval_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write(generate_summary_report(results))

    logger.info("results_saved", path=str(filepath))
    return filepath


def _save_combined_report(
    all_results: dict[str, list[AgenticEvalResult]],
    output_dir: str | None = None,
) -> Path:
    """Save combined report for all agents."""
    output_dir = Path(output_dir or "eval_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"all_agents_eval_{timestamp}.txt"

    report = f"""
{'=' * 80}
COMBINED AGENTIC EVALUATION REPORT
Generated: {datetime.now().isoformat()}
{'=' * 80}

SUMMARY BY AGENT:
{'-' * 80}
"""

    for agent_name, results in all_results.items():
        if not results:
            report += f"\n{agent_name}: No results (evaluation failed)\n"
            continue

        total = len(results)
        passed = sum(1 for r in results if r.passed)
        avg_score = sum(r.metrics.overall_score() for r in results) / total if total > 0 else 0

        report += f"""
{agent_name}:
  Cases: {total}
  Passed: {passed}/{total} ({100*passed/total:.1f}%)
  Avg Score: {avg_score:.1f}/100
"""

    # Detailed reports per agent
    for agent_name, results in all_results.items():
        if results:
            report += f"\n\n{'=' * 80}\n"
            report += generate_summary_report(results)

    with open(filepath, "w") as f:
        f.write(report)

    logger.info("combined_report_saved", path=str(filepath))
    return filepath


# CLI entry point
def main():
    """CLI entry point for running evaluations."""
    import argparse

    setup_logging()

    parser = argparse.ArgumentParser(
        description="Run agentic evaluations for causal inference agents"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Specific agent to evaluate (e.g., data_profiler, eda_agent)",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "medium", "hard"],
        default=None,
        help="Filter by difficulty level",
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=[c.value for c in EvalCategory],
        default=None,
        help="Filter by evaluation category",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Disable LLM-based reasoning evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory for output files",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all agent evaluations",
    )

    args = parser.parse_args()

    async def run():
        if args.all or args.agent is None:
            agents = [args.agent] if args.agent else None
            results = await run_all_evals(
                agents=agents,
                difficulty=args.difficulty,
                use_llm_judge=not args.no_llm_judge,
                output_dir=args.output_dir,
            )

            # Print summary
            total_passed = sum(
                sum(1 for r in res if r.passed)
                for res in results.values()
            )
            total_cases = sum(len(res) for res in results.values())
            print(f"\nTotal: {total_passed}/{total_cases} passed")

        else:
            results = await run_agent_evals(
                agent_name=args.agent,
                difficulty=args.difficulty,
                category=args.category,
                use_llm_judge=not args.no_llm_judge,
                output_dir=args.output_dir,
            )

            # Print summary
            print(generate_summary_report(results))

    asyncio.run(run())


if __name__ == "__main__":
    main()
