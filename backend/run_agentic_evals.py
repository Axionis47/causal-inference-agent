#!/usr/bin/env python3
"""Run agentic evaluations for all causal inference agents.

This script runs the comprehensive agentic evaluation suite that tests
agent reasoning capabilities, not just statistical accuracy.

Usage:
    # Run all evaluations
    python run_agentic_evals.py

    # Run specific agent
    python run_agentic_evals.py --agent data_profiler

    # Run with specific difficulty
    python run_agentic_evals.py --agent eda_agent --difficulty easy

    # Run without LLM reasoning evaluation (faster)
    python run_agentic_evals.py --no-llm-judge

Environment:
    Requires CLAUDE_API_KEY in .env file for LLM-based evaluation.
    Set LLM_PROVIDER=claude in .env (default).
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent / ".env")

from src.benchmarks.agentic_evals import (
    run_agent_evals,
    run_all_evals,
    generate_summary_report,
    EvalDifficulty,
    EvalCategory,
)
from src.logging_config.structured import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run agentic evaluations for causal inference agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all agent evaluations
    python run_agentic_evals.py --all

    # Run specific agent
    python run_agentic_evals.py --agent data_profiler
    python run_agentic_evals.py --agent eda_agent
    python run_agentic_evals.py --agent effect_estimator

    # Filter by difficulty
    python run_agentic_evals.py --agent data_profiler --difficulty easy
    python run_agentic_evals.py --agent data_profiler --difficulty hard

    # Filter by category
    python run_agentic_evals.py --agent effect_estimator --category reasoning_quality

    # Skip LLM-based reasoning evaluation (faster, less comprehensive)
    python run_agentic_evals.py --agent data_profiler --no-llm-judge

Available agents:
    - data_profiler      : Tests data profiling capabilities
    - eda_agent          : Tests exploratory data analysis
    - causal_discovery   : Tests causal structure learning
    - effect_estimator   : Tests treatment effect estimation
    - sensitivity_analyst: Tests sensitivity analysis
    - critique           : Tests quality control and review
    - orchestrator       : Tests pipeline coordination
        """,
    )

    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        choices=[
            "data_profiler",
            "eda_agent",
            "causal_discovery",
            "effect_estimator",
            "sensitivity_analyst",
            "critique",
            "orchestrator",
        ],
        help="Specific agent to evaluate",
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
        help="Disable LLM-based reasoning evaluation (faster but less comprehensive)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory for output files (default: eval_results)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all agent evaluations",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("CAUSAL ORCHESTRATOR - AGENTIC EVALUATION SUITE")
    print("=" * 70)
    print()

    use_llm = not args.no_llm_judge

    if args.all or args.agent is None:
        print(f"Running evaluations for ALL agents...")
        print(f"LLM-based reasoning evaluation: {'Enabled' if use_llm else 'Disabled'}")
        print()

        agents = [args.agent] if args.agent else None
        all_results = await run_all_evals(
            agents=agents,
            difficulty=args.difficulty,
            use_llm_judge=use_llm,
            output_dir=args.output_dir,
        )

        # Print summary
        print()
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)

        total_cases = 0
        total_passed = 0

        for agent_name, results in all_results.items():
            if results:
                passed = sum(1 for r in results if r.passed)
                total = len(results)
                total_cases += total
                total_passed += passed

                avg_score = sum(r.metrics.overall_score() for r in results) / total
                print(f"{agent_name:25s}: {passed:2d}/{total:2d} passed ({100*passed/total:5.1f}%), avg score: {avg_score:5.1f}")
            else:
                print(f"{agent_name:25s}: Failed to run")

        print("-" * 70)
        print(f"{'TOTAL':25s}: {total_passed:2d}/{total_cases:2d} passed ({100*total_passed/total_cases if total_cases > 0 else 0:5.1f}%)")

    else:
        print(f"Running evaluations for: {args.agent}")
        print(f"Difficulty filter: {args.difficulty or 'All'}")
        print(f"Category filter: {args.category or 'All'}")
        print(f"LLM-based reasoning evaluation: {'Enabled' if use_llm else 'Disabled'}")
        print()

        results = await run_agent_evals(
            agent_name=args.agent,
            difficulty=args.difficulty,
            category=args.category,
            use_llm_judge=use_llm,
            output_dir=args.output_dir,
        )

        # Print detailed report
        print(generate_summary_report(results))

    print()
    print(f"Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
