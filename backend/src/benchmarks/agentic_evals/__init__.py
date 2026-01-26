"""Agentic evaluation framework for causal inference agents.

This module provides comprehensive evaluation of agent reasoning capabilities,
not just statistical accuracy. Evaluations test:
- Tool selection quality
- Reasoning chain effectiveness
- Task completion efficiency
- Error recovery capability
- Context utilization

Usage:
    # Run all agent evals
    python -m src.benchmarks.agentic_evals.eval_runner --all

    # Run specific agent eval
    python -m src.benchmarks.agentic_evals.eval_runner --agent data_profiler

    # Run with specific difficulty
    python -m src.benchmarks.agentic_evals.eval_runner --agent eda_agent --difficulty easy

    # Programmatic usage
    from src.benchmarks.agentic_evals import run_agent_evals, run_all_evals
    import asyncio

    results = asyncio.run(run_agent_evals("data_profiler"))
"""

from .base import (
    AgenticEvalResult,
    AgenticEvalMetrics,
    AgenticEvaluator,
    EvalCase,
    EvalCategory,
    EvalDifficulty,
    EvalSuite,
    generate_summary_report,
)
from .eval_runner import run_agent_evals, run_all_evals, get_evaluator

# Import individual evaluators for direct access
from .data_profiler_eval import DataProfilerEvaluator
from .eda_agent_eval import EDAAgentEvaluator
from .causal_discovery_eval import CausalDiscoveryEvaluator
from .effect_estimator_eval import EffectEstimatorEvaluator
from .sensitivity_analyst_eval import SensitivityAnalystEvaluator
from .critique_agent_eval import CritiqueAgentEvaluator
from .orchestrator_eval import OrchestratorEvaluator

__all__ = [
    # Base classes
    "AgenticEvalResult",
    "AgenticEvalMetrics",
    "AgenticEvaluator",
    "EvalCase",
    "EvalCategory",
    "EvalDifficulty",
    "EvalSuite",
    "generate_summary_report",
    # Runner functions
    "run_agent_evals",
    "run_all_evals",
    "get_evaluator",
    # Individual evaluators
    "DataProfilerEvaluator",
    "EDAAgentEvaluator",
    "CausalDiscoveryEvaluator",
    "EffectEstimatorEvaluator",
    "SensitivityAnalystEvaluator",
    "CritiqueAgentEvaluator",
    "OrchestratorEvaluator",
]
