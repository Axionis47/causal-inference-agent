"""Effect Estimation package — decomposed from monolithic effect_estimator.py.

Backward-compatible: `from src.agents.specialists.effect_estimation import EffectEstimatorAgent`
"""

from .agent import EffectEstimatorAgent
from .diagnostics import (
    check_influence_diagnostics,
    check_residual_diagnostics,
    check_specification_diagnostics,
)
from .estimation_methods import run_method
from .method_selector import SampleSizeThresholds, find_closest_column

__all__ = [
    "EffectEstimatorAgent",
    "SampleSizeThresholds",
    "find_closest_column",
    "run_method",
    "check_residual_diagnostics",
    "check_influence_diagnostics",
    "check_specification_diagnostics",
]
