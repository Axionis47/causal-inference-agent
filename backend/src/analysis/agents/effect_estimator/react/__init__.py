"""ReAct variant of the effect estimator.

The react orchestrator swaps EffectEstimatorReActAgent in for the
default EffectEstimatorAgent under the same `effect_estimator` key.
See orchestrator/react/specialists.py for the swap mechanism.
"""

from .agent import EffectEstimatorReActAgent

__all__ = ["EffectEstimatorReActAgent"]
