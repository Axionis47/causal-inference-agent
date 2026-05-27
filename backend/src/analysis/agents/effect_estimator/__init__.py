"""Effect Estimator agent package.

Exports only the typed output model. Agent class is imported directly
from .agent to avoid the base.state -> effect_estimator.agent -> base
cycle.
"""

from .output import TreatmentEffectResult

__all__ = ["TreatmentEffectResult"]
