"""Estimation method implementations -- delegates to src.causal.methods via EffectEstimatorEngine.

Backward-compatible: run_method() has the same signature as before.
The actual implementations live in src/causal/methods/ (BaseCausalMethod subclasses).

Note: Imports are lazy to avoid circular import through src.agents → specialists
→ effect_estimation → src.causal → src.agents.base → src.agents.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from src.agents.base.state import AnalysisState, TreatmentEffectResult

_engine = None


def _get_engine():
    """Lazily create the EffectEstimatorEngine to avoid circular imports."""
    global _engine
    if _engine is None:
        from src.causal.estimators.effect_estimator import EffectEstimatorEngine
        _engine = EffectEstimatorEngine()
    return _engine


def run_method(
    method: str,
    treatment_var: str,
    outcome_var: str,
    covariates: list[str],
    df: pd.DataFrame,
    current_state: AnalysisState | None = None,
) -> TreatmentEffectResult | None:
    """Run a specific estimation method.

    Delegates to EffectEstimatorEngine.run_method_safe() which handles data prep,
    sample-size gating, state-dependent kwargs for DID/IV/RDD, and MethodResult
    to TreatmentEffectResult conversion.
    """
    return _get_engine().run_method_safe(
        method=method,
        df=df,
        treatment_col=treatment_var,
        outcome_col=outcome_var,
        covariates=covariates,
        current_state=current_state,
    )
