"""Causal inference methods and utilities."""

from .dag.discovery import CausalDiscovery
from .estimators.effect_estimator import EffectEstimatorEngine
from .methods.base import BaseCausalMethod, MethodResult
from .methods.causal_forest import CausalForestMethod
from .methods.did import DifferenceInDifferencesMethod
from .methods.double_ml import DoubleMLMethod
from .methods.iv import InstrumentalVariablesMethod
from .methods.metalearners import SLearner, TLearner, XLearner
from .methods.ols import OLSMethod
from .methods.propensity import AIPWMethod, IPWMethod, PSMMethod
from .methods.rdd import RegressionDiscontinuityMethod

__all__ = [
    # Base
    "BaseCausalMethod",
    "MethodResult",
    # Methods
    "OLSMethod",
    "PSMMethod",
    "IPWMethod",
    "AIPWMethod",
    "DifferenceInDifferencesMethod",
    "InstrumentalVariablesMethod",
    "RegressionDiscontinuityMethod",
    "SLearner",
    "TLearner",
    "XLearner",
    "CausalForestMethod",
    "DoubleMLMethod",
    # Discovery
    "CausalDiscovery",
    # Engine
    "EffectEstimatorEngine",
]
