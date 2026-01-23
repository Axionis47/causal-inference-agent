"""Causal inference methods and utilities."""

from .methods.base import BaseCausalMethod, MethodResult
from .methods.ols import OLSMethod
from .methods.propensity import PSMMethod, IPWMethod, AIPWMethod
from .methods.did import DifferenceInDifferencesMethod
from .methods.iv import InstrumentalVariablesMethod
from .methods.rdd import RegressionDiscontinuityMethod
from .methods.metalearners import SLearner, TLearner, XLearner
from .methods.causal_forest import CausalForestMethod
from .methods.double_ml import DoubleMLMethod
from .dag.discovery import CausalDiscovery
from .estimators.effect_estimator import EffectEstimatorEngine

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
