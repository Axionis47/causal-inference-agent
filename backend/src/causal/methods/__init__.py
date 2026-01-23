"""Causal inference methods."""

from .base import BaseCausalMethod, MethodResult
from .causal_forest import CausalForestMethod
from .did import DifferenceInDifferencesMethod
from .double_ml import DoubleMLMethod
from .iv import InstrumentalVariablesMethod
from .metalearners import SLearner, TLearner, XLearner
from .ols import OLSMethod
from .propensity import AIPWMethod, IPWMethod, PSMMethod
from .rdd import RegressionDiscontinuityMethod

__all__ = [
    "BaseCausalMethod",
    "MethodResult",
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
]
