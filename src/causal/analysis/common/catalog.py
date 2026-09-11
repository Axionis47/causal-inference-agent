"""Exact lookup in the four installed method definitions; no search service."""
from importlib import import_module
from types import ModuleType

METHODS = ("randomized", "aipw", "did", "rdd")


def method_module(method: str) -> ModuleType:
    if method not in METHODS:
        raise ValueError(f"Unsupported method {method!r}; available methods: {', '.join(METHODS)}")
    return import_module(f"causal.analysis.methods.{method}.specification")


def diagnostics_module(method: str) -> ModuleType:
    method_module(method)
    return import_module(f"causal.analysis.methods.{method}.diagnostic_catalog")
