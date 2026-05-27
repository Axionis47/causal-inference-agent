"""EDA tool registration index.

Each module exposes a SCHEMA dict and an async handle(agent, state, **kwargs)
function. The agent iterates MODULES at init time to wire them into the ReAct
tool registry, so adding a tool means dropping a new file here and appending
it to MODULES.
"""

from . import (
    analyze_variable,
    check_balance,
    check_missing,
    compute_correlations,
    compute_vif,
    detect_outliers,
    finalize,
    get_overview,
)

MODULES = [
    get_overview,
    analyze_variable,
    detect_outliers,
    compute_correlations,
    compute_vif,
    check_balance,
    check_missing,
    finalize,
]

__all__ = ["MODULES"]
