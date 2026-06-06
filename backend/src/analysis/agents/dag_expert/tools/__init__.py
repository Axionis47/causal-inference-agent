"""Tool registration index for the dag_expert agent."""

from . import (
    analyze_domain,
    classify_variable_role,
    fuse_and_validate,
    get_adjustment_set,
    get_discovery_edges,
    mark_forbidden_edge,
    propose_edge,
)

MODULES = [
    analyze_domain,
    classify_variable_role,
    propose_edge,
    mark_forbidden_edge,
    get_discovery_edges,
    fuse_and_validate,
    get_adjustment_set,
]

__all__ = ["MODULES"]
