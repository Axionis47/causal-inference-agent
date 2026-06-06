"""Tool registration index for the causal_discovery agent."""

from . import (
    compare_algorithms,
    finalize_discovery,
    get_data_characteristics,
    inspect_graph,
    run_discovery_algorithm,
    validate_graph,
)

MODULES = [
    get_data_characteristics,
    run_discovery_algorithm,
    inspect_graph,
    validate_graph,
    compare_algorithms,
    finalize_discovery,
]

__all__ = ["MODULES"]
