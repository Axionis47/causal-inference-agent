"""API utility functions."""

from src.api.utils.dataset_view import (
    build_from_persisted as build_dataset_view_from_persisted,
)
from src.api.utils.dataset_view import (
    build_from_state as build_dataset_view_from_state,
)
from src.api.utils.results_synthesis import (
    build_data_context,
    calculate_method_consensus,
    generate_executive_summary,
    generate_narrative_summary,
)

__all__ = [
    "build_data_context",
    "build_dataset_view_from_persisted",
    "build_dataset_view_from_state",
    "calculate_method_consensus",
    "generate_executive_summary",
    "generate_narrative_summary",
]
