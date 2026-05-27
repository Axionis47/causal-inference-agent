"""Tool registry for the domain knowledge agent.

Each tool module exports a SCHEMA dict (name, description, parameters) and an
async handle(agent, state, **kwargs) function. The agent's __init__ iterates
MODULES in this list and registers each one.
"""

from . import (
    flag_uncertainty,
    get_tags,
    hypothesize,
    investigate_column,
    list_columns,
    mark_immutable,
    read_description,
    revise_hypothesis,
    search_metadata,
    set_temporal_ordering,
)

MODULES = [
    read_description,
    list_columns,
    investigate_column,
    search_metadata,
    get_tags,
    hypothesize,
    revise_hypothesis,
    set_temporal_ordering,
    mark_immutable,
    flag_uncertainty,
]
