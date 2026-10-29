"""Sandboxed diagnostic code for choosing among already eligible estimators."""

from __future__ import annotations

from pathlib import Path

from causal_agent.common.addresses import key
from causal_agent.explore.diagnostics import DiagnosticTools, tools_for
from causal_agent.lane import nodes as L


def diagnostic_tools(state, candidates) -> DiagnosticTools:
    """Use the admitted analysis rows and only treatment plus identified adjustment columns."""
    handoff = state["handoff"]
    treatment, outcome = key(handoff.treatment), key(handoff.outcome)
    covariates = {column for candidate in candidates for column in candidate.estimand.adjustment_set}
    columns = [treatment, *sorted(covariates - {treatment, outcome})]
    table = L.table(state)
    return tools_for(table[columns], Path(state["run_dir"]) / "selection", dropped_name=outcome)
