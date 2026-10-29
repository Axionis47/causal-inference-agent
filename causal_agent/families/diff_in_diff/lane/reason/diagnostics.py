"""The probe's outcome-restricted projection, independent of effect values and missingness."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from causal_agent.common.addresses import key
from causal_agent.common.contracts import Handoff
from causal_agent.explore.diagnostics import DiagnosticTools
from causal_agent.families.diff_in_diff.design import DidDesign
from causal_agent.families.diff_in_diff.lane.contracts import ProbeReading
from causal_agent.families.diff_in_diff.lane.shape import CANON
from causal_agent.lane.episode import EpisodeLog


class DiagnosticScopeError(ValueError):
    def __init__(self, column: str, reason: str):
        super().__init__(f"{column}: {reason}")
        self.addresses = [f"col:{column}"]


def validate_metadata(handoff: Handoff, block: DidDesign) -> None:
    """Reject outcome aliases before building canonical rows or exposing any read-only tools."""
    outcomes = {key(c) for c in (handoff.outcome, block.before_column, block.after_column) if c}
    sources = {
        "unit": block.unit,
        "time": block.time,
        "assignment": block.treated_group.get("column"),
        "adoption": block.treated_group.get("cohort_column"),
        "cluster": block.cluster_level,
    }
    for role, source in sources.items():
        if source is None:
            continue
        column = key(source)
        card = handoff.column(column)
        if column in outcomes or card is not None and (card.measures_outcome is True or card.role == "outcome"):
            raise DiagnosticScopeError(column, f"{role} metadata cannot expose an outcome measurement")
        if role != "assignment" and card is not None and card.moved_by_change is True:
            raise DiagnosticScopeError(column, f"{role} identity cannot be treatment-affected")
    reserved = {*CANON, "cluster", "row_id"}
    authoritative = {"cluster": key(block.cluster_level)} if block.cluster_level else {}
    if block.time:
        authoritative["time"] = key(block.time)
        if block.unit:
            authoritative["unit"] = key(block.unit)
    for card in handoff.columns:
        if card.role == "candidate" and card.key in reserved and authoritative.get(card.key) != card.key:
            raise DiagnosticScopeError(card.key, "a candidate collides with canonical diagnostic metadata; resolve its declared source role")


def record_errors(record: ProbeReading, log: EpisodeLog, tools: DiagnosticTools, resolves: Callable[[str], bool]) -> list[str]:
    """Diagnostic findings need not manufacture a new causal ledger operation."""
    cites = record.diagnostic_cites + [e.address for op in record.operations for e in op.evidence]
    return [f"{address}: unresolved diagnostic citation" for address in record.diagnostic_cites if not resolves(address)] + tools.evidence_errors(cites, log)


def diagnostic_table(handoff: Handoff, panel: pd.DataFrame) -> pd.DataFrame:
    """Keep all assignment rows; release outcomes and safe covariates only before any adoption.

    An untreated or not-yet-treated row after the first adoption is still hidden. No outcome
    filtering or candidate support operation may decide which metadata rows survive here.
    """
    block = handoff.design
    assert isinstance(block, DidDesign)
    validate_metadata(handoff, block)
    excluded = {key(c) for c in (handoff.outcome, handoff.treatment, block.before_column, block.after_column) if c}
    metadata = [c for c in [*CANON[1:], "cluster", "row_id"] if c in panel]
    covariates = []
    for card in handoff.columns:
        if card.role != "candidate" or card.key in excluded or card.key in metadata or card.key not in panel:
            continue
        if card.measures_outcome is True or card.moved_by_change is True:
            continue
        timing = card.provenance.get("when")
        unaffected = card.provenance.get("moved_by_change")
        safe = card.when == "before" and timing is not None and timing.status == "confirmed"
        safe |= (
            card.when == "at"
            and timing is not None
            and timing.status == "confirmed"
            and card.moved_by_change is False
            and unaffected is not None
            and unaffected.status == "confirmed"
        )
        if safe:
            covariates.append(card.key)
    projected = panel.loc[:, [*metadata, "y", *covariates]].copy()
    projected.loc[panel.post != 0, ["y", *covariates]] = None
    return projected
