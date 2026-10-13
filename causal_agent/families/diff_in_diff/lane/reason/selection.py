"""Deterministic pre-effect lead selection over sealed targets and sourced assumptions."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from causal_agent.common.contracts import Handoff
from causal_agent.families.diff_in_diff.design import DidDesign
from causal_agent.families.diff_in_diff.lane.contracts import Design
from causal_agent.families.diff_in_diff.lane.knowledge import ENDPOINTS, estimator, load_checks
from causal_agent.families.diff_in_diff.lane.shape import period_label
from causal_agent.families.diff_in_diff.lane.targets import clean_projection_support, target_groups
from causal_agent.lane.ledger import Ledger


def _identity(units: set[str]) -> dict:
    return {"units": len(units), "unit_set_sha256": hashlib.sha256(json.dumps(sorted(units)).encode()).hexdigest()}


def select_lead(handoff: Handoff, designs: dict[str, Design], ledger: Ledger, panel: pd.DataFrame) -> dict:
    """Rank using only assignment/time metadata, target masks, checks and assumption provenance."""
    block = handoff.design
    assert isinstance(block, DidDesign)
    if block.target_weighting is None:
        raise ValueError("lead selection requires the admitted averaging policy")
    metadata = panel[["unit", "time", "time_index", "cohort", "rel_time", "treat", "row_id"]].copy()
    prefix = block.prefix_scope_accepted and not (metadata.cohort == 0).any()
    requested_support = "pre_last_adoption_prefix" if prefix else "admitted_window"
    if prefix:
        last = int(metadata.cohort.max())
        metadata = metadata.loc[metadata.time_index < last].copy()
        metadata.loc[metadata.cohort == last, ["cohort", "treat"]] = 0
        metadata.loc[metadata.cohort == 0, "rel_time"] = -1
    labels = {
        label: str(int(row.time_index))
        for row in metadata[["time", "time_index"]].drop_duplicates().itertuples()
        for label in {str(row.time), period_label(row.time)}
    }
    requested = {
        "aggregation": block.aggregation,
        "population": "treated",
        "horizons": block.horizons or sorted(int(x) for x in metadata.loc[metadata.treat == 1, "rel_time"].unique() if x >= 0),
        "cohorts": [labels[str(label)] for label in block.cohorts],
        "support": requested_support,
        "weighting": block.target_weighting,
    }
    expected = {label: set(metadata.loc[mask, "unit"].astype(str)) for label, mask in target_groups(metadata, requested)}
    if block.target_weighting == "clean_adopter_projection":
        if not block.horizons:
            requested["horizons"] = [h for h in requested["horizons"] if h <= load_checks()["projection"]["max_horizon"]]
        # Clean adopter populations are defined by complete declared windows, not observed outcome values.
        clean = metadata.assign(y=0.0)
        expected = {}
        for label, _mask in target_groups(metadata, requested):
            hs = [int(label.split(":")[1])] if label.startswith("horizon:") else requested["horizons"]
            units, _weights = clean_projection_support(clean, hs)
            expected[label] = set(units)
    endpoint_ranks = {endpoint.id: i for i, endpoint in enumerate(ENDPOINTS.endpoints)}
    ranking = []
    for design in designs.values():
        target = design.target
        actual = {label: set(component.get("included_units", [])) for label, component in target.get("components", {}).items()}
        component_penalty = len(set(expected) ^ set(actual))
        unit_penalty = sum(len(units ^ actual.get(label, set())) for label, units in expected.items())
        summary_penalty = int(target.get("aggregation") != requested["aggregation"])
        population_penalty = int(target.get("population") != "treated")
        horizon_penalty = int(sorted(target.get("horizons", [])) != sorted(requested["horizons"]))
        cohort_penalty = int(sorted(target.get("cohorts", [])) != sorted(requested["cohorts"]))
        scope_penalty = int(target.get("support") != requested_support)
        paired_equivalence = (
            design.endpoint_id == "paired_two_wave_att"
            and block.target_weighting == "treated_cell_shares"
            and bool(design.weights)
            and all(abs(weight - 1 / len(design.weights)) < 1e-8 for weight in design.weights.values())
        )
        weighting_penalty = int(target.get("weighting") != block.target_weighting and not paired_equivalence)
        assumptions = [ledger.entries.get(address) for address in design.assumptions]
        assumption_penalty = sum(entry is None or entry.status not in {"confirmed", "supported"} or entry.value is None for entry in assumptions)
        contested_penalty = sum(entry is not None and entry.status in {"contested", "rejected"} for entry in assumptions)
        flags = design.checks.flags
        support_penalty = [sum(check.level == "hard" for check in flags), sum(check.level == "soft" for check in flags)]
        target_penalties = [
            summary_penalty,
            population_penalty,
            component_penalty,
            unit_penalty,
            horizon_penalty,
            cohort_penalty,
            scope_penalty,
            weighting_penalty,
        ]
        rank = [
            *target_penalties,
            contested_penalty,
            assumption_penalty,
            *support_penalty,
            estimator(design.estimator).rank,
            endpoint_ranks[design.endpoint_id],
            design.id,
        ]
        ranking.append(
            {
                "design_id": design.id,
                "endpoint_id": design.endpoint_id,
                "rank": rank,
                "target_aligned": not any(target_penalties),
                "target_summary": {key: target.get(key) for key in ["aggregation", "population", "horizons", "cohorts", "support", "weighting"]},
                "component_support": {label: _identity(units) for label, units in sorted(actual.items())},
                "target_penalties": dict(
                    zip(
                        ["summary", "population", "component_labels", "unit_support", "horizons", "cohorts", "scope", "weighting"],
                        target_penalties,
                        strict=True,
                    )
                ),
                "assumption_support": [
                    {"address": address, "status": entry.status if entry else "missing", "evidence": [e.model_dump() for e in entry.evidence] if entry else []}
                    for address, entry in zip(design.assumptions, assumptions, strict=True)
                ],
                "support_flags": [check.model_dump() for check in flags],
                "catalogue_rank": {"estimator": estimator(design.estimator).rank, "endpoint": endpoint_ranks[design.endpoint_id]},
                "cites": [f"design:{design.id}.target", f"design:{design.id}.support", *design.assumptions, *[check.address for check in flags]],
            }
        )
    ranking.sort(key=lambda item: tuple(item["rank"]))
    aligned = [item for item in ranking if item["target_aligned"]]
    return {
        "policy": "Exact admitted summary, population, component support, horizons/cohorts, accepted scope and averaging policy; then supported assumptions, support flags, estimator/endpoint catalogue ranks and stable design ID. Different estimands remain companions.",
        "stage": "freeze_design",
        "effect_blind": True,
        "chosen_id": aligned[0]["design_id"] if aligned else None,
        "requested": {**requested, "component_support": {label: _identity(units) for label, units in sorted(expected.items())}},
        "requested_target_identified": bool(aligned),
        "reason": "The first exactly aligned frozen design wins the recorded pre-effect ranking; different supports or weights remain companions."
        if aligned
        else "No frozen design identifies the exact requested target and averaging policy. Different-target designs remain companions, with no replacement lead.",
        "ranking": ranking,
        "cites": ["claim:target.population", "claim:target.weighting", "claim:grain.row_is", "claim:assignment.treated_level"],
    }
