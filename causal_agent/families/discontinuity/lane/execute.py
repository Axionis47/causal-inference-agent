"""Execution reads only a frozen design and its canonical table. It cannot redesign."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from causal_agent.common import trace
from causal_agent.common.contracts import Estimate
from causal_agent.families.discontinuity.lane.contracts import Design
from causal_agent.families.discontinuity.lane.engine import fit as F
from causal_agent.families.discontinuity.lane.engine import postchecks
from causal_agent.families.discontinuity.lane.engine.shape import covcol
from causal_agent.families.discontinuity.lane.knowledge import load_falsifications, load_sensitivities


def _attempt(d, estimates):
    errors = [e.error for e in estimates if e.error]
    return {
        "design_id": d.id,
        "endpoint_id": d.endpoint_id,
        "context_id": d.context_id,
        "target": d.target,
        "estimator": d.estimator,
        "inference": d.inference,
        "table_fingerprint": d.table_fingerprint,
        "row_ids": d.rows,
        "status": "failed" if errors else "succeeded",
        "errors": errors,
        "estimates": [e.model_dump() for e in estimates],
    }


def _failure(d, reason):
    estimates = [Estimate(design_id=d.id, endpoint_id=d.endpoint_id, target=d.target, contrast=d.contrast.key, method=d.estimator, error=reason)]
    return {
        "estimates": estimates,
        "execution_attempts": [_attempt(d, estimates)],
        "candidate_history": [{"stage": "execute", "candidate_id": d.id, "execution_status": "failed", "errors": [reason]}],
    }


def execute(task):
    d = Design.model_validate(task["design"])
    d.assert_integrity()
    path = Path(task["canon_path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != d.table_fingerprint:
        return _failure(d, "canonical table changed after freeze")
    table = pd.read_csv(path)
    table = table[table.row.isin(d.rows)].copy()
    if set(table.row.astype(int)) != set(d.rows):
        return _failure(d, "frozen row set does not match the canonical table")
    with trace.span("estimate", inputs={"design": d.model_dump(), "rows": len(table)}):
        primary = postchecks.refit(d, table)
    estimate = F.to_estimate(primary, d.contrast.key, d.estimator, d.estimand).model_copy(
        update={"design_id": d.id, "endpoint_id": d.endpoint_id, "target": d.target}
    )
    estimates = [estimate]
    falsifications = []
    sensitivities = []
    if not primary.error:
        for entry in load_falsifications():
            if entry.name in d.falsifications:
                with trace.span(f"falsification:{entry.name}", inputs={"design_id": d.id}):
                    falsifications.append(postchecks.falsify(d, table, entry))
        for entry in load_sensitivities():
            if entry.name in d.sensitivities:
                with trace.span(f"sensitivity:{entry.name}", inputs={"design_id": d.id}):
                    sensitivities.append(postchecks.sensitivity(d, table, entry))
        for modifier in d.modifiers:
            col = covcol(modifier)
            if col not in table:
                continue
            levels = sorted(table[col].dropna().unique())
            if len(levels) > 4:
                continue
            for level in levels:
                sub = table[table[col] == level]
                fitted = postchecks.refit(d, sub)
                estimates.append(
                    F.to_estimate(fitted, d.contrast.key, d.estimator, d.estimand, secondary=True).model_copy(
                        update={
                            "design_id": d.id,
                            "endpoint_id": d.endpoint_id,
                            "target": {**d.target, "modifier": modifier, "level": str(level)},
                            "modifier": modifier,
                            "level": str(level),
                        }
                    )
                )
    return {
        "estimates": estimates,
        "falsifications": falsifications,
        "sensitivities": sensitivities,
        "execution_attempts": [_attempt(d, estimates)],
        "candidate_history": [
            {
                "stage": "execute",
                "candidate_id": d.id,
                "execution_status": "failed" if estimate.error else "succeeded",
                "estimate_addresses": [e.tag for e in estimates],
            }
        ],
        "reasoning_history": [{"node": "execute", "design_id": d.id, "status": "failed" if estimate.error else "succeeded"}],
    }
