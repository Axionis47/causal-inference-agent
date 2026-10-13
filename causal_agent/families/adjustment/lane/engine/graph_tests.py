"""Post-freeze independence diagnostics, never a design-selection signal."""

from __future__ import annotations

import itertools
import json

import networkx as nx
import numpy as np
import pandas as pd

from causal_agent.common.contracts import Falsification
from causal_agent.families.adjustment.lane.contracts import Design
from causal_agent.families.adjustment.lane.engine.model import TREATED


def run(table: pd.DataFrame, d: Design, contrast: str) -> Falsification:
    from dowhy.gcm import independence_test
    from statsmodels.stats.multitest import multipletests

    g = nx.relabel_nodes(d.graph.to_networkx(), {d.graph.treatment: TREATED})
    observed = sorted(set(table.columns) & set(g.nodes))
    constraints = []
    for a, b in itertools.combinations(observed, 2):
        others = [c for c in observed if c not in (a, b)]
        for conditioning in [(), *((c,) for c in others)]:
            if nx.is_d_separator(g, {a}, {b}, set(conditioning)):
                constraints.append((a, b, conditioning))
    selected = constraints[:16]
    if not selected:
        return Falsification(
            design_id=d.id,
            contrast=contrast,
            refuter="graph_independences",
            detail="no independences with zero or one conditioning variable were available; no validity claim",
            passed=None,
        )
    sample = table.sample(n=min(300, len(table)), random_state=d.bootstrap_seed)
    results = []
    failures = []
    for a, b, cond in selected:
        try:
            p = float(
                independence_test(
                    sample[[a]].to_numpy(),
                    sample[[b]].to_numpy(),
                    conditioned_on=sample[list(cond)].to_numpy() if cond else None,
                    method="kernel",
                    bootstrap_n_jobs=1,
                    max_num_samples_run=300,
                )
            )
            if not np.isfinite(p):
                raise ValueError("nonfinite independence p value")
            results.append({"x": a, "y": b, "given": list(cond), "p_value": p})
        except Exception as ex:
            failures.append({"constraint": [a, b, list(cond)], "error": str(ex)[:250]})
    corrected = multipletests([r["p_value"] for r in results], method="holm")[1] if results else []
    for record, p in zip(results, corrected, strict=True):
        record["adjusted_p_value"] = float(p)
    rejected = [r for r in results if r["adjusted_p_value"] < 0.05]
    return Falsification(
        design_id=d.id,
        contrast=contrast,
        refuter="graph_independences",
        p_value=min(corrected) if len(corrected) else None,
        passed=not rejected if results and not failures else None,
        detail=json.dumps(
            {
                "tests": results,
                "failures": failures,
                "rejected_constraints": len(rejected),
                "unexamined_constraints": max(0, len(constraints) - 16),
                "policy": "diagnostic only, no causal-edge attribution or post-outcome redesign",
            }
        ),
    )
