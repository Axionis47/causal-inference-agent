"""Execution consumes frozen designs and their frozen panel files only."""

from __future__ import annotations

import pandas as pd
from langgraph.types import Command

from causal_agent.common import trace
from causal_agent.families.diff_in_diff.lane.engine import estimators, falsify, sensitivity
from causal_agent.families.diff_in_diff.lane.engine.diagnostics import holm
from causal_agent.families.diff_in_diff.lane.history import event
from causal_agent.families.diff_in_diff.lane.state import SpecialistState
from causal_agent.lane.nodes import stop


def execute(state: SpecialistState) -> Command:
    estimates, falsifications, sensitivities = [], [], []
    candidate_history = []
    candidates = {c.id: c for c in state["candidates"]}
    for d in state["designs"].values():
        try:
            d.assert_integrity()
            panel = pd.read_csv(d.panel_path).set_index("row_id", drop=False)
        except (ValueError, OSError, KeyError) as ex:
            return stop(
                "execute",
                "frozen design input could not be verified",
                [d.id, str(ex)],
                "the exact sealed specification and input panel",
                terminal_kind="execution_failure",
            )
        if tuple(int(x) for x in panel.index) != d.row_ids:
            return stop("execute", "frozen row identity mismatch", [d.id], "the exact frozen input panel", terminal_kind="execution_failure")
        with trace.span(f"execute:{d.id}", inputs={"design": d}) as span:
            fitted = estimators.run(d, panel)
            if d.inference == "inference_unavailable":
                fitted = [
                    e.model_copy(
                        update={
                            "ci_low": None,
                            "ci_high": None,
                            "p_value": None,
                            "p_value_source": "fewer than three independent assignment clusters; inference unavailable",
                        }
                    )
                    for e in fitted
                ]
            elif d.target["aggregation"].endswith("profile"):
                adj = holm([e.p_value for e in fitted])
                fitted = [
                    e.model_copy(update={"p_value": pv, "p_value_source": (e.p_value_source or "inference unavailable") + "; Holm across requested profile"})
                    for e, pv in zip(fitted, adj, strict=True)
                ]
            estimates += fitted
            before = candidates[d.id]
            success = any(e.error is None for e in fitted)
            if not success:
                candidates[d.id] = before.model_copy(update={"status": "failed", "reason": "; ".join(e.error or "" for e in fitted)})
            candidate_history.append(
                event(
                    state,
                    "execute",
                    candidate_id=d.id,
                    before=before.model_dump(),
                    after=candidates[d.id].model_dump(),
                    execution_status="succeeded" if success else "failed",
                    estimates=[e.model_dump() for e in fitted],
                )
            )
            # A failed fit leaves a failed design; it never revisits the reasoning.
            if any(e.error is None for e in fitted):
                falsifications += [falsify.run(d, panel, name) for name in d.falsifications]
                sensitivities += [sensitivity.run(d, panel, name) for name in d.sensitivities]
            span.end(outputs={"estimates": [e.model_dump() for e in fitted]})
    extra = {
        "estimates": estimates,
        "falsifications": falsifications,
        "sensitivities": sensitivities,
        "candidate_history": candidate_history,
        "candidates": list(candidates.values()),
        "reasoning_history": [event(state, "execute", designs=list(state["designs"]))],
    }
    if not any(e.error is None for e in estimates):
        return stop(
            "execute",
            "every frozen design failed to fit",
            [e.error or "" for e in estimates],
            "an executable supported design",
            extra,
            terminal_kind="execution_failure",
        )
    return Command(goto="interpret", update=extra)
