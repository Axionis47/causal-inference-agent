"""Per frozen design and contrast: fit, diagnostics and deterministic fit fallback."""

from __future__ import annotations

from threading import Lock
from typing import Literal

import numpy as np
import pandas as pd
from langgraph.types import Command, Send

from causal_agent.common import trace
from causal_agent.families.adjustment.lane.contracts import Design
from causal_agent.families.adjustment.lane.engine import estimators, falsify, graph_tests, sensitivity
from causal_agent.families.adjustment.lane.state import ContrastTask, SpecialistState
from causal_agent.families.adjustment.lane.support import contrast_table
from causal_agent.lane import nodes as L

_MODEL_RNG_LOCK = Lock()


def fan_out_analyse(state: SpecialistState):
    return [
        Send("analyse", ContrastTask(contrast=c.key, design=d.model_dump(), table_path=state["table_path"]))
        for d in state["designs"].values()
        for c in d.contrasts
    ]


def analyse(task: ContrastTask) -> dict:
    d = Design.model_validate(task["design"])
    contrast = next(c for c in d.contrasts if c.key == task["contrast"])
    sub = contrast_table(pd.read_csv(task["table_path"]), d.graph.treatment, contrast)
    with _MODEL_RNG_LOCK, trace.span("frozen_design_fit", inputs={"design": d.model_dump(), "contrast": contrast.key}) as span:
        np.random.seed(d.bootstrap_seed)
        entry = d.estimator
        primary, bundle = estimators.estimate(sub, d, entry, contrast.key)
        records = [primary]
        falsifications = []
        sensitivities = []
        if primary.error and not primary.error.startswith("ValueError: design:") and d.fallback:
            trace.gate([f"fit fallback declared before freeze: {primary.error}"])
            records[0] = primary.model_copy(update={"secondary": True})
            entry = d.fallback
            primary, bundle = estimators.estimate(sub, d, entry, contrast.key)
            records.append(primary)
        if not primary.error:
            if isinstance(bundle, tuple):
                falsifications = [falsify.run(bundle[0], bundle, primary, r, contrast.key) for r in d.falsifications]
                sensitivities = [sensitivity.run(bundle[0], bundle, r, contrast.key, d.id).model_copy(update={"target": d.target}) for r in d.sensitivities]
        if d.modifiers:
            records += estimators.conditional_estimates(sub, d, contrast.key, entry=entry, bundle=bundle, error=primary.error)
        if d.estimand.kind != "zero":
            falsifications.append(graph_tests.run(sub, d, contrast.key))
        span.end(
            outputs={
                "estimates": [e.model_dump() for e in records],
                "falsifications": [r.model_dump() for r in falsifications],
                "sensitivities": [r.model_dump() for r in sensitivities],
            }
        )
    return {
        "estimates": records,
        "falsifications": falsifications,
        "sensitivities": sensitivities,
        "execution_failures": {e.tag: e.error for e in records if e.error},
        "execution_attempts": [
            {
                "design_id": e.design_id,
                "endpoint_id": e.endpoint_id,
                "contrast": e.contrast,
                "method": e.method,
                "target": e.target,
                "modifier": e.modifier,
                "level": e.level,
                "status": "failed" if e.error else "succeeded",
                "error": e.error,
                "value": e.value,
                "secondary": e.secondary,
                "address": e.tag + (".error" if e.error else ".value"),
            }
            for e in records
        ],
        "reasoning_history": [{"node": "analyse", "design_id": d.id, "contrast": contrast.key, "records": [e.model_dump() for e in records]}],
    }


def after_analyse(state: SpecialistState) -> Command[Literal["interpret", "feasibility"]]:
    good = [e for e in state.get("estimates", []) if e.error is None]
    if not good:
        return L.stop(
            "analyse",
            "all frozen designs failed to execute",
            list((state.get("execution_failures") or {}).values()),
            "an executable estimator under the original frozen assumptions",
            terminal_kind="execution_failure",
        )
    return Command(goto="interpret")
