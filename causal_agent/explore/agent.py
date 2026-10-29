"""One exploration, in this process: the request in, the episode over the one tool, the record gated, the trail out. The
caller (the A2A server, or a test) owns the folder and what is kept of the trail."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from causal_agent.explore import prompts as P
from causal_agent.explore import sandbox
from causal_agent.explore.contracts import Exploration, ExploreRequest, Trail, TransformRequest
from causal_agent.explore.gates import gate_record
from causal_agent.explore.inputs import prepare
from causal_agent.explore.inspection import ExplorationTools
from causal_agent.explore.tools import CodeTools
from causal_agent.lane.episode import run_episode

CHECKS = Path(__file__).with_name("checks.yaml")


@lru_cache(maxsize=1)
def checks() -> dict:
    return dict(yaml.safe_load(CHECKS.read_text()) or {})


def transform(req: TransformRequest, cancel=None) -> Trail:
    """Look at the table as asked and answer one record, or say why there is none after the tries declared."""
    sandbox.require_runtime()
    cfg = checks()
    tools = CodeTools(
        csv=req.csv, out_dir=req.out_dir, timeout=float(cfg["timeout"]), stdout_tail=int(cfg["stdout_tail"]), dropped_name=req.dropped_name(), cancel=cancel
    )
    columns = "\n".join(f"- {k} -> {v}" for k, v in req.columns.items())
    dropped = P.DROPPED.format(name=req.dropped_name()) if req.dropped_name() else ""
    user = P.USER.format(ask=req.ask, context=req.context, columns=columns, dropped=dropped)
    if req.product == "table":
        user += P.TABLE.format(keep=", ".join(repr(c) for c in req.keep) or "(none)")
    record, log, thoughts, errors = run_episode(
        Exploration,
        P.SYSTEM,
        user,
        tools=tools,
        budget=req.budget or int(cfg["budget"]),
        gate=lambda r, _log: gate_record(r, req, tools),
        node="explore",
        tries=int(cfg["tries"]),
        commit_system=P.COMMIT_SYSTEM,
    )
    return Trail(
        record=record,
        why="; ".join(errors),
        thoughts=thoughts,
        steps=tools.steps,
        calls=log.calls,
        refusals=[r.reason for r in log.refusals],
        sandbox=tools.steps[-1].kind if tools.steps else "not_run",
    )


def explore(req: ExploreRequest, cancel=None) -> Trail:
    """One whole-dataset task. The caller is the sole authority over supplied resources."""
    sandbox.require_runtime()
    cfg = checks()
    csv, inputs, manifest = prepare(req)
    tools = ExplorationTools(
        csv=csv,
        inputs=inputs,
        out_dir=req.out_dir,
        timeout=float(cfg["timeout"]),
        stdout_tail=int(cfg["stdout_tail"]),
        manifest=manifest,
        code_budget=int(cfg["budget"]),
        inspection_budget=int(cfg["inspections"]),
        cancel=cancel,
    )
    user = f"QUESTION\n{req.question}\n\nSUPPLIED CONTEXT\n{req.context}\n\nINPUT MANIFEST\n{manifest}"
    record, log, thoughts, errors = run_episode(
        Exploration,
        P.EXPLORE_SYSTEM,
        user,
        tools=tools,
        budget=tools.code_budget + tools.inspection_budget,
        gate=lambda record, _log: gate_record(record, req, tools),
        node="explore",
        tries=int(cfg["tries"]),
        commit_system=P.EXPLORE_COMMIT,
    )
    return Trail(
        record=record,
        why="; ".join(errors),
        thoughts=thoughts,
        steps=tools.steps,
        calls=log.calls,
        refusals=[r.reason for r in log.refusals],
        sandbox=tools.steps[-1].kind if tools.steps else "not_run",
    )
