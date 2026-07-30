"""The minimal governed workflow for the Streamlit causal studio.

LangGraph is used only where it earns its keep: checkpointed execution and a
human publication interrupt.  Profiling, estimators, checks, and policy remain
plain deterministic functions that are easy to test independently.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

try:  # optional in a developer environment; pinned in requirements.txt
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:  # Streamlit's cached graph still preserves in-process runs
    SqliteSaver = None

from . import lanes
from .checks import run as run_check
from .estimate import Estimate
from .monitoring import alerts as monitoring_alerts
from .monitoring import summarize as monitoring_summary
from .report import claim_strength, headline
from .studio_policy import evaluate, render_report
from .studio_prep import file_fingerprint
from .studio_protocols import build_contract, contract_hash, get_protocol


CHECKPOINTS = Path(__file__).parent.parent / "studio_jobs.sqlite"

LANE_EXECUTORS = {
    "observational": lanes.observational,
    "matching": lanes.matching,
    "iv": lanes.iv,
    "survival": lanes.survival,
    "did": lanes.did,
    "rdd": lanes.rdd,
    "mediation": lanes.mediation,
    "time_series": lanes.time_series,
}
# Backward-compatible name for callers/tests; these are controlled numerical
# executors, not tools exposed to the preparation ReAct agent.
LANE_FUNCTIONS = LANE_EXECUTORS

class StudioState(TypedDict, total=False):
    run_id: str
    parent_run_id: str
    csv_path: str
    source: str
    lane: str
    kwargs: dict[str, Any]
    context: dict[str, Any]
    repairs: list[dict[str, Any]]
    data_version: dict[str, Any]
    data_quality: dict[str, Any]
    preparation: dict[str, Any]
    prompt_versions: dict[str, Any]
    cohort: dict[str, Any] | None
    interaction_events: list[dict[str, Any]]
    design_contract: dict[str, Any]
    design_approval: dict[str, Any]
    preflight: list[dict[str, Any]]
    protocol: dict[str, Any]
    estimate: dict[str, Any]
    strength: str
    headline: str
    diagnostics: list[dict[str, Any]]
    policy: dict[str, Any]
    approval: dict[str, Any]
    report: str
    monitoring: dict[str, Any]
    monitoring_alerts: list[dict[str, str]]
    error: str
    events: list[dict[str, Any]]


def _event(state: StudioState, stage: str, status: str, detail: str = "") -> list[dict]:
    return [*state.get("events", []), {"stage": stage, "status": status, "detail": detail}]


def _clean_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        key: tuple(value) if key == "covariates" and isinstance(value, list) else value
        for key, value in kwargs.items()
        if not str(key).startswith("_") and value not in ("", None)
    }


def _execution_contract_error(state: StudioState) -> str:
    """Return why a lane must not run, or an empty string when dispatch is safe."""
    lane = state.get("lane", "")
    contract = state.get("design_contract") or {}
    approval = state.get("design_approval") or contract.get("approval") or {}
    data_version = state.get("data_version") or {}
    if not data_version.get("version_id") or not data_version.get("prepared_fingerprint"):
        return "Prepared dataset has no immutable data-version fingerprint."
    if (contract.get("data_version") or {}).get("version_id") != data_version["version_id"]:
        return "Frozen design contract does not match the prepared dataset version."
    if contract.get("lane") != lane:
        return "Frozen design contract names a different analysis lane."
    if not approval.get("approved") or not approval.get("role_ledger_and_map_confirmed"):
        return "Frozen design contract lacks the required human approval."
    contract_content = {
        key: value for key, value in contract.items()
        if key not in {
            "contract_hash", "revision", "parent_contract_hash", "change_reason",
            "revision_timing", "approval",
        }
    }
    if not contract.get("contract_hash") or contract_hash(contract_content) != contract["contract_hash"]:
        return "Frozen design contract content hash is invalid."
    try:
        expected = build_contract(
            dataset_id=contract.get("dataset_id", ""),
            lane=lane,
            kwargs=state.get("kwargs") or {},
            context=state.get("context") or {},
            cohort=state.get("cohort"),
            data_version=data_version,
        )
    except Exception as exc:
        return f"Design contract could not be reconstructed: {type(exc).__name__}: {exc}"
    if contract.get("configuration_hash") != contract_hash(expected):
        return "Frozen design contract does not match the requested roles, context, or cohort."
    expected_checks = set(get_protocol(lane).pre_checks)
    completed_checks = {item.get("check") for item in state.get("preflight", [])}
    if not expected_checks <= completed_checks:
        return "The complete lane-specific preflight protocol was not supplied."
    failures = [item.get("check", "unknown") for item in state.get("preflight", []) if item.get("verdict") == "fail"]
    if failures:
        return "Preflight failures block lane execution: " + ", ".join(failures)
    try:
        actual_fingerprint = file_fingerprint(state["csv_path"])
    except Exception as exc:
        return f"Prepared dataset artifact could not be verified: {type(exc).__name__}: {exc}"
    if actual_fingerprint != data_version["prepared_fingerprint"]:
        return "Prepared dataset artifact changed after its version was approved."
    return ""


def node_estimate(state: StudioState) -> dict:
    lane = state.get("lane", "")
    fn = LANE_EXECUTORS.get(lane)
    if fn is None:
        message = f"Unsupported analysis lane: {lane!r}"
        return {"error": message, "events": _event(state, "estimate", "failed", message)}
    contract_error = _execution_contract_error(state)
    if contract_error:
        return {
            "error": contract_error,
            "events": _event(state, "execution_guard", "blocked", contract_error),
        }
    try:
        estimate = fn(pd.read_csv(state["csv_path"]), **_clean_kwargs(state["kwargs"]))
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        return {"error": message, "events": _event(state, "estimate", "failed", message)}
    exposure = state["context"].get("treatment") or state["context"].get("exposure") or "exposure"
    outcome = state["context"].get("outcome", "outcome")
    strength = claim_strength(lane, estimate)
    return {
        "estimate": estimate.__dict__,
        "strength": strength,
        "headline": headline(lane, estimate, exposure, outcome, strength),
        "events": _event(state, "estimate", "completed", str(estimate)),
    }


def node_sensitivity(state: StudioState) -> dict:
    if state.get("error") or not state.get("estimate"):
        return {}
    df = pd.read_csv(state["csv_path"])
    names = list(get_protocol(state["lane"]).post_checks)
    findings = [
        run_check(name, df, state["lane"], state["kwargs"], state["estimate"]).__dict__
        for name in names
    ]
    failed = any(f["verdict"] == "fail" for f in findings)
    estimate = Estimate(**state["estimate"])
    strength = claim_strength(state["lane"], estimate, diagnostics_failed=failed)
    exposure = state["context"].get("treatment") or state["context"].get("exposure") or "exposure"
    outcome = state["context"].get("outcome", "outcome")
    return {
        "diagnostics": findings,
        "strength": strength,
        "headline": headline(state["lane"], estimate, exposure, outcome, strength),
        "events": _event(state, "sensitivity", "completed", f"{len(findings)} checks"),
    }


def node_policy(state: StudioState) -> dict:
    policy = evaluate(dict(state))
    summary = monitoring_summary(dict(state) | {"policy": policy})
    return {
        "policy": policy,
        "monitoring": summary,
        "monitoring_alerts": monitoring_alerts(summary),
        "events": _event(state, "policy", policy["decision"], policy["version"]),
    }


def route_after_policy(state: StudioState) -> str:
    decision = (state.get("policy") or {}).get("decision")
    if decision == "block":
        return "blocked"
    if decision == "review":
        return "review"
    return "publish"


def node_human_gate(state: StudioState) -> dict:
    decision = interrupt({
        "kind": "publication_review",
        "policy": state["policy"],
        "headline": state.get("headline", ""),
        "instruction": "Approve, reject, or add a reviewer note.",
    })
    approved = bool((decision or {}).get("approved"))
    approval = {
        "approved": approved,
        "note": str((decision or {}).get("note", "")).strip(),
        "reviewer": str((decision or {}).get("reviewer", "human reviewer")),
    }
    if not approved:
        error = "Publication rejected by human reviewer."
        summary = monitoring_summary(dict(state) | {"approval": approval, "error": error})
        return {
            "approval": approval,
            "error": error,
            "monitoring": summary,
            "monitoring_alerts": monitoring_alerts(summary),
            "events": _event(state, "human_review", "rejected", approval["note"]),
        }
    policy = dict(state["policy"])
    policy["decision"] = "approved"
    summary = monitoring_summary(dict(state) | {"approval": approval, "policy": policy})
    return {
        "approval": approval,
        "policy": policy,
        "monitoring": summary,
        "monitoring_alerts": monitoring_alerts(summary),
        "events": _event(state, "human_review", "approved", approval["note"]),
    }


def route_after_human(state: StudioState) -> str:
    return "publish" if (state.get("approval") or {}).get("approved") else "blocked"


def node_report(state: StudioState) -> dict:
    report = render_report(dict(state))
    summary = monitoring_summary(dict(state) | {"report": report})
    return {
        "report": report,
        "monitoring": summary,
        "monitoring_alerts": monitoring_alerts(summary),
        "events": _event(state, "report", "completed", "publishable report generated"),
    }


def build(checkpointer=None):
    graph = StateGraph(StudioState)
    graph.add_node("estimate", node_estimate)
    graph.add_node("sensitivity", node_sensitivity)
    graph.add_node("policy", node_policy)
    graph.add_node("human_gate", node_human_gate)
    graph.add_node("report", node_report)
    graph.add_edge(START, "estimate")
    graph.add_edge("estimate", "sensitivity")
    graph.add_edge("sensitivity", "policy")
    graph.add_conditional_edges(
        "policy",
        route_after_policy,
        {"blocked": END, "review": "human_gate", "publish": "report"},
    )
    graph.add_conditional_edges(
        "human_gate", route_after_human, {"blocked": END, "publish": "report"}
    )
    graph.add_edge("report", END)
    if checkpointer is None:
        if SqliteSaver is None:
            checkpointer = InMemorySaver()
        else:
            connection = sqlite3.connect(CHECKPOINTS, check_same_thread=False)
            checkpointer = SqliteSaver(connection)
    return graph.compile(checkpointer=checkpointer)


def build_in_memory():
    """Test/demo graph that never writes a checkpoint file."""
    return build(InMemorySaver())
