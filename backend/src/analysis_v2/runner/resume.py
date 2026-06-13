"""The S6 resume path: the analyst's plan decision re-enters the spine.

Confirm applies whitelisted edits to the spec, re-derives eligibility,
candidates, and the method plan deterministically (the same rules the
design stage used), records the S6 transition, and relaunches the spine.
Reject fails the run with the analyst's reason. Edits never bypass the
rules: a lane that stays disabled after edits is refused, not run.
"""
from __future__ import annotations

from typing import Any

import structlog

from src.analysis_v2.agents.design_detection.candidates import build_candidates
from src.analysis_v2.agents.design_detection.dag import apply_dag_adjustment
from src.analysis_v2.agents.design_detection.resolve import resolve_spec
from src.analysis_v2.agents.design_detection.rules import evaluate_all_lanes
from src.analysis_v2.agents.plan_critic.plan_builder import build_method_plan
from src.analysis_v2.core import AnalysisStage, GateResult, RunStatus
from src.analysis_v2.persistence import load_run, save_run
from src.analysis_v2.spec import EligibilityState, MethodLane, VariableRef
from src.analysis_v2.state import JobStatus

logger = structlog.get_logger(__name__)

_COLUMN_FIELDS = (
    "outcome", "treatment", "time_column", "group_column",
    "instrument", "mediator", "duration_column", "event_column",
)
_ACK_FIELDS = ("confirm_reshape", "confirm_derived_treatment")


class NoAnalysisRun(LookupError):
    """No persisted run for this job."""


class NotWaitingForUser(RuntimeError):
    """The run is not parked at the plan gate."""


class InvalidPlanEdits(ValueError):
    """An edit failed validation; the message says which and why."""


async def apply_plan_decision(
    job_id: str,
    manager: Any,
    *,
    decision: str,
    edits: dict[str, str] | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    run = await load_run(job_id)
    if run is None:
        raise NoAnalysisRun(f"job {job_id} has no analysis run")
    if run.status != RunStatus.WAITING_FOR_USER:
        raise NotWaitingForUser(
            f"job {job_id} is {run.status.value}, not waiting_for_user"
        )
    state = await manager.firestore.load_parked_state(job_id)
    if state is None:
        raise NoAnalysisRun(f"job {job_id} has no parked record to resume")

    if decision == "reject":
        message = f"plan rejected by the analyst: {(reason or '').strip() or 'no reason given'}"
        run.mark_failed(message)
        await save_run(run)
        state.mark_failed(message, "plan_gate")
        await manager.firestore.update_job(state)
        state.push_sse_event("analysis_failed", {"headline": message, "error": message})
        logger.info("plan_rejected", job_id=job_id)
        return {"job_id": job_id, "status": state.status.value}

    merged = _with_card_defaults(run, dict(edits or {}))
    applied = _apply_edits(run, merged)
    run.record_transition(
        to_state=AnalysisStage.S6_USER_CONFIRMED_OR_AUTO_APPROVED,
        agent_name=None,
        gate_result=GateResult.advance(),
        warnings=[f"user confirmed the plan; edits: {applied or 'none'}"],
    )
    run.status = RunStatus.RUNNING
    await save_run(run)

    from . import entry

    ack = await entry.start(state, manager)
    logger.info("plan_confirmed", job_id=job_id, edits=applied)
    return {"job_id": job_id, "status": ack["status"]}


def _with_card_defaults(run, edits: dict[str, str]) -> dict[str, str]:
    """Confirming the card accepts its prefilled values: any required item
    the user did not edit falls back to its current_value."""
    critique = run.plan_critique
    card = critique.confirmation_card if critique is not None else None
    if card is None:
        return edits
    for item in card.items:
        if item.field not in edits and item.current_value:
            edits[item.field] = item.current_value
    missing = [
        item.field for item in card.items
        if item.required and not str(edits.get(item.field, "")).strip()
    ]
    if missing:
        raise InvalidPlanEdits(
            "these required items need a value to confirm: " + ", ".join(missing)
        )
    return edits


def _apply_edits(run, edits: dict[str, str]) -> dict[str, str]:
    """Validate and apply edits, then re-derive the plan. Returns what applied."""
    spec = run.causal_spec.model_copy(deep=True)
    profile = run.dataset_profile
    columns = set(profile.column_names()) if profile is not None else set()
    applied: dict[str, str] = {}
    lane_choice: MethodLane | None = None

    for field, raw in edits.items():
        value = str(raw).strip()
        if field in _ACK_FIELDS:
            if value.lower() not in ("yes", "true", "1"):
                raise InvalidPlanEdits(
                    f"{field}: to decline the plan, use decision=reject with a reason"
                )
            applied[field] = "yes"
        elif field in _COLUMN_FIELDS:
            if value not in columns:
                raise InvalidPlanEdits(f"{field}: '{value}' is not a dataset column")
            setattr(spec, field, VariableRef(column=value))
            applied[field] = value
        elif field == "cutoff_value":
            try:
                spec.cutoff_value = float(value)
            except ValueError:
                raise InvalidPlanEdits(f"cutoff_value: '{value}' is not a number")
            applied[field] = value
        elif field == "intervention_date":
            if len(value) < 8:
                raise InvalidPlanEdits(f"intervention_date: '{value}' is not an ISO date")
            spec.intervention_date = value
            applied[field] = value
        elif field == "lane":
            try:
                lane_choice = MethodLane(value)
            except ValueError:
                raise InvalidPlanEdits(f"lane: '{value}' is not a method lane")
            applied[field] = value
        else:
            raise InvalidPlanEdits(f"unknown edit field '{field}'")

    refined, _notes = resolve_spec(spec, profile)
    dag = apply_dag_adjustment(refined, run.dataset_dossier)
    eligibility = evaluate_all_lanes(refined, profile)
    candidates = build_candidates(refined, eligibility)
    if lane_choice is not None:
        entry_state = eligibility.for_lane(lane_choice)
        if entry_state is None or entry_state.state == EligibilityState.DISABLED:
            raise InvalidPlanEdits(
                f"lane '{lane_choice.value}' is disabled: "
                f"{entry_state.reason if entry_state else 'unknown lane'}"
            )
        chosen = next((c for c in candidates if c.lane == lane_choice), None)
        if chosen is None:
            raise InvalidPlanEdits(
                f"lane '{lane_choice.value}' is not an offered candidate"
            )
    else:
        chosen = candidates[0] if candidates else None
    if chosen is None:
        raise InvalidPlanEdits("no executable design remains after these edits")

    run.causal_spec = refined
    run.causal_dag = dag
    run.tool_eligibility = eligibility
    run.design_candidates = candidates
    run.selected_design = chosen
    run.method_plan = build_method_plan(refined, chosen, run.dataset_dossier)
    return applied
