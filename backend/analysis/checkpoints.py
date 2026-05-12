"""Read-only checkpoint definitions and views for preserved historical runs."""








from backend.analysis.contracts import (
    CheckpointDefinition,
    CheckpointReceipt,
    CheckpointView,
)
from backend.models import AnalysisRun

DEFINITIONS = [
    CheckpointDefinition(
        id="bind",
        title="Bind the task",
        question="Which question, data and context are we analyzing?",
        phase="semantic",
        prerequisites=[],
        tools=[],
    ),
    CheckpointDefinition(
        id="target",
        title="Establish the target",
        question="What effect is asked about, for whom and over what period?",
        phase="semantic",
        prerequisites=["bind"],
        tools=["read_context"],
    ),
    CheckpointDefinition(
        id="treatments",
        title="Establish treatments",
        question="What changes, from what to what, how and when?",
        phase="semantic",
        prerequisites=["target"],
        tools=["read_context"],
    ),
    CheckpointDefinition(
        id="roles",
        title="Resolve other roles",
        question="How do other variables relate to this target and treatment set?",
        phase="semantic",
        prerequisites=["target", "treatments"],
        tools=["read_context"],
    ),
    CheckpointDefinition(
        id="causal",
        title="Establish the causal account",
        question="Under which assumptions can the requested effects be identified?",
        phase="semantic",
        prerequisites=["target", "treatments", "roles"],
        tools=["read_context"],
    ),
    CheckpointDefinition(
        id="diagnostics",
        title="Establish feasibility",
        question="Do permitted diagnostics support carrying out this specification?",
        phase="computation",
        prerequisites=["causal"],
        tools=["run_diagnostic"],
    ),
    CheckpointDefinition(
        id="freeze",
        title="Freeze the executable analysis",
        question="Exactly what will be estimated and how?",
        phase="semantic",
        prerequisites=["diagnostics"],
        tools=["write_code", "review_code"],
    ),
    CheckpointDefinition(
        id="execute",
        title="Verify execution",
        question="Did the approved program execute against the approved inputs?",
        phase="execution",
        prerequisites=["freeze"],
        tools=["run_approved_program"],
    ),
    CheckpointDefinition(
        id="report",
        title="Bound the conclusions",
        question="What do the results support under the frozen assumptions?",
        phase="interpretation",
        prerequisites=["execute"],
        tools=[],
    ),
]

BY_ID = {item.id: item for item in DEFINITIONS}

def budgets(run: AnalysisRun) -> dict[str, int]:
    if run.model_limit is not None:
        return {"model_calls": run.calls, "model_calls_limit": run.model_limit,
                "shared_calls": run.shared_calls, "shared_calls_limit": run.model_limit,
                "diagnostic_runs": sum(key.startswith("diagnostic:") for key in run.receipts),
                "diagnostic_runs_limit": 4, "repairs": max(0, sum(key.startswith("program:") for key in run.receipts) - 1),
                "repairs_limit": 2}
    return {
        "model_calls": run.calls,
        "model_calls_limit": 27 + 16 * int(str(run.inputs.get("column_count", 0))) + 2 * int(str(run.inputs.get("relationship_count", 0))),
        "shared_calls": run.shared_calls or 0,
        "shared_calls_limit": 27,
        "column_calls_limit": 16,
        **{
            name: sum(key.startswith(prefix) for key in run.receipts)
            for name, prefix in (
                ("context_reads", "read:"),
                ("diagnostic_runs", "diagnostic_request:"),
                ("repairs", "repair_request:"),
            )
        },
        "context_reads_limit": 6,
        "diagnostic_runs_limit": 4,
        "repairs_limit": 2,
    }

def receipts(run: AnalysisRun) -> list[CheckpointReceipt]:
    return sorted(
        [
            CheckpointReceipt.model_validate(value)
            for key, value in run.receipts.items()
            if key.startswith("checkpoint:")
        ],
        key=lambda value: (value.completed_at, value.id),
    )

def views(run: AnalysisRun) -> list[CheckpointView]:
    records = receipts(run)
    states: dict[str, str] = {}
    result = []
    for definition in DEFINITIONS:
        attempts = sorted(
            [r for r in records if r.checkpoint == definition.id], key=lambda r: (r.completed_at, r.id)
        )
        status: str = attempts[-1].status if attempts else "pending"
        if attempts and not attempts[-1].complete and status == "accepted":
            status = "running"
        if run.stage == definition.id and run.status in {"running", "queued"}:
            status = "running"
        elif run.stage == definition.id and run.status == "failed" and status != "accepted":
            status = "failed"
        states[definition.id] = status
        result.append(
            CheckpointView.model_validate(
                {
                    **definition.model_dump(),
                    "status": status,
                    "attempts": [r.model_dump() for r in attempts],
                    "blocked_by": [key for key in definition.prerequisites if states.get(key) != "accepted"],
                }
            )
        )
    return result

