"""The projection: the graph's state and the run records as the page's view models. Field by field, by hand, so a rename upstream
fails here and not on the page."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from causal_agent.common.contracts import Decline, Falsification, RunRecord, Sensitivity
from causal_agent.server.models import (
    CheckView,
    DecisionView,
    DeclineView,
    EstimateView,
    FeasibilityView,
    InterpretationView,
    RunView,
)
from causal_agent.viz.spec import FigureSpec

if TYPE_CHECKING:
    from causal_agent.server.settings import Settings


def run_view(r: RunRecord, s: Settings | None = None) -> RunView:
    sr = r.specialist_result or {}
    checks = [CheckView.model_validate(c) for c in sr.get("checks") or []]
    flags = [c for c in checks if c.level != "pass"]
    falsifications = [Falsification.model_validate(x) for x in sr.get("falsifications") or []]
    sensitivities = [Sensitivity.model_validate(x) for x in sr.get("sensitivities") or []]
    interps = [
        InterpretationView(
            design_id=i.get("design_id", ""),
            contrast=i.get("contrast"),
            answer=str(i.get("answer") or ""),
            caveats=list(i.get("caveats") or []),
            cites=list(i.get("cites") or []),
        )
        for i in (sr.get("interpretations") or r.artifacts.get("interpretations") or [])
    ]
    ests = [EstimateView.model_validate(e) for e in sr.get("estimates") or []]
    run_id = Path(r.run_dir).name if r.run_dir else None
    files = sorted(p.name for p in Path(r.run_dir).iterdir() if p.is_file()) if r.run_dir and Path(r.run_dir).is_dir() else []
    declines = []
    for d in sr.get("declines") or r.artifacts.get("declines") or []:
        try:
            dd = Decline.model_validate(d)
        except Exception:
            continue
        declines.append(
            DeclineView(
                address=dd.address, stage=dd.stage, kind=dd.kind, about=dd.about, pack_value=dd.pack_value, took=dd.took, reason=dd.reason, check=dd.check
            )
        )
    return RunView(
        index=r.index,
        question=r.question,
        family=r.family,
        specialist=r.specialist,
        status=r.status,
        run_id=run_id,
        effect=r.effect,
        ci_low=r.ci_low,
        ci_high=r.ci_high,
        estimator=r.estimator,
        decision=DecisionView.model_validate({k: v for k, v in (r.decision or {}).items() if k in DecisionView.model_fields}),
        decision_record=r.decision_record or "",
        flags=flags,
        checks=checks,
        lead_design_id=r.lead_design_id,
        lead_selection=sr.get("lead_selection") or {},
        context_id=sr.get("context_id") or "",
        dataset_identity=sr.get("dataset_identity") or "",
        context_admission=sr.get("context_admission"),
        context_deficiency=sr.get("context_deficiency"),
        terminal=sr.get("terminal") or {},
        candidate_history=sr.get("candidate_history") or [],
        reasoning_history=sr.get("reasoning_history") or [],
        context_facts=sr.get("context_facts") or {},
        designs=sr.get("designs") or {},
        ledger=sr.get("ledger") or {},
        candidates=sr.get("candidates") or [],
        episode_visits=sr.get("episodes") or {},
        comparisons=sr.get("comparisons") or [],
        execution_attempts=sr.get("execution_attempts") or [],
        falsifications=falsifications,
        sensitivities=sensitivities,
        interpretations=interps,
        estimates=ests,
        feasibility=FeasibilityView.model_validate(feas) if (feas := sr.get("feasibility")) else None,
        files=files,
        what_if=dict(r.what_if or {}),
        differs=list(r.differs or []),
        figures=[FigureSpec.model_validate(f) for f in r.figures or []],
        declines=declines,
    )
