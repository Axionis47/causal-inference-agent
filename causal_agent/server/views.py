"""The projection: the graph's state and the run records as the page's view models. Field by field, by hand, so a rename upstream
fails here and not on the page."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from causal_agent.common.contracts import Decline, Falsification, RunRecord, Sensitivity
from causal_agent.desk import pipeline
from causal_agent.memory import journal as J
from causal_agent.memory import store as MS
from causal_agent.memory import versions as VS
from causal_agent.server import artifacts as AV
from causal_agent.server import datasets as DS
from causal_agent.server.models import (
    Activity,
    AskItemView,
    CheckView,
    ClaimView,
    DecisionView,
    DeclineView,
    EstimateView,
    FeasibilityView,
    GapView,
    HandoffField,
    HandoffView,
    InterpretationView,
    JudgeView,
    Prompt,
    QuestionView,
    RunView,
    SessionView,
    StatusView,
    StepView,
    VersionView,
)
from causal_agent.viz.spec import FigureSpec

if TYPE_CHECKING:
    from causal_agent.server.sessions import SessionManager
    from causal_agent.server.settings import Settings


def disk_records(s: Settings, name: str) -> list[RunRecord]:
    """Every design the dataset ran, from the records beside them, in design order."""
    d = MS.home(name, s.root) / "designs"
    if not d.is_dir():
        return []
    out = []
    for p in sorted((p for p in d.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)):
        try:
            rec = pipeline.load_record(p)
        except Exception:
            rec = None
        if rec is not None:
            out.append(rec)
    return out


def all_runs(s: Settings, name: str, live: list[RunRecord]) -> list[RunRecord]:
    """The dataset's runs across every analysis: the records on disk, the live thread's winning for the same design."""
    by_index = {r.index: r for r in disk_records(s, name)}
    by_index.update({r.index: r for r in live})
    return [by_index[i] for i in sorted(by_index)]


_DESIGN_DIR = re.compile(r"^designs/(\d+)(?:/|$)")


def step_view(s: J.Step) -> StepView:
    run = next((int(m.group(1)) for p in s.left if (m := _DESIGN_DIR.match(p))), None)
    run_id = next((Path(p).name for p in s.left if Path(p).is_absolute()), None)
    return StepView(
        n=s.n,
        address=s.address,
        kind=s.kind,
        by=s.by,
        at=s.at,
        memory_version=s.memory_version,
        design=s.design,
        read=list(s.read),
        left=list(s.left),
        note=s.note,
        run=run,
        run_id=run_id,
    )


def journal_view(s: Settings, name: str, analysis: str | None) -> list[StepView]:
    return [step_view(x) for x in J.open_journal(name, analysis, s.root).steps()] if analysis else []


def session_view(mgr: SessionManager, name: str) -> SessionView:
    """The conversation as the page shows it: the stage, the prompt, the questions, the claims, the runs, the transcript."""
    sess = mgr.get(name)
    meta = DS.read_meta(mgr.s, name) or {}
    title, question = meta.get("title") or name, meta.get("question")
    if not sess.thread_id:
        runs0 = [run_view(r, mgr.s) for r in all_runs(mgr.s, name, [])]
        return SessionView(
            name=name, title=title, question=question, stage="new", runs=runs0, transcript=mgr.transcript(name), artifacts=AV.of_memory(mgr.s, name)
        )
    snap = mgr._snapshot(sess)
    values = dict(snap.values or {})
    interrupted = False
    for t in snap.tasks or ():
        if getattr(t, "interrupts", None):
            interrupted = True
        sub = getattr(t, "state", None)
        if sub is not None and hasattr(sub, "values"):
            values.update(sub.values or {})
    prompt_raw = sess.last_payload
    if prompt_raw is None:
        for t in snap.tasks or ():
            if getattr(t, "interrupts", None):
                prompt_raw = dict(t.interrupts[0].value or {})
                break
    if prompt_raw is None:
        prompt_raw = meta.get("last_prompt")
    stage: Literal["busy", "waiting", "ended", "stale", "error", "new"]
    if sess.busy:
        stage = "busy"
    elif sess.error:
        stage = "error"
    elif sess.ended or (not snap.next and values):
        stage = "ended"
    elif interrupted:
        stage = "waiting"
    elif snap.next:
        stage = "stale"  # a node was pending when the process died; Resume continues it
    else:
        stage = "new"
    phase = values.get("phase") or (prompt_raw or {}).get("phase") or "before"
    prompt = (
        Prompt(
            text=(prompt_raw or {}).get("text") or "",
            status=(prompt_raw or {}).get("status") or "",
            ready=bool((prompt_raw or {}).get("ready")),
            open=list((prompt_raw or {}).get("open") or []),
            runs=int((prompt_raw or {}).get("runs") or 0),
            phase=(prompt_raw or {}).get("phase") or "before",
            kind=(prompt_raw or {}).get("kind"),
        )
        if prompt_raw
        else None
    )
    questions: list[QuestionView] = []
    ask = values.get("ask")
    if phase == "before" and ask is not None and stage == "waiting" and (prompt_raw or {}).get("kind") == "ask":
        questions = [
            QuestionView(
                keys=list(ask.addresses),
                kind=ask.kind,
                text=ask.text,
                options=list(ask.options or []),
                recommended=ask.recommended,
                items=[AskItemView(**i.model_dump()) for i in ask.items],
                evidence_cites=list(ask.evidence or []),
                because=list(ask.because or []),
            )
        ]
    claims: list[ClaimView] = []
    data_version = 0
    if MS.exists(name, mgr.s.root):
        try:
            mem = MS.load(name, mgr.s.root)
            data_version = mem.data_version
            claims = [ClaimView(**c.model_dump()) for c in mem.to_claims().claims.values() if c.status != "empty"]
        except Exception:
            claims = []
    st = values.get("status")
    status = StatusView(**st.model_dump()) if st is not None else None
    runs = [run_view(r, mgr.s) for r in all_runs(mgr.s, name, list(values.get("runs") or []))]
    activity = Activity(node=sess.activity[0], since=sess.activity[1]) if sess.activity else None
    analysis = values.get("analysis") or sess.analysis or None
    preview = values.get("preview")
    handoff = (
        handoff_view(preview, values.get("readiness")) if phase == "before" and prompt is not None and prompt.kind == "ready" and preview is not None else None
    )
    return SessionView(
        name=name,
        title=title,
        question=values.get("question") or question,
        stage=stage,
        phase=phase,
        activity=activity,
        ready=bool(prompt and prompt.ready) if phase == "before" else True,
        prompt=prompt,
        questions=questions,
        claims=claims,
        status=status,
        runs=runs,
        brief=values.get("opening") or "",
        transcript=mgr.transcript(name),
        error=sess.error,
        analysis=analysis,
        journal=journal_view(mgr.s, name, analysis),
        artifacts=AV.of_memory(mgr.s, name),
        handoff=handoff,
        data_version=data_version,
        versions=[
            VersionView(
                v=x.v, parent=x.parent, address=x.address, ask=x.ask, summary=x.summary, rows=x.rows, added=x.added, dropped=x.dropped, made_at=x.made_at
            )
            for x in VS.list_versions(name, mgr.s.root)
        ],
    )


def handoff_view(h, readiness=None) -> HandoffView:
    """The pack's facts as fields the page can show and correct: the claim fields first, then the columns' fields; the judge's
    reading beside them when it gave one."""
    facts = h.facts
    order = [a for a in facts if a.startswith("claim:")] + [a for a in facts if a.startswith("col:")]
    fields = [
        HandoffField(address=a, value=facts[a].value, status=facts[a].status, source=facts[a].source, said=facts[a].said, reason=facts[a].reason) for a in order
    ]
    return HandoffView(
        family=h.family,
        specialist=h.specialist,
        outcome=h.outcome,
        treatment=h.treatment,
        bets_on=h.brief.bets_on if h.brief is not None else h.chosen_assumption,
        memory_version=h.memory_version,
        context_id=h.context_id,
        dataset_identity=h.dataset_identity,
        data_version=h.data_version,
        admission=h.admission,
        fields=fields,
        judge=JudgeView(
            verdict=readiness.verdict,
            summary=readiness.summary,
            gaps=[GapView(address=g.address, reason=g.reason, cites=list(g.cites)) for g in readiness.gaps],
        )
        if readiness is not None
        else None,
    )


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
        artifacts=AV.of_design(s, r.dataset, int(r.index)) if s is not None else [],
        declines=declines,
    )
