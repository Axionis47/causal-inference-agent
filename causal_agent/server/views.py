"""The projection: the graph's state and the run records as the page's view models. Field by field, by hand, so a rename upstream
fails here and not on the page."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from causal_agent.common.contracts import Decline, RunRecord
from causal_agent.desk import pipeline
from causal_agent.memory import store as MS
from causal_agent.server import datasets as DS
from causal_agent.server.models import (
    Activity,
    CheckView,
    ClaimView,
    DecisionView,
    DeclineView,
    EstimateView,
    FeasibilityView,
    InterpretationView,
    Prompt,
    QuestionView,
    RefutationView,
    RunView,
    SessionView,
    StatusView,
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


def session_view(mgr: SessionManager, name: str) -> SessionView:
    """The conversation as the page shows it: the stage, the prompt, the questions, the claims, the runs, the transcript."""
    sess = mgr.get(name)
    meta = DS.read_meta(mgr.s, name) or {}
    title, question = meta.get("title") or name, meta.get("question")
    if not sess.thread_id:
        runs0 = [run_view(r) for r in all_runs(mgr.s, name, [])]
        return SessionView(name=name, title=title, question=question, stage="new", runs=runs0, transcript=mgr.transcript(name))
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
                evidence_cites=list(ask.evidence or []),
                because=list(ask.because or []),
            )
        ]
    claims: list[ClaimView] = []
    if MS.exists(name, mgr.s.root):
        try:
            table = MS.load(name, mgr.s.root).to_claims()
            claims = [ClaimView(**c.model_dump()) for c in table.claims.values() if c.status != "empty"]
        except Exception:
            claims = []
    st = values.get("status")
    status = StatusView(**st.model_dump()) if st is not None else None
    runs = [run_view(r) for r in all_runs(mgr.s, name, list(values.get("runs") or []))]
    activity = Activity(node=sess.activity[0], since=sess.activity[1]) if sess.activity else None
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
        brief=values.get("brief") or "",
        transcript=mgr.transcript(name),
        error=sess.error,
    )


def run_view(r: RunRecord) -> RunView:
    sr = r.specialist_result or {}
    design = sr.get("design") or {}
    results = (design.get("checks") or {}).get("results") or [] if isinstance(design, dict) else []
    checks = [
        CheckView(
            contrast=c.get("contrast"),
            name=str(c.get("name")),
            level=c.get("level"),
            value=c.get("value"),
            threshold=c.get("threshold"),
            detail=c.get("detail"),
        )
        for c in results
    ]
    flags = [c for c in checks if c.level != "pass"]
    refs = [
        RefutationView(
            contrast=x.get("contrast"),
            refuter=str(x.get("refuter")),
            kind=x.get("kind"),
            passed=x.get("passed"),
            p_value=x.get("p_value"),
            new_effect=x.get("new_effect"),
            detail=x.get("detail"),
        )
        for x in (sr.get("refutations") or r.artifacts.get("refutations") or [])
    ]
    interps = [
        InterpretationView(
            contrast=i.get("contrast"), answer=str(i.get("answer") or ""), caveats=list(i.get("caveats") or []), cites=list(i.get("cites") or [])
        )
        for i in (sr.get("interpretations") or r.artifacts.get("interpretations") or [])
    ]
    ests = [
        EstimateView(
            contrast=e.get("contrast"),
            method=e.get("method"),
            value=e.get("value"),
            ci_low=e.get("ci_low"),
            ci_high=e.get("ci_high"),
            n=e.get("n"),
            n_treated=e.get("n_treated"),
            n_control=e.get("n_control"),
            secondary=bool(e.get("secondary")),
            error=e.get("error"),
        )
        for e in (sr.get("estimates") or r.artifacts.get("estimates") or [])
    ]
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
        refutations=refs,
        interpretations=interps,
        estimates=ests,
        feasibility=FeasibilityView.model_validate(feas) if (feas := sr.get("feasibility") or r.artifacts.get("feasibility")) else None,
        files=files,
        what_if=dict(r.what_if or {}),
        differs=list(r.differs or []),
        figures=[FigureSpec.model_validate(f) for f in r.figures or []],
        declines=declines,
    )
