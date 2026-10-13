"""The adjustment lane's artifacts: what the run leaves behind, read from the frozen design and the results. The interpretation
per contrast (a judgement, gated on the numbers it copies and the addresses it cites), the figures, checked against the
addresses the run produced, and the report and result the desk reads."""

from __future__ import annotations

from typing import Any

from langgraph.types import Send

from causal_agent.common.contracts import Feasibility, Interpretation
from causal_agent.families.adjustment.lane import prompts as P
from causal_agent.families.adjustment.lane.contracts import Design
from causal_agent.families.adjustment.lane.knowledge import load_checks
from causal_agent.families.adjustment.lane.state import InterpretTask, SpecialistState, episodes_of, ladder_of, latest_primary
from causal_agent.lane import figures as LF
from causal_agent.lane import nodes as L
from causal_agent.lane import records
from causal_agent.lane.episode import EpisodeLog, run_episode
from causal_agent.viz.postviz import common as PV

_writer, _question = L.writer, L.question


# ------------------------------------------------------------------ interpret (judgement, fan-out per contrast)


def _addresses(state: SpecialistState, contrast_key: str) -> list[str]:
    d: Design = state["design"]
    out = ["design.estimand.adjustment_set", "design.assumption"] + [b.address for b in state["handoff"].beliefs.values() if b.known() or b.status == "unknown"]
    if state["handoff"].brief is not None:
        out += sorted(state["handoff"].brief.addresses())
    out += [r.address for r in d.checks.results if r.contrast in (contrast_key, "all")]
    out += [x.address for x in state.get("declines") or []]
    out += sorted(ladder_of(state).addresses())
    out += [f.address for log in episodes_of(state).values() for f in log.facts]
    for e in state.get("estimates") or []:
        if e.contrast == contrast_key and e.error is None:
            out += [f"{e.tag}.value", f"{e.tag}.ci", f"{e.tag}.n"]
    for r in state.get("refutations") or []:
        if r.contrast == contrast_key:
            out += [f"refute:{contrast_key}.{r.refuter}.p_value", f"refute:{contrast_key}.{r.refuter}.new_effect", f"refute:{contrast_key}.{r.refuter}.passed"]
    return out


def _required(state: SpecialistState, contrast_key: str) -> list[str]:
    """What the interpretation must cite: every flagged check (the person's flags among them) and the estimate's interval."""
    d: Design = state["design"]
    out = [r.address for r in d.checks.results if r.contrast in (contrast_key, "all") and r.level != "pass"]
    if any(e.contrast == contrast_key and e.error is None and not e.secondary and e.modifier is None for e in state.get("estimates") or []):
        out.append(f"estimate:{contrast_key}.ci")
    out += [f"{e.tag}.value" for e in state.get("estimates") or [] if e.contrast == contrast_key and e.modifier is not None and e.error is None]
    out += [x.address for x in state.get("declines") or [] if x.about == "scope.target"]  # the target delivered is not the one asked
    return out


def _material(state: SpecialistState, contrast_key: str) -> str:
    d: Design = state["design"]
    c = next(x for x in d.contrasts if x.key == contrast_key)
    names = state.get("columns") or {}
    lines = [f"[design.assumption] the design bets on: {state['handoff'].chosen_assumption}"]
    if state["handoff"].brief is not None:
        lines.append(state["handoff"].brief.render())
    lines += [b.render() for b in state["handoff"].beliefs.values() if b.known() or b.status == "unknown"]
    lines += [
        f"[design.estimand.adjustment_set] adjusted for: {', '.join(names.get(k, k) for k in d.estimand.adjustment_set) or 'nothing (no confounders in the graph)'}",
        f"comparison: {names.get(d.graph.treatment, d.graph.treatment)} = {c.treated!r} versus {c.control!r}; outcome: {names.get(d.graph.outcome, d.graph.outcome)}",
    ]
    for r in d.checks.results:
        if r.contrast in (contrast_key, "all"):
            lines.append(f"[{r.address}] {r.level}: {r.detail}")
    lines += [x.render() for x in state.get("declines") or []]
    lines += [f"[{a}] {text}" for a, text in ladder_of(state).lines()]
    lines += [f.render() for log in episodes_of(state).values() for f in log.facts]
    for e in state.get("estimates") or []:
        if e.contrast != contrast_key:
            continue
        tag = e.tag
        if e.modifier is not None and e.error is not None:
            lines.append(f"[{tag}.error] within {names.get(e.modifier, e.modifier)} = {e.level}: {e.error}")
            continue
        if e.error is not None:
            continue
        where = f"within {names.get(e.modifier, e.modifier)} = {e.level}, " if e.modifier is not None else ""
        lines.append(f"[{tag}.value] {e.value:.4g} ({where}{'secondary' if e.secondary else 'primary'} estimator {e.method}, target {e.target_units})")
        lines.append(f"[{tag}.ci] 95% interval {e.ci_low:.4g} to {e.ci_high:.4g}" if e.ci_low is not None else f"[{tag}.ci] no interval")
        lines.append(f"[{tag}.n] {e.n_treated} treated, {e.n_control} control")
    for r in state.get("refutations") or []:
        if r.contrast == contrast_key:
            lines.append(
                f"[refute:{contrast_key}.{r.refuter}.passed] {r.passed}  [refute:{contrast_key}.{r.refuter}.new_effect] {r.new_effect}  "
                f"[refute:{contrast_key}.{r.refuter}.p_value] {r.p_value}  ({r.detail})"
            )
    return "\n".join(lines)


def fan_out_interpret(state: SpecialistState):
    return [
        Send(
            "interpret",
            InterpretTask(
                question=_question(state),
                frame=L.frame_text(state),
                contrast=c.key,
                material=_material(state, c.key),
                addresses="\n".join(_addresses(state, c.key)),
                required="\n".join(_required(state, c.key)),
                primary_value=latest_primary(state)[c.key].value,
                tolerance=state["design"].effect_tolerance,
            ),
        )
        for c in state["contrasts"]
        if c.key in latest_primary(state)
    ]


def interpret(task: InterpretTask) -> dict:
    """One episode per contrast, no tools. After three refusals the last answer still stands in the report, marked as having
    failed its gate, so the reader sees what the model wrote and why it was refused."""
    allowed = set(task["addresses"].splitlines())
    required = [a for a in task["required"].splitlines() if a]
    contrast = task["contrast"]

    def gate(r: Interpretation, _log: EpisodeLog) -> list[str]:
        errs = [f"{c} is not an address you may cite" for c in r.cites if c not in allowed]
        if not r.cites:
            errs.append("no citations given")
        missing = [a for a in required if a not in r.cites]
        if missing:
            errs.append("these must be cited: " + ", ".join(missing))
        v = task["primary_value"]
        if v is not None and abs(r.effect_stated - v) > max(abs(v), 1e-9) * task["tolerance"]:
            errs.append(f"effect_stated {r.effect_stated} does not match the primary estimate {v:.4g}")
        return errs

    user = P.INTERPRET_USER.format(
        frame=task["frame"],
        question=task["question"],
        contrast=contrast,
        material=task["material"],
        addresses=task["addresses"],
        required="\n".join(required) or "(none)",
        errors="",
    )
    node = f"interpret:{contrast}"
    parsed, log, thoughts, errors = run_episode(Interpretation, P.INTERPRET_SYSTEM, user, tools=None, budget=0, gate=gate, node=node)
    if parsed is None:
        parsed = Interpretation.model_validate(log.tries[-1].answer)
    out: dict[str, Any] = {"interpretations": [parsed.model_copy(update={"contrast": contrast})], "debug": thoughts, "episodes": {node: log}}
    if errors:
        out["interpret_errors"] = {contrast: errors}
    return out


# ------------------------------------------------------------------ feasibility, figures, assemble (facts)


def figures(state: SpecialistState) -> dict:
    """What this run drew, checked against the addresses it produced: the graph it built, the balance of every adjustment
    column before and after weighting on the score, and the estimate against its falsifications."""
    from causal_agent.families.adjustment import postviz as PA

    h = state["handoff"]
    names = state.get("columns") or {}
    g = state.get("graph")
    specs = [PA.causal_graph(g.model_dump() if g is not None else {}, names)]
    thr = float(load_checks()["balance"]["smd"]["soft"])
    for c, facts in ((state.get("check_facts") or {}).get("balance") or {}).items():
        specs.append(PA.balance(facts, c, thr, names))
    ests = [e.model_dump() for e in state.get("estimates") or []]
    specs.append(PV.effect_and_refutations([e for e in ests if e.get("modifier") is None], [r.model_dump() for r in state.get("refutations") or []], "refute"))
    d = state.get("design")
    for c in d.contrasts if d is not None else []:
        specs.append(PV.effect_by_modifier(ests, c.key, names))
    kept, declines = LF.write(state.get("run_dir"), specs, LF.ok_addresses(h, state, "refute"))
    _writer()({"figures": [s.id for s in kept]})
    return {"figures": [s.model_dump() for s in kept], "declines": declines}


def assemble(state: SpecialistState) -> dict:
    h = state["handoff"]
    names = state.get("columns") or {}
    d: Design | None = state.get("design")
    f: Feasibility | None = state.get("feasibility")
    lines = [f"QUESTION     {_question(state)}", f"LANE         {h.family} → {h.specialist}", f"OUTCOME      {h.outcome}    TREATMENT   {h.treatment}", ""]
    lad = ladder_of(state)
    if lad.lines():
        lines += ["THE LADDER"] + [f"  [{a}] {text}" for a, text in lad.lines()] + [""]
    episodes = episodes_of(state)
    if any(log.facts or log.refusals or any(t.errors for t in log.tries) for log in episodes.values()):
        lines.append("WHAT THE EPISODES LOOKED AT")
        for name, log in episodes.items():
            lines.append(f"  {name}: {log.calls} call{'s' if log.calls != 1 else ''}, {len(log.tries)} tr{'ies' if len(log.tries) != 1 else 'y'}")
            lines += [f"    {f.render()}" for f in log.facts]
            lines += [f"    refused {r.tool}({', '.join(f'{k}={v!r}' for k, v in r.args.items())}): {r.reason}" for r in log.refusals]
            lines += [f"    try {t.n} refused: " + "; ".join(t.errors) for t in log.tries if t.errors]
        lines.append("")
    if d:
        lines += d.render().splitlines()
        lines.append("")
        lines.append("RESULTS")
        for c in d.contrasts:
            lines.append(f"  {names.get(d.graph.treatment, d.graph.treatment)} = {c.treated!r} vs {c.control!r}")
            for e in state.get("estimates") or []:
                if e.contrast == c.key:
                    label = f"{e.method:34}" if e.modifier is None else f"  within {names.get(e.modifier, e.modifier)} = {e.level:<16}"[:34].ljust(34)
                    if e.error:
                        lines.append(f"    {label} FAILED: {e.error}")
                    else:
                        ci = f" [{e.ci_low:.3g}, {e.ci_high:.3g}]" if e.ci_low is not None else ""
                        kind = "within a level" if e.modifier is not None else "secondary" if e.secondary else "primary"
                        lines.append(f"    {label} {e.value:+.4g}{ci}  n={e.n_treated}/{e.n_control}  {kind}")
            words = {rf.name: rf.in_words for rf in d.refuters}
            for r in state.get("refutations") or []:
                if r.contrast == c.key:
                    lines.append(f"    refute {r.refuter:27} {r.detail}  ({words.get(r.refuter, '')})")
            for i in state.get("interpretations") or []:
                if i.contrast == c.key:
                    lines.append(f"    ANSWER  {i.answer}")
                    for cv in i.caveats:
                        lines.append(f"    CAVEAT  {cv}")
                    lines.append(f"    CITES   {', '.join(i.cites)}")
            errs = (state.get("interpret_errors") or {}).get(c.key)
            if errs:
                lines.append(f"    INTERPRETATION GATE FAILED: {'; '.join(errs)}")
    if f:
        lines += ["", f"STOPPED AT   {f.stage}", f"REASON       {f.reason}"]
        lines += [f"FACT         {x}" for x in f.facts]
        lines.append(f"WOULD FIX    {f.what_would_fix}")
    lines += records.report_tail(state)
    debug = state.get("debug") or []
    if any(t.text for t in debug):
        lines += ["", "MODEL THOUGHTS (debug only)"]
        for t in debug:
            if t.text:
                lines.append(f"  [{t.node}] {t.text.strip()[:2000]}")
    report = "\n".join(lines)
    g = state.get("graph")
    records.write(state.get("run_dir"), records.artifacts(state, {"graph": g.model_dump() if g else None, "estimand": state.get("estimand")}), report)
    result = records.result(
        state,
        report,
        {
            "ladder": [[a, text] for a, text in lad.lines()],
            "facts": [{"address": f.address, "text": f.render().split("] ", 1)[1], "value": f.value} for log in episodes.values() for f in log.facts],
            "relations": [
                {
                    "column": x.column,
                    "stands_for": x.stands_for,
                    "redundant_with": x.redundant_with,
                    "nested_in": x.nested_in,
                    "modifier_candidate": x.modifier_candidate,
                }
                for x in (lad.roles.items if lad.roles is not None else [])
            ],
        },
    )
    _writer()({"report": report})
    return {"report": report, "specialist_result": result}
