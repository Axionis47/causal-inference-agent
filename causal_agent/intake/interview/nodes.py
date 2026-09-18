"""Interview nodes. Facts: load, check, probe, status, write_pack. Judgements, one model call each with a gate:
extract, respond. listen is the interrupt. No node names a column, a dataset, or a method."""

from __future__ import annotations

import re
from typing import Any, Literal

from langgraph.types import Command, interrupt

from causal_agent.common.contracts import Thought
from causal_agent.common.llm import structured
from causal_agent.memory import checks as C
from causal_agent.profile import data as D
from causal_agent.intake.interview import prompts as P
from causal_agent.memory import table as T
from causal_agent.memory.claims import Claim, ClaimTable, ClaimUpdate, Extraction, Question, Reply, Status
from causal_agent.memory.probes import run_probes
from causal_agent.intake.interview.state import InterviewState
from causal_agent.intake.interview.writer import write_dataset
from causal_agent.memory.catalogue import Catalogue, ClaimKind, load_catalogue, load_thresholds
from causal_agent.common.addresses import key as _key
from causal_agent.knowledge import load_registry

CAT: Catalogue = load_catalogue()
TH: dict = load_thresholds()

METRIC_RE = re.compile(r"\bhow many\b|\bhow much of\b|\bwhat (share|percent|percentage|proportion|fraction|rate)\b|\bnumber of rows\b|\brow count\b", re.I)


# ------------------------------------------------------------------ helpers


def _entity_time(table: ClaimTable, df) -> tuple[list[str] | None, str | None]:
    """Profiler flags from the settled claims: the change's period column as time, the other key columns as entity."""
    g, ch = table.get("grain"), table.get("change")
    time = D.column(df, ch.fields.get("date_column")) if ch and ch.status != "empty" else None
    if time is not None and not (df[time].dtype.kind in "iuf" or _looks_like_dates(df[time])):
        time = None
    keys = [D.column(df, k) for k in (g.fields.get("key_columns") or [])] if g and g.status != "empty" else []
    entity = [k for k in keys if k is not None and k != time]
    if not (g and g.fields.get("panel") is True):  # an entity is a unit seen more than once; a one-row-per-unit file declares none
        entity = []
    return (entity or None), time


def _looks_like_dates(s) -> bool:
    try:
        import pandas as pd

        sample = s.dropna().astype(str).head(200)
        return bool(len(sample)) and pd.to_datetime(sample, errors="coerce").notna().mean() >= 0.9
    except Exception:
        return False


def _data(state: InterviewState, table: ClaimTable | None = None):
    df, _ = D.load(state["csv"])
    entity, time = _entity_time(table, df) if table else (None, None)
    try:
        return D.load(state["csv"], entity, time)
    except ValueError:
        return D.load(state["csv"])


def _kinds_text() -> str:
    out = []
    for k in CAT.ordered():
        fields = "; ".join(
            f"{name} ({spec.type}{': ' + ', '.join(f'{o} = {spec.about[str(o)]}' if str(o) in spec.about else str(o) for o in spec.options) if spec.options else ''}{', optional' if spec.optional else ''})"
            + (f" = {spec.hint}" if spec.hint else "")
            for name, spec in k.fields.items()
        )
        tag = " [per column: give column]" if k.per_column else " [uncheckable: only on the person's word]" if k.uncheckable else ""
        cues = f" Reading their words: {k.cues}" if k.cues else ""
        out.append(f"- {k.name}{tag}: {k.about}. Fields: {fields}.{cues}")
    return "\n".join(out)


def _seed(prof) -> ClaimTable:
    t = ClaimTable()
    for k in CAT.ordered():
        if k.per_column:
            for cp in prof.columns:
                if not cp.constant:
                    t.claims[f"col:{cp.key}"] = Claim(kind=k.name, key=f"col:{cp.key}")
        else:
            t.claims[k.name] = Claim(kind=k.name, key=k.name)
    return t


def _truthy(v: Any) -> bool | None:
    return C._truthy(v)


def _coerce(kind: ClaimKind, name: str, raw: str, df) -> tuple[Any, str | None]:
    spec = kind.fields.get(name)
    if spec is None:
        return None, f"{kind.name} has no field {name!r}"
    s = str(raw).strip()
    if s.lower() in {"", "null", "none"}:
        return None, None if spec.optional else f"{kind.name}.{name} needs a value"
    if spec.type == "text":
        return s, None
    if spec.type == "choice":
        for o in spec.options:
            if s.lower() == str(o).lower():
                return o, None
        return None, f"{kind.name}.{name} must be one of {spec.options}, not {s!r}"
    if spec.type == "bool":
        b = _truthy(s)
        return (b, None) if b is not None else (None, f"{kind.name}.{name} must be true or false, not {s!r}")
    if spec.type == "number":
        try:
            return float(s), None
        except ValueError:
            return None, f"{kind.name}.{name} must be a number, not {s!r}"
    if spec.type == "column":
        c = D.column(df, s)
        return (c, None) if c is not None else (None, f"{kind.name}.{name}: {s!r} is not a column in the file")
    if spec.type == "columns":
        cols, bad = [], []
        for part in [p.strip() for p in s.split(",") if p.strip()]:
            c = D.column(df, part)
            (cols if c is not None else bad).append(c or part)
        return (cols, None) if not bad else (None, f"{kind.name}.{name}: not columns in the file: {bad}")
    return None, f"unknown field type {spec.type}"


def _apply(table: ClaimTable, up: ClaimUpdate, df, allowed_cites: set[str], turn: int) -> str | None:
    kind = CAT.kinds.get(up.kind)
    if kind is None:
        return f"unknown claim kind {up.kind!r}"
    bad = [c for c in up.cites if c not in allowed_cites and not c.startswith("col:") and not c.startswith("dataset")]
    bad += [c for c in up.cites if (c.startswith("col:") or c.startswith("dataset")) and not _card_ok(c, df)]
    if bad:
        return f"{up.kind}: cites do not resolve: {bad}"
    grounded = [c for c in up.cites if c.startswith("doc:") or c.startswith("user:turn:")]
    if not grounded:
        return f"{up.kind}: needs a doc or user cite; the file's numbers alone do not make a claim"
    if kind.per_column:
        if not up.column:
            return f"{kind.name}: a per-column claim needs a column"
        col = D.column(df, up.column)
        if col is None:
            return f"{kind.name}: {up.column!r} is not a column in the file"
        key = f"col:{_key(col)}"
    else:
        key = kind.name
    claim = table.claims.get(key)
    if claim is None:
        return f"no claim slot for {key}"
    from_user = any(c.startswith("user:turn:") for c in up.cites)
    if claim.status == "confirmed" and not from_user:
        return f"{key} was confirmed by the person; only their word changes it"
    if kind.uncheckable and not from_user and not up.unknown:
        return f"{key} is asked of the person, never read from a description"
    source = next((c for c in up.cites if c.startswith("user:turn:")), None) or next(c for c in up.cites if c.startswith("doc:"))
    if up.unknown:
        claim.status, claim.source = "unknown", source
        claim.evidence = list(dict.fromkeys(claim.evidence + up.cites))
        return None
    new: dict[str, Any] = {}
    for fv in up.values:
        v, err = _coerce(kind, fv.name, fv.value, df)
        if err:
            return err
        new[fv.name] = v
    if not new:
        return f"{key}: no values given"
    claim.fields.update(new)
    claim.status = "confirmed" if from_user else "drafted"
    claim.source = source
    claim.evidence = list(dict.fromkeys(claim.evidence + up.cites))
    claim.check_detail = None
    return None


def _card_ok(address: str, df) -> bool:
    if address.startswith("dataset"):
        return True
    name = address[4:].partition(".")[0]
    return D.column(df, name) is not None


def _allowed_cites(state: InterviewState) -> set[str]:
    out = {f"doc:{n}" for n in (state.get("docs") or {})}
    out |= {f"user:turn:{i}" for i in range(0, int(state.get("turn", 0)) + 1)}
    return out


def _norm_key(key: str) -> str:
    """The model writes keys as claim:x, x.field, x:field, or col:Name.when; the claim key is x or col:<key>."""
    k = str(key).strip().removeprefix("claim:")
    if k.startswith("col:"):
        return "col:" + _key(k[4:].split(".")[0].split(":")[0])
    return re.split(r"[.:]", k)[0]


def _family_words() -> list[str]:
    words = list(CAT.method_words)
    for f in load_registry():
        words += [f.name, f.name.replace("_", " "), f.name.replace("_", "-")]
    return [w.lower() for w in words]


def _legal_options(kind: ClaimKind, field: str | None) -> list[str] | None:
    if field is None or field not in kind.fields:
        return None
    spec = kind.fields[field]
    if spec.type == "choice":
        return [str(o) for o in spec.options]
    if spec.type == "bool":
        return ["yes", "no"]
    return None


def _open_text(table: ClaimTable, status: Status) -> str:
    lines = []
    for key in status.open:
        c = table.claims[key]
        k = CAT.kinds[c.kind]
        lines.append(c.render())
        lines.append(f"  about: {k.frame}")
        for name, spec in k.fields.items():
            opts = _legal_options(k, name)
            lines.append(f"  field {name}: {spec.type}{' ' + str(opts) if opts else ''}{' (optional)' if spec.optional else ''}")
        lines.append(f"  ask as: {'confirm' if c.status in {'drafted', 'refuted'} else 'choose or open'}")
    return "\n".join(lines) or "(nothing open)"


def _fallback_reply(table: ClaimTable, status: Status) -> Reply:
    qs: list[Question] = []
    measured = [k for k in status.open if table.claims[k].kind == "measured"]
    others = [k for k in status.open if k not in measured]
    for key in others:
        c = table.claims[key]
        k = CAT.kinds[c.kind]
        if c.status in {"drafted", "refuted"}:
            vals = ", ".join(f"{a} = {b}" for a, b in c.fields.items() if b is not None)
            ev = [c.evidence[-1]] if c.evidence else []
            text = f"I read {k.name} as: {vals}." + (f" But the file says: {c.check_detail}." if c.status == "refuted" and c.check_detail else "") + " Is that right, or what should it be?"
            qs.append(Question(keys=[key], field=None, kind="confirm", text=text, evidence_cites=ev))
        else:
            field = next((n for n, s in k.fields.items() if not s.optional), next(iter(k.fields)))
            opts = _legal_options(k, field)
            if opts:
                qs.append(Question(keys=[key], field=field, kind="choose", text=f"{k.frame.capitalize()}? Options: {', '.join(opts)}.", options=opts))
            else:
                qs.append(Question(keys=[key], field=field, kind="open", text=f"{k.frame.capitalize()}? One sentence is enough."))
    if measured:
        drafted = [k for k in measured if table.claims[k].status in {"drafted", "refuted"}]
        empty = [k for k in measured if k not in drafted]
        if drafted:
            desc = "; ".join(f"{k[4:]}: {table.claims[k].fields.get('when')}" for k in drafted)
            qs.append(Question(keys=drafted, field="when", kind="confirm", text=f"I read these columns as set, relative to the change: {desc}. Any wrong?",
                               evidence_cites=[table.claims[k].address for k in drafted]))
        if empty:
            names = ", ".join(k[4:] for k in empty)
            qs.append(Question(keys=empty, field="when", kind="choose", text=f"For {names}: what does each measure, and was it fixed before the change, set at it, or measured after? Options: before, at, after, unknown.",
                               options=["before", "at", "after", "unknown"]))
    text = "\n".join(f"{i + 1}. {q.text}" for i, q in enumerate(qs))
    return Reply(questions=qs, text=text)


def _reply_gate(reply: Reply, table: ClaimTable, status: Status) -> list[str]:
    errors: list[str] = []
    open_set = set(status.open)
    covered: set[str] = set()
    words = _family_words()
    low = reply.text.lower()
    for w in words:
        if w in low:
            errors.append(f"the message names a method or a kind of study ({w!r}); ask about the world instead")
            break
    if METRIC_RE.search(reply.text):
        errors.append("the message asks for a count, share, or rate; the file answers those, never the person")
    for q in reply.questions:
        q.keys = [_norm_key(k) for k in q.keys]
        for key in q.keys:
            if key not in table.claims:
                errors.append(f"question names no claim: {key!r}")
                continue
            if key not in open_set:
                errors.append(f"question about {key}, which is settled; ask only about open claims")
            covered.add(key)
        if not q.keys:
            errors.append("a question names no claim key")
            continue
        claims = [table.claims[k] for k in q.keys if k in table.claims]
        if not claims:
            continue
        kind = CAT.kinds[claims[0].kind]
        if q.kind == "confirm" and any(c.status not in {"drafted", "refuted"} for c in claims):
            errors.append(f"confirm question for {q.keys} but there is no draft to confirm; use choose or open")
        if q.kind == "choose":
            if q.field is None:
                fixed = [n for n, sp in kind.fields.items() if sp.type in {"choice", "bool"} and not sp.optional] or [n for n, sp in kind.fields.items() if sp.type in {"choice", "bool"}]
                if len(fixed) == 1:
                    q.field = fixed[0]
            legal = _legal_options(kind, q.field)
            if legal is None:
                errors.append(f"choose question for {q.keys} on field {q.field!r}, which has no fixed options")
            elif {o.lower() for o in q.options} != {o.lower() for o in legal}:
                errors.append(f"choose question for {q.keys} must list exactly {legal}")
        if q.kind == "open" and q.field and q.field in kind.fields and kind.fields[q.field].type in {"choice", "bool"}:
            errors.append(f"open question for {q.keys} on {q.field}, which has fixed options; use choose")
        for c in claims:
            if c.status == "refuted" and c.evidence and not any(e.startswith("check:") for e in q.evidence_cites):
                errors.append(f"{c.key} was refuted by the file; the question must cite {c.evidence[-1]} and show its number")
    missing = open_set - covered
    if missing:
        errors.append(f"no question for open claims: {sorted(missing)}")
    measured_open = [k for k in status.open if table.claims[k].kind == "measured"]
    if len(measured_open) > int(TH["interview"]["group_measured_above"]):
        n = sum(1 for q in reply.questions if any(table.claims.get(k) and table.claims[k].kind == "measured" for k in q.keys))
        if n > 3:
            errors.append(f"{len(measured_open)} columns are open; group them into at most three questions, not {n}")
    return errors


# ------------------------------------------------------------------ nodes


def load(state: InterviewState) -> dict:
    existing = state.get("claims")
    if existing is not None and existing.claims:  # re-entry after a revision: keep the claims, read the new message
        return {"probes": state.get("probes") or [], "reply": None, "extract_errors": [], "extract_attempts": 0, "respond_errors": [], "respond_attempts": 0,
                "handoff_ready": False, "written": None}
    df, prof = D.load(state["csv"])
    docs = state.get("docs") or {}
    name = next(iter(docs), "context")
    text = docs.get(name, "")
    return {
        "turn": 0,
        "claims": _seed(prof),
        "probes": [],
        "status": None,
        "reply": None,
        "last_message": text,
        "last_source": f"doc:{name}",
        "messages": [{"role": "user", "turn": 0, "text": text}],
        "extract_errors": [],
        "extract_attempts": 0,
        "respond_errors": [],
        "respond_attempts": 0,
        "debug": [],
        "handoff_ready": False,
        "written": None,
    }


def extract(state: InterviewState) -> Command[Literal["extract", "check"]]:
    table = state["claims"].model_copy(deep=True)
    df, prof = _data(state, table)
    cards = D.cards(state["dataset"], prof)
    errs = state.get("extract_errors") or []
    errors = ("\nPREVIOUS UPDATES WERE REJECTED:\n" + "\n".join(f"- {e}" for e in errs) + "\nFix them and return the full set again.\n") if errs else ""
    prev = state.get("reply")
    asked = "\n".join(f"- about {', '.join(q.keys)}{' (' + q.field + ')' if q.field else ''}: {q.text}" for q in (prev.questions if prev else [])) or "(none: this is the description)"
    user = P.EXTRACT_USER.format(kinds=_kinds_text(), claims=table.render(), cards="\n".join(c.render() for c in cards.columns) + "\n" + cards.dataset.render(),
                                 asked=asked, source=state["last_source"], material=state["last_message"] or "(empty)", errors=errors)
    out, thought = structured(Extraction, P.EXTRACT_SYSTEM, user, node=f"extract:{state.get('turn', 0)}")
    allowed = _allowed_cites(state)
    rejected = [e for up in out.updates if (e := _apply(table, up, df, allowed, int(state.get("turn", 0))))]
    turn = int(state.get("turn", 0))
    for key in out.confirmed:
        c = table.claims.get(_norm_key(key)) or table.claims.get(f"col:{_key(key)}")
        if c is None:
            rejected.append(f"confirmed key {key!r} names no claim")
        elif not str(state.get("last_source", "")).startswith("user:"):
            rejected.append(f"{key}: a description cannot confirm a draft; only the person can")
        elif (missing := [n for n, sp in CAT.kinds[c.kind].fields.items() if not sp.optional and c.fields.get(n) is None]):
            rejected.append(f"{c.key} cannot be confirmed while {missing} is unset; give it as an update if the person's words say it, else leave it to be asked")
        elif c.status in {"drafted", "refuted"}:
            c.status, c.source, c.check_detail = "confirmed", f"user:turn:{turn}", None
            c.evidence = list(dict.fromkeys(c.evidence + [f"user:turn:{turn}"]))
    attempts = int(state.get("extract_attempts", 0)) + 1
    if rejected and attempts < int(TH["interview"]["extract_attempts"]):
        return Command(goto="extract", update={"claims": table, "extract_errors": rejected, "extract_attempts": attempts, "debug": [thought]})
    thoughts = [thought]
    # a second, focused pass over what was asked and is still open: one long turn settles many claims and drops one
    asked_keys = [k for q in (prev.questions if prev else []) for k in q.keys] or [c.key for c in table.claims.values() if not CAT.kinds[c.kind].uncheckable]
    still = [k for k in dict.fromkeys(asked_keys) if (c := table.claims.get(k)) and c.status in {"empty", "drafted", "refuted"}]
    if still and state["last_message"]:
        user2 = P.SWEEP_USER.format(open=table.render(still), asked=asked, source=state["last_source"], material=state["last_message"])
        out2, thought2 = structured(Extraction, P.EXTRACT_SYSTEM, user2, node=f"extract_sweep:{turn}")
        thoughts.append(thought2)
        for up in out2.updates:
            _apply(table, up, df, allowed, turn)  # a rejected sweep update is dropped; the first pass already had its retries
        for key in out2.confirmed:
            c = table.claims.get(_norm_key(key))
            if c is not None and c.status in {"drafted", "refuted"} and str(state.get("last_source", "")).startswith("user:") \
                    and not [n for n, sp in CAT.kinds[c.kind].fields.items() if not sp.optional and c.fields.get(n) is None]:
                c.status, c.source, c.check_detail = "confirmed", f"user:turn:{turn}", None
    return Command(goto="check", update={"claims": table, "extract_errors": rejected, "extract_attempts": 0, "debug": thoughts})


def check(state: InterviewState) -> dict:
    table = state["claims"].model_copy(deep=True)
    df, prof = _data(state, table)
    max_asks = int(TH["interview"]["max_asks"])
    for claim in table.claims.values():
        kind = CAT.kinds[claim.kind]
        if kind.check == "none":
            continue
        if claim.status == "empty" and claim.kind == "missing":  # informational before it is answered: the reply shows the gaps
            out = C.run(kind.check, claim, table, df, prof, TH)
            if out is not None and not any(cp.nulls for cp in prof.columns) is False:
                claim.check_detail, claim.evidence = out.detail, list(dict.fromkeys(claim.evidence + [out.address(claim)]))
            continue
        if claim.status not in {"drafted", "confirmed"}:
            continue
        out = C.run(kind.check, claim, table, df, prof, TH)
        if out is None:
            continue
        addr = out.address(claim)
        claim.evidence = list(dict.fromkeys([e for e in claim.evidence if not e.startswith("check:")] + [addr]))
        claim.check_detail = out.detail
        if out.passed is False:
            claim.refutations += 1
            claim.status = "contradiction" if claim.refutations >= max_asks else "refuted"
    m = table.get("missing")
    if m and m.status == "empty" and not any(cp.nulls for cp in prof.columns):
        m.fields, m.status, m.source = {"why": "none"}, "confirmed", "data"
        m.check_detail = "no column has missing values"
    return {"claims": table}


def probe(state: InterviewState) -> dict:
    table = state["claims"]
    a, ch = table.get("assignment"), table.get("change")
    if not (a and a.status in {"confirmed", "drafted"} and ch and ch.status in {"confirmed", "drafted"}):
        return {"probes": []}
    df, _ = _data(state, table)
    return {"probes": run_probes(df, table, list(CAT.families), TH)}


def status(state: InterviewState) -> dict:
    return {"status": T.compute(CAT, state["claims"], state.get("probes") or [])}


def respond(state: InterviewState) -> Command[Literal["respond", "listen", "write_pack"]]:
    table, st = state["claims"], state["status"]
    assert st is not None
    if st.ready and state.get("run_requested"):
        return Command(goto="write_pack", update={"run_requested": False})
    if not st.open:
        text = "Everything the analysis needs is settled." + (" Say run to hand off, or tell me anything to change." if st.ready else " No design fits yet: " + "; ".join(f"{f} ({w})" for f, w in st.struck.items()))
        return Command(goto="listen", update={"reply": Reply(questions=[], text=text), "respond_errors": [], "respond_attempts": 0})
    prev = [m for m in (state.get("messages") or []) if m["role"] == "user"]
    turn = int(state.get("turn", 0))
    tag = f"turn:{turn}" if turn else "doc:"
    settled_now = [c.render() for c in table.claims.values() if c.source and (c.source.endswith(tag) if turn else c.source.startswith(tag)) and c.status in {"confirmed", "drafted", "unknown"}] or ["(nothing new)"]
    errs = state.get("respond_errors") or []
    errors = ("\nPREVIOUS REPLY WAS REJECTED:\n" + "\n".join(f"- {e}" for e in errs) + "\n") if errs else ""
    n_cols = sum(1 for k in st.open if table.claims[k].kind == "measured")
    rule = f"\n{n_cols} columns are open: cover them in at most three questions, grouped by what they share.\n" if n_cols > int(TH["interview"]["group_measured_above"]) else ""
    last = prev[-1]["text"] if prev else "(none)"
    if state.get("run_requested"):
        last += f"\n(they asked to run; the drafts were taken on their word, but these claims are still open and must be settled first: {', '.join(st.open)})"
    user = P.RESPOND_USER.format(settled=("READ FROM THE DESCRIPTION, AS DRAFTS TO CONFIRM\n" if not turn else "") + "\n".join(settled_now), open=_open_text(table, st) + rule, last=last, errors=errors)
    out, thought = structured(Reply, P.RESPOND_SYSTEM, user, node=f"respond:{turn}")
    gate = _reply_gate(out, table, st)
    attempts = int(state.get("respond_attempts", 0)) + 1
    if gate and attempts < int(TH["interview"]["respond_attempts"]):
        return Command(goto="respond", update={"respond_errors": gate, "respond_attempts": attempts, "debug": [thought]})
    if gate:
        out = _fallback_reply(table, st)
        gate = [f"fell back to templated questions after {attempts} tries: " + "; ".join(gate)]
    out = _compose(out)
    return Command(goto="listen", update={"reply": out, "respond_errors": gate, "respond_attempts": 0, "debug": [thought]})


def _compose(reply: Reply) -> Reply:
    """The person sees every gated question, numbered, after the model's text; a question already in the text is not repeated."""
    missing = [q for q in reply.questions if q.text.strip() and q.text.strip()[:40] not in reply.text]
    if not missing:
        return reply
    lines = [reply.text.strip()] if reply.text.strip() else []
    lines += [f"{i + 1}. {q.text.strip()}" + (f" ({' / '.join(q.options)})" if q.kind == "choose" and q.options and not any(o in q.text for o in q.options) else "") for i, q in enumerate(missing)]
    return reply.model_copy(update={"text": "\n\n".join(lines)})


def listen(state: InterviewState) -> Command[Literal["extract", "write_pack", "__end__"]]:
    st, reply = state["status"], state["reply"]
    payload = {"text": reply.text if reply else "", "status": st.render(list(CAT.kinds)) if st else "", "ready": bool(st and st.ready), "open": list(st.open) if st else []}
    answer = str(interrupt(payload) or "").strip()
    low = answer.lower()
    if low in {"quit", "exit"}:
        return Command(goto="__end__", update={"handoff_ready": False})
    if low in {"run", "go"} and st and st.ready:
        return Command(goto="write_pack")
    turn = int(state.get("turn", 0)) + 1
    if low in {"run", "go"}:
        # "run" is the person's word that the drafts read from their description stand; empty and refuted claims stay open
        table = state["claims"].model_copy(deep=True)
        confirmed = []
        for c in table.claims.values():
            if c.status == "drafted" and not [n for n, sp in CAT.kinds[c.kind].fields.items() if not sp.optional and c.fields.get(n) is None]:
                c.status, c.source, c.check_detail = "confirmed", f"user:turn:{turn}", None
                confirmed.append(c.key)
        return Command(goto="check", update={"claims": table, "turn": turn, "last_message": answer, "last_source": f"user:turn:{turn}", "run_requested": True,
                                             "messages": [{"role": "assistant", "turn": turn - 1, "text": reply.text if reply else ""}, {"role": "user", "turn": turn, "text": answer}],
                                             "extract_errors": [], "extract_attempts": 0, "respond_errors": [], "respond_attempts": 0})
    return Command(goto="extract", update={"turn": turn, "last_message": answer, "last_source": f"user:turn:{turn}", "run_requested": False,
                                           "messages": [{"role": "assistant", "turn": turn - 1, "text": reply.text if reply else ""}, {"role": "user", "turn": turn, "text": answer}],
                                           "extract_errors": [], "extract_attempts": 0, "respond_errors": [], "respond_attempts": 0})


def write_pack(state: InterviewState) -> dict:
    table = state["claims"]
    df, prof = _data(state, table)
    entity, time = _entity_time(table, df)
    written = write_dataset(state["dataset"], state["csv"], prof, table, state.get("probes") or [], entity=entity, time=time)
    from causal_agent.common.contracts import Said
    from causal_agent.memory import store
    from causal_agent.memory.records import Memory

    said = [Said(turn=int(m["turn"]), about="", text=str(m["text"])) for m in (state.get("messages") or []) if m.get("role") == "user" and m.get("text")]
    store.save(Memory.from_claims(state["dataset"], table, profile=prof, csv=written["entry"]["csv"], said=said))
    return {"written": written, "handoff_ready": True}
