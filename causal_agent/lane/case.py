"""The pack weighed by code. A confirmed field is a fact the lane takes; a drafted or empty one is open and may be asked
of the model; a refuted or contradicted one is contested. The person's beliefs, the unknowns, and the contradictions
become flags by the lane's `knowledge/beliefs.yaml`, and a flag is a check the assessment must answer.

    case = weigh(h, load_beliefs())
    case.facts["col:lunch.when"] == "before"
    checks += as_checks(case)
    action, payload, checks = decide_by_code(case, checks, rules, h)      # "stop" | "ask" | "proceed"

beliefs.yaml, per lane:

    beliefs:
      <kind>:                              # a key of Handoff.beliefs, or with `from: assignment.movable` a dataset field
        value_field: believed              # the field the value sits in; the address is claim:<kind>.<value_field>
        by_status:                         # confirmed_true, confirmed_false, drafted, unknown, contradiction, empty
          confirmed_false: {level: hard, caveat: "..."}
          empty: {level: soft, caveat: "...", ask: {question: "...", options: [yes, no]}}
        with_check:                        # the belief against a data check of the same run
          <check name>:
            - {belief: confirmed_true, check: hard, then: ask, once: true, question: "...", options: [yes, no]}
            - {belief: confirmed_true, check: hard, asked: true, then: soften, caveat: "..."}
            - {belief: confirmed_false, check: hard, then: stop, reason: "..."}
            - {belief: confirmed_true, check: soft, then: harden, unless_fact: "col:{score}.set_by", check_value_lt: 0.10, caveat: "..."}
    unknowns:
      "col:*.when": {level: soft, caveat: "..."}
      "claim:change.period_value": {level: hard, ask: {question: "..."}}
    contradictions:
      "col:*.when": {level: soft, caveat: "..."}

`{value}` is the check's value, `{said}` the person's sentence, `{column}` the column, `{check}` the check address,
`{score}` and the like any key of the family block. No dataset name belongs in a beliefs file."""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from causal_agent.common.contracts import Belief, CheckResult, ColumnBrief, Handoff, LaneAsk

Weight = Literal["fact", "draft", "open", "contested"]
Level = Literal["pass", "soft", "hard", "stop"]
FIELDS = ("when", "moved_by_change", "measures_outcome", "set_by")
_FIELD_ADDRESS = {"moved_by_change": "moved"}  # the brief renders col:x.moved; the memory holds col:x.moved_by_change


def load_beliefs(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text()) or {}


# ------------------------------------------------------------------ weights


def weigh_provenance(status: str | None, source: str | None) -> Weight:
    """How much a field's provenance is worth to code."""
    if status == "confirmed":
        return "fact" if (source or "").split(":")[0] in ("user", "doc", "data", "code") or source is None else "draft"
    if status == "drafted":
        return "draft"
    if status in ("refuted", "contradiction"):
        return "contested"
    return "open"


def settled(brief: ColumnBrief, field: str) -> tuple[Weight, Any]:
    """The weight and the value of one field of a brief."""
    value = getattr(brief, field, None)
    p = brief.provenance.get(field)
    if value is None or (field == "when" and value == "unknown"):
        return ("contested" if p and p.status in ("refuted", "contradiction") else "open"), None
    if p is None:
        return ("fact" if brief.source and brief.source.split(":")[0] in ("user", "doc", "data") else "draft"), value
    return weigh_provenance(p.status, p.source), value


def belief_status(b: Belief | None) -> str:
    """confirmed_true | confirmed_false | drafted | unknown | contradiction | empty."""
    if b is None or b.status in ("empty", None):
        return "empty"
    if b.status == "confirmed" and b.value is not None:
        return "confirmed_true" if b.value else "confirmed_false"
    if b.status == "drafted":
        return "drafted"
    if b.status == "unknown":
        return "unknown"
    if b.status in ("refuted", "contradiction"):
        return "contradiction"
    return "empty"


# ------------------------------------------------------------------ the case


class Flag(BaseModel):
    name: str  # belief.<kind> | unknown.<address> | contradiction.<address>
    level: Level = "soft"
    caveat: str = ""
    cites: list[str] = Field(default_factory=list)
    ask: LaneAsk | None = None
    address: str = ""
    status: str = ""  # the belief status key the flag came from


class Case(BaseModel):
    facts: dict[str, Any] = Field(default_factory=dict)
    drafts: dict[str, Any] = Field(default_factory=dict)
    open: list[str] = Field(default_factory=list)
    contested: list[str] = Field(default_factory=list)
    flags: list[Flag] = Field(default_factory=list)
    beliefs: dict[str, str] = Field(default_factory=dict)  # kind -> status key

    def fact(self, address: str, default: Any = None) -> Any:
        return self.facts.get(address, default)

    def is_fact(self, address: str) -> bool:
        return address in self.facts

    def flag(self, name: str) -> Flag | None:
        return next((f for f in self.flags if f.name == name), None)

    def render(self) -> str:
        lines = ["SETTLED BY THE PACK (facts; do not answer these again)"]
        lines += [f"  [{a}] {v}" for a, v in self.facts.items()] or ["  (nothing)"]
        if self.drafts or self.open:
            lines.append("OPEN (drafted or not said; a judgement may fill these)")
            lines += [f"  [{a}] drafted {v}" for a, v in self.drafts.items()]
            lines += [f"  [{a}] not said" for a in self.open]
        if self.contested:
            lines.append("CONTESTED (the file and the person disagree; weigh it and say which you took)")
            lines += [f"  [{a}]" for a in self.contested]
        if self.flags:
            lines.append("FLAGS FROM WHAT THE PERSON SAID")
            lines += [f"  {f.name} ({f.level}): {f.caveat}" for f in self.flags]
        return "\n".join(lines)


def _address(kind: str, spec: dict) -> str:
    if spec.get("address"):
        return spec["address"]
    if str(spec.get("from") or "").startswith("design."):
        return str(spec["from"])
    if spec.get("from"):
        return f"claim:{spec['from']}"
    return f"claim:{kind}.{spec.get('value_field', 'believed')}"


def _belief(h: Handoff, kind: str, spec: dict) -> Belief | None:
    """The belief by kind, or a dataset field read as one (`from: assignment.movable`), or a family-block field
    (`from: design.score_fixed_before`), whose status is confirmed when the desk could say and empty when it could not."""
    if str(spec.get("from") or "").startswith("design."):
        fld = str(spec["from"]).split(".", 1)[1]
        value = getattr(h.design, fld, None) if h.design is not None else None
        return Belief(kind=kind, value=bool(value) if value is not None else None, status="confirmed" if value is not None else "empty")
    if spec.get("from"):
        claim, _, fld = str(spec["from"]).partition(".")
        d = getattr(h, claim, None) or {}
        c = (h.claims or {}).get(claim) or {}
        value = d.get(fld) if isinstance(d, dict) else None
        addr = f"claim:{claim}.{fld}"
        own = (c.get("fields_status") or {}).get(fld)  # the field's own status when the pack carries it; the claim's is a poor proxy
        status = (
            "unknown"
            if addr in h.unknowns
            else "contradiction"
            if addr in h.contradictions
            else own or ("empty" if value is None else c.get("status") or "confirmed")
        )
        return Belief(kind=kind, value=bool(value) if value is not None else None, status=status, source=c.get("source"))
    return h.beliefs.get(kind)


def _fill(text: str, h: Handoff, **extra) -> str:
    """The yaml's {placeholders} from the family block, the pack, and the caller."""
    values: dict[str, Any] = {}
    if h.design is not None:
        values.update({k: v for k, v in h.design.model_dump().items() if isinstance(v, (str, int, float)) or v is None})
    values.update({"treatment": h.treatment, "outcome": h.outcome})
    values.update(extra)

    class _Safe(dict):
        def __missing__(self, k):
            return "{" + k + "}"

    try:
        return text.format_map(_Safe({k: ("" if v is None else v) for k, v in values.items()}))
    except (ValueError, IndexError):
        return text


def _ask(spec: dict, address: str, h: Handoff, *, because: str = "", evidence: list[str] | None = None, **fill) -> LaneAsk:
    options = ["yes" if o is True else "no" if o is False else str(o) for o in spec.get("options") or []]  # yaml reads a bare yes as True
    return LaneAsk(
        address=address,
        question=_fill(str(spec.get("question") or ""), h, **fill),
        options=options,
        because=_fill(because or str(spec.get("because") or ""), h, **fill),
        evidence=list(evidence or []),
    )


def weigh(h: Handoff, rules: dict | None) -> Case:
    """The pack as facts, drafts, open fields, contested fields, and flags."""
    rules = rules or {}
    case = Case()
    for b in h.columns:
        if b.role in ("outcome", "treatment"):
            continue
        for fld in FIELDS:
            w, v = settled(b, fld)
            addr = f"{b.address}.{fld}"
            if w == "fact":
                case.facts[addr] = v
            elif w == "draft":
                case.drafts[addr] = v
            elif w == "contested":
                case.contested.append(addr)
            elif fld != "set_by":
                case.open.append(addr)
    for kind, spec in (rules.get("beliefs") or {}).items():
        bel = _belief(h, kind, spec)
        st = belief_status(bel)
        case.beliefs[kind] = st
        addr = _address(kind, spec)
        if st.startswith("confirmed") and bel is not None:
            case.facts[addr] = bel.value
        by = (spec.get("by_status") or {}).get(st) or {}
        if by:
            said = (bel.said if bel else None) or ""
            caveat = _fill(str(by.get("caveat") or ""), h, said=said, column="")
            ask = _ask(by["ask"], addr, h, said=said) if by.get("ask") else None
            case.flags.append(Flag(name=f"belief.{kind}", level=by.get("level", "soft"), caveat=caveat, cites=[addr], ask=ask, address=addr, status=st))
    for addr in h.unknowns:
        spec = _match(rules.get("unknowns") or {}, addr)
        if spec:
            col = addr.split(":", 1)[-1].split(".")[0] if addr.startswith("col:") else ""
            ask = _ask(spec["ask"], addr, h, column=col) if spec.get("ask") else None
            case.flags.append(
                Flag(
                    name=f"unknown.{addr}",
                    level=spec.get("level", "soft"),
                    caveat=_fill(str(spec.get("caveat") or ""), h, column=col, check=""),
                    cites=[addr],
                    ask=ask,
                    address=addr,
                    status="unknown",
                )
            )
    for addr in h.contradictions:
        spec = _match(rules.get("contradictions") or {}, addr)
        if spec:
            col = addr.split(":", 1)[-1].split(".")[0] if addr.startswith("col:") else ""
            case.flags.append(
                Flag(
                    name=f"contradiction.{addr}",
                    level=spec.get("level", "soft"),
                    caveat=_fill(str(spec.get("caveat") or ""), h, column=col, check=""),
                    cites=[addr],
                    address=addr,
                    status="contradiction",
                )
            )
    return case


def _match(table: dict, address: str) -> dict | None:
    for pattern, spec in table.items():
        if fnmatch.fnmatchcase(address, pattern):
            return spec
    return None


# ------------------------------------------------------------------ flags as checks, and the code's decision


def as_checks(case: Case) -> list[CheckResult]:
    """A flag is a check with contrast 'all'; a stop-level flag is a hard check here and a stop in decide_by_code."""
    out = []
    for f in case.flags:
        if f.level == "pass":
            continue
        out.append(CheckResult(contrast="all", name=f.name, level="hard" if f.level == "stop" else f.level, detail=f.caveat))
    return out


def already_asked(h: Handoff, address: str) -> bool:
    """The desk remembers a turn that answered a lane's question as about lane:<address>; an address asked once is not asked
    again across designs. The person's own earlier sentence about the field does not count as an answer to the lane."""
    return any(f"lane:{address}" in (s.about or "") for s in h.said)


def decide_by_code(case: Case, checks: list[CheckResult], rules: dict | None, h: Handoff) -> tuple[str, Any, list[CheckResult]]:
    """What the yaml decides before any judgement: ("stop", Feasibility-like dict), ("ask", LaneAsk), or ("proceed", None) with
    the checks re-levelled. Order: stops, then asks, then level edits and caveats."""
    rules = rules or {}
    checks = [c.model_copy() for c in checks]
    by_name: dict[str, list[CheckResult]] = {}
    for c in checks:
        by_name.setdefault(c.name, []).append(c)
    stops: list[dict] = []
    asks: list[LaneAsk] = []
    edits: list[tuple[CheckResult, str, str]] = []
    for f in case.flags:
        if f.level == "stop":
            stops.append(
                {
                    "stage": "assess",
                    "reason": f.caveat,
                    "facts": [f"[{f.address}] {f.status}"],
                    "what_would_fix": f"a different answer at [{f.address}]",
                    "flag": f.name,
                }
            )
    for kind, spec in (rules.get("beliefs") or {}).items():
        st = case.beliefs.get(kind, "empty")
        addr = _address(kind, spec)
        b = _belief(h, kind, spec)
        said = (b.said if b else None) or ""
        for check_name, rows in (spec.get("with_check") or {}).items():
            for c in by_name.get(check_name, []):
                for row in rows:
                    if row.get("belief") != st or row.get("check") != c.level:
                        continue
                    if "check_value_lt" in row and (c.value is None or c.value >= float(row["check_value_lt"])):
                        continue  # the check is flagged for another reason (uninformative, not computable); the number is not a jump
                    asked = already_asked(h, addr)
                    if row.get("asked") is True and not asked:
                        continue
                    if row.get("once") and asked and row.get("then") == "ask":
                        continue
                    unless = row.get("unless_fact")
                    if unless and case.is_fact(_fill(str(unless), h)):
                        continue
                    fill = dict(value=f"{c.value:.3g}" if isinstance(c.value, (int, float)) else c.value, said=said, check=c.address, column="")
                    then = row.get("then")
                    if then == "stop":
                        flag = case.flag(f"belief.{kind}")
                        stops.append(
                            {
                                "stage": "assess",
                                "reason": _fill(str(row.get("reason") or (flag.caveat if flag else "")), h, **fill),
                                "facts": [f"[{c.address}] {c.level}: {c.detail}", f"[{addr}] {st}" + (f' said "{said}"' if said else "")],
                                "what_would_fix": f"a different answer at [{addr}], or data on which [{c.address}] passes",
                                "flag": f"belief.{kind}",
                            }
                        )
                    elif then == "ask":
                        asks.append(_ask(row, addr, h, because=row.get("because") or f"[{c.address}] {c.detail}", evidence=[c.address], **fill))
                    elif then in ("soften", "harden", "caveat"):
                        edits.append((c, then, _fill(str(row.get("caveat") or ""), h, **fill)))
    for f in case.flags:
        if f.ask is not None and f.level != "stop" and not already_asked(h, f.address):
            asks.append(f.ask)
    if stops:
        return "stop", stops[0], checks
    if asks:
        return "ask", asks[0], checks
    for c, then, caveat in edits:
        if then == "soften":
            c.level = "soft"
        elif then == "harden":
            c.level = "hard"
        if caveat:
            c.detail = (c.detail + " · " + caveat) if c.detail else caveat
    return "proceed", None, checks
