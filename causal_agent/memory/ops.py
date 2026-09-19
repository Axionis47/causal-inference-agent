"""Operations on a memory. Facts, all of them: the model never touches a memory except through `apply`, which gates what
a judgement returned. `seed` from the profile, `apply` for writes, `check` for the data facts and the consistency rules,
`probe` and `fit` for the families, `open` for what is still vague. `roles` is a view: what each column is to the question,
computed from the map every time and never written back.

The checks, probes, and family table still run on the claim-table view (`Memory.to_claims`) until the desk is rewritten on
the memory itself; the consistency rules and `open` run on the map directly."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field as PField

from causal_agent.common.addresses import key as _key
from causal_agent.memory import checks as C
from causal_agent.memory import table as T
from causal_agent.memory.catalogue import Catalogue, ClaimKind, load_catalogue, load_thresholds
from causal_agent.memory.claims import ProbeResult, Status
from causal_agent.memory.probes import run_probes
from causal_agent.memory.records import COLUMN_KIND, Field, Memory

WriteStatus = Literal["drafted", "confirmed", "unknown"]

# ------------------------------------------------------------------ seed


def seed_facts(m: Memory, profile) -> None:
    """The two dataset fields the profile settles on its own, written only where nothing is known yet."""
    if not any(cp.nulls for cp in profile.columns) and m.value("claim:missing.why") is None:
        m.set("claim:missing.why", "none", status="confirmed", source="data")
    es = profile.dataset.entity_summary
    if es is not None:
        if m.value("claim:grain.panel") is None:
            m.set("claim:grain.panel", bool(es.rows_per_entity_max > 1), status="confirmed", source="data")
        if m.value("claim:grain.key_columns") is None:
            m.set("claim:grain.key_columns", list(es.columns) + ([profile.dataset.time_coverage.column] if profile.dataset.time_coverage else []),
                  status="drafted", source="data")


def seed(name: str, profile, csv: str | None = None, cat: Catalogue | None = None) -> Memory:
    """A memory from the file alone: the facts on every column, and the dataset fields the profile settles. No field is
    written until someone says something about it."""
    from causal_agent.memory.claims import ClaimTable

    m = Memory.from_claims(name, ClaimTable(), profile=profile, csv=csv, cat=cat or load_catalogue())
    seed_facts(m, profile)
    return m


# ------------------------------------------------------------------ apply (the one write path)


class Update(BaseModel):
    """What a judgement, a check, or the person asks to write."""

    address: str = PField(description="col:<key>.<field> or claim:<kind>.<field>")
    value: Any = None
    status: WriteStatus = "drafted"
    source: str = PField(description="user:turn:<n> | doc:<name> | model:<node> | code:<rule> | data")
    said: str | None = None
    reason: str = ""
    evidence: list[str] = PField(default_factory=list)


def _coerce(kind: ClaimKind, name: str, raw: Any, columns: dict[str, str]) -> tuple[Any, str | None]:
    spec = kind.fields.get(name)
    if spec is None:
        return None, f"{kind.name} has no field {name!r}"
    if raw is None or (isinstance(raw, str) and raw.strip().lower() in {"", "null", "none"}):
        return None, None if spec.optional else f"{kind.name}.{name} needs a value"
    s = raw if not isinstance(raw, str) else raw.strip()
    if spec.type == "text":
        return str(s), None
    if spec.type == "choice":
        for o in spec.options:
            if str(s).lower() == str(o).lower():
                return o, None
        return None, f"{kind.name}.{name} must be one of {spec.options}, not {s!r}"
    if spec.type == "bool":
        b = C._truthy(s)
        return (b, None) if b is not None else (None, f"{kind.name}.{name} must be true or false, not {s!r}")
    if spec.type == "number":
        try:
            return float(s), None
        except (TypeError, ValueError):
            return None, f"{kind.name}.{name} must be a number, not {s!r}"
    if spec.type == "column":
        c = columns.get(_key(str(s)))
        return (c, None) if c is not None else (None, f"{kind.name}.{name}: {s!r} is not a column in the file")
    if spec.type == "columns":
        parts = s if isinstance(s, list) else [p.strip() for p in str(s).split(",") if p.strip()]
        cols, bad = [], []
        for part in parts:
            c = columns.get(_key(str(part)))
            (cols if c is not None else bad).append(c or part)
        return (cols, None) if not bad else (None, f"{kind.name}.{name}: not columns in the file: {bad}")
    return None, f"unknown field type {spec.type}"


def apply(memory: Memory, updates: list[Update], cat: Catalogue | None = None) -> list[str]:
    """Write what passes the gate; return why each rejected update was refused. The gate:
    a source is required; a model may only draft; a confirmed field changes only on the person's word or a data check;
    a belief (an uncheckable kind) is written only on the person's word; the value must fit the field; the column must exist."""
    cat = cat or load_catalogue()
    columns = {c.key: c.name for c in memory.columns.values()}
    rejected: list[str] = []
    for up in updates:
        where, name, field = Memory.parse(up.address)
        if field is None:
            rejected.append(f"{up.address}: names no field")
            continue
        if not up.source:
            rejected.append(f"{up.address}: no source; a write with no source is not made")
            continue
        from_person = up.source.startswith("user:turn:") or up.source.startswith("doc:")
        from_data = up.source == "data" or up.source.startswith("code:")
        from_model = up.source.startswith("model:")
        if where == "col":
            if memory.column(name) is None:
                rejected.append(f"{up.address}: {name!r} is not a column in the file")
                continue
            kind = cat.kinds[COLUMN_KIND]
        else:
            kind = cat.kinds.get(name)
            if kind is None:
                rejected.append(f"{up.address}: unknown claim kind {name!r}")
                continue
        if field not in kind.fields:
            rejected.append(f"{up.address}: {kind.name} has no field {field!r}")
            continue
        current = memory.field(up.address)
        if from_model and up.status != "drafted":
            rejected.append(f"{up.address}: a model may only draft; {up.status!r} needs the person's word")
            continue
        if kind.uncheckable and not from_person:
            rejected.append(f"{up.address}: {kind.name} is a belief; only the person can set it")
            continue
        if current is not None and current.status == "confirmed" and not (from_person or from_data):
            rejected.append(f"{up.address}: confirmed by the person; only their word or a data check changes it")
            continue
        if up.status == "unknown":
            memory.set(up.address, None, status="unknown", source=up.source, said=up.said, evidence=up.evidence)
            continue
        value, err = _coerce(kind, field, up.value, columns)
        if err:
            rejected.append(f"{up.address}: {err}")
            continue
        if value is None:
            rejected.append(f"{up.address}: no value given")
            continue
        memory.set(up.address, value, status=up.status, source=up.source, said=up.said, evidence=up.evidence)
    return rejected


# ------------------------------------------------------------------ roles (a view)


def roles(memory: Memory, outcome: str | None = None, treatment: str | None = None, time: str | None = None) -> dict[str, str]:
    """What each column is to the question, by key, from the dataset fields and the frame. A view: computed every time,
    never written. A column with no entry is a candidate when the frame names it and out of play otherwise. `time` is the
    period column when the caller knows it and the memory does not yet (the index entry's flag)."""
    a, ch, g = memory.values_of("claim:assignment"), memory.values_of("claim:change"), memory.values_of("claim:grain")
    excl, med = memory.values_of("claim:exclusion"), memory.values_of("claim:mediator")
    out: dict[str, str] = {}

    def put(col: str | None, role: str) -> None:
        if col and memory.column(col) is not None:
            out.setdefault(memory.column(col).key, role)

    time = ch.get("date_column") or time
    put(outcome, "outcome")
    put(treatment or a.get("treatment_column"), "treatment")
    put(a.get("score_column"), "score")
    put(time, "time")
    if g.get("panel") is True:
        for c in g.get("key_columns") or []:
            if _key(c) != _key(time or ""):
                put(c, "unit")
    put(a.get("level_column"), "group")
    if excl.get("exists") and excl.get("column"):
        put(excl["column"], "instrument")
    if med.get("exists") and med.get("column"):
        put(med["column"], "mediator")
    for d in a.get("depends_on") or []:
        put(d, "depends_on")
    return out


# ------------------------------------------------------------------ check: data facts and consistency


class Finding(BaseModel):
    address: str
    rule: str
    passed: bool | None
    detail: str

    @property
    def evidence(self) -> str:
        return f"check:{self.address.removeprefix('claim:')}.{self.rule}"


_CHECK_FIELD = {  # which field a legacy check speaks about
    "key_unique": ("grain", "key_columns"), "date_column": ("change", "date_column"), "period_value": ("change", "period_value"),
    "treatment_column": ("assignment", "treatment_column"), "treated_level": ("assignment", "treated_level"),
    "rows_by_side": ("assignment", "cutoff"), "takeup_by_side": ("assignment", "cutoff"), "treatment_varies": ("assignment", "treatment_column"),
    "fixed_within_unit": (COLUMN_KIND, "when"),
}


def _refute(memory: Memory, address: str, finding: Finding) -> None:
    f = memory.field(address)
    if f is None:
        return
    if f.status != "contradiction":  # a contradiction stands; the person kept their answer against the file twice
        f.status = "refuted"
    f.evidence = list(dict.fromkeys(f.evidence + [finding.evidence]))


def consistency(memory: Memory, outcome: str | None = None, treatment: str | None = None) -> list[Finding]:
    """The rules that hold between fields, never overwriting a value: a failed rule marks the field refuted with the rule as
    evidence, and the desk asks about it next."""
    out: list[Finding] = []
    role = roles(memory, outcome, treatment)
    for c in memory.columns.values():
        v = memory.values_of(c.address)
        when, moved, r = v.get("when"), v.get("moved_by_change"), role.get(c.key)
        if r == "depends_on" and when in ("after", "at"):
            fd = Finding(address=f"{c.address}.when", rule="depends_on_before", passed=False, detail=f"the offer or the rule looked at {c.name!r}, so it was set before the change, not {when}")
            out.append(fd); _refute(memory, fd.address, fd)
        if r == "score" and when in ("after", "at"):
            fd = Finding(address=f"{c.address}.when", rule="score_before", passed=False, detail=f"{c.name!r} is the score the rule was applied to, so it was set before the decision, not {when}")
            out.append(fd); _refute(memory, fd.address, fd)
        if r == "outcome" and when in ("before", "at"):
            fd = Finding(address=f"{c.address}.when", rule="outcome_after", passed=False, detail=f"{c.name!r} is the outcome, so it was measured after the change, not {when}")
            out.append(fd); _refute(memory, fd.address, fd)
        if moved is True and when == "before":
            fd = Finding(address=f"{c.address}.moved_by_change", rule="moved_not_before", passed=False, detail=f"{c.name!r} is fixed before the change, so the change could not have moved it")
            out.append(fd); _refute(memory, fd.address, fd)
    return out


def check(memory: Memory, df: pd.DataFrame, profile, th: dict | None = None, cat: Catalogue | None = None,
          outcome: str | None = None, treatment: str | None = None) -> list[Finding]:
    """The data facts on every drafted or confirmed field the file can check, then the consistency rules."""
    cat, th = cat or load_catalogue(), th or load_thresholds()
    table = memory.to_claims(cat)
    out: list[Finding] = []
    for claim in table.claims.values():
        kind = cat.kinds[claim.kind]
        if kind.check == "none" or claim.status not in {"drafted", "confirmed"}:
            continue
        res = C.run(kind.check, claim, table, df, profile, th)
        if res is None:
            continue
        where = _CHECK_FIELD.get(res.name)
        address = (f"{claim.key}.{where[1]}" if claim.kind == COLUMN_KIND else f"claim:{where[0]}.{where[1]}") if where else f"claim:{claim.key}"
        fd = Finding(address=address, rule=res.name, passed=res.passed, detail=res.detail)
        out.append(fd)
        f = memory.field(address)
        if f is not None:
            f.evidence = list(dict.fromkeys(f.evidence + [fd.evidence]))
            if res.passed is False and f.status != "contradiction":
                f.status = "refuted"
    out += consistency(memory, outcome, treatment)
    return out


# ------------------------------------------------------------------ probe, fit, open


def probe(memory: Memory, df: pd.DataFrame, th: dict | None = None, cat: Catalogue | None = None) -> list[ProbeResult]:
    cat, th = cat or load_catalogue(), th or load_thresholds()
    return run_probes(df, memory.to_claims(cat), list(cat.families), th)


def fit(memory: Memory, probes: list[ProbeResult], cat: Catalogue | None = None, columns: list[str] | None = None) -> Status:
    """The family grid. `columns` limits the per-column needs to the columns in play; a column nobody named never blocks."""
    cat = cat or load_catalogue()
    return T.compute(cat, memory.to_claims(cat, columns=columns), probes)


class Open(BaseModel):
    """One vague field a surviving family needs: the question engine's unit."""

    address: str
    kind: str
    field: str
    status: str
    options: list[str] = PField(default_factory=list)
    optional: bool = False
    because: list[str] = PField(default_factory=list, description="the surviving families that need this kind")
    frame: str = ""


def _options(kind: ClaimKind, field: str) -> list[str]:
    spec = kind.fields[field]
    if spec.type == "choice":
        return [str(o) for o in spec.options]
    if spec.type == "bool":
        return ["yes", "no"]
    return []


def open(memory: Memory, status: Status, cat: Catalogue | None = None) -> list[Open]:
    """Every field still vague on a claim a survivor needs: first the open claims (what blocks readiness, required fields first),
    then the drafts the model left on settled claims (what the person has not confirmed yet, never blocking), in the table's order."""
    cat = cat or load_catalogue()
    needs = {f: fam.requires for f, fam in cat.families.items()}
    out: list[Open] = []
    for key in list(status.open) + [k for k in status.settled if k not in status.open]:
        blocking = key in status.open
        if key.startswith("col:"):
            if memory.column(key[4:]) is None:
                continue
            kind, prefix = cat.kinds[COLUMN_KIND], key
        else:
            kind, prefix = cat.kinds[key], f"claim:{key}"
        fields = memory.fields_of(prefix)
        because = [f for f in status.surviving if kind.name in needs.get(f, [])]
        here: list[Open] = []
        for name, spec in kind.fields.items():
            f = fields.get(name) or Field()
            vague = (blocking and f.status in {"empty", "refuted"} and not spec.optional) or f.status == "drafted"
            if vague:
                here.append(Open(address=f"{prefix}.{name}", kind=kind.name, field=name, status=f.status, options=_options(kind, name),
                                 optional=spec.optional, because=because, frame=kind.frame))
        here.sort(key=lambda o: (o.optional, o.status == "drafted"))  # within a claim: required and empty first, then drafts
        out.extend(here)
    return out
