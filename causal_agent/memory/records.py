"""The memory of a dataset: one record per column and one for the dataset, every field with its own status, source, the
person's words, and the checks that touched it. The claim table (`claims.py`) is the older, per-claim view the interview
and the router still run on; a memory converts to it and back without loss until they are rewritten on the memory itself.

Addresses: col:<key>.<field> for a column field, claim:<kind>.<field> for a dataset field (claim:<kind> alone for the kind).
"""

from __future__ import annotations

import copy
from typing import Any, Literal

from pydantic import BaseModel, Field as PField

from causal_agent.common.addresses import key as _key
from causal_agent.common.contracts import ColumnFacts, Said
from causal_agent.memory.catalogue import Catalogue, load_catalogue
from causal_agent.memory.claims import Claim, ClaimTable

Status = Literal["empty", "drafted", "confirmed", "refuted", "unknown", "contradiction"]
SETTLED = {"confirmed", "unknown", "contradiction"}
COLUMN_KIND = "measured"
LEGACY_NAMES = {"affected_by_treatment": "moved_by_change"}


class Field(BaseModel):
    """One thing known, or not, about a column or the dataset."""

    value: Any = None
    status: Status = "empty"
    source: str | None = PField(default=None, description="data | user:turn:<n> | doc:<name> | model:<node> | code:<rule>")
    said: str | None = PField(default=None, description="the person's sentence this rests on, verbatim")
    evidence: list[str] = PField(default_factory=list, description="check addresses that touched this field")

    def settled(self) -> bool:
        return self.status in SETTLED

    def known(self) -> bool:
        return self.value is not None and self.status in {"drafted", "confirmed"}

    def render(self, address: str) -> str:
        v = "(empty)" if self.value is None else self.value
        tail = f" · {self.status}" + (f" · {self.source}" if self.source else "") + (f' · said "{self.said}"' if self.said else "")
        return f"[{address}] {v}{tail}"


class KindRecord(BaseModel):
    """The fields of one dataset claim kind (grain, change, assignment, a belief), plus the interview's bookkeeping."""

    fields: dict[str, Field] = PField(default_factory=dict)
    status: Status = PField(default="empty", description="the kind as a whole, when no field carries it (an unknown belief)")
    source: str | None = None
    check_detail: str | None = None
    asked: int = 0
    refutations: int = 0


class ColumnRecord(BaseModel):
    name: str
    key: str
    facts: ColumnFacts = PField(default_factory=ColumnFacts)
    fields: dict[str, Field] = PField(default_factory=dict)
    status: Status = "empty"
    source: str | None = None
    check_detail: str | None = None
    asked: int = 0
    refutations: int = 0

    @property
    def address(self) -> str:
        return f"col:{self.key}"

    def field(self, name: str) -> Field:
        return self.fields.setdefault(name, Field())

    def value(self, name: str) -> Any:
        f = self.fields.get(name)
        return None if f is None else f.value

    def role(self) -> str | None:
        return self.value("role")

    def render(self, known_only: bool = True) -> str:
        lines = [f"[{self.address}] column {self.name!r}" + (f" ({self.role()})" if self.role() else "")]
        for name, f in self.fields.items():
            if known_only and f.value is None and f.status == "empty":
                continue
            lines.append("  " + f.render(f"{self.address}.{name}"))
        return "\n".join(lines)


class DatasetRecord(BaseModel):
    kinds: dict[str, KindRecord] = PField(default_factory=dict)

    def kind(self, name: str) -> KindRecord:
        return self.kinds.setdefault(name, KindRecord())

    def field(self, kind: str, name: str) -> Field:
        return self.kind(kind).fields.setdefault(name, Field())

    def value(self, kind: str, name: str) -> Any:
        k = self.kinds.get(kind)
        f = k.fields.get(name) if k else None
        return None if f is None else f.value


class Memory(BaseModel):
    """Everything known about one dataset. Versioned: every accepted write bumps `version`."""

    name: str
    version: int = 0
    csv: str | None = None
    dataset_facts: dict = PField(default_factory=dict)
    columns: dict[str, ColumnRecord] = PField(default_factory=dict)
    dataset: DatasetRecord = PField(default_factory=DatasetRecord)
    transcript: list[Said] = PField(default_factory=list)

    # ------------------------------------------------------------- lookup
    def column(self, name_or_key: str) -> ColumnRecord | None:
        k = _key(name_or_key)
        for c in self.columns.values():
            if c.key == k or c.name == name_or_key:
                return c
        return None

    @staticmethod
    def parse(address: str) -> tuple[str, str | None, str | None]:
        """('col', key, field) or ('kind', kind, field). Accepts claim:col:x.when, col:x.when, claim:assignment.kind, assignment.kind."""
        a = str(address).strip().removeprefix("claim:")
        if a.startswith("col:"):
            rest = a[4:]
            name, _, field = rest.partition(".")
            return "col", _key(name), field or None
        kind, _, field = a.partition(".")
        return "kind", kind, field or None

    def field(self, address: str) -> Field | None:
        where, name, field = self.parse(address)
        if field is None:
            return None
        if where == "col":
            c = self.column(name)
            return c.fields.get(field) if c else None
        k = self.dataset.kinds.get(name)
        return k.fields.get(field) if k else None

    def set(self, address: str, value: Any, *, status: Status, source: str | None, said: str | None = None, evidence: list[str] | None = None) -> Field:
        """A raw write. The gate is `ops.apply`; nothing else should call this from a judgement."""
        where, name, field = self.parse(address)
        if field is None:
            raise KeyError(f"{address!r} names no field")
        if where == "col":
            c = self.column(name)
            if c is None:
                raise KeyError(f"{name!r} is not a column in the memory")
            f = c.field(field)
        else:
            f = self.dataset.field(name, field)
        f.value, f.status, f.source = value, status, source
        if said:
            f.said = said
        if evidence:
            f.evidence = list(dict.fromkeys(f.evidence + evidence))
        self.version += 1
        return f

    def addresses(self) -> set[str]:
        out: set[str] = set()
        for c in self.columns.values():
            out.add(c.address)
            out.update(f"{c.address}.{n}" for n in c.fields)
        for kind, k in self.dataset.kinds.items():
            out.add(f"claim:{kind}")
            out.update(f"claim:{kind}.{n}" for n in k.fields)
        return out

    def render(self, known_only: bool = True) -> str:
        lines = []
        for kind, k in self.dataset.kinds.items():
            shown = [(n, f) for n, f in k.fields.items() if not known_only or f.value is not None or f.status != "empty"]
            if not shown and k.status == "empty":
                continue
            if not shown:
                lines.append(f"[claim:{kind}] {k.status}" + (f" · {k.source}" if k.source else ""))
            for n, f in shown:
                lines.append(f.render(f"claim:{kind}.{n}"))
        for c in self.columns.values():
            lines.append(c.render(known_only))
        return "\n".join(lines)

    # ------------------------------------------------------------- copies
    def snapshot(self) -> "Memory":
        return copy.deepcopy(self)

    def fork(self) -> "Memory":
        m = copy.deepcopy(self)
        m.version += 1
        return m

    # ------------------------------------------------------------- the claim table, both ways
    @classmethod
    def from_claims(cls, name: str, table: ClaimTable, *, profile=None, csv: str | None = None, cat: Catalogue | None = None,
                    transcript: list[Said] | None = None) -> "Memory":
        """Every claim's status and source spread to each of its fields; a column record for every profiled column."""
        cat = cat or load_catalogue()
        m = cls(name=name, csv=csv, transcript=list(transcript or []))
        if profile is not None:
            dp = profile.dataset
            m.dataset_facts = {"rows": dp.rows, "columns": dp.columns, "duplicate_rows": dp.duplicate_rows, "grain": list(dp.grain or []),
                               "time_coverage": dp.time_coverage.model_dump() if dp.time_coverage else None,
                               "entity_summary": dp.entity_summary.model_dump() if dp.entity_summary else None, "format_issues": list(dp.format_issues or [])}
            for cp in profile.columns:
                m.columns[cp.key] = ColumnRecord(name=cp.name, key=cp.key, facts=ColumnFacts.from_profile(cp))
        for kind in cat.ordered():
            if kind.per_column:
                continue
            m.dataset.kinds[kind.name] = KindRecord(fields={n: Field() for n in kind.fields})
        col_kind = cat.kinds[COLUMN_KIND]
        for c in m.columns.values():
            c.fields = {n: Field() for n in col_kind.fields}
        for claim in table.claims.values():
            fields = {LEGACY_NAMES.get(n, n): v for n, v in claim.fields.items()}
            if claim.kind == COLUMN_KIND:
                k = claim.key.removeprefix("col:")
                rec = m.column(k) or m.columns.setdefault(k, ColumnRecord(name=k, key=k, fields={n: Field() for n in col_kind.fields}))
                target, meta = rec.fields, rec
            else:
                kr = m.dataset.kind(claim.kind)
                target, meta = kr.fields, kr
            meta.status, meta.source, meta.check_detail, meta.asked, meta.refutations = claim.status, claim.source, claim.check_detail, claim.asked, claim.refutations
            for n, v in fields.items():
                if v is None:
                    continue
                target[n] = Field(value=v, status=claim.status if claim.status != "empty" else "drafted", source=claim.source, evidence=list(claim.evidence))
        return m

    def to_claims(self, cat: Catalogue | None = None) -> ClaimTable:
        cat = cat or load_catalogue()
        t = ClaimTable()
        for kind_name, kr in self.dataset.kinds.items():
            spec = cat.kinds.get(kind_name)
            required = [n for n, s in spec.fields.items() if not s.optional] if spec else []
            t.claims[kind_name] = _claim(kind_name, kind_name, kr.fields, kr, required)
        for c in self.columns.values():
            spec = cat.kinds[COLUMN_KIND]
            required = [n for n, s in spec.fields.items() if not s.optional]
            if c.facts.constant:
                continue
            t.claims[c.address] = _claim(COLUMN_KIND, c.address, c.fields, c, required)
        return t


def _collapse(fields: dict[str, Field], required: list[str], fallback: Status) -> tuple[Status, str | None, list[str]]:
    known = {n: f for n, f in fields.items() if f.value is not None or f.status != "empty"}
    if not known:
        return fallback, None, []
    statuses = {f.status for f in known.values()}
    req = {f.status for n, f in known.items() if n in required} or statuses
    if "contradiction" in statuses:
        st: Status = "contradiction"
    elif "refuted" in statuses:
        st = "refuted"
    elif "drafted" in req:
        st = "drafted"
    elif "confirmed" in req:
        st = "confirmed"
    elif "unknown" in req:
        st = "unknown"
    elif "drafted" in statuses:
        st = "drafted"
    else:
        st = fallback
    if st == "confirmed" and any(n not in known for n in required):  # a required field still empty: the claim is not settled
        st = "drafted"
    user = [f.source for f in known.values() if f.source and f.source.startswith("user:turn:")]
    source = max(user, key=lambda s: int(s.rsplit(":", 1)[1])) if user else next((f.source for f in known.values() if f.source), None)
    evidence = list(dict.fromkeys(e for f in known.values() for e in f.evidence))
    return st, source, evidence


def _claim(kind: str, key: str, fields: dict[str, Field], meta, required: list[str]) -> Claim:
    status, source, evidence = _collapse(fields, required, meta.status)
    return Claim(kind=kind, key=key, fields={n: f.value for n, f in fields.items() if f.value is not None}, status=status,
                 source=source or meta.source, evidence=evidence, check_detail=meta.check_detail, asked=meta.asked, refutations=meta.refutations)
