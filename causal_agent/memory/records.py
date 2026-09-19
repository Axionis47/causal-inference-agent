"""The memory of a dataset: one map from address to field, the file's facts beside it, the person's words beside that.

    fields   {address: Field}      everything known, or asked and not known, with a status, a source, and the sentence behind it
    columns  {key: Column}         the file's facts on every column, from code, never asked and never written to
    facts    {..}                  the file's facts on the dataset as a whole
    said     [Said]                the person's words, verbatim, by turn

Addresses: col:<key>.<field> for a column field, claim:<kind>.<field> for a dataset field. Roles, what is in play, what is
open, the family fit, and the pack are all views computed from the map (`ops.py`, `desk/handoff.py`); nothing derived is
written back. The claim table (`claims.py`) is the older, per-claim view the interview, the checks, the probes, and the fit
table still run on; `to_claims` projects it and `from_claims` reads one in, and both go when those are rewritten on the map.
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


class Column(BaseModel):
    """A column as the file has it: its name, its key, the profiler's facts. Nothing here is ever asked or written."""

    name: str
    key: str
    facts: ColumnFacts = PField(default_factory=ColumnFacts)

    @property
    def address(self) -> str:
        return f"col:{self.key}"


class Memory(BaseModel):
    """Everything known about one dataset. Versioned: every accepted write bumps `version`."""

    name: str
    version: int = 0
    csv: str | None = None
    facts: dict = PField(default_factory=dict, description="the dataset facts from the profile: rows, columns, grain, time coverage")
    columns: dict[str, Column] = PField(default_factory=dict, description="key -> the file's facts on the column")
    fields: dict[str, Field] = PField(default_factory=dict, description="address -> the field; absent means nothing said yet")
    said: list[Said] = PField(default_factory=list)

    # ------------------------------------------------------------- addresses
    @staticmethod
    def parse(address: str) -> tuple[str, str, str | None]:
        """('col', key, field) or ('claim', kind, field). Accepts claim:col:x.when, col:x.when, claim:assignment.kind, assignment.kind."""
        a = str(address).strip().removeprefix("claim:")
        if a.startswith("col:"):
            name, _, field = a[4:].partition(".")
            return "col", _key(name), field or None
        kind, _, field = a.partition(".")
        return "claim", kind, field or None

    @classmethod
    def canonical(cls, address: str) -> str:
        where, name, field = cls.parse(address)
        return f"{where}:{name}" + (f".{field}" if field else "")

    def column(self, name_or_key: str) -> Column | None:
        k = _key(name_or_key)
        c = self.columns.get(k)
        if c is not None:
            return c
        return next((c for c in self.columns.values() if c.name == name_or_key), None)

    # ------------------------------------------------------------- reads
    def field(self, address: str) -> Field | None:
        """The stored field, or None when nothing has been said at this address."""
        return self.fields.get(self.canonical(address))

    def value(self, address: str) -> Any:
        f = self.field(address)
        return None if f is None else f.value

    def fields_of(self, prefix: str) -> dict[str, Field]:
        """The fields under one column or one claim kind, by field name: fields_of('col:lunch'), fields_of('claim:assignment')."""
        p = self.canonical(prefix) + "."
        return {a[len(p):]: f for a, f in self.fields.items() if a.startswith(p)}

    def values_of(self, prefix: str) -> dict[str, Any]:
        """The known values under one column or one claim kind."""
        return {n: f.value for n, f in self.fields_of(prefix).items() if f.value is not None}

    def addresses(self) -> set[str]:
        out = set(self.fields)
        out.update(c.address for c in self.columns.values())
        out.update(a.rsplit(".", 1)[0] for a in self.fields)
        return out

    # ------------------------------------------------------------- the one raw write
    def set(self, address: str, value: Any, *, status: Status, source: str | None, said: str | None = None, evidence: list[str] | None = None) -> Field:
        """A raw write. The gate is `ops.apply`; nothing else should call this from a judgement."""
        where, name, field = self.parse(address)
        if field is None:
            raise KeyError(f"{address!r} names no field")
        if where == "col" and self.column(name) is None:
            raise KeyError(f"{name!r} is not a column in the memory")
        f = self.fields.setdefault(f"{where}:{name}.{field}", Field())
        f.value, f.status, f.source = value, status, source
        if said:
            f.said = said
        if evidence:
            f.evidence = list(dict.fromkeys(f.evidence + evidence))
        self.version += 1
        return f

    # ------------------------------------------------------------- text
    def render(self, known_only: bool = True, roles: dict[str, str] | None = None) -> str:
        lines = []
        for a, f in self.fields.items():
            if a.startswith("claim:") and (not known_only or f.value is not None or f.status != "empty"):
                lines.append(f.render(a))
        for c in self.columns.values():
            fs = {n: f for n, f in self.fields_of(c.address).items() if not known_only or f.value is not None or f.status != "empty"}
            role = (roles or {}).get(c.key)
            lines.append(f"[{c.address}] column {c.name!r}" + (f" ({role})" if role else ""))
            lines.extend("  " + f.render(f"{c.address}.{n}") for n, f in fs.items())
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
                    said: list[Said] | None = None) -> "Memory":
        """Every claim's status and source spread to each of its fields; a column for every profiled column."""
        cat = cat or load_catalogue()
        m = cls(name=name, csv=csv, said=list(said or []))
        if profile is not None:
            dp = profile.dataset
            m.facts = {"rows": dp.rows, "columns": dp.columns, "duplicate_rows": dp.duplicate_rows, "grain": list(dp.grain or []),
                       "time_coverage": dp.time_coverage.model_dump() if dp.time_coverage else None,
                       "entity_summary": dp.entity_summary.model_dump() if dp.entity_summary else None, "format_issues": list(dp.format_issues or [])}
            for cp in profile.columns:
                m.columns[cp.key] = Column(name=cp.name, key=cp.key, facts=ColumnFacts.from_profile(cp))
        for claim in table.claims.values():
            spec = cat.kinds.get(claim.kind)
            if claim.kind == COLUMN_KIND:
                k = _key(claim.key.removeprefix("col:"))
                m.columns.setdefault(k, Column(name=claim.key.removeprefix("col:"), key=k))
                prefix = f"col:{k}"
            else:
                prefix = f"claim:{claim.kind}"
            values = {LEGACY_NAMES.get(n, n): v for n, v in claim.fields.items() if v is not None}
            status: Status = claim.status if claim.status != "empty" else "drafted"
            for n, v in values.items():
                m.fields[f"{prefix}.{n}"] = Field(value=v, status=status, source=claim.source, evidence=list(claim.evidence))
            if not values and claim.status in {"unknown", "refuted", "contradiction"} and spec is not None:
                for n, s in spec.fields.items():  # the claim as a whole was asked and not settled: its required fields carry that
                    if not s.optional:
                        m.fields[f"{prefix}.{n}"] = Field(status=claim.status, source=claim.source, evidence=list(claim.evidence))
        return m

    def to_claims(self, cat: Catalogue | None = None, columns: list[str] | None = None) -> ClaimTable:
        """The claim-table view. `columns` limits the per-column claims to the columns in play (by name or key); None means all."""
        cat = cat or load_catalogue()
        keys = None if columns is None else {c.key for n in columns if (c := self.column(n)) is not None}
        t = ClaimTable()
        for kind in cat.ordered():
            if kind.per_column:
                continue
            required = [n for n, s in kind.fields.items() if not s.optional]
            t.claims[kind.name] = _claim(kind.name, kind.name, self.fields_of(f"claim:{kind.name}"), required)
        col_kind = cat.kinds[COLUMN_KIND]
        required = [n for n, s in col_kind.fields.items() if not s.optional]
        for c in self.columns.values():
            if c.facts.constant or (keys is not None and c.key not in keys):
                continue
            t.claims[c.address] = _claim(COLUMN_KIND, c.address, self.fields_of(c.address), required)
        return t


def _collapse(fields: dict[str, Field], required: list[str]) -> tuple[Status, str | None, list[str]]:
    """One status for a set of fields: contradiction, then refuted, then drafted, confirmed, unknown on the required fields;
    a required field still empty keeps a claim from being confirmed."""
    known = {n: f for n, f in fields.items() if f.value is not None or f.status != "empty"}
    if not known:
        return "empty", None, []
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
        st = "empty"
    if st == "confirmed" and any(n not in known for n in required):
        st = "drafted"
    user = [f.source for f in known.values() if f.source and f.source.startswith("user:turn:")]
    source = max(user, key=lambda s: int(s.rsplit(":", 1)[1])) if user else next((f.source for f in known.values() if f.source), None)
    evidence = list(dict.fromkeys(e for f in known.values() for e in f.evidence))
    return st, source, evidence


def _claim(kind: str, key: str, fields: dict[str, Field], required: list[str]) -> Claim:
    status, source, evidence = _collapse(fields, required)
    return Claim(kind=kind, key=key, fields={n: f.value for n, f in fields.items() if f.value is not None}, status=status, source=source, evidence=evidence)
