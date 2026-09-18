"""Join a semantic note and a deterministic profile into citable cards.

A pack is one dataset card, one card per change, one card per column, and, when the interview ran, one card
per settled claim and per probe. Every card has an address an agent can cite:

    dataset.note
    change:1.note
    col:lunch.note
    col:lunch.profile.varies_over
    claim:assignment.kind
    claim:col:lunch.when
    probe:discontinuity.rows_by_side

Addresses are the whole citation system. A claim that cites an address that
does not exist fails the gate.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

from causal_agent.common.addresses import key as _key, norm_address as _norm_address
from causal_agent.intake.profiler import ColumnProfile, DatasetProfile, Profile


class ChangeCard(BaseModel):
    index: int
    title: str
    note: str
    source: str | None = None

    @property
    def address(self) -> str:
        return f"change:{self.index}"


class ColumnCard(BaseModel):
    name: str
    key: str
    note: str | None
    source: str | None
    profile: ColumnProfile

    @property
    def address(self) -> str:
        return f"col:{self.key}"

    def first_sentence(self) -> str:
        if not self.note:
            return "(no note)"
        return re.split(r"(?<=[.!?])\s", self.note.strip(), maxsplit=1)[0]

    def render(self) -> str:
        """The full card as an agent sees it, with addresses on every line."""
        p = self.profile
        lines = [f"[{self.address}] column {self.name!r}"]
        lines.append(f"  [{self.address}.note] {self.note or '(no note)'}")
        lines.append(f"  [{self.address}.profile.kind] {p.kind}")
        lines.append(f"  [{self.address}.profile.nulls] {p.nulls} ({p.null_rate:.1%})")
        lines.append(f"  [{self.address}.profile.distinct] {p.distinct}{' (constant)' if p.constant else ''}")
        lines.append(f"  [{self.address}.profile.varies_over] {p.varies_over}")
        if p.numeric:
            n = p.numeric
            lines.append(f"  [{self.address}.profile.numeric] min {n.min:g}, p50 {n.p50:g}, max {n.max:g}, mean {n.mean:g}")
        if p.top_values:
            tv = ", ".join(f"{t.value}={t.share:.0%}" for t in p.top_values[:6])
            lines.append(f"  [{self.address}.profile.top_values] {tv}")
        if p.datetime:
            lines.append(f"  [{self.address}.profile.datetime] {p.datetime.first} to {p.datetime.last}, {p.datetime.inferred_frequency}")
        if p.switch:
            s = p.switch
            lines.append(
                f"  [{self.address}.profile.switch] {s.entities_that_switch} entities switch, "
                f"{s.entities_never_on} never on, first on {s.first_on}"
            )
        if p.observed_sentinels:
            lines.append(f"  [{self.address}.profile.sentinels] " + "; ".join(f"{s.value} x{s.count} ({s.reason})" for s in p.observed_sentinels))
        if p.format_issues:
            lines.append(f"  [{self.address}.profile.format_issues] " + "; ".join(p.format_issues))
        return "\n".join(lines)


class DatasetCard(BaseModel):
    name: str
    note: str
    source: str | None
    profile: DatasetProfile

    @property
    def address(self) -> str:
        return "dataset"

    def render(self) -> str:
        p = self.profile
        lines = [f"[dataset] {self.name}", f"  [dataset.note] {self.note}"]
        lines.append(f"  [dataset.profile.rows] {p.rows} rows, {p.columns} columns, {p.duplicate_rows} duplicate rows")
        lines.append(f"  [dataset.profile.grain] {' + '.join(p.grain) if p.grain else 'no key column found'}")
        if p.time_coverage:
            t = p.time_coverage
            lines.append(f"  [dataset.profile.time_coverage] {t.column}: {t.first} to {t.last}, {t.inferred_frequency}, {t.gaps} gaps")
        else:
            lines.append("  [dataset.profile.time_coverage] no time column")
        if p.entity_summary:
            e = p.entity_summary
            lines.append(
                f"  [dataset.profile.entity_summary] {' + '.join(e.columns)}: {e.entities} entities, "
                f"{e.rows_per_entity_min} to {e.rows_per_entity_max} rows each"
            )
        else:
            lines.append("  [dataset.profile.entity_summary] no entity column declared")
        if p.format_issues:
            lines.append("  [dataset.profile.format_issues] " + "; ".join(p.format_issues))
        return "\n".join(lines)


class ClaimCard(BaseModel):
    """A claim the interview settled: a typed statement about the world with its status and who made it."""

    kind: str
    key: str  # kind name, or col:<key> for a per-column claim
    fields: dict = Field(default_factory=dict)
    status: str
    source: str | None = None
    check_detail: str | None = None

    @property
    def address(self) -> str:
        return f"claim:{self.key}"

    def render(self) -> str:
        vals = "; ".join(f"[{self.address}.{k}] {v}" for k, v in self.fields.items() if v is not None)
        head = f"[{self.address}] {self.kind} ({self.status}" + (f", {self.source}" if self.source else "") + ")"
        out = head + ("\n  " + vals.replace("; ", "\n  ") if vals else "")
        if self.check_detail:
            out += f"\n  [{self.address}.check] {self.check_detail}"
        return out


class ProbeCard(BaseModel):
    family: str
    name: str
    passed: bool | None
    detail: str

    @property
    def address(self) -> str:
        return f"probe:{self.family}.{self.name}"

    def render(self) -> str:
        v = "pass" if self.passed else "FAIL" if self.passed is False else "n/a"
        return f"[{self.address}] {v}: {self.detail}"


class Pack(BaseModel):
    name: str
    dataset: DatasetCard
    changes: list[ChangeCard]
    columns: list[ColumnCard]
    claims: list[ClaimCard] = Field(default_factory=list)
    probes: list[ProbeCard] = Field(default_factory=list)
    unnoted_columns: list[str] = Field(default_factory=list)  # in the file, not in the note
    unprofiled_notes: list[str] = Field(default_factory=list)  # in the note, not in the file

    # ------------------------------------------------------------- addresses
    def addresses(self) -> set[str]:
        out = {"dataset", "dataset.note"}
        for f in ("rows", "grain", "time_coverage", "entity_summary", "format_issues"):
            out.add(f"dataset.profile.{f}")
        for ch in self.changes:
            out.update({ch.address, f"{ch.address}.note"})
        for c in self.columns:
            a = c.address
            out.update({a, f"{a}.note"})
            for f in ("kind", "nulls", "distinct", "varies_over", "numeric", "top_values", "datetime", "switch", "sentinels", "format_issues"):
                out.add(f"{a}.profile.{f}")
        for cl in self.claims:
            out.update({cl.address, f"{cl.address}.check"})
            out.update(f"{cl.address}.{k}" for k in cl.fields)
        for pr in self.probes:
            out.add(pr.address)
        return out

    def resolve(self, address: str) -> bool:
        """An address names a card; the column segment is compared as a key, so a citation written with the
        column's original spelling (col:Total_Emp.note) resolves to the same card as col:total_emp.note."""
        return _norm_address(address) in {_norm_address(a) for a in self.addresses()}

    # ------------------------------------------------------------- queries
    def column(self, name_or_key: str) -> ColumnCard | None:
        k = _key(name_or_key)
        for c in self.columns:
            if c.key == k or c.name == name_or_key:
                return c
        return None

    def column_index(self) -> str:
        """Every column as one line: address, name, first sentence of its note."""
        return "\n".join(f"[{c.address}] {c.name!r}: {c.first_sentence()}" for c in self.columns)

    def render_changes(self) -> str:
        if not self.changes:
            return "[changes] none recorded"
        return "\n".join(f"[{ch.address}.note] {ch.title}. {ch.note}" for ch in self.changes)

    def render_claims(self) -> str:
        if not self.claims:
            return ""
        lines = [c.render() for c in self.claims if c.status != "empty"]
        lines += [p.render() for p in self.probes]
        return "\n".join(lines)

    def digest(self) -> str:
        parts = [self.dataset.render(), self.render_changes()]
        if self.claims:
            parts.append("CLAIMS SETTLED AT INTAKE\n" + self.render_claims())
        return "\n\n".join(parts)


# ------------------------------------------------------------------ parsing


def _split_sections(md: str) -> dict[str, str]:
    parts = re.split(r"^##\s+", md, flags=re.M)
    out: dict[str, str] = {}
    for part in parts[1:]:
        title, _, body = part.partition("\n")
        out[title.strip().lower()] = body.strip()
    return out


def _source(text: str) -> tuple[str, str | None]:
    m = re.search(r"\s*\[([^\]]+)\]\s*$", text)
    if m:
        return text[: m.start()].strip(), m.group(1)
    return text.strip(), None


def _parse_changes(body: str) -> list[ChangeCard]:
    cards = []
    for i, para in enumerate([p for p in re.split(r"\n\s*\n", body) if p.strip()], start=1):
        m = re.match(r"\*\*(.+?)\*\*\s*(.*)", para.strip(), flags=re.S)
        title = m.group(1).rstrip(".") if m else f"change {i}"
        text = m.group(2) if m else para
        note, src = _source(" ".join(text.split()))
        cards.append(ChangeCard(index=i, title=title, note=note, source=src))
    return cards


def _parse_columns(body: str) -> dict[str, tuple[str, str | None]]:
    """Map column key -> (note, source). A bold header may name several columns."""
    out: dict[str, tuple[str, str | None]] = {}
    for para in [p for p in re.split(r"\n\s*\n", body) if p.strip()]:
        m = re.match(r"\*\*(.+?)\*\*\s*[—–-]?\s*(.*)", para.strip(), flags=re.S)
        if not m:
            continue
        names = [n.strip() for n in m.group(1).split(",")]
        note, src = _source(" ".join(m.group(2).split()))
        for n in names:
            out[_key(n)] = (note, src)
    return out


def _load_claims(path: str | Path | None) -> tuple[list[ClaimCard], list[ProbeCard]]:
    if path is None or not Path(path).exists():
        return [], []
    import yaml

    raw = yaml.safe_load(Path(path).read_text()) or {}
    claims = [ClaimCard(kind=c["kind"], key=c["key"], fields=c.get("fields") or {}, status=c.get("status", "empty"), source=c.get("source"), check_detail=c.get("check_detail"))
              for c in raw.get("claims") or []]
    probes = [ProbeCard(family=p["family"], name=p["name"], passed=p.get("passed"), detail=p.get("detail", "")) for p in raw.get("probes") or []]
    return claims, probes


def load_pack(name: str, note_path: str | Path, profile_path: str | Path, claims_path: str | Path | None = None) -> Pack:
    md = Path(note_path).read_text()
    prof = Profile.model_validate(json.loads(Path(profile_path).read_text()))
    claims, probes = _load_claims(claims_path)
    sections = _split_sections(md)

    ds_note, ds_src = _source(" ".join(sections.get("about the dataset", "").split()))
    changes = _parse_changes(sections.get("what changed", ""))
    col_notes = _parse_columns(sections.get("about each column", ""))

    columns: list[ColumnCard] = []
    seen: set[str] = set()
    for cp in prof.columns:
        k = _key(cp.name)
        note, src = col_notes.get(k, (None, None))
        seen.add(k)
        columns.append(ColumnCard(name=cp.name, key=k, note=note, source=src, profile=cp))

    return Pack(
        name=name,
        dataset=DatasetCard(name=name, note=ds_note, source=ds_src, profile=prof.dataset),
        changes=changes,
        columns=columns,
        claims=claims,
        probes=probes,
        unnoted_columns=[c.name for c in columns if c.note is None],
        unprofiled_notes=[k for k in col_notes if k not in seen],
    )
