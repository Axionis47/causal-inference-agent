"""Typed artifacts of the interview. Model outputs (Extraction, Reply) are flat so any structured-output backend
accepts them; the claim table and the status are code-owned."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from causal_agent.common.contracts import Cited

ClaimStatus = Literal["empty", "drafted", "confirmed", "refuted", "unknown", "contradiction"]
Cell = Literal["fits", "does_not_fit", "unknown", "not_needed"]


class Claim(BaseModel):
    kind: str
    key: str  # the kind name, or "col:<key>" for a per-column claim
    fields: dict[str, Any] = Field(default_factory=dict)
    status: ClaimStatus = "empty"
    source: str | None = None
    evidence: list[str] = Field(default_factory=list)
    check_detail: str | None = None
    asked: int = 0
    refutations: int = 0

    @property
    def address(self) -> str:
        return f"claim:{self.key}"

    def settled(self) -> bool:
        """Drafted is not settled: a derivation of the model's is confirmed by the person before it counts."""
        return self.status in {"confirmed", "unknown", "contradiction"}

    def render(self) -> str:
        vals = ", ".join(f"{k}={v!r}" for k, v in self.fields.items() if v is not None) or "(no values)"
        line = f"[{self.address}] {self.kind} {self.status}"
        if self.source:
            line += f" (from {self.source})"
        line += f": {vals}"
        if self.check_detail:
            line += f"\n  [{self.evidence[-1] if self.evidence else self.address + '.check'}] {self.check_detail}"
        return line


class ClaimTable(BaseModel):
    claims: dict[str, Claim] = Field(default_factory=dict)

    def get(self, key: str) -> Claim | None:
        return self.claims.get(key)

    def of_kind(self, kind: str) -> list[Claim]:
        return [c for c in self.claims.values() if c.kind == kind]

    def value(self, key: str, field: str) -> Any:
        c = self.claims.get(key)
        return None if c is None else c.fields.get(field)

    def render(self, keys: list[str] | None = None) -> str:
        cs = [self.claims[k] for k in keys] if keys else list(self.claims.values())
        return "\n".join(c.render() for c in cs) or "(no claims yet)"


class FieldValue(BaseModel):
    name: str = Field(description="the field name, exactly as listed for the kind")
    value: str = Field(description="the value as text; true/false for yes-no fields; a column name for column fields; comma-separated names for column lists")


class ClaimUpdate(BaseModel):
    kind: str = Field(description="one of the claim kinds")
    column: str | None = Field(default=None, description="for the per-column kind, the column this claim is about; otherwise null")
    values: list[FieldValue] = Field(default_factory=list)
    unknown: bool = Field(default=False, description="true when the person said they do not know; values are then ignored")
    reason: str = Field(description="one sentence: what in the material says this")
    cites: list[str] = Field(description="doc:<name>, user:turn:<n>, or a card address the reason rests on")


class Extraction(BaseModel):
    updates: list[ClaimUpdate]
    confirmed: list[str] = Field(
        default_factory=list,
        description="claim keys the person confirmed as they stand this turn (e.g. change, sampling, col:age); only from the person's words, never from a description",
    )
    notes: list[Cited] = Field(default_factory=list, description="anything read that fits no claim, with cites")


QuestionKind = Literal["confirm", "choose", "open"]


class Question(BaseModel):
    keys: list[str] = Field(description="the claim keys this question settles, e.g. assignment or col:age")
    field: str | None = Field(default=None, description="the field asked about, or null when the whole claim is")
    kind: QuestionKind = Field(description="confirm a draft; choose among options; open one-sentence answer")
    text: str
    options: list[str] = Field(default_factory=list, description="for choose: the legal options, exactly")
    evidence_cites: list[str] = Field(default_factory=list, description="addresses of the draft, the check, or the card shown beside the question")


class Reply(BaseModel):
    questions: list[Question]
    text: str = Field(description="the whole message to the person: what was settled this turn, then the questions")


class ProbeResult(BaseModel):
    family: str
    name: str
    value: float | None = None
    passed: bool | None  # None: not computable with what is settled
    detail: str

    @property
    def address(self) -> str:
        return f"probe:{self.family}.{self.name}"

    def render(self) -> str:
        v = "pass" if self.passed else "FAIL" if self.passed is False else "n/a"
        return f"[{self.address}] {v}: {self.detail}"


class Status(BaseModel):
    table: dict[str, dict[str, Cell]]
    surviving: list[str]
    struck: dict[str, str] = Field(default_factory=dict)
    required: list[str]
    settled: list[str]
    open: list[str]
    ready: bool
    contradictions: list[str] = Field(default_factory=list)

    def render(self, kinds: list[str]) -> str:
        fams = list(self.table)
        mark = {"fits": "✓", "does_not_fit": "✗", "unknown": "?", "not_needed": "·"}
        w = max(len(k) for k in kinds) + 2
        head = " " * w + "  ".join(f"{f[:14]:<14}" for f in fams)
        rows = [head]
        for k in kinds:
            rows.append(f"{k:<{w}}" + "  ".join(f"{mark[self.table[f].get(k, 'not_needed')]:<14}" for f in fams))
        struck = "; ".join(f"{f}: {why}" for f, why in self.struck.items())
        line = f"settled {len(self.settled)}/{len(self.required)} · open: {', '.join(self.open) or 'none'} · in play: {', '.join(self.surviving) or 'none'}"
        line += " · READY" if self.ready else " · not ready"
        if struck:
            line += f"\nstruck out: {struck}"
        return "\n".join(rows) + "\n" + line
