"""Shared contracts: what the desk writes for a lane (the context pack), what the lanes write back (the artifacts),
and the question frame and decision in between. Every claim carries the addresses it cites."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, SerializeAsAny, field_validator

Intent = Literal["effect_of_change", "driver_search", "root_cause", "not_causal"]


class Cited(BaseModel):
    reason: str = Field(description="one or two sentences")
    cites: list[str] = Field(description="addresses from the pack, e.g. col:lunch.note, change:1.note, dataset.profile.grain")


class Candidate(Cited):
    column: str


class Scope(BaseModel):
    population_filter: str | None = Field(default=None, description="a filter on rows the question implies, in words, or null")
    window: str | None = Field(default=None, description="time window the question implies, or null")
    contrast: Literal["switch", "dose", "level_vs_level", "none"] = "switch"
    target: Literal["average", "on_treated", "on_untreated", "conditional", "counterfactual"] = "average"


class QuestionFrame(BaseModel):
    intent: Intent
    decision_served: str = Field(description="what decision the answer would inform, in one sentence")
    outcome_candidates: list[Candidate] = Field(description="ranked, best first")
    cause_candidates: list[Candidate] = Field(description="ranked, best first; empty if intent is driver_search or root_cause")
    scope: Scope
    relevant_columns: list[Candidate] = Field(
        description="every column that matters to this question, including the outcome and cause, each with why and a citation"
    )
    reasons: list[Cited] = Field(description="anything else the reader should know about how the question was read")

    @property
    def outcome(self) -> str | None:
        return self.outcome_candidates[0].column if self.outcome_candidates else None

    @property
    def cause(self) -> str | None:
        return self.cause_candidates[0].column if self.cause_candidates else None


class PrefilterVote(Cited):
    column: str
    relevant: bool


class NeedCheck(BaseModel):
    need: str
    met: bool
    cites: list[str] = Field(default_factory=list)
    note: str = Field(default="", description="why met or unmet, one sentence")


class FamilyVerdict(BaseModel):
    family: str
    admissible: bool
    needs: list[NeedCheck]
    concern: str = Field(default="", description="if admissible, the weak_when condition that applies here, or empty")


class Rejection(BaseModel):
    family: str
    reason: str
    cites: list[str] = Field(default_factory=list)


class FamilyDecision(BaseModel):
    admissible: list[str]
    chosen: str = Field(description="one of admissible; the literal string 'none' if no family is admissible")
    chosen_assumption: str = Field(description="the assumption this choice bets on, for this data; 'none' if no family chosen")
    why_over_alternatives: str = Field(description="if more than one family was admissible, why this one; else 'only admissible family'")
    rejected: list[Rejection]
    cites: list[str] = Field(default_factory=list, description="pack addresses supporting the choice; may be empty")


# ----------------------------------------------------------------- the context pack
# What a lane receives. Built once by causal_agent.desk.handoff; the lanes read it and nothing else about the data.
# Every line a brief renders carries an address, so a lane's citations resolve against the pack alone.

Role = Literal["outcome", "treatment", "depends_on", "score", "unit", "time", "group", "instrument", "mediator", "candidate"]
When = Literal["before", "at", "after", "unknown"]

_WHEN_WORDS = {
    "before": "fixed before the change",
    "at": "set at the change",
    "after": "measured after the change",
    "unknown": "when it was set is not known",
}


class ColumnFacts(BaseModel):
    """What the profiler found about one column. A projection of the profile, so this module needs no profiler import."""

    kind: str = "unknown"
    nulls: int = 0
    null_rate: float = 0.0
    distinct: int = 0
    constant: bool = False
    varies_over: str = "unknown"
    numeric: dict[str, float] | None = None
    top_values: list[dict] = Field(default_factory=list, description="{value, count, share}, most common first")
    datetime: dict | None = None
    switch: dict | None = None
    sentinels: list[str] = Field(default_factory=list)
    format_issues: list[str] = Field(default_factory=list)
    binary_like: bool = False
    bounds: list[str] | None = None
    role_hints: list[str] = Field(default_factory=list)

    @classmethod
    def from_profile(cls, p) -> ColumnFacts:
        """From a profiler ColumnProfile (duck-typed)."""
        num = p.numeric.model_dump() if getattr(p, "numeric", None) else None
        return cls(
            kind=str(p.kind),
            nulls=int(p.nulls),
            null_rate=float(p.null_rate),
            distinct=int(p.distinct),
            constant=bool(p.constant),
            varies_over=str(getattr(p, "varies_over", "unknown")),
            numeric=num,
            top_values=[t.model_dump() for t in (p.top_values or [])],
            datetime=p.datetime.model_dump() if getattr(p, "datetime", None) else None,
            switch=p.switch.model_dump() if getattr(p, "switch", None) else None,
            sentinels=[f"{s.value} x{s.count} ({s.reason})" for s in (getattr(p, "observed_sentinels", None) or [])],
            format_issues=list(getattr(p, "format_issues", None) or []),
            binary_like=bool(getattr(p, "binary_like", False)),
            bounds=list(getattr(p, "bounds", None) or []) or None,
            role_hints=list(getattr(p, "role_hints", None) or []),
        )

    def levels(self) -> list[str]:
        return [str(t["value"]) for t in self.top_values]


class Provenance(BaseModel):
    """How one field of a brief was settled: its status, who set it, and the sentence it rests on."""

    status: str = "empty"
    source: str | None = None
    said: str | None = None

    def tail(self) -> str:
        return f" · {self.status}" + (f" · {self.source}" if self.source else "") + (f' · said "{self.said}"' if self.said else "")


class ColumnBrief(BaseModel):
    """One column as the lane reads it: the person's word about it beside the profiler's facts, with an address on every line
    and, on every field the memory holds, how it was settled."""

    name: str
    key: str
    role: Role = "candidate"
    meaning: str | None = None
    stands_for: str | None = None
    proxy: str | None = None
    when: When = "unknown"
    set_by: str | None = None
    moved_by_change: bool | None = None
    measures_outcome: bool | None = None
    source: str | None = Field(default=None, description="where the meaning and timing came from: user:turn:<n>, doc:<name>, or data")
    provenance: dict[str, Provenance] = Field(default_factory=dict, description="field name -> how it was settled")
    facts: ColumnFacts = Field(default_factory=ColumnFacts)

    def how(self, field: str) -> str:
        p = self.provenance.get(field)
        return p.tail() if p else ""

    @property
    def address(self) -> str:
        return f"col:{self.key}"

    def first_sentence(self) -> str:
        import re

        if not self.meaning:
            return "(not described)"
        return re.split(r"(?<=[.!?])\s", self.meaning.strip(), maxsplit=1)[0]

    def line(self) -> str:
        """One line for an index: address, name, kind and what the shape allows, distinct, nulls, the bounds or a few examples, the meaning."""
        f = self.facts
        kind = f.kind + (f" ({', '.join(f.role_hints)})" if f.role_hints else "")
        ex = ", ".join(f.levels()[:3])
        span = f"{f.bounds[0]} to {f.bounds[1]}" if f.bounds else (f"e.g. {ex}" if ex else "")
        return (
            f"[{self.address}] {self.name!r} · {kind} · {f.distinct} distinct · {f.nulls} null"
            + (f" · {span}" if span else "")
            + (f" · {self.first_sentence()}" if self.meaning else "")
        )

    def render(self) -> str:
        f = self.facts
        a = self.address
        lines = [f"[{a}] column {self.name!r}" + (f" ({self.role})" if self.role != "candidate" else "")]
        lines.append(f"  [{a}.note] {self.meaning or '(not described)'}" + self.how("meaning"))
        if self.stands_for:
            lines.append(f"  [{a}.stands_for] {self.stands_for}" + (f" ({self.proxy})" if self.proxy else "") + self.how("stands_for"))
        lines.append(f"  [{a}.when] {_WHEN_WORDS[self.when]}" + self.how("when"))
        if self.set_by:
            lines.append(f"  [{a}.set_by] set by {self.set_by}" + self.how("set_by"))
        if self.moved_by_change is not None:
            lines.append(
                f"  [{a}.moved] the change {'could have moved it' if self.moved_by_change else 'could not have moved it'}" + self.how("moved_by_change")
            )
        if self.measures_outcome is not None:
            lines.append(
                f"  [{a}.measures_outcome] {'another measure of the outcome' if self.measures_outcome else 'not a measure of the outcome'}"
                + self.how("measures_outcome")
            )
        lines.append(f"  [{a}.profile.kind] {f.kind}")
        lines.append(f"  [{a}.profile.nulls] {f.nulls} ({f.null_rate:.1%})")
        lines.append(f"  [{a}.profile.distinct] {f.distinct}{' (constant)' if f.constant else ''}")
        lines.append(f"  [{a}.profile.varies_over] {f.varies_over}")
        if f.role_hints:
            lines.append(f"  [{a}.profile.hints] {', '.join(f.role_hints)}")
        if f.bounds:
            lines.append(f"  [{a}.profile.bounds] {f.bounds[0]} to {f.bounds[1]}")
        if f.numeric:
            n = f.numeric
            lines.append(f"  [{a}.profile.numeric] min {n.get('min', 0):g}, p50 {n.get('p50', 0):g}, max {n.get('max', 0):g}, mean {n.get('mean', 0):g}")
        if f.top_values:
            tv = ", ".join(f"{t['value']}={t['share']:.0%}" for t in f.top_values[:6])
            lines.append(f"  [{a}.profile.top_values] {tv}")
        if f.datetime:
            d = f.datetime
            lines.append(f"  [{a}.profile.datetime] {d.get('first')} to {d.get('last')}, {d.get('inferred_frequency')}")
        if f.switch:
            s = f.switch
            lines.append(
                f"  [{a}.profile.switch] {s.get('entities_that_switch')} entities switch, {s.get('entities_never_on')} never on, first on {s.get('first_on')}"
            )
        if f.sentinels:
            lines.append(f"  [{a}.profile.sentinels] " + "; ".join(f.sentinels))
        if f.format_issues:
            lines.append(f"  [{a}.profile.format_issues] " + "; ".join(f.format_issues))
        return "\n".join(lines)


class Belief(BaseModel):
    """An uncheckable claim as the person left it: the value, their reason, and how settled it is."""

    kind: str
    value: bool | None = None
    what: str | None = None
    why: str | None = None
    column: str | None = None
    status: str = "empty"
    source: str | None = None
    said: str | None = None

    @property
    def address(self) -> str:
        return f"claim:{self.kind}"

    def known(self) -> bool:
        return self.status in {"confirmed", "drafted"} and self.value is not None

    def render(self) -> str:
        words = {
            "unobserved": (
                "something not in the file drove both who got the change and the outcome",
                "nothing outside the file drove both who got the change and the outcome",
            ),
            "exclusion": (
                "a column pushed units toward the change without touching the outcome any other way",
                "no column pushed units toward the change without touching the outcome",
            ),
            "spillover": (
                "a unit that got the change could affect the outcome of one that did not",
                "units that got the change could not affect the outcomes of those that did not",
            ),
            "trend_continues": (
                "without the change, the treated group would have kept moving with the others",
                "apart from the change, the treated group would have moved differently",
            ),
            "cutoff_only": ("nothing else switches at the cutoff", "something else also switches at the cutoff"),
        }.get(self.kind, (f"{self.kind}: yes", f"{self.kind}: no"))
        if self.status == "unknown":
            body = f"the person does not know ({self.kind.replace('_', ' ')})"
        elif not self.known():
            body = f"not asked ({self.kind.replace('_', ' ')})"
        else:
            body = words[0] if self.value else words[1]
        extra = " ".join(x for x in (self.what, self.why) if x)
        col = f"; the column is {self.column!r}" if self.column else ""
        return f"[{self.address}] {body}{col}" + (f". {extra}" if extra else "") + (f" [{self.source}]" if self.source else "")


class Said(BaseModel):
    """The person's own words, verbatim, so a lane's judgement sees their reasons and not only field values."""

    turn: int
    about: str = Field(description="which claim or question the words were about")
    text: str


class Probe(BaseModel):
    """A family probe the desk ran on the table: one number against one threshold."""

    family: str
    name: str
    value: float | None = None
    passed: bool | None = None
    detail: str = ""

    @property
    def address(self) -> str:
        return f"probe:{self.family}.{self.name}"

    def render(self) -> str:
        v = "pass" if self.passed else "FAIL" if self.passed is False else "n/a"
        return f"[{self.address}] {v}: {self.detail}"


class Design(BaseModel):
    """The family block: what a lane must not guess, as the desk filled it. Each family subclasses this and registers
    the subclass, so a pack read from JSON comes back as the family's own block without the pack naming any family."""

    kind: str

    def columns(self) -> list[str]:
        """Every column the block names, by the pack's name; the harness loads them even when the frame forgot one."""
        return []

    def time_column(self) -> str | None:
        return None

    def render(self) -> str:
        return "  (no fields)"


DESIGNS: dict[str, type[Design]] = {}


def register_design(cls: type[Design]) -> type[Design]:
    DESIGNS[cls.model_fields["kind"].default] = cls
    return cls


def parse_design(value: object) -> object:
    """A design as it arrives: already a block, or a dict whose kind names a registered block."""
    if value is None or isinstance(value, Design):
        return value
    if isinstance(value, dict):
        kind = value.get("kind")
        cls = DESIGNS.get(str(kind))
        if cls is None:
            raise ValueError(f"no family block is registered for kind {kind!r}; registered: {sorted(DESIGNS)}")
        return cls.model_validate(value)
    return value


_DATASET_FACETS = ("rows", "grain", "time_coverage", "entity_summary", "format_issues")
_COLUMN_FACETS = ("kind", "nulls", "distinct", "varies_over", "numeric", "top_values", "datetime", "switch", "sentinels", "format_issues")


class Handoff(BaseModel):
    """The context pack. The question, the decision, the columns that matter with the person's word on each, how the change
    happened, what the person believes and could not say, the probe numbers, the claim snapshot, and one family block."""

    # the decision (the fields every lane and the chat already read)
    family: str
    specialist: str
    supported_now: bool
    outcome: str = Field(description="the outcome column, by name")
    treatment: str | None = Field(description="the column that records who got the change, by name; null when the change is the rule itself")
    scope: Scope
    pack_name: str
    relevant_columns: list[Candidate] = Field(description="the slice the specialist starts from, with why each column matters")
    chosen_assumption: str
    reasons: list[Cited]
    # the question
    question: str = ""
    intent: Intent = "effect_of_change"
    design_id: int = Field(default=0, description="which design of this dataset the pack is; 0 for a forced hand-off")
    memory_version: int = Field(default=0, description="the memory version the pack was projected from")
    why: str = Field(default="", description="why this family over the others")
    over: dict[str, str] = Field(default_factory=dict, description="rejected family -> reason")
    # the data
    csv: str | None = None
    docs: dict[str, str] = Field(default_factory=dict)
    dataset_facts: dict = Field(default_factory=dict, description="rows, columns, duplicate_rows, grain, time_coverage, entity_summary, format_issues")
    grain: dict = Field(default_factory=dict)
    sampling: dict = Field(default_factory=dict)
    missing: dict = Field(default_factory=dict)
    # the columns that matter
    treated_level: str | None = None
    control_level: str | None = None
    columns: list[ColumnBrief] = Field(default_factory=list)
    # how the change happened
    change: dict = Field(default_factory=dict, description="what, to_whom, when, date_column, period_value")
    assignment: dict = Field(
        default_factory=dict,
        description="kind, rule, depends_on, treatment_column, treated_level, score_column, cutoff, treated_side, cutoff_value_treated, level_column, movable",
    )
    # what the person believes and could not say
    beliefs: dict[str, Belief] = Field(default_factory=dict)
    unknowns: list[str] = Field(default_factory=list, description="addresses the person could not settle")
    contradictions: list[str] = Field(default_factory=list, description="addresses the data refuted and the person kept, or two confirmed fields at odds")
    said: list[Said] = Field(default_factory=list)
    # evidence and the record
    probes: list[Probe] = Field(default_factory=list)
    claims: dict = Field(default_factory=dict, description="claim key -> {kind, fields, status, source}")
    # the family block
    design: SerializeAsAny[Design] | None = None

    @field_validator("design", mode="before")
    @classmethod
    def _design(cls, v: object) -> object:
        return parse_design(v)

    # ------------------------------------------------------------- columns
    def brief(self, name_or_key: str) -> ColumnBrief | None:
        from causal_agent.common.addresses import key as _k

        k = _k(name_or_key)
        for b in self.columns:
            if b.key == k or b.name == name_or_key:
                return b
        return None

    def brief_text(self, name_or_key: str) -> str:
        b = self.brief(name_or_key)
        return b.render() if b else f"[col:{name_or_key}] (no brief)"

    def render_columns(self) -> str:
        return "\n\n".join(b.render() for b in self.columns)

    def column_index(self) -> str:
        return "\n".join(b.line() for b in self.columns)

    # ------------------------------------------------------------- context
    def render_dataset(self) -> str:
        return render_dataset_text(self.pack_name, self.dataset_facts, self.grain, self.sampling, self.missing, self.docs)

    def render_change(self) -> str:
        return render_change_text(self.change, self.assignment)

    def render_beliefs(self) -> str:
        return "\n".join(b.render() for b in self.beliefs.values()) or "(no beliefs recorded)"

    def render_said(self) -> str:
        return "\n".join(f"[said:{s.turn}] about {s.about}: {s.text}" for s in self.said) or "(nothing quoted)"

    def render_design(self) -> str:
        return self.design.render() if self.design else "(no family block)"

    def render_open(self) -> str:
        """What the memory could not settle: the fields the person could not say, and the ones the data refuted and they kept."""
        lines = []
        if self.unknowns:
            lines.append("UNKNOWN (the person could not say): " + ", ".join(f"[{a}]" for a in self.unknowns))
        if self.contradictions:
            lines.append(
                "CONTRADICTION (the file disagrees and the person kept it; weigh it, and say which you took): "
                + ", ".join(f"[{a}]" for a in self.contradictions)
            )
        return "\n".join(lines)

    def render_words(self) -> str:
        """The person's words and the open fields, for every judgement a lane makes."""
        parts = ["WHAT THE PERSON SAID\n" + self.render_said()]
        if self.render_open():
            parts.append(self.render_open())
        return "\n\n".join(parts)

    def render_context(self) -> str:
        """The dataset, the change, the beliefs, the family block, the person's words, and what is open: what pack.digest() used to be."""
        return "\n\n".join(
            [self.render_dataset(), self.render_change(), "BELIEFS\n" + self.render_beliefs(), "FAMILY BLOCK\n" + self.render_design(), self.render_words()]
        )

    # ------------------------------------------------------------- addresses
    def addresses(self) -> set[str]:
        out = {"dataset", "dataset.note", "change:1", "change:1.note"}
        out.update(f"dataset.profile.{f}" for f in _DATASET_FACETS)
        for b in self.columns:
            a = b.address
            out.update({a, f"{a}.note", f"{a}.stands_for", f"{a}.when", f"{a}.set_by", f"{a}.moved", f"{a}.measures_outcome", f"{a}.role"})
            out.update(f"{a}.profile.{f}" for f in _COLUMN_FACETS)
        for k, c in self.claims.items():
            out.update({f"claim:{k}", f"claim:{k}.check"})
            out.update(f"claim:{k}.{f}" for f in (c.get("fields") or {}))
        for bl in self.beliefs.values():
            out.update(
                {
                    bl.address,
                    f"{bl.address}.exists",
                    f"{bl.address}.possible",
                    f"{bl.address}.believed",
                    f"{bl.address}.column",
                    f"{bl.address}.what",
                    f"{bl.address}.why",
                }
            )
        out.update(p.address for p in self.probes)
        out.update(f"said:{s.turn}" for s in self.said)
        return out

    def resolve(self, address: str) -> bool:
        from causal_agent.common.addresses import norm_address

        return norm_address(address) in {norm_address(a) for a in self.addresses()}


class Thought(BaseModel):
    """Debug only. Never gated, cited, or read by another node."""

    node: str
    text: str
    thinking_tokens: int | None = None
    output_tokens: int | None = None


# ----------------------------------------------------------------- specialist artifacts
# Lane-invariant. Every specialist produces these shapes; lane-specific parts sit inside.


class Contrast(Cited):
    """One comparison: rows at `control` versus rows at `treated`."""

    control: str
    treated: str

    @property
    def key(self) -> str:
        return f"{_slug(self.treated)}_vs_{_slug(self.control)}"


class CheckResult(BaseModel):
    contrast: str = Field(description="contrast key, or 'all' when the check does not depend on the contrast")
    name: str
    level: Literal["pass", "soft", "hard"]
    value: float | None = None
    threshold: float | None = None
    detail: str = ""

    @property
    def address(self) -> str:
        return f"check:{self.contrast}.{self.name}"


class Checks(BaseModel):
    results: list[CheckResult] = Field(default_factory=list)

    @property
    def flags(self) -> list[CheckResult]:
        return [r for r in self.results if r.level != "pass"]

    @property
    def hard(self) -> list[CheckResult]:
        return [r for r in self.results if r.level == "hard"]


class Estimate(BaseModel):
    contrast: str
    method: str
    value: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    n_treated: int = 0
    n_control: int = 0
    target_units: str = "ate"
    error: str | None = None
    secondary: bool = False


class Refutation(BaseModel):
    contrast: str
    refuter: str
    kind: Literal["falsification", "sensitivity"]
    new_effect: float | None = None
    range_low: float | None = None
    range_high: float | None = None
    p_value: float | None = None
    passed: bool | None = Field(default=None, description="None for sensitivity; it reports a range, not a verdict")
    detail: str = ""


class Interpretation(BaseModel):
    contrast: str
    answer: str = Field(description="the answer to the question for this contrast, two or three sentences, in the outcome's units")
    effect_stated: float = Field(description="the effect size you are reporting, copied from the estimate")
    caveats: list[str] = Field(description="what the reader must know: the assumption bet on, flags, refuters that failed")
    cites: list[str] = Field(description="artifact addresses such as estimate:<contrast>.value, check:<contrast>.overlap, refute:<contrast>.<refuter>.p_value")


class Feasibility(BaseModel):
    """An honest stop. Which stage, why, the facts, and what would make the analysis possible."""

    stage: str
    reason: str
    facts: list[str] = Field(default_factory=list)
    what_would_fix: str = ""


class Decline(BaseModel):
    """The lane did not take a pack field as given. Every override of the pack is one of these, with an address, so the
    brief and the page can show where the lane and the desk disagreed."""

    stage: str = Field(description="the node that declined")
    kind: Literal["declined", "replaced", "substituted"] = Field(
        description="declined: could not apply it; replaced: took another value; substituted: answered a near thing"
    )
    about: str = Field(description="the pack address: scope.window, design.cluster_level, claim:assignment.cutoff, col:<key>")
    pack_value: str | None = None
    took: str | None = None
    reason: str
    check: str = Field(description="the code rule that decided: intake.filter_unparsed, groups.level_observed, shape.pre_periods")
    cites: list[str] = Field(default_factory=list)

    @property
    def address(self) -> str:
        return f"decline:{self.stage}.{_slug(self.about)}"

    def render(self) -> str:
        pack = f" · pack said {self.pack_value!r}" if self.pack_value is not None else ""
        took = f" · lane took {self.took!r}" if self.took is not None else ""
        return f"[{self.address}] {self.kind}: {self.about}{pack}{took} · {self.reason} ({self.check})"


class LaneAsk(BaseModel):
    """One question back to the desk, keyed to a memory address the desk can settle. The desk asks it as it asks
    everything else, and the lane runs again on the memory as it then stands."""

    address: str
    question: str
    options: list[str] = Field(default_factory=list)
    because: str = Field(default="", description="the check or belief that opened the question, in words")
    evidence: list[str] = Field(default_factory=list, description="check addresses shown beside the question")
    stage: str = ""


class RunRecord(BaseModel):
    """What one run left behind, as the desk keeps it in state and the server shows it."""

    index: int
    dataset: str
    question: str
    family: str | None = None
    specialist: str | None = None
    status: str = "no_handoff"
    run_dir: str | None = None
    design_dir: str | None = None
    figures: list[dict] = Field(default_factory=list, description="FigureSpecs the run left behind, the ready-moment figure first")
    what_if: dict[str, str] = Field(default_factory=dict, description="for a what-if design: the fields changed on the fork, address -> value")
    differs: list[str] = Field(default_factory=list, description="the fields that differ from the design before, by address")
    effect: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    estimator: str | None = None
    decision_record: str = ""
    decision: dict = Field(default_factory=dict)  # chosen, chosen_assumption, why, over: {family: reason}
    specialist_result: dict = Field(default_factory=dict)
    artifacts: dict = Field(default_factory=dict)  # artifacts.json when the run dir holds one


def _slug(s: str) -> str:
    import re

    out = re.sub(r"[^0-9a-zA-Z]+", "_", str(s).strip()).strip("_").lower()
    return out or "x"


def render_dataset_text(name: str, facts: dict, grain: dict, sampling: dict, missing: dict, docs: dict | None = None) -> str:
    """The dataset card: what the memory says about the rows, then the profile's facts, every line addressed."""
    docs = docs or {}
    about = []
    if grain.get("row_is"):
        about.append(f"each row is {str(grain['row_is']).rstrip('.')}")
    if grain.get("panel") is True:
        about.append("the same unit appears in more than one period")
    if grain.get("nesting"):
        about.append(f"units sit inside {', '.join(grain['nesting'])}")
    if sampling.get("how"):
        about.append(f"rows were chosen: {sampling['how']}" + (f", {sampling['detail']}" if sampling.get("detail") else ""))
    if missing.get("why"):
        about.append(f"missing values: {missing['why']}")
    lines = [f"[dataset] {name}", f"  [dataset.note] {'. '.join(about) if about else docs.get('about') or '(nothing stated about the rows)'}"]
    lines.append(
        f"  [dataset.profile.rows] {facts.get('rows', '?')} rows, {facts.get('columns', '?')} columns, {facts.get('duplicate_rows', 0)} duplicate rows"
    )
    g = facts.get("grain") or []
    lines.append(f"  [dataset.profile.grain] {' + '.join(g) if g else 'no key column found'}")
    t = facts.get("time_coverage")
    lines.append(
        f"  [dataset.profile.time_coverage] {t['column']}: {t['first']} to {t['last']}, {t.get('inferred_frequency')}, {t.get('gaps')} gaps"
        if t
        else "  [dataset.profile.time_coverage] no time column"
    )
    e = facts.get("entity_summary")
    lines.append(
        f"  [dataset.profile.entity_summary] {' + '.join(e['columns'])}: {e['entities']} entities, {e['rows_per_entity_min']} to {e['rows_per_entity_max']} rows each"
        if e
        else "  [dataset.profile.entity_summary] no entity column declared"
    )
    if facts.get("format_issues"):
        lines.append("  [dataset.profile.format_issues] " + "; ".join(facts["format_issues"]))
    return "\n".join(lines)


def render_change_text(ch: dict, a: dict) -> str:
    """The change card, one paragraph, addressed change:1.note."""
    if not ch.get("what") and not a.get("rule"):
        return "[change:1.note] not stated"
    parts = []
    if ch.get("what"):
        parts.append(
            f"{str(ch['what']).rstrip('.')}." + (f" It reached {ch['to_whom']}," if ch.get("to_whom") else "") + (f" {ch['when']}." if ch.get("when") else "")
        )
    if ch.get("date_column"):
        parts.append(
            f"The period column is {ch['date_column']!r}" + (f", the change took effect at {ch['period_value']!r}." if ch.get("period_value") else ".")
        )
    kind_words = {
        "lottery": "a random draw decided who got it",
        "cutoff_rule": "a line on a measured score decided who got it",
        "own_choice": "units chose whether to take it, with or without an offer",
        "date_by_others": "it reached some units on a date by a decision made above them",
        "third_party": "someone else picked the units one by one",
    }
    if a.get("kind"):
        parts.append(f"How it was assigned: {kind_words.get(a['kind'], a['kind'])}.")
    if a.get("rule"):
        parts.append(f"The rule, in the person's words: {str(a['rule']).rstrip('.')}.")
    if a.get("depends_on"):
        parts.append(f"The decision or the offer depended on {', '.join(a['depends_on'])}.")
    if a.get("treatment_column"):
        parts.append(
            f"{a['treatment_column']!r} records who got it" + (f", {a['treated_level']!r} meaning yes." if a.get("treated_level") is not None else ".")
        )
    if a.get("score_column"):
        side = a.get("treated_side") or "above"
        parts.append(
            f"The score is {a['score_column']!r}, treated {side} {a.get('cutoff')}"
            + (", the cutoff value itself treated." if a.get("cutoff_value_treated") else ".")
        )
    if a.get("level_column"):
        parts.append(f"It was assigned at the level of {a['level_column']!r}.")
    if a.get("movable") is not None:
        parts.append("A unit could change what the rule looked at." if a["movable"] else "A unit could not change what the rule looked at.")
    return "[change:1.note] " + " ".join(parts)
