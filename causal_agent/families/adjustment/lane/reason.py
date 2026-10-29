"""Admitted context to evidence ledger, rival certificates, and frozen designs."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Literal

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from langgraph.types import Command
from pydantic import Field, create_model

from causal_agent.common.addresses import key
from causal_agent.common.contracts import CheckResult, Checks, Contrast, Feasibility, Handoff
from causal_agent.common.contracts.base import _slug
from causal_agent.families.adjustment.design import AdjustmentDesign
from causal_agent.families.adjustment.lane import checks as CK
from causal_agent.families.adjustment.lane import prompts as P
from causal_agent.families.adjustment.lane import support as S
from causal_agent.families.adjustment.lane.contracts import (
    Candidate,
    Contrasts,
    Decision,
    Design,
    Edge,
    Excluded,
    Graph,
    Ladder,
    Mechanism,
    Relationships,
)
from causal_agent.families.adjustment.lane.engine import model as engine
from causal_agent.families.adjustment.lane.knowledge import SPEC, load_checks, load_endpoint_catalogue, load_estimators, load_falsifications, load_sensitivities
from causal_agent.families.adjustment.lane.selection import diagnostic_tools
from causal_agent.families.adjustment.lane.state import SpecialistState, ladder_of
from causal_agent.families.adjustment.lane.views import build as view_for
from causal_agent.lane import intake
from causal_agent.lane import nodes as L
from causal_agent.lane.endpoints import design_identity, validate_context
from causal_agent.lane.episode import run_episode
from causal_agent.lane.ledger import Entry, Evidence, Ledger, apply


def case(state):
    return {"check_facts": {"admission": state["handoff"].admission.model_dump()}}


feasibility = L.feasibility
_CFG = yaml.safe_load((Path(__file__).parent / "knowledge" / "rivals.yaml").read_text())


class RestrictedTools:
    def __init__(self, tools, names):
        self.tools, self.allowed = tools, set(names)

    def names(self):
        return tuple(n for n in self.tools.names() if n in self.allowed)

    def schemas(self):
        return [s for s in self.tools.schemas() if s.name in self.allowed]

    def call(self, name, args):
        from causal_agent.lane.tools import Refused

        if name not in self.allowed:
            raise Refused("tool is not permitted by this decision declaration")
        if name != "describe":
            columns = [v for k, value in args.items() if k in {"column", "a", "b", "columns"} for v in (value if isinstance(value, list) else [value])]
            if any(isinstance(v, str) and self.tools.column(v) == self.tools.outcome for v in columns):
                raise Refused(
                    "outcome relational diagnostics are permitted only after design freeze; before freeze describe the outcome univariately for scale/shape only"
                )
        return self.tools.call(name, args)


def _stop(node, reason, facts=(), extra=None, *, kind=None):
    kind = (
        kind
        or {
            "load": "unsupported_method",
            "pair": "exhausted_reasoning",
            "mechanism": "exhausted_reasoning",
            "relate": "exhausted_reasoning",
            "rivals": "unidentified_effect",
            "check_design": "inadequate_support",
            "pick_estimator": "exhausted_reasoning",
        }[node]
    )
    f = Feasibility(
        terminal_kind=kind, stage=node, reason=reason, facts=list(facts), what_would_fix="resolve the named context, identification or computational deficiency"
    )
    return Command(goto="feasibility", update={"feasibility": f, **(extra or {})})


def _transitions(state, candidates, stage):
    previous = {c.id: c for c in state.get("candidates") or []}
    catalogue = load_endpoint_catalogue()
    admission = {e.endpoint_id: e for e in state["handoff"].admission.endpoints}
    changes = []
    for c in candidates:
        endpoint = catalogue.endpoint(c.endpoint_id)
        obligations = [o.model_dump() for o in admission[c.endpoint_id].obligations]
        obligations += [{"id": "identification", "status": "passed", "reason": c.estimand.symbolic, "cites": c.cites}]
        obligations += [
            {"id": r.name, "status": "failed" if r.level == "hard" else "passed" if r.level == "pass" else "assumed", "reason": r.detail, "cites": [r.address]}
            for r in c.checks.results
        ]
        obligations += [
            {
                "id": "estimator",
                "status": "passed" if c.chosen_estimator else "pending",
                "reason": c.chosen_estimator or ",".join(c.allowed_estimators),
                "cites": [],
            }
        ]
        changes.append(
            {
                "candidate_id": c.id,
                "endpoint_id": c.endpoint_id,
                "stage": stage,
                "ledger_revision": state["ledger"].revision,
                "from_status": previous[c.id].status if c.id in previous else "proposed",
                "to_status": c.status,
                "reason": c.reason,
                "obligations": obligations,
                "declared_trajectory": endpoint.trajectory,
            }
        )
    return changes


def load(state: SpecialistState) -> Command[Literal["case", "feasibility"]]:
    h = Handoff.model_validate(state["handoff"])
    deficiency = validate_context(h, load_endpoint_catalogue())
    if deficiency:
        return _stop("load", deficiency.reason, deficiency.addresses, {"handoff": h, "context_deficiency": deficiency}, kind="context_deficiency")
    if h.intent != "effect_of_change" or not h.treatment:
        return _stop("load", "requires one treatment and one causal effect question", extra={"handoff": h})
    try:
        it = intake.load(h, "dowhy")
    except intake.IntakeStop as ex:
        return Command(goto="feasibility", update={"handoff": h, "feasibility": ex.feasibility})
    t, y = key(h.treatment), key(h.outcome)
    if t != S.TREATED and S.TREATED in it.table.columns:
        return _stop(
            "load",
            "reserved internal treatment column 'treated' collides with an unrelated admitted column; rename that source column before a newly authorized run",
            [f"col:{S.TREATED}"],
            {"handoff": h, "run_dir": str(it.run_dir)},
        )
    kind = CK.outcome_kind(it.table[y])
    nlevels = it.table[t].nunique()
    if kind is None or nlevels < 2 or nlevels > 12:
        return _stop(
            "load",
            "unsupported outcome or treatment levels; explicit binary contrasts require 2 to 12 levels",
            [f"outcome={kind}, levels={nlevels}"],
            {"handoff": h, "run_dir": str(it.run_dir)},
        )
    if kind == "binary" and not set(it.table[y].dropna().unique()).issubset({0, 1}):
        return _stop("load", "binary outcomes must be explicitly encoded 0/1 before admission", extra={"handoff": h, "run_dir": str(it.run_dir)})
    return Command(
        goto="case",
        update={
            "handoff": h,
            "run_dir": str(it.run_dir),
            "table_path": str(it.table_path),
            "columns": it.columns,
            "declines": it.declines,
            "check_facts": {"intake": it.facts},
            "outcome_kind": kind,
            "source_row_ids": [int(i) for i in it.table.index],
            "ledger": Ledger(),
            "ladder": Ladder(),
            "relate_retries": 0,
            "candidates": [],
            "designs": {},
        },
    )


def _episode(state, node, schema, system, view, gate, *, tools=None, evidence_gate=None, before_commit=None):
    visit = 1 + sum(k.startswith(node + "_") for k in state.get("episodes", {}))
    name = f"{node}_{visit}"
    if tools is None and SPEC.budget(node):
        t, _, _ = L.keys(state)
        table = L.table(state)
        cs = state.get("contrasts") or []
        mask = table[t].astype(str) == str(cs[0].treated) if cs else None
        tools = RestrictedTools(L.data_tools(state, mask), SPEC.rung(node).tools)
    result, log, thoughts, errors = run_episode(
        schema,
        system,
        view.text,
        tools=tools,
        budget=SPEC.budget(node),
        gate=lambda r, log: gate(r, lambda a: view.resolve(a) or log.resolve(a)) + (evidence_gate(r, log) if evidence_gate else []),
        node=name,
        notes=True,
        before_commit=before_commit,
    )
    return (
        result,
        {
            "episodes": {name: log},
            "debug": thoughts,
            "reasoning_history": [
                {
                    "node": node,
                    "visit": log.visit,
                    "view": view.text,
                    "record": result.model_dump() if result else None,
                    "log": log.model_dump(),
                    "errors": errors,
                }
            ],
        },
        errors,
    )


def pair(state: SpecialistState) -> Command[Literal["seed", "feasibility"]]:
    h = state["handoff"]
    t, y, _ = L.keys(state)
    levels = sorted(L.table(state)[t].astype(str).unique())
    cs = []
    if h.treated_level in levels:
        controls = [h.control_level] if h.control_level in levels else [v for v in levels if v != h.treated_level]
        cs = [Contrast(treated=h.treated_level, control=v, reason="admitted treatment comparison", cites=["pair.levels"]) for v in controls]
    if not cs:
        view = view_for("pair", h, state["ledger"], material="OBSERVED LEVELS\n" + repr(levels), columns=[t, y])

        def gate(r, resolve):
            errs = []
            for c in r.items:
                if c.treated not in levels or c.control not in levels or c.treated == c.control:
                    errs.append("comparison must use two different observed levels")
                errs += [f"unresolved citation {a}" for a in c.cites if not resolve(a)]
            if not r.items or len(r.items) > 11:
                errs.append("one to eleven explicit comparisons required")
            if not S.unique_contrast_addresses(r.items):
                errs.append("raw contrast labels collide at their artifact addresses; use distinct source labels")
            return errs

        rec, updates, errors = _episode(state, "pair", Contrasts, P.PAIR, view, gate)
        if rec is None:
            return _stop("pair", "comparison judgement exhausted", errors, updates)
        cs = rec.items
    else:
        updates = {}
    lad = ladder_of(state).model_copy(
        update={
            "pair": Decision(
                reason="comparison fixed before outcome effects", values={"pair.contrasts": [c.model_dump() for c in cs], "pair.target": h.scope.target}
            )
        }
    )
    if not S.unique_contrast_addresses(cs):
        return _stop(
            "pair",
            "raw contrast labels collide at their artifact addresses; relabel them before a newly authorized run",
            extra=updates,
            kind="unsupported_method",
        )
    return Command(goto="seed", update={**updates, "contrasts": cs, "ladder": lad})


def seed(state: SpecialistState) -> dict:
    h = state["handoff"]
    t, y, _ = L.keys(state)
    ledger = Ledger()
    entries = []

    def add(kind, src, dst, value, address, confirmed=True):
        evidence = _evidence({**state, "ledger": ledger}, address, kind)
        source_confirmation = evidence.source in {"pack", "person"} and evidence.category == "study_account"
        measurement_confirmation = kind == "when" and evidence.source == "data" and evidence.category == "measurement_timing"
        entries.append(
            Entry(
                id=f"rel:{kind}.{src}.{dst or 'value'}",
                kind=kind,
                src=src,
                dst=dst,
                value=value,
                status="confirmed" if confirmed and (source_confirmation or measurement_confirmation) else "proposed",
                evidence=[evidence],
                reason="literal relationship or timing reported in admitted context",
            )
        )

    for b in h.columns:
        if b.key not in state["columns"]:
            continue
        for field, kind, dst in [("feeds_treatment", "edge", t), ("moves_outcome", "edge", y), ("moved_by_change", "edge", b.key)]:
            a = f"col:{b.key}.{field}"
            f = h.fact(a)
            if f is None:
                continue
            if b.key in (t, y):
                continue
            src = t if field == "moved_by_change" else b.key
            if f.value is not None and f.status == "confirmed":
                add(kind if f.value else "no_edge", src, dst, bool(f.value), a)
        for field, kind in [
            ("when", "when"),
            ("same_as", "same_as"),
            ("nested_in", "nested_in"),
            ("measures_outcome", "outcome_measure"),
            ("may_modify", "modifies"),
        ]:
            a = f"col:{b.key}.{field}"
            f = h.fact(a)
            if f and f.value is not None and f.status == "confirmed":
                add(kind, b.key, None, f.value, a)
    if _fact_true(h, "claim:change.no_causal_path"):
        add("no_edge", t, y, False, "claim:change.no_causal_path")
    f = h.fact("claim:unobserved.exists")
    if f and f.value is True:
        add("hidden", "hidden:unmeasured", None, [t, y], "claim:unobserved.exists", f.status == "confirmed")
    ledger, errors = apply(ledger, entries, by="seed", resolves=h.resolve)
    if errors:
        raise ValueError("; ".join(errors))
    return {"ledger": ledger}


def mechanism(state: SpecialistState) -> Command[Literal["relate", "feasibility"]]:
    h = state["handoff"]
    t, _, _ = L.keys(state)
    view = view_for("mechanism", h, state["ledger"], columns=[t, *[key(x) for x in h.assignment.get("depends_on") or []]])

    def gate(r, resolve):
        return (
            [f"unknown driver {c}" for c in r.drivers if c not in state["columns"]]
            + [f"unresolved citation {a}" for a in r.cites if not resolve(a)]
            + ([] if r.reason.strip() else ["reason required"])
        )

    rec, updates, errors = _episode(state, "mechanism", Mechanism, P.MECHANISM, view, gate)
    if rec is None:
        return _stop("mechanism", "assignment reading exhausted", errors, updates)
    entries = []
    for c in rec.drivers:
        ident = f"rel:edge.{c}.{t}"
        if ident not in state["ledger"].entries:
            entries.append(
                Entry(
                    id=ident,
                    kind="edge",
                    src=c,
                    dst=t,
                    status="supported" if any(_evidence(state, a, "edge").category == "study_account" for a in rec.cites) else "proposed",
                    reason=rec.reason,
                    evidence=[_evidence(state, a, "edge") for a in rec.cites],
                )
            )
    ledger, errors = apply(state["ledger"], entries, by="mechanism", resolves=view.resolve)
    if errors:
        return _stop("mechanism", "mechanism ledger update failed", errors, updates)
    lad = ladder_of(state).model_copy(update={"mechanism": Decision(reason=rec.reason, cites=rec.cites, values={"mechanism.record": rec.model_dump()})})
    return Command(goto="relate", update={**updates, "ledger": ledger, "ladder": lad})


def _evidence(state, address, kind):
    if address.startswith("pair."):
        source, category = "judgement", "assumption"
    elif address.startswith("probe:") or ".profile." in address:
        source, category = "data", "association"
    elif address in state["ledger"].entries:
        old = state["ledger"].entries[address]
        accounts = [e for e in old.evidence if e.source in {"pack", "person"} and e.category == "study_account"]
        if old.status == "confirmed" and accounts:
            source, category = "pack", "study_account"
        elif old.status == "confirmed" and any(e.source == "data" for e in old.evidence):
            source, category = "data", "measurement_timing" if kind == "when" else "association"
        else:
            source, category = "judgement", "assumption"
    else:
        fact = state["handoff"].fact(address)
        if fact is None and address.startswith("col:") and address.endswith(".note"):
            fact = state["handoff"].fact(address[:-5] + ".meaning")
        provenance = (fact.source or "") if fact else ""
        if provenance.startswith("data"):
            source, category = "data", "measurement_timing" if kind == "when" else "association"
        elif fact and fact.status == "confirmed" and provenance.startswith(("user:", "doc:", "fixture:", "eval:")):
            source, category = "pack", "study_account"
        else:
            source, category = "judgement", "assumption"
    return Evidence(source=source, address=address, category=category)


def _draft_entries(state, record):
    entries = []
    for proposal in record.entries:
        destination = proposal.dst or None
        identity = f"rel:{proposal.kind}.{proposal.src}.{destination or 'value'}"
        value = True if proposal.kind == "edge" else False if proposal.kind == "no_edge" else proposal.value
        evidence = [_evidence(state, address, proposal.kind) for address in proposal.cites]
        entry = Entry(
            id=identity, kind=proposal.kind, src=proposal.src, dst=destination, value=value, status=proposal.status, reason=proposal.reason, evidence=evidence
        )
        old = state["ledger"].entries.get(identity)
        if old and old.status == "confirmed" and old.kind == entry.kind and old.src == entry.src and old.dst == entry.dst and old.value == entry.value:
            continue  # a consistent read is not a new write or a promotion
        entries.append(entry)
    return entries


def _relationship_gate(state, record, resolve, *, all_columns):
    _, _, cols = L.keys(state)
    errors = []
    if all_columns and set(record.considered) != set(cols):
        errors.append(f"consider exactly these non-pair raw column keys: {cols}")
    if len(record.considered) != len(set(record.considered)):
        errors.append("duplicate considered columns")
    allowed = set(state["columns"])
    for entry in _draft_entries(state, record):
        if entry.kind in {"edge", "no_edge", "hidden"} and entry.status in {"supported", "confirmed"}:
            if not any(e.source in {"person", "pack"} and e.category == "study_account" for e in entry.evidence):
                errors.append(f"{entry.id}: causal support requires a specific study account; pair/profile/timing alone only permit proposed assumptions")
        if entry.src not in allowed and not (entry.src or "").startswith("hidden:"):
            errors.append(f"source must be a raw column key, not an address: {entry.src}")
        if entry.dst is not None and entry.dst not in allowed and not entry.dst.startswith("hidden:"):
            errors.append(f"destination must be a raw column key, not an address: {entry.dst}")
        if entry.kind == "hidden" and (not isinstance(entry.value, list) or any(v not in allowed for v in entry.value)):
            errors.append("hidden value must be a JSON list of affected raw column keys")
        if entry.kind in ("outcome_measure", "modifies") and not isinstance(entry.value, bool):
            errors.append(f"{entry.kind} value must be true or false")
        old = state["ledger"].entries.get(entry.id)
        if old and old.status == "confirmed":
            errors.append(f"cannot overwrite the confirmed source assertion {entry.id}; retain it and name a separate conflict")
        if entry.kind in ("edge", "no_edge") and (not entry.dst or entry.src == entry.dst):
            errors.append(
                f"{entry.kind} from {entry.src}: populate dst with a distinct raw destination column key. For t to m use src=t, dst=m; for m to y use src=m, dst=y."
            )
    _, ledger_errors = apply(state["ledger"], _draft_entries(state, record), by="gate", resolves=resolve)
    return errors + ledger_errors + [f"unresolved citation {a}" for a in record.cites if not resolve(a)]


def relate(state: SpecialistState) -> Command[Literal["probe", "feasibility"]]:
    h = state["handoff"]
    _, _, cols = L.keys(state)
    view = view_for(
        "relate",
        h,
        state["ledger"],
        material="CONSIDER THESE NON-PAIR COLUMNS\n" + repr(cols) + "\nPrevious structural errors\n" + repr(state.get("reasoning_errors") or []),
    )
    rec, updates, errors = _episode(state, "relate", Relationships, P.RELATE, view, lambda r, resolve: _relationship_gate(state, r, resolve, all_columns=True))
    if rec is None:
        return _stop("relate", "relationship reasoning exhausted without an admissible record", errors, updates)
    ledger, errors = apply(
        state["ledger"],
        _draft_entries(state, rec),
        by="relate",
        resolves=lambda a: view.resolve(a) or any(log.resolve(a) for log in updates["episodes"].values()),
    )
    return Command(
        goto="probe",
        update={
            **updates,
            "ledger": ledger,
            "ladder": ladder_of(state).model_copy(
                update={"relationships": Decision(reason=rec.reason, cites=rec.cites, values={"relationships.summary": rec.reason})}
            ),
        },
    )


def probe(state: SpecialistState) -> dict:
    uncertain = [
        e for e in state["ledger"].entries.values() if e.kind in ("edge", "hidden", "no_edge", "outcome_measure") and e.status in ("proposed", "contested")
    ]
    if not uncertain:
        return {}
    cols = sorted({v for e in uncertain for v in (e.src, e.dst) if v in state["columns"]})
    view = view_for(
        "probe", state["handoff"], state["ledger"], material="UNSETTLED STRUCTURAL ENTRIES\n" + "\n".join(e.model_dump_json() for e in uncertain), columns=cols
    )
    rec, updates, errors = _episode(state, "probe", Relationships, P.PROBE, view, lambda r, resolve: _relationship_gate(state, r, resolve, all_columns=False))
    if rec is None:
        return {**updates, "reasoning_errors": errors}
    ledger, errors = apply(
        state["ledger"],
        _draft_entries(state, rec),
        by="probe",
        resolves=lambda a: view.resolve(a) or any(log.resolve(a) for log in updates["episodes"].values()),
    )
    return {
        **updates,
        "ledger": ledger,
        "ladder": ladder_of(state).model_copy(update={"probe": Decision(reason=rec.reason, cites=rec.cites, values={"probe.summary": rec.reason})}),
    }


def _graphs(state) -> tuple[list[Graph], list[str]]:
    ledger = state["ledger"]
    t, y, _ = L.keys(state)
    structural = [e for e in ledger.entries.values() if e.kind in ("edge", "no_edge", "hidden", "outcome_measure") and e.status != "rejected"]
    uncertain = [e for e in structural if e.status in ("proposed", "contested")]
    if len(uncertain) > _CFG["max_unsettled"]:
        return [], [f"reasoning.exhausted: {len(uncertain)} structural uncertainties exceed {_CFG['max_unsettled']}; nothing silently omitted"]
    excluded = [
        Excluded(column=e.src, why=e.reason) for e in ledger.entries.values() if e.kind == "outcome_measure" and e.value is True and e.src not in (t, y)
    ]
    out = []
    errors = []
    seen = set()
    for bits in itertools.product((False, True), repeat=len(uncertain)):
        use = {e.id: bit for e, bit in zip(uncertain, bits, strict=True)}
        edges = []
        absences = set()
        branch_excluded = [
            x for x in excluded if use.get(next((entry.id for entry in structural if entry.kind == "outcome_measure" and entry.src == x.column), ""), True)
        ]
        omitted = {x.column for x in branch_excluded}
        nodes = set(state["columns"]) - omitted
        for e in structural:
            active = use.get(e.id, True)
            if e.kind == "outcome_measure":
                continue
            if e.kind == "no_edge":
                if active:
                    absences.add((e.src, e.dst))
                elif e.src not in omitted and e.dst not in omitted:
                    edges.append(Edge(src=e.src, dst=e.dst, cites=[x.address for x in e.evidence], status="proposed"))
                continue
            if not active:
                continue
            if e.kind == "hidden":
                nodes.add(e.src)
                edges += [Edge(src=e.src, dst=v, cites=[x.address for x in e.evidence], status=e.status) for v in e.value or [] if v in nodes]
            elif e.src not in omitted and e.dst not in omitted:
                edges.append(Edge(src=e.src, dst=e.dst, cites=[x.address for x in e.evidence], status=e.status))
        if any((e.src, e.dst) in absences for e in edges):
            errors.append("contradictory edge and no-edge assumptions")
            continue
        g = Graph(treatment=t, outcome=y, nodes=sorted(nodes), edges=sorted(edges, key=lambda e: (e.src, e.dst)), excluded=branch_excluded)
        if _fact_true(state["handoff"], "claim:change.no_causal_path") and nx.has_path(g.to_networkx(), t, y):
            errors.append("graph contradicts the explicit supplied absence of every treatment-outcome path")
            continue
        if not nx.is_directed_acyclic_graph(g.to_networkx()):
            errors.append("candidate cycle")
            continue
        ordering = {"before": 0, "at": 1, "after": 2}
        if any(
            ordering.get(state["handoff"].column(e.src).when, 0) > ordering.get(state["handoff"].column(e.dst).when, 2)
            for e in g.edges
            if state["handoff"].column(e.src) and state["handoff"].column(e.dst)
        ):
            errors.append("edge contradicts recorded temporal ordering")
            continue
        stamp = engine.fingerprint(g)
        if stamp not in seen:
            out.append(g)
            seen.add(stamp)
    return out, errors


def _fact_true(h, address):
    f = h.fact(address)
    return bool(f and f.status == "confirmed" and f.value is True)


def rivals(state: SpecialistState) -> Command[Literal["relate", "check_design", "feasibility"]]:
    h = state["handoff"]
    graphs, errors = _graphs(state)
    if not graphs:
        if not any("exhausted" in e for e in errors) and state.get("relate_retries", 0) < 1:
            return Command(goto="relate", update={"relate_retries": 1, "reasoning_errors": errors})
        return _stop("rivals", "no admissible graph candidate", errors, kind="exhausted_reasoning")
    catalogue = load_endpoint_catalogue()
    admitted = {e.endpoint_id for e in h.admission.endpoints if e.admitted}
    table = L.table(state)
    block = h.design if isinstance(h.design, AdjustmentDesign) else AdjustmentDesign()
    instruments = block.instruments or ([block.instrument] if block.instrument else [])
    mediator = block.mediator
    modifiers = block.modifiers or [b.key for b in h.columns if b.may_modify and b.when == "before"]
    target = {
        "average": "ate",
        "on_treated": "att",
        "on_untreated": "atc",
        "local": "late",
        "natural_direct": "nde",
        "natural_indirect": "nie",
        "conditional": "conditional",
    }.get(h.scope.target, h.scope.target)
    candidates = []
    diagnostics = []
    seen = set()
    for graph in graphs:
        sub = S.contrast_table(table, graph.treatment, state["contrasts"][0])
        certs, why = engine.identify_candidates(graph, sub, _CFG["backdoor_methods"], mediator=mediator, instruments=instruments, controls=block.iv_controls)
        gid = engine.fingerprint(graph)
        diagnostics.extend({"graph_id": gid, "status": "excluded", "reason": reason} for reason in why)
        for e in certs:
            end = None
            if e.kind == "zero":
                if not _fact_true(h, "claim:change.no_causal_path"):
                    diagnostics.append(
                        {
                            "graph_id": gid,
                            "endpoint_id": "structural.zero",
                            "status": "unidentified",
                            "reason": "unspecified edges cannot establish structural zero; explicit no-causal-path evidence is absent",
                        }
                    )
                    continue
                end = "structural.zero"
            elif e.kind in ("backdoor", "general_adjustment"):
                if (
                    target == "conditional"
                    and modifiers
                    and engine.valid_model_adjustment(graph, sorted(set(e.adjustment_set) | set(modifiers)))
                    and state["outcome_kind"] == "continuous"
                ):
                    end = "adjustment.linear_conditional"
                    e.model_covariates = sorted(set(e.adjustment_set) | set(modifiers))
                elif target in ("ate", "att", "atc"):
                    end = "total.general_adjustment" if e.kind == "general_adjustment" else "total.backdoor" if e.adjustment_set else "total.unadjusted"
            elif e.kind == "frontdoor" and target == "ate" and len(e.frontdoor_set) == 1 and state["outcome_kind"] == "continuous":
                end = "frontdoor.linear_total"
            elif (
                e.kind == "mediation"
                and target == e.target
                and state["outcome_kind"] == "continuous"
                and _fact_true(h, "claim:mediator.sequential_ignorability")
                and _fact_true(h, "claim:mediator.no_treatment_confounder")
                and _fact_true(h, "claim:mediator.linear_additive")
            ):
                end = "mediation.linear_" + e.target
            elif e.kind == "iv":
                if target == "late" and len(e.instruments) == 1 and sub[e.instruments[0]].nunique() == 2 and _fact_true(h, "claim:monotonicity.believed"):
                    end = "iv.wald_late"
                elif target == "ate" and _fact_true(h, "claim:exclusion.homogeneous"):
                    end = "iv.linear_conditional" if e.adjustment_set else "iv.linear_multiple" if len(e.instruments) > 1 else "iv.linear_single"
            if end not in admitted:
                diagnostics.append(
                    {
                        "graph_id": gid,
                        "endpoint_id": end or e.kind,
                        "status": "unsupported",
                        "reason": "requested target, admitted assumptions or engine shape excludes this certificate",
                    }
                )
                continue
            definition = catalogue.endpoint(end)
            e.target = target
            meaning = {
                "type": target,
                "population": h.scope.population_filter or "scoped dataset",
                "outcome": graph.outcome,
                "scale": "probability_difference" if state["outcome_kind"] == "binary" else "outcome_units",
                "scope": h.scope.target,
            }
            if target in ("att", "atc"):
                meaning["population"] = ("treated units within " if target == "att" else "untreated units within ") + meaning["population"]
            if target == "late":
                meaning["population"] = "compliers for the stated instrument"
                meaning["instruments"] = e.instruments
            if target in ("nie", "nde"):
                meaning["mediator"] = e.mediator
            stamp = json.dumps([gid, end, e.adjustment_set, e.instruments, e.frontdoor_set, e.mediator, target], sort_keys=True)
            if stamp in seen:
                continue
            seen.add(stamp)
            ident = design_identity(end, {"graph": gid, "certificate": e.model_dump(), "target": meaning}, h.context_id)
            assumptions = [*definition.assumptions, "every omitted identification-relevant edge is a conditional graph assumption, not an observed absence"]
            candidates.append(
                Candidate(
                    id=ident,
                    graph_id=gid,
                    endpoint_id=end,
                    graph=graph,
                    estimand=e,
                    target=meaning,
                    assumptions=assumptions,
                    cites=[a for edge in graph.edges for a in edge.cites],
                )
            )
        diagnostics.append({"graph_id": gid, "endpoint_id": "identification.id_expression", **engine.id_diagnostic(graph, set(table.columns))})
    if len(candidates) > _CFG["max_designs"]:
        return _stop(
            "rivals",
            "candidate design budget exhausted",
            [f"{len(candidates)} designs exceed {_CFG['max_designs']}; no omitted candidate is treated as absent"],
            {"candidates": candidates, "diagnostic_candidates": diagnostics},
            kind="exhausted_reasoning",
        )
    if not candidates:
        return _stop(
            "rivals",
            "no executable endpoint identifies the requested target",
            [d.get("reason", "") for d in diagnostics],
            {"candidates": [], "diagnostic_candidates": diagnostics},
            kind="unsupported_method"
            if any(d.get("status") == "unsupported" and d.get("endpoint_id") != "identification.id_expression" for d in diagnostics)
            else "unidentified_effect",
        )
    candidates.sort(key=lambda c: (sum(e.status in ("proposed", "contested") for e in c.graph.edges), c.endpoint_id, c.id))
    lad = ladder_of(state).model_copy(
        update={
            "rivals": Decision(
                reason="all surviving admitted graph/road configurations recorded before effects", values={"rivals.candidates": [c.id for c in candidates]}
            )
        }
    )
    return Command(
        goto="check_design",
        update={"candidates": candidates, "diagnostic_candidates": diagnostics, "ladder": lad, "candidate_history": _transitions(state, candidates, "rivals")},
    )


def check_design(state: SpecialistState) -> Command[Literal["pick_estimator", "feasibility"]]:
    h = state["handoff"]
    table = L.table(state)
    cfg = load_checks()
    candidates = []
    all_checks = []
    facts = {}
    for candidate in state["candidates"]:
        results = []
        e = candidate.estimand
        for contrast in state["contrasts"]:
            sub = S.contrast_table(table, candidate.graph.treatment, contrast)

            def check(name, valid, detail, value=None, soft=False, *, results=results, candidate=candidate, contrast=contrast):
                results.append(
                    CheckResult(
                        design_id=candidate.id,
                        contrast=contrast.key,
                        name=name,
                        level="pass" if valid else "soft" if soft else "hard",
                        detail=detail,
                        value=value,
                    )
                )

            if e.kind in ("backdoor", "general_adjustment"):
                res, cf = CK.run_checks(sub, e.adjustment_set, contrast.key, cfg)
                # Baseline imbalance is a threat adjustment addresses, not proof a valid set is wrong.
                res = [
                    r.model_copy(update={"design_id": candidate.id, "level": "soft" if r.name.startswith("balance.") and r.level == "hard" else r.level})
                    for r in res
                ]
                results += res
                facts[candidate.id + "." + contrast.key] = cf
                check(
                    "dimensionality",
                    len(e.adjustment_set) <= min((sub[engine.TREATED] == 1).sum(), (sub[engine.TREATED] == 0).sum()) / 10,
                    "at least ten rows in the smaller arm per adjustment column",
                )
            else:
                floor = min((sub[engine.TREATED] == 1).sum(), (sub[engine.TREATED] == 0).sum())
                check("arms", floor >= 20, "at least twenty rows per comparison arm", float(floor))
            if e.kind == "iv":
                z = sub[e.instruments]
                numeric = all(pd.api.types.is_numeric_dtype(z[c]) for c in z)
                check("instrument_encoding", numeric, "instruments are explicit numeric columns")
                if candidate.endpoint_id == "iv.wald_late" or (len(e.instruments) == 1 and z.iloc[:, 0].nunique() == 2):
                    check("instrument_levels", set(z.iloc[:, 0].dropna().unique()) == {0, 1}, "binary instrument must be explicitly encoded 0/1")
                if numeric:
                    from statsmodels.api import OLS, add_constant

                    controls = pd.get_dummies(sub[e.adjustment_set], drop_first=True, dtype=float) if e.adjustment_set else pd.DataFrame(index=sub.index)
                    X = add_constant(pd.concat([controls, z], axis=1).astype(float), has_constant="add")
                    check("instrument_rank", np.linalg.matrix_rank(X) == X.shape[1], "instrument matrix including intercept and controls has full rank")
                    if np.linalg.matrix_rank(X) == X.shape[1]:
                        fit = OLS(sub[engine.TREATED].astype(float), X).fit()
                        restriction = np.zeros((len(z.columns), X.shape[1]))
                        for j, c in enumerate(z.columns):
                            restriction[j, X.columns.get_loc(c)] = 1
                        strength = float(fit.f_test(restriction).fvalue)
                        check("relevance", strength > 1e-10, "nonzero conditional first stage", strength)
                        check(
                            "weak_instrument",
                            strength >= 10,
                            "first-stage F below ten makes current percentile inference unsupported; heuristic, not IV validity evidence",
                            strength,
                            soft=False,
                        )
            if e.kind in ("frontdoor", "mediation"):
                m = e.frontdoor_set[0] if e.kind == "frontdoor" else e.mediator
                check("mediator_numeric", pd.api.types.is_numeric_dtype(sub[m]), "singleton mediator must be numeric")
                check("mediator_variation", sub[m].nunique() > 1, "mediator varies in the scoped table")
                if e.kind == "frontdoor":
                    check("linear_assumption", _fact_true(h, "claim:mediator.linear_additive"), "explicit additive linear-stage assumption admitted")
            if candidate.endpoint_id == "adjustment.linear_conditional":
                requested_modifiers = h.scope.modifiers
                check(
                    "baseline_modifiers",
                    bool(requested_modifiers) and all(h.column(m) is not None and h.column(m).when == "before" for m in requested_modifiers),
                    "provided named baseline modifiers: " + repr(requested_modifiers) + "; all must be measured before treatment",
                )
                try:
                    _groups(table, requested_modifiers)
                except ValueError as ex:
                    check("modifier_addresses", False, str(ex))
            if _fact_true(h, "claim:spillover.possible"):
                check("interference", False, "reported spillover is incompatible with these no-interference endpoints")
            if h.sampling.get("how") == "by_outcome":
                check("selection", False, "selection on outcome prevents the requested population effect with these endpoints")
        checks = Checks(results=results)
        allowed = [
            x.name
            for x in load_estimators()
            if candidate.endpoint_id in x.endpoints
            and state["outcome_kind"] in x.outcomes
            and candidate.target["type"] in x.targets
            and (x.name != "linear_interaction" or e.kind == "backdoor")
            and (x.name != "general_linear_interaction" or e.kind == "general_adjustment")
        ]
        if "propensity_score_stratification" in allowed:
            params = next(entry.params for entry in load_estimators() if entry.name == "propensity_score_stratification")
            for contrast in state["contrasts"]:
                sub = S.contrast_table(table, candidate.graph.treatment, contrast)
                try:
                    support = engine.stratification_support(sub, candidate.graph, e, params)
                    valid = support["retained_rows"] == support["input_rows"]
                    detail = (
                        f"propensity stratification would retain {support['retained_rows']} of {support['input_rows']} rows; outcome values were withheld; "
                        + (
                            "the method preserves the input population"
                            if valid
                            else "this unused method is EXCLUDED; no rows were removed from other candidate estimators"
                        )
                    )
                except Exception as ex:
                    valid = False
                    detail = f"propensity stratification support unavailable without changing population: {ex}"
                results.append(
                    CheckResult(design_id=candidate.id, contrast=contrast.key, name="stratification_support", level="pass" if valid else "soft", detail=detail)
                )
                if not valid:
                    allowed.remove("propensity_score_stratification")
                    break
            checks = Checks(results=results)
        dropped = bool(checks.hard) or not allowed
        candidates.append(
            candidate.model_copy(
                update={
                    "checks": checks,
                    "allowed_estimators": allowed,
                    "status": "dropped" if dropped else "identified",
                    "reason": "; ".join(r.detail for r in checks.hard) if checks.hard else "no eligible estimator for exact target" if not allowed else "",
                }
            )
        )
        all_checks += results
    updates = {"candidates": candidates, "checks": all_checks, "check_facts": facts, "candidate_history": _transitions(state, candidates, "check_design")}
    if not any(c.status == "identified" for c in candidates):
        return _stop(
            "check_design",
            "every identified design failed a support, assumption or estimator check",
            [c.reason for c in candidates],
            updates,
            kind="inadequate_support" if any(c.checks.hard for c in candidates) else "unsupported_method",
        )
    return Command(goto="pick_estimator", update=updates)


def pick_estimator(state: SpecialistState) -> Command[Literal["freeze_design", "feasibility"]]:
    candidates = [c for c in state["candidates"] if c.status == "identified"]
    automatic = {c.id: c.allowed_estimators[0] for c in candidates if len(c.allowed_estimators) == 1}
    candidates = [c for c in candidates if len(c.allowed_estimators) > 1]
    if not candidates:
        picked = [c.model_copy(update={"chosen_estimator": automatic.get(c.id)}) for c in state["candidates"]]
        return Command(goto="freeze_design", update={"candidates": picked, "candidate_history": _transitions(state, picked, "pick_estimator")})
    catalogue = {x.name: x for x in load_estimators()}
    parts = ["ADMITTED TREATMENT CONTRASTS\n" + "\n".join(str(contrast.model_dump()) for contrast in state["contrasts"])]
    for c in candidates:
        parts.append(
            f"DESIGN {c.id}: endpoint {c.endpoint_id}; target {c.target}\n"
            + f"[{'design:' + c.id + '.identification'}] {c.estimand.model_dump()}\n"
            + "\n".join(f"[{r.address}] {r.level}: {r.detail}" for r in c.checks.results)
            + "\n"
            + "\n".join(catalogue[n].render() + f"; frozen parameters: {catalogue[n].params}" for n in c.allowed_estimators)
            + f"\nALLOWED {c.id}: {c.allowed_estimators}"
        )
    view = view_for("pick_estimator", state["handoff"], state["ledger"], material="\n\n".join(parts))
    from causal_agent.families.adjustment.lane.contracts import EstimatorChoice

    AllowedName = Literal[tuple(sorted({n for c in candidates for n in c.allowed_estimators}))]
    AllowedDesign = Literal[tuple(c.id for c in candidates)]
    Choice = create_model("AllowedEstimatorChoice", __base__=EstimatorChoice, name=(AllowedName, ...), design_id=(AllowedDesign, ...))
    Picks = create_model("AllowedEstimatorPicks", items=(list[Choice], Field(min_length=len(candidates), max_length=len(candidates))))

    def gate(r, resolve):
        errors = []
        if set(p.design_id for p in r.items) != set(c.id for c in candidates) or len(r.items) != len(candidates):
            errors.append("exactly one pick per displayed candidate required")
        for p in r.items:
            c = next((x for x in candidates if x.id == p.design_id), None)
            if c is None or p.name not in c.allowed_estimators:
                errors.append(
                    f"{p.design_id}: name must be one of the plain offered names {c.allowed_estimators if c else []}; estimator: prefixes belong only in cites"
                )
            errors += [f"unresolved citation {a}" for a in p.cites if not resolve(a)]
        return errors

    tools = diagnostic_tools(state, candidates)

    def evidence_gate(record, log):
        return [f"{pick.design_id}: {error}" for pick in record.items for error in tools.evidence_errors(pick.cites, log)]

    rec, updates, errors = _episode(
        state, "pick_estimator", Picks, P.PICK, view, gate, tools=tools, evidence_gate=evidence_gate, before_commit=tools.completion_errors
    )
    if rec is None:
        return _stop("pick_estimator", "estimator selection exhausted", errors, updates)
    chosen = {**automatic, **{p.design_id: p.name for p in rec.items}}
    picked = [c.model_copy(update={"chosen_estimator": chosen.get(c.id)}) for c in state["candidates"]]
    return Command(
        goto="freeze_design",
        update={
            **updates,
            "candidates": picked,
            "candidate_history": _transitions(state, picked, "pick_estimator"),
            "ladder": ladder_of(state).model_copy(
                update={"picks": Decision(reason="eligible choices frozen before estimation", values={"picks.estimators": chosen})}
            ),
        },
    )


def _groups(table, columns):
    out = {}
    for c in columns:
        if table[c].nunique() > 4 and pd.api.types.is_numeric_dtype(table[c]):
            edges = np.unique(table[c].quantile([0, 0.25, 0.5, 0.75, 1]).to_numpy())
            out[c] = [
                {"label": f"{lo:g} to {hi:g}", "low": float(lo), "high": float(hi), "include_low": j == 0}
                for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True))
            ]
        else:
            if table[c].nunique() > 4:
                raise ValueError(f"modifier {c} has more than four categorical levels; no arbitrary subset of levels is supported")
            out[c] = [{"label": str(v), "values": [str(v)]} for v in sorted(table[c].dropna().unique(), key=str)[:4]]
        if len({_slug(group["label"]) for group in out[c]}) != len(out[c]):
            raise ValueError(f"modifier {c} has distinct raw labels that collide at artifact addresses; relabel them before a newly authorized run")
    return out


def _lead_rank(candidate, catalogue):
    checks = candidate.checks.results
    overlaps = [check.value for check in checks if check.name == "overlap" and check.value is not None]
    arms = [check.value for check in checks if check.name == "arms" and check.value is not None]
    return (
        len(candidate.checks.hard),
        sum(check.level == "soft" for check in checks if check.name != "stratification_support"),
        -min(overlaps, default=1.0),
        -min(arms, default=0.0),
        sum(edge.status != "confirmed" for edge in candidate.graph.edges) + len(candidate.assumptions),
        next(index for index, endpoint in enumerate(catalogue.endpoints) if endpoint.id == candidate.endpoint_id),
        candidate.id,
    )


def freeze_design(state: SpecialistState) -> dict:
    h = state["handoff"]
    entries = {e.name: e for e in load_estimators()}
    designs = {}
    candidates = []
    table = L.table(state)
    support = S.freeze(table, key(h.treatment), state["contrasts"], state["source_row_ids"], h.context_id)
    catalogue = load_endpoint_catalogue()
    ranked = sorted(
        (candidate for candidate in state["candidates"] if candidate.status == "identified"), key=lambda candidate: _lead_rank(candidate, catalogue)
    )
    for c in state["candidates"]:
        if c.status != "identified":
            candidates.append(c)
            continue
        entry = entries[c.chosen_estimator]
        alternatives = sorted([entries[n] for n in c.allowed_estimators if n != entry.name], key=lambda e: (e.rank, e.name))
        modifiers = h.design.modifiers if isinstance(h.design, AdjustmentDesign) else []
        if c.endpoint_id == "adjustment.linear_conditional":
            modifiers = modifiers or [b.key for b in h.columns if b.may_modify and b.when == "before"]
        else:
            modifiers = []
        groups = _groups(L.table(state), modifiers)
        d = Design(
            id=c.id,
            endpoint_id=c.endpoint_id,
            context_id=h.context_id,
            ledger_revision=state["ledger"].revision,
            target={
                **c.target,
                "analysis_population_ids": {k: v["analysis_population_id"] for k, v in support.items()},
                "primary_summaries": S.primary_summaries(state["contrasts"], groups),
                "reference_summaries": S.primary_summaries(state["contrasts"], {}) if groups else [],
            },
            contrasts=state["contrasts"],
            identification_certificate=c.estimand.model_dump(),
            support={
                "checks": c.checks.model_dump(),
                "population": c.target["population"],
                "by_contrast": support,
                "missingness_policy": "one common complete-case population across all admitted loaded columns",
                "complete_case_columns": list(table.columns),
            },
            parameters=entry.params,
            inference={
                "method": "graph_implication" if c.estimand.kind == "zero" else "percentile_bootstrap",
                "draws": 100,
                "confidence_level": 0.95,
                "seed": 7,
            },
            weight_policy={
                "kind": "estimator_specific",
                "frozen_parameters": entry.params,
                "input_rows": {k: v["source_row_ids"] for k, v in support.items()},
                "reestimate_in_bootstrap": True,
            },
            graph=c.graph,
            estimand=c.estimand.model_copy(update={"assumptions": c.assumptions}),
            checks=c.checks,
            estimator=entry,
            fallback=alternatives[0] if alternatives else None,
            falsifications=[r for r in load_falsifications() if c.endpoint_id in r.endpoints],
            sensitivities=[r for r in load_sensitivities() if c.endpoint_id in r.endpoints and state["outcome_kind"] == "continuous"],
            modifiers=modifiers,
            modifier_groups=groups,
            target_units=c.target["type"],
        )
        designs[c.id] = d
        candidates.append(c.model_copy(update={"status": "frozen"}))
    return {
        "designs": designs,
        "lead_design_id": ranked[0].id,
        "lead_selection": {
            "policy": "hard failures, soft support flags, minimum overlap, minimum arm size, assumption burden, endpoint catalogue order, stable candidate ID",
            "ranking": [{"design_id": candidate.id, "rank": list(_lead_rank(candidate, catalogue))} for candidate in ranked],
            "selected_before_effect_fitting": True,
        },
        "candidates": candidates,
        "candidate_history": _transitions(state, candidates, "freeze_design"),
    }
