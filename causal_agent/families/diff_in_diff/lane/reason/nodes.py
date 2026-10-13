"""The DiD reasoning trajectory. Context is admitted by the desk, designs are sealed here."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from langgraph.types import Command

from causal_agent.common import trace
from causal_agent.common.addresses import key
from causal_agent.common.contracts import CheckResult, Checks, ContextDeficiency, Contrast, Feasibility, Handoff
from causal_agent.families.diff_in_diff.design import DidDesign
from causal_agent.families.diff_in_diff.lane import shape as SH
from causal_agent.families.diff_in_diff.lane.contracts import (
    ComparisonReading,
    ControlReading,
    Design,
    EstimatorPicks,
    Groups,
    MechanismReading,
    Periods,
    ProbeReading,
    Reading,
)
from causal_agent.families.diff_in_diff.lane.engine.diagnostics import holm, pre_trends, regular_support
from causal_agent.families.diff_in_diff.lane.history import event, revisions
from causal_agent.families.diff_in_diff.lane.knowledge import ENDPOINTS, SPEC, estimator, load_checks, load_falsifications, load_sensitivities
from causal_agent.families.diff_in_diff.lane.reason import ledger as RL
from causal_agent.families.diff_in_diff.lane.reason.rivals import eligible_control, enumerate_candidates, support
from causal_agent.families.diff_in_diff.lane.reason.selection import select_lead
from causal_agent.families.diff_in_diff.lane.reason.views import build
from causal_agent.families.diff_in_diff.lane.state import SpecialistState
from causal_agent.families.diff_in_diff.lane.targets import target_groups, weights_of
from causal_agent.lane import intake
from causal_agent.lane import nodes as L
from causal_agent.lane.endpoints import validate_context
from causal_agent.lane.episode import run_episode
from causal_agent.lane.ledger import Entry, Evidence, apply
from causal_agent.lane.ledger import seed as seed_ledger
from causal_agent.lane.nodes import stop

STAGE_RULES = {
    "mechanism": "Read who chose the treated group and why. Propose selection, anticipation, exchangeability entries. Unknown is explicit. Never infer random assignment from a data association.",
    "comparison": "Record parallel_trends, spillover, other_shock and composition assumptions. A parallel_trends value may be {requires_controls: [column keys]}. Report the identifying account, never certify parallel trends from a non-significant test.",
    "controls": "Classify only listed candidate columns as control or modifier. control value true means eligible predating treatment or confirmed unaffected covariate. Supported status needs domain evidence of its role, not correlation. Proposed uncertainty generates rival inclusion only when code finds timing safe.",
    "probe": "Resolve only consequential uncertainty in existing proposed entries. Keep unknown relationships proposed. Never invent human study facts, dates, group identity or anticipation windows. Do not confirm or erase the person's account.",
}


def _h(state: SpecialistState) -> Handoff:
    h = state["handoff"]
    assert isinstance(h, Handoff)
    return h


def _block(state: SpecialistState) -> DidDesign:
    b = _h(state).design
    assert isinstance(b, DidDesign)
    return b


def _deficiency(state: SpecialistState, stage: str, addresses: list[str], reason: str, extra: dict | None = None) -> Command:
    d = ContextDeficiency(stage=stage, addresses=addresses, reason=reason)
    return stop(
        stage,
        reason,
        addresses,
        "the desk must complete and admit a new context",
        {"context_deficiency": d.model_dump(), "reasoning_history": [event(state, stage, deficiency=d.model_dump())], **(extra or {})},
        terminal_kind="context_deficiency",
    )


def load(state: SpecialistState) -> Command:
    h = _h(state)
    missing = validate_context(h, ENDPOINTS)
    if missing:
        return _deficiency(state, "load", missing.addresses, missing.reason)
    if not isinstance(h.design, DidDesign):
        return _deficiency(state, "load", ["design"], "the admitted family block is missing")
    b = h.design
    group = b.treated_group
    if not group.get("column") or group.get("level") is None:
        return _deficiency(
            state, "load", ["claim:assignment.treatment_column", "claim:assignment.treated_level"], "treated group identity must be supplied by the desk"
        )
    if not b.cluster_level:
        return _deficiency(state, "load", ["claim:assignment.level_column"], "assignment clustering must be explicit")
    if b.time:
        if not b.unit or (not b.change_period and b.adoption_representation == "label"):
            return _deficiency(state, "load", ["claim:grain.key_columns", "claim:change.period_value"], "unit identity and treatment timing are required")
        periods = Periods(
            kind="long", time_column=b.time, first_post=b.change_period, reason="admitted temporal mapping", cites=["claim:change.date_column"], by="pack"
        )
    else:
        if not b.before_column or not b.after_column:
            return _deficiency(state, "load", ["col:<outcome>.measures_outcome", "col:<outcome>.when"], "paired dated outcome columns must be explicit")
        periods = Periods(
            kind="wide",
            before_column=b.before_column,
            after_column=b.after_column,
            reason="admitted paired measurements",
            cites=[f"col:{b.before_column}.when"],
            by="pack",
        )
    loaded = None
    try:
        it = intake.load(h, "did", extra=b.columns(), dropna=False)
        loaded = it
        table = it.table
        if b.cluster_level not in table:
            raise SH.ShapeError("assignment cluster column is missing", [b.cluster_level], "the declared cluster column")
        groups = Groups(
            column=key(group["column"]),
            treated_level=str(group["level"]),
            reason="admitted treatment contrast",
            cites=["claim:assignment.treated_level"],
            by="pack",
        )
        if periods.kind == "long" and not periods.first_post:
            if b.adoption_representation == "cohort_column":
                values = pd.to_numeric(table[group["cohort_column"]], errors="coerce")
                first = values[values > 0].min()
            else:
                first = table.loc[table[groups.column].astype(str) == groups.treated_level, b.time].min()
            periods = periods.model_copy(update={"first_post": str(first)})
        candidates = [c.key for c in h.columns if c.role == "candidate"]
        panel, facts = SH.canonical(
            table, groups, periods, key(h.outcome), candidates, unit_column=b.unit, cluster_column=b.cluster_level, cohort_column=group.get("cohort_column")
        )
        if facts.units_control == 0:
            if facts.cohorts < 2:
                return stop(
                    "load",
                    "all units adopt together without an untreated comparison",
                    [],
                    "an observed supported comparison group",
                    {"run_dir": str(it.run_dir), "shape": facts},
                    terminal_kind="inadequate_support",
                )
            prefix_admitted = any(e.endpoint_id == "untreated_prefix_imputation_att" and e.admitted for e in h.admission.endpoints)
            if not b.prefix_scope_accepted or not prefix_admitted:
                return _deficiency(
                    state,
                    "load",
                    ["claim:target.prefix_scope_accepted"],
                    "without never-treated units, the desk must explicitly admit the supported before-last-adoption target scope",
                    {"run_dir": str(it.run_dir), "shape": facts},
                )
        panel["row_id"] = np.arange(len(panel))
        path = it.run_dir / "panel.csv"
        trace.write(path, panel.to_csv(index=False))
    except intake.IntakeStop as ex:
        return Command(
            goto="feasibility", update={"feasibility": ex.feasibility, "reasoning_history": [event(state, "load", terminal=ex.feasibility.model_dump())]}
        )
    except (SH.ShapeError, KeyError, ValueError) as ex:
        return stop(
            "load",
            str(ex),
            getattr(ex, "facts", []),
            getattr(ex, "fix", "valid declared panel mapping"),
            {"run_dir": str(loaded.run_dir), "columns": loaded.columns, "table_path": str(loaded.table_path)} if loaded else {},
            terminal_kind="inadequate_support",
        )
    return Command(
        goto="seed",
        update={
            "run_dir": str(it.run_dir),
            "table_path": str(it.table_path),
            "panel_path": str(path),
            "columns": it.columns,
            "groups": groups,
            "periods": periods,
            "shape": facts,
            "ledger": seed_ledger(h),
            "declines": it.declines,
            "reasoning_history": [event(state, "load", shape=facts.model_dump(), groups=groups.model_dump(), periods=periods.model_dump())],
        },
    )


def seed(state: SpecialistState) -> Command:
    h = _h(state)
    b = _block(state)
    ledger = state["ledger"]
    entries = []
    for kind, address in [("parallel_trends", "claim:trend_continues.believed"), ("spillover", "claim:spillover.possible")]:
        f = h.fact(address)
        if f:
            entries.append(
                Entry(
                    id=f"rel:{kind}.account",
                    kind=kind,
                    src="account",
                    value=f.value,
                    status="confirmed" if f.status == "confirmed" else "proposed",
                    evidence=[Evidence(source="pack", address=address, category="study_account", note=f.said or f.reason or "explicit unknown")],
                    reason=f"the desk carried status {f.status}",
                )
            )
    ledger, errors = apply(ledger, entries, by="seed", resolves=h.resolve)
    if errors:
        return _deficiency(state, "seed", [], "; ".join(errors))
    return Command(goto="mechanism", update={"ledger": ledger, "reasoning_history": [event(state, "seed", revision=ledger.revision, block=b.model_dump())]})


def _judge(state: SpecialistState, stage: str, next_stage: str) -> Command:
    view = build(state, stage)
    panel = pd.read_csv(state["panel_path"])
    if _block(state).prefix_scope_accepted and not (panel.cohort == 0).any():
        from causal_agent.families.diff_in_diff.lane.contracts import Candidate

        panel = support(panel, Candidate(id="view", endpoint_id="untreated_prefix_imputation_att", estimator_options=[], support="pre_last_adoption_prefix"))
    # Even describe(y) sees no post-treatment values. Stronger than only refusing joint requests.
    pre = panel.loc[panel.post == 0].copy()
    h = _h(state)
    aliases = {key(h.outcome): "y", key(h.treatment or ""): "treated"}
    for source, column in [(_block(state).time, "time"), (_block(state).unit, "unit"), (_block(state).cluster_level, "cluster")]:
        if source:
            aliases.setdefault(key(source), column)
    tools = L.data_tools(
        state,
        pre.treated == 1,
        table_=pre,
        aliases=aliases,
        allow_outcome_rows=pd.Series(True, index=pre.index),
        allow_outcome_words="globally pre-treatment periods",
    )
    allowed_kinds = {
        "mechanism": {"selection", "anticipation", "exchangeability"},
        "comparison": {"parallel_trends", "spillover", "other_shock", "composition"},
        "controls": {"control", "modifier"},
        "probe": {"selection", "parallel_trends", "anticipation", "spillover", "other_shock", "composition", "control", "modifier", "exchangeability"},
    }[stage]

    def gate(record: Reading, log):
        errors = []

        def resolves(a):
            return view.resolves(a) or log.resolve(a)

        for op in record.operations:
            if op.kind == "selection" and (not isinstance(op.value, str) or op.value not in {"levels", "trends", "neither", "unknown"}):
                errors.append("selection value must be levels, trends, neither, or unknown")
            if op.kind == "anticipation" and op.value is not None and (not isinstance(op.value, int) or isinstance(op.value, bool) or op.value < 0):
                errors.append("anticipation must be a stated nonnegative integer period count or null")
            if (
                op.kind in {"control", "modifier", "exchangeability", "spillover", "other_shock", "composition"}
                and op.value is not None
                and not isinstance(op.value, bool)
            ):
                errors.append(f"{op.kind} value must be boolean or null")
            if op.kind not in allowed_kinds:
                errors.append(f"{stage} cannot write {op.kind}")
            if op.kind in {"control", "modifier"} and (h.column(op.subject) is None or h.column(op.subject).role != "candidate"):
                errors.append(f"unknown column {op.subject}")
            if op.kind == "control" and op.value is True and op.status == "supported" and not eligible_control(h, panel, op.subject):
                errors.append(f"{op.subject} lacks safe confirmed timing or varies only within fixed effects")
        _, more = RL.commit(state["ledger"], record, stage, resolves, handoff=h, log=log)
        return errors + more

    record, log, thoughts, errors = run_episode(
        {"mechanism": MechanismReading, "comparison": ComparisonReading, "controls": ControlReading, "probe": ProbeReading}[stage],
        STAGE_RULES[stage]
        + " You have no post-treatment outcomes here. Do not estimate or describe programme effects; pre-period differences are only design evidence."
        + " Write only NEW operations of kinds "
        + ", ".join(sorted(allowed_kinds))
        + ". Prior ledger entries are input evidence and must not be repeated as outputs."
        + " Data/probe/profile citations must keep source=data and category=association, diagnostic or measurement_timing; they cannot become study accounts or certify parallel trends."
        + " Evidence.address must exactly match a bracket address from ALLOWED EVIDENCE ADDRESSES, such as claim:assignment.kind, or dataset.note for source-document statements. File paths and plain column names are not evidence addresses. For a domain statement use source=pack, category=study_account. Proposed is permitted; certainty is not required.",
        view.text + "\n\nALLOWED EVIDENCE ADDRESSES\n" + "\n".join(sorted(view.addresses)),
        tools=tools,
        budget=SPEC.budget(stage),
        gate=gate,
        node=stage,
    )
    history = event(state, stage, view=view.text, episode=log.model_dump(), record=record.model_dump() if record else None, errors=errors)
    extra = {"reasoning_history": [history], "episodes": {f"{stage}:1": log}, "debug": thoughts}
    if record is None:
        # Failed schema reasoning must never silently become an omitted assumption or control.
        return stop(stage, "reasoning record failed its bounded gate", errors, "a valid source-backed reading", extra, terminal_kind="exhausted_reasoning")
    ledger, _ = RL.commit(state["ledger"], record, stage, lambda a: view.resolves(a) or log.resolve(a), handoff=h, log=log)
    return Command(goto=next_stage, update={"ledger": ledger, **extra})


def mechanism(state: SpecialistState) -> Command:
    return _judge(state, "mechanism", "comparison")


def comparison(state: SpecialistState) -> Command:
    return _judge(state, "comparison", "controls")


def controls(state: SpecialistState) -> Command:
    h = _h(state)
    panel = pd.read_csv(state["panel_path"])
    candidates = [c.key for c in h.columns if c.role == "candidate"]
    if not any(eligible_control(h, panel, column) for column in candidates):
        return Command(
            goto="probe",
            update={
                "reasoning_history": [
                    event(
                        state,
                        "controls",
                        reason="no admitted covariate has safe timing and within-panel variation after fixed effects",
                        candidate_columns=candidates,
                        operations=[],
                    )
                ]
            },
        )
    return _judge(state, "controls", "probe")


def probe(state: SpecialistState) -> Command:
    panel = pd.read_csv(state["panel_path"])
    consequential = [
        e
        for e in state["ledger"].entries.values()
        if e.status in {"proposed", "contested"}
        and (
            e.kind == "control"
            and e.value is True
            and eligible_control(_h(state), panel, e.src or "")
            or e.kind == "anticipation"
            and isinstance(e.value, int)
            and e.value > 0
            or e.kind == "parallel_trends"
            and isinstance(e.value, dict)
            and e.value.get("requires_controls")
        )
    ]
    if not consequential:
        return Command(goto="rivals", update={"reasoning_history": [event(state, "probe", reason="no unresolved load-bearing entries", entries=[])]})
    return _judge(state, "probe", "rivals")


def rivals(state: SpecialistState) -> Command:
    panel = pd.read_csv(state["panel_path"])
    try:
        candidates, omitted = enumerate_candidates(_h(state), _block(state), panel, state["ledger"], _h(state).context_id)
    except ValueError as ex:
        return _deficiency(state, "rivals", ["claim:target.cohorts"], str(ex))
    if not any(c.status == "proposed" for c in candidates):
        return stop(
            "rivals",
            "no admitted endpoint supports this target and comparison",
            [c.reason for c in candidates],
            "supported target context",
            {"candidates": candidates, "candidate_history": revisions(state, "rivals", candidates)},
            terminal_kind="unsupported_method",
        )
    return Command(
        goto="check_design",
        update={
            "candidates": candidates,
            "candidate_history": revisions(state, "rivals", candidates),
            "reasoning_history": [event(state, "rivals", unexpanded_controls=omitted)],
        },
    )


def check_design(state: SpecialistState) -> Command:
    panel = pd.read_csv(state["panel_path"])
    cfg = load_checks()
    proposed = [c for c in state["candidates"] if c.status == "proposed"]
    diagnostics = []
    for candidate in proposed:
        with trace.span(f"pre_trends:{candidate.id}", run_type="tool", inputs={"candidate": candidate, "outcome_scope": "globally pre-adoption only"}) as span:
            diagnostic = pre_trends(support(panel, candidate), candidate.controls, "cluster")
            diagnostics.append(diagnostic)
            span.end(outputs=diagnostic)
    adjusted = holm([x["p"] for x in diagnostics])
    checks = []
    candidates = []
    ledger = state["ledger"]
    for c, facts, pv in zip(proposed, diagnostics, adjusted, strict=True):
        facts["p"] = pv
        level = (
            "soft" if pv is None else "hard" if pv < cfg["pre_trends"]["p_value"]["hard"] else "soft" if pv < cfg["pre_trends"]["p_value"]["soft"] else "pass"
        )
        r = CheckResult(
            design_id=c.id,
            contrast="all",
            name="pre_trends",
            level=level,
            value=pv,
            detail=cfg["in_words"]["pre_trends"]
            + f"; globally pre-adoption evidence; raw p={facts['raw_p']}; Holm p={pv}; "
            + (facts["error"] or "passing does not prove parallel trends"),
        )
        checks.append(r)
        failed_belief = any(e.kind == "parallel_trends" and e.status == "confirmed" and e.value is False for e in ledger.entries.values())
        selected_trend = any(e.kind == "selection" and e.status in {"supported", "confirmed"} and e.value == "trends" for e in ledger.entries.values())
        anticipation = any(
            e.kind == "anticipation" and e.status in {"supported", "confirmed"} and isinstance(e.value, (int, float)) and e.value > 0
            for e in ledger.entries.values()
        )
        other_shock = any(e.kind == "other_shock" and e.status in {"supported", "confirmed"} and e.value is True for e in ledger.entries.values())
        spillover = any(e.kind == "spillover" and e.status in {"supported", "confirmed"} and e.value is True for e in ledger.entries.values())
        if level == "hard" or failed_belief or selected_trend or anticipation or other_shock or spillover:
            c = c.model_copy(
                update={
                    "status": "excluded",
                    "reason": "hard pretrend failure or unsupported assignment, anticipation, shock or spillover account",
                    "checks": [r],
                }
            )
        else:
            c = c.model_copy(update={"status": "ready", "checks": [r]})
        candidates.append(c)
        if level == "hard":
            entry = Entry(
                id=f"rel:diagnostic.{c.id}",
                kind="parallel_trends",
                src=c.id,
                value=False,
                status="contested",
                evidence=[Evidence(source="test", address=r.address, category="diagnostic", note=r.detail)],
                reason="the candidate's pretrend implication failed",
            )
            ledger, _ = apply(ledger, [entry], by="check_design", resolves=lambda a, address=r.address: a == address)
    candidates += [c for c in state["candidates"] if c.status != "proposed"]
    extra = {
        "candidates": candidates,
        "checks": checks,
        "ledger": ledger,
        "candidate_history": revisions(state, "check_design", candidates),
        "reasoning_history": [event(state, "check_design", diagnostics=diagnostics)],
    }
    if not any(c.status == "ready" for c in candidates):
        return stop(
            "check_design",
            "every sealed candidate failed its design checks",
            [r.detail for r in checks],
            "a credible supported comparison",
            extra,
            terminal_kind="unidentified_effect",
        )
    return Command(goto="pick_estimator", update=extra)


def pick_estimator(state: SpecialistState) -> Command:
    ready = {c.id: c for c in state["candidates"] if c.status == "ready"}
    view = build(state, "pick_estimator")
    text = view.text + "\n" + "\n".join(estimator(n).render() for c in ready.values() for n in c.estimator_options)

    def gate(record: EstimatorPicks, log):
        errors = []
        if {p.candidate_id for p in record.choices} != set(ready) or len(record.choices) != len(ready):
            errors.append("choose exactly once for every ready candidate")
        for pick in record.choices:
            if pick.candidate_id not in ready or pick.estimator not in ready[pick.candidate_id].estimator_options:
                errors.append("estimator must be an offered entry for this candidate")
            if not pick.cites or any(not view.resolves(a) and a != f"estimator:{pick.estimator}" for a in pick.cites):
                errors.append(
                    f"cites must be bare address strings, e.g. candidate:{pick.candidate_id} or estimator:{pick.estimator}; omit brackets and all rendered prose"
                )
        return errors

    record, log, thoughts, errors = run_episode(
        EstimatorPicks,
        "Choose one offered estimator per candidate. Do not alter targets, controls or support. Explain using candidate evidence. Each cites entry is ONLY the exact address inside brackets, such as candidate:<id> or estimator:<name>. Never copy a bracketed line or its prose into cites.",
        text,
        tools=None,
        budget=0,
        gate=gate,
        node="pick_estimator",
    )
    extra = {"episodes": {"pick_estimator:1": log}, "debug": thoughts, "reasoning_history": [event(state, "pick_estimator", episode=log.model_dump())]}
    if record is None:
        return stop("pick_estimator", "no validated estimator configuration", errors, "an offered estimator", extra, terminal_kind="exhausted_reasoning")
    return Command(goto="freeze_design", update={"chosen_estimators": {p.candidate_id: p.estimator for p in record.choices}, **extra})


def _inference(engine: str, p: pd.DataFrame, wide: bool, dynamic: bool) -> tuple[str, str | dict]:
    cfg = load_checks()["inference"]
    clusters = int(p.cluster.nunique())
    if clusters < cfg["min_clusters"]:
        return "inference_unavailable", {"CRV1": "cluster"}
    if wide and p.cluster.nunique() == p.unit.nunique():
        return "row_robust", "hetero"
    if engine == "did2s":
        return "native_gmm", {"CRV1": "cluster"}
    if engine == "saturated":
        return "native_crv1", {"CRV1": "cluster"}
    if engine == "lpdid":
        return ("native_crv3", {"CRV3": "cluster"}) if clusters < cfg["moderate_clusters"] else ("native_crv1", {"CRV1": "cluster"})
    if dynamic:
        return ("cluster_crv3", {"CRV3": "cluster"}) if clusters < cfg["moderate_clusters"] else ("cluster_crv1", {"CRV1": "cluster"})
    if clusters < cfg["few_clusters"]:
        return "wild_webb", {"CRV1": "cluster"}
    if clusters < cfg["moderate_clusters"]:
        return "wild_rademacher", {"CRV3": "cluster"}
    return "cluster_crv1", {"CRV1": "cluster"}


def freeze_design(state: SpecialistState) -> Command:
    # Outcome values are never read here, only declared support, masks and metadata.
    from causal_agent.families.diff_in_diff.lane.targets import clean_projection_support, horizon_projection_weights, projection_weights

    h = _h(state)
    panel = pd.read_csv(state["panel_path"])
    designs: dict[str, Design] = {}
    candidates = []
    for c in state["candidates"]:
        if c.status != "ready":
            candidates.append(c)
            continue
        p = support(panel, c)
        entry = estimator(state["chosen_estimators"][c.id])
        target = dict(c.target)
        target["cohort_labels"] = {str(int(row.time_index)): SH.period_label(row.time) for row in p[["time_index", "time"]].drop_duplicates().itertuples()}
        available_horizons = sorted(int(x) for x in p.loc[p.treat == 1, "rel_time"].unique() if x >= 0)
        explicit_horizons = bool(target["horizons"])
        if target["aggregation"] == "pooled" and explicit_horizons and entry.engine != "lpdid" and sorted(target["horizons"]) != available_horizons:
            candidates.append(c.model_copy(update={"status": "excluded", "reason": "this pooled estimator cannot restrict horizons; request a horizon target"}))
            continue
        if not target["horizons"]:
            target["horizons"] = available_horizons
        if entry.engine == "lpdid":
            max_h = min(load_checks()["projection"]["max_horizon"], max(target["horizons"]))
            if not (p.cohort == 0).any():
                starts = sorted(p.loc[p.cohort > 0, "cohort"].unique())
                max_h = min(max_h, min(np.diff(starts)) - 1)
            if explicit_horizons and any(x > max_h for x in target["horizons"]):
                candidates.append(c.model_copy(update={"status": "excluded", "reason": "requested projection horizons exceed supported clean window"}))
                continue
            if target["aggregation"] == "pooled" and explicit_horizons and sorted(target["horizons"]) != list(range(max_h + 1)):
                candidates.append(
                    c.model_copy(update={"status": "excluded", "reason": "pooled projection requires a contiguous horizon window beginning at zero"})
                )
                continue
            target["horizons"] = [int(x) for x in target["horizons"] if x <= max_h]
            if not target["horizons"]:
                candidates.append(c.model_copy(update={"status": "excluded", "reason": "no clean projection horizon remains"}))
                continue
        if any(not mask.any() for _, mask in target_groups(p, target)):
            candidates.append(c.model_copy(update={"status": "excluded", "reason": "the requested target has no observations"}))
            continue
        path = Path(state["run_dir"]) / "designs" / c.id / "panel.csv"
        trace.write(path, p.to_csv(index=False))
        dynamic = entry.name == "twfe_dynamic"
        spec = "y ~ " + ("i(rel_time, treated, ref=-1)" if dynamic else "treat") + "".join(f" + {k}" for k in c.controls) + " | unit+time_index"
        first = "~ " + (" + ".join(c.controls) if c.controls else "0") + " | unit+time_index"
        second = (
            "~ i(rel_time, ref=-1)"
            if entry.name == "did2s_dynamic"
            else "~ " + " + ".join(f"cohort_treat_{int(g)}" for g in sorted(p.loc[p.treat == 1, "cohort"].unique()))
            if entry.name == "did2s_cohort"
            else "~ treat"
        )
        inference, vcov = _inference(entry.engine, p, state["periods"].kind == "wide", dynamic)
        try:
            weight_map = projection_weights(p, c.controls) if entry.name == "twfe_static" else weights_of(p, target)
            horizon_weights = {int(h): horizon_projection_weights(p, c.controls, int(h)) for h in target["horizons"]} if entry.name == "twfe_dynamic" else {}
        except ValueError as ex:
            candidates.append(c.model_copy(update={"status": "excluded", "reason": str(ex)}))
            continue
        included_units = sorted(str(x) for x in p.loc[p.treat == 1, "unit"].unique())
        if entry.engine == "lpdid":
            if not regular_support(p):
                candidates.append(c.model_copy(update={"status": "excluded", "reason": "missing outcomes broke regular projection support"}))
                continue
            try:
                included_units, weight_map = clean_projection_support(p, target["horizons"])
            except ValueError as ex:
                candidates.append(c.model_copy(update={"status": "excluded", "reason": str(ex)}))
                continue
            target["within_unit_horizon_weights"] = (
                {str(h): 1 / (max(target["horizons"]) + 1) for h in range(max(target["horizons"]) + 1)}
                if target["aggregation"] == "pooled"
                else {str(h): 1.0 for h in target["horizons"]}
            )
            if target["aggregation"] != "pooled":
                target["components"] = {}
                for horizon in target["horizons"]:
                    units, component_weights = clean_projection_support(p, [horizon])
                    target["components"][f"horizon:{horizon}"] = {"included_units": units, "weights": component_weights, "horizons": [horizon]}

        if entry.engine != "lpdid":
            target["components"] = {}
            for label, mask in target_groups(p, target):
                component = {"included_units": sorted(str(x) for x in p.loc[mask, "unit"].unique())}
                if label.startswith("horizon:"):
                    horizon = int(label.split(":")[1])
                    component["horizons"] = [horizon]
                    if entry.name == "twfe_dynamic":
                        component["weighting"] = "linear_response_horizon_projection"
                        component["weights"] = horizon_weights[horizon]
                    else:
                        component["weights"] = weights_of(p, {**target, "aggregation": "horizon", "horizons": [horizon]})
                elif label.startswith("cohort:"):
                    selected_cohort = label.split(":")[1]
                    component["cohorts"] = [selected_cohort]
                    component["weights"] = weights_of(
                        p, {**target, "aggregation": "cohort_horizon" if ":horizon:" in label else "cohort", "cohorts": [selected_cohort]}
                    )
                target["components"][label] = component
        target.update(
            weighting="clean_adopter_projection"
            if entry.engine == "lpdid"
            else "residualized_treatment_projection"
            if entry.name == "twfe_static"
            else "linear_response_horizon_projection"
            if entry.name == "twfe_dynamic"
            else "treated_cell_shares",
            weights=weight_map,
            included_units=included_units,
            support=c.support,
        )
        facts = dict(engine=entry.engine, cohorts=state["shape"].cohorts, periods_pre=state["shape"].periods_pre, units_treated=state["shape"].units_treated)
        static = entry.name == "twfe_static"
        d = Design(
            id=c.id,
            endpoint_id=c.endpoint_id,
            context_id=h.context_id,
            ledger_revision=state["ledger"].revision,
            identification_certificate={
                "endpoint_id": c.endpoint_id,
                "comparison": ENDPOINTS.endpoint(c.endpoint_id).configuration.get("comparison", "untreated"),
                "assumptions": [state["ledger"].entries[a].model_dump() for a in c.assumptions],
                "support": c.support,
                "row_ids": list(int(x) for x in p.row_id),
                "checks": [r.model_dump() for r in c.checks],
                "controls": c.controls,
                "interpretation": "identification is conditional on these assumptions; data diagnostics do not certify them",
            },
            target=target,
            groups=state["groups"],
            periods=state["periods"],
            shape=state["shape"],
            contrast=Contrast(treated=state["groups"].treated_level, control="comparison", reason="admitted comparison", cites=state["groups"].cites),
            controls=tuple(c.controls),
            checks=Checks(results=c.checks),
            estimator=entry.name,
            engine=entry.engine,
            formula=spec,
            first_stage=first,
            second_stage=second,
            inference=inference,
            cluster_column="cluster",
            vcov=vcov,
            panel_path=str(path),
            row_ids=tuple(int(x) for x in p.row_id),
            weights=weight_map,
            assumptions=tuple(c.assumptions),
            falsifications=tuple(e.name for e in load_falsifications() if static and e.applies(**facts)),
            sensitivities=tuple(e.name for e in load_sensitivities() if (static or e.name == "twfe_naive") and e.applies(**facts)),
        )
        designs[d.id] = d
        trace.write(path.parent / "design.json", d.model_dump_json(indent=2))
        candidates.append(c.model_copy(update={"status": "frozen"}))
    if not designs:
        return stop(
            "freeze_design",
            "no exact supported target could be frozen",
            [],
            "supported target support",
            {"candidates": candidates, "candidate_history": revisions(state, "freeze_design", candidates)},
            terminal_kind="inadequate_support",
        )
    lead_selection = select_lead(h, designs, state["ledger"], panel)
    trace.write(Path(state["run_dir"]) / "lead_selection.json", json.dumps(lead_selection, indent=2))
    feasibility = (
        {}
        if lead_selection["chosen_id"] is not None
        else {
            "feasibility": Feasibility(
                stage="freeze_design", reason=lead_selection["reason"], facts=lead_selection["cites"], terminal_kind="unidentified_effect"
            )
        }
    )
    return Command(
        goto="execute",
        update={
            "designs": designs,
            "lead_design_id": lead_selection["chosen_id"],
            "lead_selection": lead_selection,
            "candidates": candidates,
            "candidate_history": revisions(state, "freeze_design", candidates),
            "reasoning_history": [event(state, "freeze_design", designs=list(designs), lead_selection=lead_selection)],
            **feasibility,
        },
    )
