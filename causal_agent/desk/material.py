"""Everything a finished run left behind, as lines with addresses and a table of numbers. Reads the run record, not
the lane state, so it works for any stored run. Address formats match the lanes' own interpret addresses."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from causal_agent.desk.contracts import RunRecord
from causal_agent.memory.records import Memory


@dataclass
class Material:
    lines: list[str] = field(default_factory=list)
    addresses: set[str] = field(default_factory=set)
    numbers: dict[str, float] = field(default_factory=dict)
    by_address: dict[str, str] = field(default_factory=dict)

    def add(self, address: str, text: str, value: float | None = None) -> None:
        self.lines.append(f"[{address}] {text}")
        self.addresses.add(address)
        self.by_address[address] = text
        if value is not None and isinstance(value, (int, float)) and value == value:
            self.numbers[address] = float(value)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _g(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4g}"


def _contrast_key(design: dict) -> str | None:
    c = design.get("contrast")
    if isinstance(c, dict):
        return _slug(c.get("treated")) + "_vs_" + _slug(c.get("control"))
    cs = design.get("contrasts")
    if isinstance(cs, list) and cs:
        return _slug(cs[0].get("treated")) + "_vs_" + _slug(cs[0].get("control"))
    return None


def _slug(s) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(s or "")).strip("_").lower() or "x"


def render(run: RunRecord, memory: Memory | None = None, previous: RunRecord | None = None) -> Material:
    m = Material()
    dec = run.decision or {}
    m.add("run.question", run.question)
    m.add("decision.family", f"{run.family or 'none'} via {run.specialist or 'none'}; status {run.status}")
    if dec.get("chosen_assumption"):
        m.add("decision.assumption", dec["chosen_assumption"])
        m.add("design.assumption", dec["chosen_assumption"])
    if previous is not None:
        m.add(f"run:{previous.index}.question", previous.question)
        m.add(f"run:{previous.index}.family", f"{previous.family or 'none'}; status {previous.status}")
        if previous.effect is not None:
            m.add(f"run:{previous.index}.effect", f"{_g(previous.effect)}, interval {_g(previous.ci_low)} to {_g(previous.ci_high)} by {previous.estimator}", previous.effect)
            m.numbers[f"run:{previous.index}.ci_low"], m.numbers[f"run:{previous.index}.ci_high"] = float(previous.ci_low), float(previous.ci_high)
            m.addresses.update({f"run:{previous.index}.ci_low", f"run:{previous.index}.ci_high"})
    if dec.get("why"):
        m.add("decision.why", dec["why"])
    for fam, why in (dec.get("over") or {}).items():
        m.add(f"decision.over:{fam}", why)
    sr = run.specialist_result or {}
    design = sr.get("design") or {}
    c = _contrast_key(design) or "all"
    if design:
        for k in ("estimator", "estimand", "formula", "inference", "vce", "cluster", "target_units", "sharp_bandwidth_used"):
            if design.get(k) is not None:
                m.add(f"design.{k}", str(design[k]))
        if isinstance(design.get("estimand"), dict):
            m.add("design.estimand.adjustment_set", str(design["estimand"].get("adjustment_set")))
        for k in ("score", "groups", "periods", "bandwidths", "spec", "shape", "controls", "covariates"):
            if design.get(k) is not None:
                m.add(f"design.{k}", str(design[k])[:600])
        bw = design.get("bandwidths") or {}
        if isinstance(bw, dict) and bw.get("h") is not None:
            m.add("design.bandwidth", f"h = {bw['h']:.4g}, b = {_g(bw.get('b'))}, effective rows {bw.get('n_h_left')} / {bw.get('n_h_right')}", bw["h"])
        g = design.get("graph")
        if isinstance(g, dict):
            m.add("design.graph", "; ".join(f"{e.get('src')} -> {e.get('dst')}" for e in g.get("edges") or []))
        for r in (design.get("checks") or {}).get("results") or []:
            addr = f"check:{r.get('contrast')}.{r.get('name')}"
            m.add(addr, f"{r.get('level')}: {r.get('detail')} (value {_g(r.get('value'))}, threshold {_g(r.get('threshold'))})", r.get("value"))
            if r.get("threshold") is not None:
                m.numbers[addr + ".threshold"] = float(r["threshold"])
                m.addresses.add(addr + ".threshold")
    a_est = sr.get("estimates") or run.artifacts.get("estimates") or []
    for e in a_est:
        if e.get("error"):
            m.add(f"estimate:{e['contrast']}.{e['method']}.error", str(e["error"]))
            continue
        tag = f"estimate:{e['contrast']}" + (f".{e['method']}" if e.get("secondary") else "")
        m.add(f"{tag}.value", f"{e['method']}: {_g(e.get('value'))} ({e.get('target_units')})", e.get("value"))
        m.add(f"{tag}.ci", f"interval {_g(e.get('ci_low'))} to {_g(e.get('ci_high'))}", None)
        if e.get("ci_low") is not None:
            m.numbers[f"{tag}.ci_low"], m.numbers[f"{tag}.ci_high"] = float(e["ci_low"]), float(e["ci_high"])
            m.addresses.update({f"{tag}.ci_low", f"{tag}.ci_high"})
        m.add(f"{tag}.n", f"{e.get('n_treated')} treated-side and {e.get('n_control')} control-side rows", None)
        m.numbers[f"{tag}.n_treated"], m.numbers[f"{tag}.n_control"] = float(e.get("n_treated") or 0), float(e.get("n_control") or 0)
        m.addresses.update({f"{tag}.n_treated", f"{tag}.n_control"})
    prim = run.artifacts.get("primary") or {}
    for k in ("p", "h", "b", "n_h_left", "n_h_right"):
        if prim.get(k) is not None:
            m.add(f"primary.{k}", _g(prim[k]) if isinstance(prim[k], float) else str(prim[k]), float(prim[k]))
    for r in sr.get("refutations") or run.artifacts.get("refutations") or []:
        prefix = "refute" if run.specialist == "dowhy" else "placebo"
        tag = f"{prefix}:{r['contrast']}.{r['refuter']}"
        verdict = "pass" if r.get("passed") else "FAIL" if r.get("passed") is False else "no verdict"
        m.add(f"{tag}.passed", f"{verdict}: {r.get('detail')}", None)
        m.add(f"{tag}.detail", str(r.get("detail")), None)
        if r.get("p_value") is not None:
            m.add(f"{tag}.p_value", _g(r["p_value"]), r["p_value"])
        if r.get("new_effect") is not None:
            m.add(f"{tag}.new_effect", _g(r["new_effect"]), r["new_effect"])
    for i in sr.get("interpretations") or []:
        m.add(f"interpretation:{i.get('contrast')}.answer", str(i.get("answer")))
        for j, cv in enumerate(i.get("caveats") or [], start=1):
            m.add(f"interpretation:{i.get('contrast')}.caveat:{j}", str(cv))
    f = sr.get("feasibility")
    if f:
        m.add("feasibility.stage", str(f.get("stage")))
        m.add("feasibility.reason", str(f.get("reason")))
        for j, fact in enumerate(f.get("facts") or [], start=1):
            m.add(f"feasibility.fact:{j}", str(fact))
        if f.get("what_would_fix"):
            m.add("feasibility.what_would_fix", str(f["what_would_fix"]))
    for raw in run.figures or []:
        try:
            from causal_agent.viz.spec import FigureSpec

            spec = FigureSpec.model_validate(raw)
        except Exception:
            continue
        m.add(spec.address, f"{spec.kind}: {spec.title}. {spec.note}")
        for s_ in spec.series:
            for i, (x, y) in enumerate(zip(s_.x, s_.y)):
                if y is not None:
                    m.add(f"{spec.address}.{s_.key}.{i}", f"{s_.name} · {x}: {y:.4g}", float(y))
    if memory is not None:  # every field the memory holds, so the chat can cite what the design rested on
        for address, f in memory.fields.items():
            if f.value is None and f.status == "empty":
                continue
            m.add(address, f"{f.value} ({f.status}, {f.source})" + (f' said "{f.said}"' if f.said else ""),
                  float(f.value) if isinstance(f.value, (int, float)) and not isinstance(f.value, bool) else None)
        for a in list(m.addresses):
            m.addresses.add(a.rsplit(".", 1)[0])
    return m


def brief(run: RunRecord, previous: RunRecord | None, material: Material) -> str:
    """The opening message after a run: values with addresses, no model."""
    sr = run.specialist_result or {}
    lines = [f"Run {run.index}: {run.question}"]
    if run.status == "pipeline_error":
        lines.append("The analysis process failed before producing a record. [decision.family]")
        lines.append(run.decision_record[-1500:])
        lines.append("Ask again, or change something about the data; the pack was written.")
        return "\n".join(lines)
    if run.family:
        lines.append(f"Design: {run.family} via {run.specialist}." + (f" Why: {run.decision.get('why')}" if run.decision.get("why") else "") + " [decision.family]")
    else:
        lines.append("No design fit what is known and the data. [decision.family]")
    if run.status == "done" and run.effect is not None:
        lines.append(f"Effect: {_g(run.effect)}, interval {_g(run.ci_low)} to {_g(run.ci_high)}, by {run.estimator}. [estimate:{_contrast_key(sr.get('design') or {}) or 'all'}.value]")
    elif run.status == "no_handoff" or not run.family:
        lines.append("No family was admissible for this question on what is known. Each was rejected for a reason:")
        for fam, why in (run.decision.get("over") or {}).items():
            lines.append(f"  {fam}: {why} [decision.over:{fam}]")
        lines.append("Tell me if one of those reasons rests on something about the data that is wrong.")
    elif run.status != "done":
        f = sr.get("feasibility") or {}
        lines.append(f"Stopped at {f.get('stage')}: {f.get('reason')} [feasibility.reason]")
        for j, fact in enumerate(f.get("facts") or [], start=1):
            lines.append(f"  fact: {fact} [feasibility.fact:{j}]")
    flags = [r for r in ((sr.get("design") or {}).get("checks") or {}).get("results") or [] if r.get("level") != "pass"]
    if flags:
        lines.append("Flags carried: " + "; ".join(f"{r['name']} ({r['level']}) [check:{r['contrast']}.{r['name']}]" for r in flags))
    failed = [r for r in sr.get("refutations") or [] if r.get("passed") is False]
    if failed:
        prefix = "refute" if run.specialist == "dowhy" else "placebo"
        lines.append("Falsifications that failed: " + "; ".join(f"{r['refuter']} [{prefix}:{r['contrast']}.{r['refuter']}.passed]" for r in failed))
    elif sr.get("refutations"):
        lines.append(f"All {len(sr['refutations'])} falsifications passed.")
    for i in sr.get("interpretations") or []:
        lines.append(f"Reading: {i.get('answer')} [interpretation:{i.get('contrast')}.answer]")
        for j, cv in enumerate(i.get("caveats") or [], start=1):
            lines.append(f"  caveat: {cv} [interpretation:{i.get('contrast')}.caveat:{j}]")
    if previous is not None:
        lines.append(f"Then and now: run {previous.index} gave {_g(previous.effect)} ({previous.family or 'no design'}); run {run.index} gives {_g(run.effect)} ({run.family or 'no design'}).")
    lines.append("Ask anything about it, tell me something to change, ask a new question of the same data, or say done.")
    return "\n".join(lines)
