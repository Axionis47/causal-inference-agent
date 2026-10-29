"""Everything a finished run left behind, as lines with addresses and a table of numbers. Reads the run record, not
the lane state, so it works for any stored run. Address formats match the lanes' own interpret addresses."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from causal_agent.common.addresses import norm_address
from causal_agent.common.contracts import Contrast, Decline, DesignBrief, Estimate, Handoff, RunRecord
from causal_agent.desk.pull import Item, Shelf, shelf_of
from causal_agent.memory import artifacts as VA
from causal_agent.memory.journal import Step
from causal_agent.memory.matrix import Matrix
from causal_agent.memory.records import Memory


@dataclass
class Material:
    """Everything citable at one point of the conversation, as lines with addresses. Addresses are normalised on the way in,
    so a cite in another case or spelling of a column still resolves. `pull` and `shelf` are the two reads."""

    lines: list[str] = field(default_factory=list)
    addresses: set[str] = field(default_factory=set)
    numbers: dict[str, float] = field(default_factory=dict)
    by_address: dict[str, str] = field(default_factory=dict)

    def add(self, address: str, text: str, value: float | None = None) -> None:
        address = norm_address(address)
        self.lines.append(f"[{address}] {text}")
        self.addresses.add(address)
        self.by_address[address] = text
        if value is not None and isinstance(value, (int, float)) and value == value:
            self.numbers[address] = float(value)

    def close(self) -> None:
        """Every parent of an address is citable too: artifact:<id> for artifact:<id>.<fact>, estimate:<c> for its .value."""
        for a in list(self.addresses):
            self.addresses.add(a.rsplit(".", 1)[0])

    def has(self, address: str) -> bool:
        return norm_address(address) in self.addresses

    def pull(self, address: str) -> Item | None:
        """One item by address, or None when nothing on any shelf answers to it."""
        a = norm_address(address)
        if a not in self.addresses:
            return None
        return Item(address=a, shelf=shelf_of(a), text=self.by_address.get(a, ""), value=self.numbers.get(a))

    def shelf(self, kind: Shelf) -> list[Item]:
        """Everything on one shelf, in the order it was added."""
        out = []
        for line in self.lines:
            a = line[1:].partition("] ")[0]
            if shelf_of(a) == kind:
                out.append(Item(address=a, shelf=kind, text=self.by_address.get(a, ""), value=self.numbers.get(a)))
        return out

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


def _g(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.4g}"


def _brief(run: RunRecord) -> DesignBrief | None:
    """The design brief the run carried, or None."""
    raw = (run.decision or {}).get("brief")
    try:
        return DesignBrief.model_validate(raw) if raw else None
    except Exception:
        return None


def _declines(run: RunRecord) -> list[Decline]:
    """Where the lane did not take the pack as given, from the result or the artifacts on disk."""
    raw = (run.specialist_result or {}).get("declines") or run.artifacts.get("declines") or []
    out = []
    for d in raw:
        try:
            out.append(Decline.model_validate(d))
        except Exception:
            continue
    return out


def _scale(run: RunRecord, m: Material) -> None:
    """The outcome's and the treatment's scale from the run's pack, so a number can be judged against it: the bounds when the
    column has them, and min, p50, max and mean when it is numeric, each number under its own address."""
    if not run.design_dir or not (Path(run.design_dir) / "handoff.json").exists():
        return
    h = Handoff.model_validate_json((Path(run.design_dir) / "handoff.json").read_text())
    for name in (h.outcome, h.treatment):
        b = h.column(name) if name else None
        if b is None:
            continue
        f = b.facts
        if f.bounds:
            m.add(f"{b.address}.profile.bounds", f"{f.bounds[0]} to {f.bounds[1]}")
            for tag, v in zip(("low", "high"), f.bounds):
                try:
                    m.numbers[f"{b.address}.profile.bounds.{tag}"] = float(v)
                    m.addresses.add(f"{b.address}.profile.bounds.{tag}")
                except ValueError:
                    continue
        if f.numeric:
            n = f.numeric
            m.add(f"{b.address}.profile.numeric", ", ".join(f"{k} {n[k]:g}" for k in ("min", "p50", "max", "mean") if k in n))
            for k in ("min", "p50", "max", "mean"):
                if k in n:
                    m.numbers[f"{b.address}.profile.numeric.{k}"] = float(n[k])
                    m.addresses.add(f"{b.address}.profile.numeric.{k}")


def _values(m: Material, address: str, value) -> None:
    """Render structured evidence without dropping nested numbers or provenance."""
    weight_fields = {"weights", "weight_map", "projection_weights", "cohort_weights"}
    if address.split(".")[-1] in weight_fields and isinstance(value, (dict, list)) and len(value) > 256:
        weights = list(value.values()) if isinstance(value, dict) else value
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in weights):
            fingerprint = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            m.add(address, "Full frozen numerical weights are retained in result.json; these summaries describe every weight.")
            for name, number in {
                "count": len(weights),
                "sum": math.fsum(weights),
                "absolute_sum": math.fsum(abs(v) for v in weights),
                "negative_count": sum(v < 0 for v in weights),
                "zero_count": sum(v == 0 for v in weights),
                "min": min(weights),
                "max": max(weights),
            }.items():
                m.add(f"{address}.{name}", str(number), number)
            m.add(f"{address}.fingerprint", fingerprint)
            return
    if isinstance(value, dict):
        for key, item in value.items():
            _values(m, f"{address}.{key}", item)
    elif isinstance(value, list):
        path = address.split(".")
        membership = path[-1] in {"rows", "row_ids", "source_row_ids", "included_units"} or "input_rows" in path
        if membership and len(value) > 16 and all(isinstance(v, (int, str)) and not isinstance(v, bool) for v in value):
            fingerprint = hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()
            members = "units" if path[-1] == "included_units" else "rows"
            m.add(address, "Full frozen membership is retained in result.json; see the addressed count and fingerprint.")
            m.add(f"{address}.count", f"{len(value)} frozen {members}", len(value))
            m.add(f"{address}.fingerprint", fingerprint)
            return
        for index, item in enumerate(value):
            _values(m, f"{address}.{index}", item)
    elif value is not None:
        m.add(address, str(value), value if isinstance(value, (int, float)) and not isinstance(value, bool) else None)


def render(
    run: RunRecord, memory: Memory | None = None, previous: RunRecord | None = None, steps: list[Step] | None = None, matrix: Matrix | None = None
) -> Material:
    m = Material()
    for line in matrix.render() if matrix is not None else []:
        address, _, text = line[1:].partition("] ")
        m.add(address, text)
    m.add("run.question", run.question)
    m.add("decision.family", run.family or "none")
    m.add("run.status", run.status)
    context_policy = (
        "This analysis uses its frozen admitted context, recorded in context.fields with its context.id. "
        "Current durable memory, when shown at raw claim: and col: addresses, is the later dataset context and does not replace that frozen analysis source."
    )
    if run.what_if:
        context_policy += " This hypothetical run used an isolated context copy; its recorded run.what_if changes never wrote durable memory."
        m.add("run.what_if", "Hypothetical changes applied only to this run's isolated context copy.")
        _values(m, "run.what_if", run.what_if)
    m.add("run.context_policy", context_policy)
    for key, value in (run.decision or {}).items():
        _values(m, f"decision.{key}", value)
    brief_record = _brief(run)
    if brief_record:
        for address, text in brief_record.lines():
            m.add(address, text)
    sr = run.specialist_result
    m.add("context.id", str(sr.get("context_id") or ""))
    m.add("run.lead_design_id", str(sr.get("lead_design_id") or "none"))
    _values(m, "lead.selection", sr.get("lead_selection"))
    for design_id, design in (sr.get("designs") or {}).items():
        _values(m, f"design:{design_id}", design)
    _values(m, "context.admission", sr.get("context_admission"))
    _values(m, "context.fields", sr.get("context_facts"))
    _values(m, "ledger", sr.get("ledger"))
    for ident, entry in (sr.get("ledger") or {}).get("entries", {}).items():
        _values(m, ident, entry)
        m.add(ident, f"{entry.get('kind')}: {entry.get('value')}; {entry.get('status')}; {entry.get('reason')}")
    for change in (sr.get("ledger") or {}).get("history", []):
        _values(m, f"ledger:revision.{change['revision']}", change)
    for episode in (sr.get("episodes") or {}).values():
        for fact in episode.get("facts") or []:
            address = f"probe:{fact['node']}.{fact['visit']}.{fact['n']}"
            m.add(address, fact["text"], fact.get("value"))
    _values(m, "candidates", sr.get("candidates"))
    _values(m, "comparisons", sr.get("comparisons"))
    _values(m, "execution", sr.get("execution_attempts"))
    for raw in sr.get("estimates") or []:
        estimate = Estimate.model_validate(raw)
        tag = estimate.tag
        _values(m, tag, raw)
        m.add(f"{tag}.ci", f"interval {_g(estimate.ci_low)} to {_g(estimate.ci_high)}")
        m.add(f"{tag}.n", f"{estimate.n_treated} treated and {estimate.n_control} control rows")
    for raw in sr.get("falsifications") or []:
        from causal_agent.common.contracts import Falsification

        record = Falsification.model_validate(raw)
        _values(m, record.address, raw)
    for raw in sr.get("sensitivities") or []:
        from causal_agent.common.contracts import Sensitivity

        record = Sensitivity.model_validate(raw)
        _values(m, record.address, raw)
    for raw in sr.get("checks") or []:
        from causal_agent.common.contracts import CheckResult

        record = CheckResult.model_validate(raw)
        _values(m, record.address, raw)
        m.add(record.address, f"{record.level}: {record.detail}", record.value)
    for address, text in sr.get("ladder") or []:
        m.add(str(address), str(text))
    for fact in sr.get("facts") or []:
        m.add(str(fact["address"]), str(fact["text"]), fact.get("value"))
    for record in sr.get("interpretations") or []:
        tag = f"interpretation:{record.get('design_id') or 'across'}.{record.get('contrast') or 'all'}"
        m.add(f"{tag}.answer", str(record.get("answer") or ""))
        for index, caveat in enumerate(record.get("caveats") or [], 1):
            m.add(f"{tag}.caveat:{index}", str(caveat))
    for decline in _declines(run):
        m.add(decline.address, decline.render())
    _values(m, "feasibility", sr.get("feasibility"))
    _values(m, "context.deficiency", sr.get("context_deficiency"))
    for raw in run.figures:
        from causal_agent.viz.spec import FigureSpec

        figure = FigureSpec.model_validate(raw)
        for line in figure.render().splitlines():
            trimmed = line.strip()
            if trimmed.startswith("["):
                address, _, text = trimmed[1:].partition("] ")
                m.add(address, text)
        for series in figure.series:
            for index, value in enumerate(series.y):
                address = f"{figure.address}.{series.key}.{index}"
                if value is not None:
                    m.numbers[address] = value
                for suffix, values in (("lo", series.lo), ("hi", series.hi), ("n", series.n)):
                    if values is not None and values[index] is not None:
                        m.add(f"{address}.{suffix}", str(values[index]), values[index])
    if memory:
        for address, record in memory.fields.items():
            m.add(address, record.render(address), record.value if isinstance(record.value, (int, float)) and not isinstance(record.value, bool) else None)
        for said in memory.said:
            m.add(f"user:turn:{said.turn}", said.text)
        add_artifacts(m, memory.name, run.index)
    for step in steps or []:
        m.add(step.address, step.line())
    if previous:
        m.add(f"run:{previous.index}.question", previous.question)
        m.add(f"run:{previous.index}.effect", _g(previous.effect), previous.effect)
        m.add(f"run:{previous.index}.status", previous.status)
    _scale(run, m)
    m.close()
    return m


def add_artifacts(m: Material, dataset: str, design: int | None = None) -> None:
    """Every artifact about the data and, when a design is given, every one about that run: the summary at its address and
    every number it holds at its own, the table and the picture when there are any, so the chat can cite them and the gate
    can check the numbers."""
    for a in VA.about(dataset, design):
        m.add(a.address, f"{a.summary} ({a.origin}; {a.format}; {a.when()}; asked: {a.ask})")
        m.add(f"{a.address}.method", a.method)
        for i, finding in enumerate(a.findings):
            m.add(f"{a.address}.finding.{i}", finding)
        for i, limitation in enumerate(a.limitations):
            m.add(f"{a.address}.limitation.{i}", limitation)
        m.add(f"{a.address}.provenance", a.provenance())
        for name, value in a.facts.items():
            m.add(f"{a.address}.{name}", f"{name}: {value:.4g}", value)
        if a.table:
            m.add(f"{a.address}.table", f"a table: {a.table}")
        if a.figure:
            m.add(f"{a.address}.figure", "a picture")


def brief(run: RunRecord, previous: RunRecord | None, material: Material) -> str:
    """The opening is a projection of the new result, never a positional choice of an estimate."""
    sr = run.specialist_result
    lines = [f"Run {run.index}: {run.question}"]
    for interpretation in sr.get("interpretations") or []:
        tag = f"interpretation:{interpretation.get('design_id') or 'across'}.{interpretation.get('contrast') or 'all'}"
        lines.append(f"{interpretation.get('answer')} [{tag}.answer]")
        lines.extend(f"Keep in mind: {text} [{tag}.caveat:{index}]" for index, text in enumerate(interpretation.get("caveats") or [], 1))
    lead = sr.get("lead_design_id")
    design = (sr.get("designs") or {}).get(lead) or {}
    if lead:
        lines.append(f"Lead design: {design.get('endpoint_id', lead)} [design:{lead}.endpoint_id]")
    estimates = [Estimate.model_validate(e) for e in sr.get("estimates") or []]
    primary = [e for e in estimates if e.design_id == lead and not e.secondary]
    expected: list[tuple[tuple[str, str | None, str | None], str]] = []
    summaries = (design.get("target") or {}).get("primary_summaries") or []
    if summaries:
        expected = [
            ((summary["contrast"], summary.get("modifier"), summary.get("level")), f"design:{lead}.target.primary_summaries.{index}")
            for index, summary in enumerate(summaries)
        ]
    else:
        expected = [
            ((Contrast.model_validate(raw).key, None, None), f"design:{lead}.contrasts.{index}") for index, raw in enumerate(design.get("contrasts") or [])
        ]
    multiple = len(expected) > 1 or len(primary) > 1
    projected = []
    for summary, address in expected:
        contrast, modifier, level = summary
        records = [e for e in primary if (e.contrast, e.modifier, e.level) == summary]
        if not records:
            group = f", {modifier}={level}" if modifier else ""
            lines.append(f"Lead contrast {contrast}{group}: no primary estimate record was produced. [{address}]")
        projected.extend(records)
    projected.extend(e for e in primary if (e.contrast, e.modifier, e.level) not in {key for key, _ in expected})
    for estimate in projected:
        group = f", {estimate.modifier}={estimate.level}" if estimate.modifier else ""
        label = f"Lead contrast {estimate.contrast}{group}: " if multiple or group else ""
        if estimate.error:
            failure = f"{label}failed: " if multiple else "The lead design failed: "
            lines.append(f"{failure}{estimate.error} [{estimate.tag}.error]")
        elif estimate.value is not None:
            lines.append(
                f"{label}Effect {_g(estimate.value)}, interval {_g(estimate.ci_low)} to {_g(estimate.ci_high)} [{estimate.tag}.value] [{estimate.tag}.ci]"
            )
        else:
            lines.append(f"{label or 'The lead design: '}no point estimate was recorded. [{estimate.tag}]")
    deficiency = sr.get("context_deficiency")
    if deficiency:
        lines.append(f"The context needs clarification: {deficiency['reason']} [context.deficiency.reason]")
    if sr.get("feasibility"):
        lines.append(f"Analysis limitation: {sr['feasibility']['reason']} [feasibility.reason]")
    failed = [x for x in sr.get("falsifications") or [] if x.get("passed") is False]
    for item in failed:
        from causal_agent.common.contracts import Falsification

        record = Falsification.model_validate(item)
        lines.append(f"Falsification failed: {record.detail} [{record.address}.detail]")
    for comparison in sr.get("comparisons") or []:
        if comparison.get("summary"):
            lines.append(str(comparison["summary"]))
    for decline in _declines(run):
        lines.append(f"The lane disagreed with what was settled: {decline.reason} [{decline.address}]")
    for check in sr.get("checks") or []:
        if check.get("level") != "pass":
            from causal_agent.common.contracts import CheckResult

            record = CheckResult.model_validate(check)
            lines.append(f"Design flag: {record.detail} [{record.address}]")
    if run.status == "pipeline_error":
        lines.append("The analysis process failed before producing a record. [run.status]")
        lines.append(run.decision_record[-1500:])
    lines.append("Ask about the evidence, correct the context, explore further, ask a new question, or say done.")
    return "\n".join(lines)
