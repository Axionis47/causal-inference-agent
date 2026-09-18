"""Write a dataset the router can load: the CSV under data/raw, the profile, the claims file, a note rendered
from the claims in the three-heading format, and the datasets.yaml entry. Facts only; the note's sentences are
the claims' values with their sources."""

from __future__ import annotations

import datetime as dt
import json
import shutil
from pathlib import Path

import yaml

from causal_agent.intake.datasets import ROOT
from causal_agent.intake.interview.contracts import ClaimTable, ProbeResult
from causal_agent.intake.profiler import Profile


def _src(claim) -> str:
    if not claim.source:
        return "[unstated]"
    if claim.source.startswith("user:"):
        return f"[user, {dt.date.today().isoformat()}]"
    if claim.source.startswith("doc:"):
        return f"[{claim.source[4:]}]"
    return f"[{claim.source}]"


def _sentence(s: str | None) -> str:
    s = (s or "").strip()
    return s if not s or s.endswith(".") else s + "."


def render_note(name: str, prof: Profile, table: ClaimTable) -> str:
    g, samp, ch, a, miss = (table.get(k) for k in ("grain", "sampling", "change", "assignment", "missing"))
    unob, spill, trend, excl = (table.get(k) for k in ("unobserved", "spillover", "trend_continues", "exclusion"))
    ds = []
    if g and g.fields.get("row_is"):
        ds.append(f"Each row is {g.fields['row_is'].rstrip('.')}. {_src(g)}")
        if g.fields.get("key_columns"):
            ds.append(f"Rows are identified by {' + '.join(g.fields['key_columns'])}. {_src(g)}")
        if g.fields.get("panel") is True:
            ds.append(f"The same unit appears in more than one period. {_src(g)}")
    ds.append(f"The file has {prof.dataset.rows} rows and {prof.dataset.columns} columns. [profile]")
    if samp and samp.fields.get("how"):
        how = {"whole": "The file keeps every unit", "by_side": "Rows were drawn by which side of the cutoff they fell on", "by_arm": "Rows were drawn by whether the unit got the change",
               "by_group": "Rows were drawn by group", "by_period": "Rows were drawn by period", "by_outcome": "Rows were drawn by how the outcome turned out", "unknown": "How rows were chosen is not known"}[samp.fields["how"]]
        ds.append(f"{how}. {_sentence(samp.fields.get('detail'))} {_src(samp)}".replace("  ", " "))
    if miss and miss.status != "empty":
        ds.append(f"Missing values: {_sentence(miss.fields.get('why'))} {_src(miss)}")
    for c, yes, no in ((unob, "Something not in the file affected both who got the change and the outcome", "Nothing not in the file affected both who got the change and the outcome"),
                       (spill, "A unit that got the change could affect the outcome of one that did not", "Units that got the change could not affect the outcomes of those that did not"),
                       (trend, "Without the change, the treated group would have kept moving with the others", "Apart from the change, the treated group would have moved differently from the others")):
        if c and c.status in {"confirmed", "drafted"}:
            flag = c.fields.get("exists", c.fields.get("possible", c.fields.get("believed")))
            body = yes if flag else no
            extra = _sentence(c.fields.get("what") or c.fields.get("why") or c.fields.get("why_believed"))
            ds.append(f"{body}. {extra} {_src(c)}".replace("  ", " "))
        elif c and c.status == "unknown":
            ds.append(f"The author does not know: {c.kind.replace('_', ' ')}. {_src(c)}")
    if excl and excl.status in {"confirmed", "drafted"} and excl.fields.get("column"):
        ds.append(f"{excl.fields['column']} moved who got the change and touches the outcome only through it: {_sentence(excl.fields.get('why'))} {_src(excl)}")

    changed = []
    if ch and ch.fields.get("what"):
        head = f"**{ch.fields['what'][0].upper() + ch.fields['what'][1:].rstrip('.')}.**"
        body = f"It reached {ch.fields.get('to_whom', 'the units described').rstrip('.')}, {ch.fields.get('when', 'at a time not stated').rstrip('.')}."
        if ch.fields.get("date_column"):
            body += f" The period is recorded in {ch.fields['date_column']}" + (f"; the change took effect at {ch.fields['period_value']}." if ch.fields.get("period_value") else ".")
        changed.append(f"{head} {body} {_src(ch)}".replace("  ", " "))
    if a and a.fields.get("kind"):
        kind = {"lottery": "Who got the change was decided by a draw", "cutoff_rule": "Who got the change was decided by a cutoff on a score", "own_choice": "Units chose whether to take the change",
                "date_by_others": "The change reached units on a date set by someone else", "third_party": "A third party decided who got the change"}[a.fields["kind"]]
        parts = [f"{kind}: {_sentence(a.fields.get('rule'))}"]
        if a.fields.get("depends_on"):
            parts.append(f"The decision to give or offer the change depended on {', '.join(a.fields['depends_on'])}, which are columns in the file.")
        if a.fields.get("score_column") is not None and a.fields.get("cutoff") is not None:
            side = a.fields.get("treated_side") or "one side"
            tie = a.fields.get("cutoff_value_treated")
            parts.append(f"The score is {a.fields['score_column']} and the cutoff is {a.fields['cutoff']:g}; units {side} it got the change" + (", a unit exactly at the cutoff included." if tie else ", a unit exactly at the cutoff not included." if tie is False else "."))
        if a.fields.get("treatment_column"):
            parts.append(f"{a.fields['treatment_column']} records who got it, {a.fields.get('treated_level')!r} meaning yes.")
        else:
            parts.append("No column records receipt separately: the rule itself is the change.")
        if a.fields.get("movable") is True:
            parts.append("A unit could change what the rule looked at after seeing it.")
        elif a.fields.get("movable") is False:
            parts.append("A unit could not change what the rule looked at.")
        if a.check_detail:
            parts.append(f"In the file: {a.check_detail}. [profile]")
        changed.append(" ".join(parts) + f" {_src(a)}")
    if not changed:
        changed.append("**Change not stated.** [unstated]")
    changed = [" ".join(changed)]  # one paragraph: one change card

    cols = []
    for cp in prof.columns:
        c = table.get(f"col:{cp.key}")
        if c is None or not c.fields.get("meaning"):
            cols.append(f"**{cp.name}** — Not described. [unstated]")
            continue
        when = {"before": "Fixed before the change", "at": "Set at the change", "after": "Measured after the change", "unknown": "When it was set is not known"}.get(c.fields.get("when"), "")
        bits = [_sentence(c.fields["meaning"]), when + "." if when else ""]
        if c.fields.get("set_by"):
            bits.append(f"Set by {c.fields['set_by'].rstrip('.')}.")
        if c.fields.get("derived_from"):
            bits.append(f"Derived from {', '.join(c.fields['derived_from'])}.")
        if c.fields.get("affected_by_treatment") is True:
            bits.append("The change could have moved it.")
        elif c.fields.get("affected_by_treatment") is False:
            bits.append("The change could not have moved it.")
        if cp.nulls:
            bits.append(f"Missing for {cp.nulls} rows.")
        cols.append(f"**{cp.name}** — {' '.join(b for b in bits if b)} {_src(c)}")

    title = name.replace("_", " ").title()
    return f"# {title}\n\n## About the dataset\n{' '.join(ds)}\n\n## What changed\n" + "\n\n".join(changed) + "\n\n## About each column\n" + "\n\n".join(cols) + "\n"


def claims_document(table: ClaimTable, probes: list[ProbeResult]) -> dict:
    return {
        "claims": [c.model_dump() for c in table.claims.values()],
        "probes": [p.model_dump() for p in probes],
    }


def write_dataset(name: str, csv: str, prof: Profile, table: ClaimTable, probes: list[ProbeResult], *, entity: list[str] | None, time: str | None, root: Path | None = None) -> dict:
    root = Path(root or ROOT)
    src = Path(csv).resolve()
    raw_dir = root / "data" / "raw" / name
    try:
        rel = src.relative_to(root)
        csv_rel = str(rel)
    except ValueError:
        raw_dir.mkdir(parents=True, exist_ok=True)
        dest = raw_dir / src.name
        if not dest.exists():
            shutil.copy(src, dest)
        csv_rel = str(dest.relative_to(root))
    (root / "data" / "profiles").mkdir(parents=True, exist_ok=True)
    (root / "data" / "context").mkdir(parents=True, exist_ok=True)
    (root / "data" / "claims").mkdir(parents=True, exist_ok=True)
    profile_rel, note_rel, claims_rel = f"data/profiles/{name}.json", f"data/context/{name}.md", f"data/claims/{name}.yaml"
    (root / profile_rel).write_text(json.dumps(prof.model_dump(), indent=2))
    (root / note_rel).write_text(render_note(name, prof, table))
    (root / claims_rel).write_text(yaml.safe_dump(claims_document(table, probes), sort_keys=False, allow_unicode=True))
    entry: dict = {"csv": csv_rel, "note": note_rel, "profile": profile_rel, "claims": claims_rel}
    if entity:
        entry["entity"] = list(entity)
    if time:
        entry["time"] = time
    samp = table.get("sampling")
    if samp and samp.fields.get("how") == "by_side":
        entry["sampled_by_side"] = True
    index = root / "data" / "datasets.yaml"
    entries = yaml.safe_load(index.read_text()) if index.exists() else {}
    header = "# The eval datasets. Name → files and the profiler flags used to build the profile.\n"
    entries[name] = entry
    index.write_text(header + yaml.safe_dump(entries, sort_keys=False, allow_unicode=True))
    return {"dataset": name, "csv": csv_rel, "note": note_rel, "profile": profile_rel, "claims": claims_rel, "entry": entry}
