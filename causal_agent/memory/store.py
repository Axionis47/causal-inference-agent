"""Where a memory lives on disk, and how the older claims files become one.

    data/memory/<name>/meta.yaml          name, version, csv, the dataset facts
    data/memory/<name>/columns.yaml       the file's facts on every column, by key
    data/memory/<name>/fields.yaml        the map: address -> value, status, source, said, evidence
    data/memory/<name>/said.jsonl         the person's words, one turn per line
    data/memory/<name>/designs/<n>/       memory.json, frame.json, decision.json, handoff.json, run/

    uv run python -m causal_agent.memory.store migrate --all        # every dataset in data/datasets.yaml
    uv run python -m causal_agent.memory.store show students3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from causal_agent.common.contracts import Said
from causal_agent.memory.claims import Claim, ClaimTable, ProbeResult
from causal_agent.memory.records import Memory
from causal_agent.profile import datasets as DS

LEGACY = ("dataset.yaml", "transcript.jsonl")  # the nested layout, replaced by fields.yaml and said.jsonl


def home(name: str, root: Path | None = None) -> Path:
    return Path(root or DS.ROOT) / "data" / "memory" / name


def exists(name: str, root: Path | None = None) -> bool:
    d = home(name, root)
    return (d / "meta.yaml").exists() and (d / "fields.yaml").exists()


def save(memory: Memory, root: Path | None = None) -> Path:
    d = home(memory.name, root)
    d.mkdir(parents=True, exist_ok=True)
    dump = memory.model_dump(mode="json")
    (d / "meta.yaml").write_text(yaml.safe_dump({k: dump[k] for k in ("name", "version", "csv", "facts")}, sort_keys=False, allow_unicode=True))
    (d / "columns.yaml").write_text(yaml.safe_dump(dump["columns"], sort_keys=False, allow_unicode=True))
    (d / "fields.yaml").write_text(yaml.safe_dump(dump["fields"], sort_keys=False, allow_unicode=True))
    (d / "said.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in dump["said"]))
    for legacy in LEGACY:
        (d / legacy).unlink(missing_ok=True)
    return d


def load(name: str, root: Path | None = None) -> Memory:
    d = home(name, root)
    if not exists(name, root):
        raise FileNotFoundError(f"no memory for {name!r} under {d}")
    meta = yaml.safe_load((d / "meta.yaml").read_text()) or {}
    columns = yaml.safe_load((d / "columns.yaml").read_text()) or {}
    fields = yaml.safe_load((d / "fields.yaml").read_text()) or {}
    said = []
    sp = d / "said.jsonl"
    if sp.exists():
        said = [json.loads(line) for line in sp.read_text().splitlines() if line.strip()]
    return Memory.model_validate({**meta, "columns": columns, "fields": fields, "said": said})


def snapshot(memory: Memory, design_id: int, root: Path | None = None) -> Path:
    d = home(memory.name, root) / "designs" / str(design_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "memory.json").write_text(memory.model_dump_json(indent=2))
    return d


# ------------------------------------------------------------------ the claims files, and a memory by name


def load_claims(path: str | Path) -> tuple[ClaimTable, list[ProbeResult]]:
    """The claims document the interview writes: {claims: [...], probes: [...]}."""
    doc = yaml.safe_load(Path(path).read_text()) or {}
    table = ClaimTable(claims={c["key"]: Claim.model_validate(c) for c in doc.get("claims") or []})
    probes = [ProbeResult.model_validate(p) for p in doc.get("probes") or []]
    return table, probes


def memory_for(name: str, root: Path | None = None) -> Memory:
    """The memory on disk, or one made from the dataset's profile and claims files and saved."""
    if exists(name, root):
        return load(name, root)
    return migrate(name, root)


# ------------------------------------------------------------------ migration from the claims files


def _said(name: str, root: Path) -> list[Said]:
    """The web desk's transcript, user turns numbered as the interview numbered them."""
    p = Path(root) / "data" / "web" / name / "transcript.jsonl"
    if not p.exists():
        return []
    out, turn = [], 0
    for line in p.read_text().splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("role") == "user" and d.get("text"):
            turn += 1
            out.append(Said(turn=turn, about="", text=str(d["text"])))
    return out


def migrate(name: str, root: Path | None = None, *, write: bool = True) -> Memory:
    """A memory from a datasets.yaml entry: the profile for the facts, the claims file for the fields, the transcript for the words."""
    from causal_agent.profile.profiler import Profile

    root = Path(root or DS.ROOT)
    entries = DS.dataset_entries(root / "data" / "datasets.yaml")
    if name not in entries:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(entries)}")
    e = entries[name]
    prof = Profile.model_validate(json.loads((root / e["profile"]).read_text()))
    table, _ = load_claims(root / e["claims"]) if e.get("claims") and (root / e["claims"]).exists() else (ClaimTable(), [])
    m = Memory.from_claims(name, table, profile=prof, csv=e.get("csv"), said=_said(name, root))
    from causal_agent.memory.ops import seed_facts

    seed_facts(m, prof)
    if write:
        save(m, root)
    return m


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    mg = sub.add_parser("migrate")
    mg.add_argument("name", nargs="?")
    mg.add_argument("--all", action="store_true")
    sh = sub.add_parser("show")
    sh.add_argument("name")
    args = ap.parse_args(argv)
    if args.cmd == "migrate":
        names = list(DS.dataset_entries()) if args.all else [args.name]
        for n in names:
            m = migrate(n)
            print(f"{n}: {len(m.columns)} columns, {sum(1 for f in m.fields.values() if f.value is not None)} fields known, {len(m.said)} turns")
    else:
        print(load(args.name).render())


if __name__ == "__main__":
    main()
