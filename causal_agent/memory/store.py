"""Where a memory lives on disk, and how the older claims files become one.

    data/memory/<name>/meta.yaml          name, version, csv, the dataset facts
    data/memory/<name>/columns.yaml       one record per column
    data/memory/<name>/dataset.yaml       the dataset kinds
    data/memory/<name>/transcript.jsonl   the person's words, one turn per line
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
from causal_agent.memory.records import Memory
from causal_agent.profile.datasets import ROOT, dataset_entries

FILES = ("meta.yaml", "columns.yaml", "dataset.yaml")


def home(name: str, root: Path | None = None) -> Path:
    return Path(root or ROOT) / "data" / "memory" / name


def exists(name: str, root: Path | None = None) -> bool:
    return (home(name, root) / "meta.yaml").exists()


def save(memory: Memory, root: Path | None = None) -> Path:
    d = home(memory.name, root)
    d.mkdir(parents=True, exist_ok=True)
    dump = memory.model_dump(mode="json")
    (d / "meta.yaml").write_text(yaml.safe_dump({k: dump[k] for k in ("name", "version", "csv", "dataset_facts")}, sort_keys=False, allow_unicode=True))
    (d / "columns.yaml").write_text(yaml.safe_dump(dump["columns"], sort_keys=False, allow_unicode=True))
    (d / "dataset.yaml").write_text(yaml.safe_dump(dump["dataset"], sort_keys=False, allow_unicode=True))
    (d / "transcript.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in dump["transcript"]))
    return d


def load(name: str, root: Path | None = None) -> Memory:
    d = home(name, root)
    if not (d / "meta.yaml").exists():
        raise FileNotFoundError(f"no memory for {name!r} under {d}")
    meta = yaml.safe_load((d / "meta.yaml").read_text()) or {}
    columns = yaml.safe_load((d / "columns.yaml").read_text()) or {}
    dataset = yaml.safe_load((d / "dataset.yaml").read_text()) or {}
    transcript = []
    tp = d / "transcript.jsonl"
    if tp.exists():
        transcript = [json.loads(line) for line in tp.read_text().splitlines() if line.strip()]
    return Memory.model_validate({**meta, "columns": columns, "dataset": dataset, "transcript": transcript})


def snapshot(memory: Memory, design_id: int, root: Path | None = None) -> Path:
    d = home(memory.name, root) / "designs" / str(design_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "memory.json").write_text(memory.model_dump_json(indent=2))
    return d


# ------------------------------------------------------------------ migration from the claims files


def _transcript(name: str, root: Path) -> list[Said]:
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
    from causal_agent.desk.handoff import load_claims
    from causal_agent.memory.claims import ClaimTable
    from causal_agent.profile.profiler import Profile

    root = Path(root or ROOT)
    entries = dataset_entries(root / "data" / "datasets.yaml")
    if name not in entries:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(entries)}")
    e = entries[name]
    prof = Profile.model_validate(json.loads((root / e["profile"]).read_text()))
    table, _ = load_claims(root / e["claims"]) if e.get("claims") and (root / e["claims"]).exists() else (ClaimTable(), [])
    m = Memory.from_claims(name, table, profile=prof, csv=e.get("csv"), transcript=_transcript(name, root))
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
        names = list(dataset_entries()) if args.all else [args.name]
        for n in names:
            m = migrate(n)
            print(f"{n}: {len(m.columns)} columns, {sum(1 for k in m.dataset.kinds.values() for f in k.fields.values() if f.value is not None)} dataset fields, {len(m.transcript)} turns")
    else:
        print(load(args.name).render())


if __name__ == "__main__":
    main()
