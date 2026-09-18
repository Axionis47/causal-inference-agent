from __future__ import annotations

from pathlib import Path

import yaml

from causal_agent.profile.pack import Pack, load_pack

ROOT = Path(__file__).resolve().parents[2]


def dataset_entries(index_path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(index_path or ROOT / "data/datasets.yaml").read_text())


def load_dataset_pack(name: str) -> Pack:
    entries = dataset_entries()
    if name not in entries:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(entries)}")
    e = entries[name]
    return load_pack(name, ROOT / e["note"], ROOT / e["profile"], ROOT / e["claims"] if e.get("claims") else None)
