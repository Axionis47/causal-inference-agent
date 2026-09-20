from __future__ import annotations

from pathlib import Path

import yaml

from causal_agent.common.config import ROOT
from causal_agent.profile.pack import Pack, load_pack


def dataset_entries(index_path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(index_path or ROOT / "data/datasets.yaml").read_text())


def load_dataset_pack(name: str) -> Pack:
    entries = dataset_entries()
    if name not in entries:
        raise KeyError(f"unknown dataset {name!r}; known: {sorted(entries)}")
    e = entries[name]
    return load_pack(name, ROOT / e["note"], ROOT / e["profile"], ROOT / e["claims"] if e.get("claims") else None)


def csv_path(name: str, csv: str | None = None) -> Path | None:
    """Where a dataset's file is: the path given, else the index entry's, relative paths under the root. None when neither names one."""
    csv = csv or (dataset_entries().get(name) or {}).get("csv")
    if not csv:
        return None
    p = Path(csv)
    return p if p.is_absolute() else Path(ROOT) / p
