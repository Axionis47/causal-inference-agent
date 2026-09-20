"""The file and its profile, loaded once per path in this process and once per file content on disk. Column names are
matched by key, as the pack does.

The disk cache lives under .artifacts/profiles/<sha256>-<flags>-<profiler version>.json: the same bytes with the same
flags always give the same profile, so a file is profiled once however many processes read it."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from causal_agent.common import config
from causal_agent.common.addresses import key as _key
from causal_agent.profile.pack import ColumnCard, DatasetCard, Pack
from causal_agent.profile.profiler import PROFILER_VERSION, Profile, profile

_cache: dict[str, tuple[pd.DataFrame, Profile]] = {}


def cache_dir() -> Path:
    return config.get().paths.profiles


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def profile_for(csv: str | Path, entity: list[str] | None = None, time: str | None = None) -> Profile:
    """The profile of a file, from the disk cache when the same bytes were profiled before with the same flags."""
    path = Path(csv)
    flags = _key("|".join([*(entity or []), time or ""])) or "plain"
    d = cache_dir()
    p = d / f"{_digest(path)}-{flags}-{PROFILER_VERSION}.json"
    if p.exists():
        try:
            return Profile.model_validate_json(p.read_text())
        except ValueError:
            pass
    prof = profile(path, entity_columns=entity, time_column=time)
    d.mkdir(parents=True, exist_ok=True)
    p.write_text(prof.model_dump_json())
    return prof


def load(csv: str | Path, entity: list[str] | None = None, time: str | None = None) -> tuple[pd.DataFrame, Profile]:
    k = f"{Path(csv).resolve()}|{entity}|{time}"
    if k not in _cache:
        _cache[k] = (pd.read_csv(csv), profile_for(csv, entity, time))
    return _cache[k]


def clear() -> None:
    _cache.clear()


def column(df: pd.DataFrame, name: str | None) -> str | None:
    """The file's spelling of a column named by key or by name; None when absent."""
    if name is None:
        return None
    k = _key(str(name))
    for c in df.columns:
        if c == name or _key(c) == k:
            return c
    return None


def cards(name: str, prof: Profile) -> Pack:
    """A pack with no notes: the profile cards the interview shows and the addresses it lets the model cite."""
    cols = [ColumnCard(name=cp.name, key=_key(cp.name), note=None, source=None, profile=cp) for cp in prof.columns]
    return Pack(name=name, dataset=DatasetCard(name=name, note="", source=None, profile=prof.dataset), changes=[], columns=cols)
