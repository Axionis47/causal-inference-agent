"""The file and its profile, loaded once per path. Column names are matched by key, as the pack does."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from causal_agent.common.addresses import key as _key
from causal_agent.profile.pack import ColumnCard, DatasetCard, Pack
from causal_agent.profile.profiler import Profile, profile

_cache: dict[str, tuple[pd.DataFrame, Profile]] = {}


def load(csv: str | Path, entity: list[str] | None = None, time: str | None = None) -> tuple[pd.DataFrame, Profile]:
    k = f"{Path(csv).resolve()}|{entity}|{time}"
    if k not in _cache:
        _cache[k] = (pd.read_csv(csv), profile(csv, entity_columns=entity, time_column=time))
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
