"""Frozen row populations, data integrity and permissible numerical comparisons."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from causal_agent.common.contracts import Contrast
from causal_agent.common.contracts.base import _slug

TREATED = "treated"


def primary_summaries(contrasts, groups):
    if groups:
        return [
            {"contrast": contrast.key, "modifier": col, "level": group["label"]} for contrast in contrasts for col, items in groups.items() for group in items
        ]
    return [{"contrast": contrast.key, "modifier": None, "level": None} for contrast in contrasts]


def summary_key(contrast, modifier=None, level=None):
    return contrast if modifier is None else f"{contrast}.{_slug(modifier)}.{_slug(level)}"


def unique_contrast_addresses(contrasts) -> bool:
    return len({contrast.key for contrast in contrasts}) == len(contrasts)


def contrast_table(table: pd.DataFrame, treatment: str, contrast: Contrast) -> pd.DataFrame:
    if treatment != TREATED and TREATED in table.columns:
        raise ValueError("reserved internal treatment column 'treated' would overwrite an unrelated admitted column")
    col = table[treatment].astype(str)
    sub = table[col.isin([str(contrast.control), str(contrast.treated)])].copy()
    flag = (col.loc[sub.index] == str(contrast.treated)).astype(int)
    if treatment != TREATED:
        sub = sub.drop(columns=[treatment])
    sub[TREATED] = flag
    return sub


def fingerprint(table: pd.DataFrame) -> str:
    payload = pd.util.hash_pandas_object(table, index=True).values.tobytes()
    return hashlib.sha256(json.dumps(list(table.columns)).encode() + payload).hexdigest()


def freeze(table, treatment, contrasts, source_rows, context_id):
    rows = list(source_rows)
    out = {}
    for contrast in contrasts:
        sub = contrast_table(table, treatment, contrast)
        ids = [rows[int(i)] for i in sub.index]
        identity = hashlib.sha256(json.dumps([context_id, ids, contrast.model_dump()], sort_keys=True).encode()).hexdigest()[:24]
        out[contrast.key] = {"source_row_ids": ids, "table_fingerprint": fingerprint(sub), "analysis_population_id": identity}
    return out


def verify(table, design, contrast):
    expected = design.support.get("by_contrast", {}).get(contrast)
    if expected and expected["table_fingerprint"] != fingerprint(table):
        raise ValueError("design: analysis rows or values differ from the frozen support")


def signature(estimate):
    return estimate.contrast, json.dumps(estimate.target, sort_keys=True)
