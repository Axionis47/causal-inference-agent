"""Checks that can refute a claim from the file. Facts only: a number, a verdict against a declared threshold,
and an address the reply can cite. A check that cannot run with what is settled returns None."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from causal_agent.memory.claims import Claim, ClaimTable
from causal_agent.profile.data import column
from causal_agent.profile.profiler import Profile


@dataclass
class Outcome:
    name: str
    passed: bool | None
    detail: str

    def address(self, claim: Claim) -> str:
        return f"check:{claim.key}.{self.name}"


def _truthy(v) -> bool | None:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    return True if s in {"true", "yes", "1"} else False if s in {"false", "no", "0"} else None


def _share(df: pd.DataFrame, mask: pd.Series, tcol: str, level: str) -> float | None:
    sub = df.loc[mask, tcol].dropna()
    if sub.empty:
        return None
    return float((sub.astype(str).str.strip() == str(level).strip()).mean())


def key_unique(claim: Claim, table: ClaimTable, df: pd.DataFrame, prof: Profile, th: dict) -> Outcome | None:
    keys = claim.fields.get("key_columns") or []
    if not keys:
        return None
    found = [column(df, k) for k in keys]
    if any(c is None for c in found):
        return Outcome("key_unique", False, f"key column not in the file: {[k for k, c in zip(keys, found) if c is None]}")
    cols = [c for c in found if c is not None]
    dups = int(df.duplicated(subset=cols).sum())
    if dups:
        return Outcome("key_unique", False, f"{' + '.join(cols)} does not identify a row: {dups} rows share a key with another row")
    return Outcome("key_unique", True, f"{' + '.join(cols)} identifies every row ({len(df)} rows)")


def date_column_exists(claim: Claim, table: ClaimTable, df: pd.DataFrame, prof: Profile, th: dict) -> Outcome | None:
    dc = claim.fields.get("date_column")
    if not dc:
        return None
    c = column(df, dc)
    if c is None:
        return Outcome("date_column", False, f"{dc!r} is not a column in the file")
    cp = next((x for x in prof.columns if x.name == c), None)
    if cp is not None and cp.kind not in {"datetime", "numeric", "id"}:
        return Outcome("date_column", False, f"{c!r} holds {cp.kind} values, not dates or period numbers; it cannot be the period column")
    pv = claim.fields.get("period_value")
    if pv is not None:
        vals = set(df[c].dropna().astype(str).str.strip())
        if str(pv).strip() not in vals:
            lo, hi = df[c].min(), df[c].max()
            return Outcome("period_value", False, f"{pv!r} does not appear in {c!r}, whose values run {lo} to {hi}")
        return Outcome("period_value", True, f"{pv!r} appears in {c!r}")
    return Outcome("date_column", True, f"{c!r} is in the file")


def assignment_matches_data(claim: Claim, table: ClaimTable, df: pd.DataFrame, prof: Profile, th: dict) -> Outcome | None:
    f = claim.fields
    tcol = column(df, f.get("treatment_column"))
    level = f.get("treated_level")
    if f.get("treatment_column") and tcol is None:
        return Outcome("treatment_column", False, f"{f['treatment_column']!r} is not a column in the file")
    if tcol is not None and level is not None:
        observed = set(df[tcol].dropna().astype(str).str.strip())
        if str(level).strip() not in observed:
            return Outcome("treated_level", False, f"{level!r} never appears in {tcol!r}; observed values: {sorted(observed)[:8]}")
    if f.get("kind") == "cutoff_rule":
        sc = column(df, f.get("score_column"))
        if sc is None or f.get("cutoff") is None or f.get("treated_side") not in {"above", "below"}:
            return None
        x = pd.to_numeric(df[sc], errors="coerce")
        c = float(f["cutoff"])
        incl = _truthy(f.get("cutoff_value_treated"))
        if f["treated_side"] == "above":
            treated = (x >= c) if incl else (x > c)
        else:
            treated = (x <= c) if incl else (x < c)
        other = x.notna() & ~treated
        n_t, n_o = int(treated.sum()), int(other.sum())
        if n_t == 0 or n_o == 0:
            return Outcome("rows_by_side", False, f"no rows on the {'other' if n_t else 'treated'} side of {c} on {sc!r} (range {x.min():g} to {x.max():g})")
        if tcol is None or level is None:
            return Outcome("rows_by_side", True, f"{n_t} rows on the treated side of {c} on {sc!r}, {n_o} on the other")
        s_t, s_o = _share(df, treated, tcol, level), _share(df, other, tcol, level)
        if s_t is None or s_o is None:
            return None
        detail = f"share with {tcol}={level}: {s_t:.0%} on the treated side of {c}, {s_o:.0%} on the other ({n_t} and {n_o} rows)"
        if s_t - s_o < float(th["takeup"]["side_gap"]):
            return Outcome("takeup_by_side", False, "take-up does not rise across the cutoff as the rule says: " + detail)
        return Outcome("takeup_by_side", True, detail)
    if tcol is not None:
        vals = df[tcol].dropna().nunique()
        if vals < 2:
            return Outcome("treatment_varies", False, f"{tcol!r} takes one value; nobody differs on it")
        return Outcome("treatment_varies", True, f"{tcol!r} takes {vals} values")
    return None


def before_is_fixed(claim: Claim, table: ClaimTable, df: pd.DataFrame, prof: Profile, th: dict) -> Outcome | None:
    if claim.fields.get("when") != "before":
        return None
    key = claim.key.removeprefix("col:")
    cp = next((c for c in prof.columns if c.key == key or c.name == key), None)
    if cp is None or prof.dataset.entity_summary is None:  # without a unit seen more than once, "varies over time" means nothing
        return None
    if cp.varies_over in {"time", "both"}:
        return Outcome("fixed_within_unit", False, f"{cp.name!r} changes within a unit across periods, so it was not fixed before the change")
    if cp.varies_over in {"entity", "neither"}:
        return Outcome("fixed_within_unit", True, f"{cp.name!r} is constant within a unit")
    return None


def treated_mask(df: pd.DataFrame, table: ClaimTable) -> pd.Series | None:
    """Who got the change: the treatment column at its level, or, for a cutoff rule with no such column, the treated side."""
    a = table.get("assignment")
    if not a:
        return None
    tcol, level = column(df, a.fields.get("treatment_column")), a.fields.get("treated_level")
    if tcol is not None and level is not None:
        return df[tcol].astype(str).str.strip() == str(level).strip()
    if a.fields.get("kind") == "cutoff_rule":
        sc = column(df, a.fields.get("score_column"))
        if sc is None or a.fields.get("cutoff") is None or a.fields.get("treated_side") not in {"above", "below"}:
            return None
        x, c, incl = pd.to_numeric(df[sc], errors="coerce"), float(a.fields["cutoff"]), _truthy(a.fields.get("cutoff_value_treated"))
        return ((x >= c) if incl else (x > c)) if a.fields["treated_side"] == "above" else ((x <= c) if incl else (x < c))
    return None


def missing_by_arm(claim: Claim, table: ClaimTable, df: pd.DataFrame, prof: Profile, th: dict) -> Outcome | None:
    gappy = [cp.name for cp in prof.columns if cp.nulls > 0]
    if not gappy:
        return Outcome("missing_by_arm", True, "no column has missing values")
    treated = treated_mask(df, table)
    if treated is None:
        return Outcome("missing_by_arm", None, f"columns with gaps: {gappy}; who got the change is not settled yet, so gaps cannot be split by arm")
    parts, gap = [], False
    for name in gappy:
        st, so = float(df.loc[treated, name].isna().mean()), float(df.loc[~treated, name].isna().mean())
        parts.append(f"{name}: {st:.1%} missing among treated, {so:.1%} among the rest")
        gap = gap or abs(st - so) > float(th["missing"]["share_gap"])
    return Outcome("missing_by_arm", True, ("missing shares differ by arm; " if gap else "") + "; ".join(parts))


CHECKS = {
    "key_unique": key_unique,
    "date_column_exists": date_column_exists,
    "assignment_matches_data": assignment_matches_data,
    "before_is_fixed": before_is_fixed,
    "missing_by_arm": missing_by_arm,
}


def run(check: str, claim: Claim, table: ClaimTable, df: pd.DataFrame, prof: Profile, th: dict) -> Outcome | None:
    fn = CHECKS.get(check)
    return None if fn is None else fn(claim, table, df, prof, th)
