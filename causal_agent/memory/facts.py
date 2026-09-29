"""Data facts about the columns in play, computed by code before any run so the pack carries the numbers a reasoning would
otherwise have to ask for: each candidate by arm, the associations among candidates, which pairs are redundant or nested, how
each candidate stands to the outcome, and every column's timing in one line. Each fact is a `ProbeResult` under the family
name `data`, addressed `probe:data.<name>`, with no threshold and no verdict.

One rule, kept here and not in a prompt: no fact joins the outcome with the treatment. The effect is the run's to find.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from causal_agent.common.addresses import key as _key
from causal_agent.memory.checks import treated_mask
from causal_agent.memory.claims import ClaimTable, ProbeResult
from causal_agent.memory.records import Memory
from causal_agent.profile.data import column

FAMILY = "data"
MAX_LEVELS = 12  # a categorical column with more levels than this is summarised, not tabulated
NOTABLE = 0.2  # an association below this is listed, not spelled out
MAX_COVARIATES = 12  # pairwise facts stop here; the lane's tools reach the rest


def _fact(name: str, detail: str, value: float | None = None) -> ProbeResult:
    return ProbeResult(family=FAMILY, name=name, value=value, passed=None, detail=detail)


def _is_numeric(s: pd.Series) -> bool:
    return bool(pd.api.types.is_numeric_dtype(s)) and s.nunique(dropna=True) > MAX_LEVELS


def _smd(a: pd.Series, b: pd.Series) -> float | None:
    a, b = pd.to_numeric(a, errors="coerce").dropna(), pd.to_numeric(b, errors="coerce").dropna()
    if len(a) < 2 or len(b) < 2:
        return None
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float(abs(a.mean() - b.mean()) / sd) if sd > 0 else 0.0


def _cramers_v(a: pd.Series, b: pd.Series) -> float | None:
    t = pd.crosstab(a.astype(str), b.astype(str))
    if t.shape[0] < 2 or t.shape[1] < 2:
        return None
    n = t.values.sum()
    expected = np.outer(t.sum(axis=1), t.sum(axis=0)) / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((t.values - expected) ** 2 / expected)
    k = min(t.shape) - 1
    return float(np.sqrt(chi2 / (n * k))) if n and k else None


def _eta(cat: pd.Series, num: pd.Series) -> float | None:
    """The correlation ratio: how much of a number's spread the levels of a category account for, 0 to 1."""
    num = pd.to_numeric(num, errors="coerce")
    ok = num.notna() & cat.notna()
    cat, num = cat[ok].astype(str), num[ok]
    if num.nunique() < 2 or cat.nunique() < 2:
        return None
    grand = num.mean()
    between = sum(len(g) * (g.mean() - grand) ** 2 for _, g in num.groupby(cat))
    total = ((num - grand) ** 2).sum()
    return float(np.sqrt(between / total)) if total > 0 else 0.0


def association(a: pd.Series, b: pd.Series) -> tuple[float | None, str]:
    """The association two columns show, with the measure that fits their kinds: Pearson's r for two numbers, Cramér's V for
    two categories, the correlation ratio for one of each. All read 0 to 1 except r, which keeps its sign."""
    na, nb = _is_numeric(a), _is_numeric(b)
    if na and nb:
        x, y = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
        ok = x.notna() & y.notna()
        if ok.sum() < 3 or x[ok].std() == 0 or y[ok].std() == 0:
            return None, "correlation"
        return float(np.corrcoef(x[ok], y[ok])[0, 1]), "correlation"
    if not na and not nb:
        return _cramers_v(a, b), "Cramér's V"
    return (_eta(b, a) if na else _eta(a, b)), "correlation ratio"


def determines(a: pd.Series, b: pd.Series) -> bool:
    """Whether every value of `a` maps to one value of `b`: `a` is at least as fine as `b`."""
    t = pd.DataFrame({"a": a.astype(str), "b": b.astype(str)}).dropna()
    return bool(len(t) and (t.groupby("a")["b"].nunique() <= 1).all())


def redundancy(a: pd.Series, b: pd.Series) -> str | None:
    """'same' when each determines the other, 'a in b' when a is the finer one, 'b in a' the other way, None when neither."""
    ab, ba = determines(a, b), determines(b, a)
    if ab and ba:
        return "same"
    if ab:
        return "a in b"
    if ba:
        return "b in a"
    return None


def _by_arm(s: pd.Series, treated: pd.Series, name: str) -> ProbeResult:
    t, o = s[treated], s[~treated]
    if _is_numeric(s):
        d = _smd(t, o)
        tm, om = pd.to_numeric(t, errors="coerce").mean(), pd.to_numeric(o, errors="coerce").mean()
        return _fact(
            f"by_arm.{_key(name)}",
            f"{name!r} by arm: mean {tm:.3g} among the treated, {om:.3g} among the others; standardised difference {d:.2f}"
            if d is not None
            else f"{name!r} by arm: too few rows",
            d,
        )
    lv = [str(v) for v in s.astype(str).value_counts().index[:4]]
    parts, gap = [], 0.0
    for v in lv:
        st, so = float((t.astype(str) == v).mean()) if len(t) else 0.0, float((o.astype(str) == v).mean()) if len(o) else 0.0
        gap = max(gap, abs(st - so))
        parts.append(f"{v}: {st:.0%} of the treated, {so:.0%} of the others")
    more = f"; {s.nunique() - len(lv)} more levels" if s.nunique() > len(lv) else ""
    return _fact(f"by_arm.{_key(name)}", f"{name!r} by arm: " + "; ".join(parts) + more + f"; largest share gap {gap:.2f}", gap)


def facts(df: pd.DataFrame, memory: Memory, table: ClaimTable, *, outcome: str | None, treatment: str | None, columns: list[str]) -> list[ProbeResult]:
    """The data facts for the columns in play. `columns` are the names in play; the outcome and the treatment are named so no
    fact joins them and neither is treated as a candidate."""
    out: list[ProbeResult] = []
    oc, tc = column(df, outcome), column(df, treatment)
    cands = []
    for n in columns:
        c = column(df, n)
        if c is not None and c not in (oc, tc) and c not in cands and df[c].nunique(dropna=True) > 1:
            cands.append(c)
    cands = cands[:MAX_COVARIATES]
    treated = treated_mask(df, table)
    for c in cands:
        if treated is not None:
            out.append(_by_arm(df[c], treated, c))
        if oc is not None:
            v, how = association(df[c], df[oc])
            if v is not None:
                out.append(_fact(f"with_outcome.{_key(c)}", f"{c!r} against the outcome {oc!r}: {how} {v:.2f}", v))
    quiet: list[str] = []
    for a, b in combinations(cands, 2):
        v, how = association(df[a], df[b])
        pair = f"{_key(a)}~{_key(b)}"
        if v is not None and abs(v) >= NOTABLE:
            out.append(_fact(f"assoc.{pair}", f"{a!r} and {b!r}: {how} {v:.2f}", v))
        elif v is not None:
            quiet.append(f"{a!r} and {b!r} ({v:.2f})")
        r = redundancy(df[a], df[b])
        if r == "same":
            out.append(_fact(f"redundancy.{pair}", f"{a!r} and {b!r} carry the same information: each value of one maps to one value of the other", 1.0))
        elif r == "a in b":
            out.append(
                _fact(f"redundancy.{pair}", f"{a!r} sits inside {b!r}: every value of {a!r} maps to one value of {b!r}, and {b!r} has fewer levels", 1.0)
            )
        elif r == "b in a":
            out.append(
                _fact(f"redundancy.{pair}", f"{b!r} sits inside {a!r}: every value of {b!r} maps to one value of {a!r}, and {a!r} has fewer levels", 1.0)
            )
    if quiet:
        out.append(_fact("assoc.quiet", f"associations below {NOTABLE:.1f} among the candidates: " + "; ".join(quiet)))
    when: dict[str, list[str]] = {"before": [], "at": [], "after": [], "unknown": []}
    for n in columns:
        col = memory.column(n)
        if col is None:
            continue
        w = str(memory.value(f"{col.address}.when") or "unknown")
        when.setdefault(w if w in when else "unknown", []).append(col.name)
    out.append(_fact("timing", "; ".join(f"{k}: {', '.join(v) if v else 'none'}" for k, v in when.items())))
    return out
