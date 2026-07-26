"""Propose the arguments a lane needs, so a stranger's dataset is runnable.

Reading a question gives you a treatment and an outcome. Every lane needs more
than that: covariates to adjust for, a duration and an event, an instrument, a
mediator. On a dataset you have seen before you type those in. On one you have
never seen, typing them in means reading 40 column names first, and the tool
has failed you.

So this proposes a complete argument set. Deterministic wherever the data can
decide it, and every proposal is shown and editable before anything runs. A
suggestion is a starting point, not a decision.
"""
from __future__ import annotations

import re

from .profile import Profile

# Columns that are identifiers rather than measurements. Adjusting for a row id
# is meaningless and can absorb real variation, so they never become covariates.
ID_LIKE = re.compile(
    r"^(unnamed.*|.*_?id|id_?.*|index|row|key|uuid|code|no|number)$", re.I
)


def _is_id(name: str, p: Profile, n_rows: int) -> bool:
    col = p[name]
    if ID_LIKE.match(name.strip()):
        return True
    # near-unique integers are an index whatever they are called
    return col.numeric and col.n_unique > 0.95 * n_rows and not col.binary


LEAK_CORR = 0.95


def covariates(
    p: Profile,
    exclude: set[str],
    df=None,
    outcome: str = "",
    limit: int = 12,
) -> list[str]:
    """Numeric columns worth adjusting for, most complete first.

    Numeric only, because the lanes coerce and a text column would silently
    drop every row. Heavily missing columns are skipped for the same reason.

    The important filter is the last one. Give it the frame and the outcome and
    it drops anything correlating above 0.95 with the outcome, because that is
    almost never a confounder: it is the outcome restated. Card's data carries
    both `wage` and `lwage`, and adjusting the log-wage regression for wage
    would produce a confident, meaningless number. On a dataset you have read
    you would catch that. On a stranger's you would not.
    """
    out = [
        c.name
        for c in p.columns
        if c.numeric
        and c.name not in exclude
        and c.missing < 0.3
        and not _is_id(c.name, p, p.n_rows)
        and c.n_unique > 1
    ]
    if df is not None and outcome and outcome in getattr(df, "columns", []):
        import pandas as pd

        y = pd.to_numeric(df[outcome], errors="coerce")
        kept = []
        for name in out:
            x = pd.to_numeric(df[name], errors="coerce")
            pair = pd.DataFrame({"x": x, "y": y}).dropna()
            if len(pair) > 10:
                # Rank correlation, not just linear. A log or a rescaling is
                # monotonic but not linear: wage against log wage is Pearson
                # 0.947, which slips under the threshold, and Spearman exactly
                # 1.0, which does not. A real confounder like experience sits
                # near 0.04 on both, so nothing legitimate is caught.
                linear = abs(pair.x.corr(pair.y))
                monotone = abs(pair.x.corr(pair.y, method="spearman"))
                if max(linear, monotone) > LEAK_CORR:
                    continue  # a restatement of the outcome, not a confounder
            kept.append(name)
        out = kept
    out.sort(key=lambda n: (p[n].missing, -p[n].n_unique))
    return out[:limit]


def _first(names: list[str], *patterns: str) -> str:
    for pattern in patterns:
        for name in names:
            if re.search(pattern, name, re.I):
                return name
    return ""


def instrument_candidates(p: Profile, names: list[str], spoken: set[str]) -> list[str]:
    """Columns whose *name* hints at an instrument. Never auto-selected.

    An instrument must move treatment, which is testable, and reach the outcome
    by no other route, which is not. It is tempting to rank candidates by
    first-stage strength since that half is measurable. Do not: the columns
    with the strongest first stage are the ones mechanically tied to treatment,
    and those violate the exclusion restriction worst.

    Measured on Card's data, ranking by first stage picks `exper` (roughly
    age minus schooling minus six) and returns -0.006 against a published 0.13.
    The correct instrument, `nearc4`, ranks nowhere near the top. Optimising
    the testable half actively selects against the half that matters.

    So: offer names, select nothing.
    """
    # Only genuinely generic words. An earlier version had `^near`, `draft` and
    # `quarter` in here, which are not rules: they are the answers to Card 1995,
    # the Vietnam lottery and Angrist's quarter-of-birth, memorised from the
    # eval set. They would fire on nothing else. The eight datasets are a test
    # suite, and a test suite you tune against stops measuring anything.
    #
    # This list is now a weak fallback. Instrument candidates properly come
    # from roles.py, which reasons about the data in front of it.
    hints = (r"instrument", r"assign", r"lottery", r"random", r"eligib")
    out = []
    for pattern in hints:
        for name in names:
            if name not in spoken and name not in out and re.search(pattern, name, re.I):
                out.append(name)
    return out[:6]


def for_lane(lane: str, p: Profile, intake: dict, df=None, roles=None) -> dict:
    """A complete, editable argument set for `lane` on this dataset.

    With `roles` (from roles.read_roles) the covariates are the columns reasoned
    to be confounders, and mediators, colliders and outcome proxies are left
    out. Without it, the deterministic rules below apply: every numeric column
    that is not an identifier and does not correlate with the outcome. That
    fallback is worse in a specific way, and it is worth naming: it cannot tell
    a confounder from a mediator, so it will adjust away part of the effect it
    is measuring.
    """
    treatment = intake.get("treatment") or ""
    outcome = intake.get("outcome") or ""
    names = p.names()
    spoken = {treatment, outcome} - {""}

    reasoned = bool(roles and roles.by_column)

    def controls(extra_exclude: set[str] = frozenset(), for_outcome: str = "") -> list[str]:
        """Confounders when we have reasoning; the greedy rule when we do not."""
        if reasoned:
            return [c for c in roles.confounders() if c not in extra_exclude][:12]
        return covariates(p, spoken | set(extra_exclude), df, for_outcome or outcome)

    if lane in ("observational", "matching"):
        return {"outcome": outcome, "treatment": treatment,
                "covariates": controls()}

    if lane == "iv":
        # Instrument blank on purpose (see instrument_candidates), and
        # covariates blank on purpose too.
        #
        # Suggesting controls here is not merely imprecise, it is unsafe. An
        # instrument is usually something outside the causal system, and a
        # greedy control set reaches straight for its neighbours. Measured on
        # Card's data with nearc4: no controls gives a first-stage F of 63.9,
        # the paper's five controls give 16.7, and a greedy twelve including
        # the regional dummies drives it to nothing and the lane refuses. The
        # instrument is geographic; controlling for geography deletes it.
        #
        # So IV starts bare. Adding a control is a decision, and it belongs to
        # someone who knows why the instrument is valid.
        candidates = (roles.named("instrument") if reasoned
                      else instrument_candidates(p, names, spoken))
        return {"outcome": outcome, "treatment": treatment, "instrument": "",
                "covariates": [],
                "_candidates": {
                    "instrument": candidates,
                    "covariates": controls(set(candidates)),
                }}

    if lane == "mediation":
        found = roles.named("mediator") if reasoned else []
        guess = found[0] if found else _first(
            [n for n in names if n not in spoken], r"mediat", r"channel", r"pathway")
        return {"outcome": outcome, "treatment": treatment, "mediator": guess,
                "covariates": controls({guess}),
                "_candidates": {"mediator": found}}

    if lane == "survival":
        event = _first(
            [c.name for c in p.columns if c.binary and c.name not in {treatment}],
            r"death|died|event|status|churn|fail|censor")
        # "how long until X" makes the outcome the duration, so try it first
        duration = ""
        if outcome and outcome in p.names():
            col = p[outcome]
            if col.numeric and (col.low or 0) >= 0 and outcome != event:
                duration = outcome
        duration = duration or _first(
            [c.name for c in p.columns
             if c.numeric and c.name not in {treatment, event} and (c.low or 0) >= 0],
            r"time|dur|days|months|tenure|followup|follow_up")
        # the event must never become a covariate: it is the model's outcome
        return {"treatment": treatment, "duration": duration, "event": event,
                "covariates": controls({event, duration}, duration)}

    if lane == "did":
        return {"outcome": outcome,
                "group": intake.get("group", ""),
                "period": intake.get("period", ""),
                "treated_group": "",  # required, and only a person knows which
                "unit": _first(names, r"_id$|^id$|unit|store|firm|person|subject")}

    if lane == "rdd":
        running = intake.get("running_variable") or _first(
            [c.name for c in p.columns if c.numeric and c.n_unique > 20 and c.name not in spoken],
            r"score|amount|rank|distance|margin|threshold")
        cutoff = intake.get("cutoff")
        return {"outcome": outcome, "running": running,
                "cutoff": "" if cutoff is None else cutoff}

    if lane == "time_series":
        time = intake.get("time_column") or _first(
            [c.name for c in p.columns if c.datelike], r".")
        return {"outcome": outcome, "time": time,
                "intervention": ""}  # a date only a person knows

    return {}


def all_lanes(p: Profile, intake: dict, df=None, roles=None) -> dict[str, dict]:
    """Suggestions for every lane, so the form is filled the moment one is picked."""
    return {
        lane: for_lane(lane, p, intake, df, roles)
        for lane in ("observational", "matching", "iv", "did", "rdd",
                     "survival", "mediation", "time_series")
    }
