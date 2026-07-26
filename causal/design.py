"""Which designs this dataset can support, and why the others cannot.

This is the honest core of the tool. Run backwards it is a veto, checking
whether a chosen design is allowed. Run forwards, which is how it is used here,
it produces a menu: everything the data can identify, each with the assumption
it rests on, so a person picks with the tradeoff in front of them.

Two kinds of design, and the difference matters:

  detectable  - the data alone shows it is available (a two-group two-period
                panel supports DiD; a 0/1 event column supports survival)
  needs input - the data cannot reveal it. No column announces itself as a
                valid instrument; that is domain knowledge someone brings.

A design in the second group is offered as a candidate awaiting a named column,
never silently ruled out for lack of evidence that could not exist.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .profile import Profile


@dataclass
class Option:
    lane: str
    available: bool
    reason: str
    assumption: str
    needs: dict[str, str] = field(default_factory=dict)  # param -> what to supply

    def describe(self) -> str:
        mark = "yes" if self.available else "no "
        line = f"[{mark}] {self.lane}: {self.reason}"
        if self.needs:
            line += f"\n      needs: {', '.join(f'{k} ({v})' for k, v in self.needs.items())}"
        if self.available:
            line += f"\n      assumes: {self.assumption}"
        return line


# Each design's identifying assumption, stated once so the UI and the report
# cannot drift from each other.
ASSUMPTION = {
    "observational": "the covariates close every backdoor path (untestable)",
    "matching": "treated and control overlap on the covariates, and those "
                "covariates are all the confounding there is",
    "did": "parallel trends: both groups would have moved together absent "
           "the intervention",
    "rdd": "nothing else changes at the cutoff, and units cannot precisely "
           "control which side they land on",
    "iv": "the instrument moves treatment and affects the outcome by no other "
          "route (untestable)",
    "survival": "the hazard ratio is constant over time",
    "mediation": "no unmeasured confounding of treatment-mediator or "
                 "mediator-outcome",
    "time_series": "nothing else happened at the intervention point",
}


def find_panel(df, p: Profile) -> tuple[str, str, str] | None:
    """(unit, period, group) if this is a two-period panel, else None.

    A panel is repeated observations of the same units, which only the data can
    show. Two two-valued columns are not enough: a table of patients has
    several 0/1 columns and is still one row per person.
    """
    two_valued = [c.name for c in p.columns if c.n_unique == 2]
    for period in two_valued:
        counts = df[period].value_counts()
        if len(counts) != 2 or counts.min() < 10:
            continue
        for unit in p.names():
            if unit == period:
                continue
            n_units = df[unit].nunique()
            # each unit seen about once per period, and pairs nearly unique
            if not (0.4 * len(df) <= n_units <= 0.6 * len(df)):
                continue
            if df.duplicated([unit, period]).mean() > 0.05:
                continue
            for group in two_valued:
                if group in (period, unit):
                    continue
                # Units should sit in one group, but not perfectly: Card &
                # Krueger's own store ids repeat across states for a couple of
                # stores out of 409. Demanding purity here rejects real panels.
                if (df.groupby(unit)[group].nunique() == 1).mean() > 0.95:
                    return unit, period, group
    return None


def options(
    df,
    p: Profile,
    *,
    treatment: str | None = None,
    outcome: str | None = None,
) -> list[Option]:
    """Every design, with a verdict and the reason for it."""
    binary = set(p.binary_names())
    numeric = set(p.numeric_names())
    dates = p.datelike_names()
    out = []

    ok = bool(treatment and outcome and treatment in numeric and outcome in numeric)
    out.append(Option(
        "observational", ok,
        "always available once treatment and outcome are numeric" if ok
        else "needs a numeric treatment and outcome",
        ASSUMPTION["observational"],
    ))

    covs = [c for c in numeric if c not in (treatment, outcome)]
    ok = bool(treatment in binary and len(covs) >= 1 and p.n_rows >= 40)
    out.append(Option(
        "matching", ok,
        f"treatment is binary and {len(covs)} numeric covariates are available" if ok
        else "needs a binary treatment and at least one numeric covariate",
        ASSUMPTION["matching"],
    ))

    panel = find_panel(df, p)
    out.append(Option(
        "did", panel is not None,
        f"a two-period panel: unit '{panel[0]}', period '{panel[1]}', "
        f"group '{panel[2]}'" if panel
        else "not a panel: no column identifies units observed in both periods",
        ASSUMPTION["did"],
        {} if panel else {"group": "a column with 2 values", "period": "before/after"},
    ))

    ok = bool(dates) and p.n_rows >= 30
    out.append(Option(
        "time_series", ok,
        f"date column(s) {dates} over {p.n_rows} rows" if ok
        else "needs a date column and 30+ time points",
        ASSUMPTION["time_series"],
        {} if ok else {"time": "a date column"},
    ))

    # Four designs the data cannot confirm on its own. Nearly every table has a
    # 0/1 column and a non-negative number, so calling that "survival available"
    # is noise. Suggest candidates, let a person decide.
    events = [c for c in sorted(binary) if c != treatment][:4]
    durations = [
        c for c in sorted(numeric)
        if c not in (treatment, outcome) and (p[c].low or 0) >= 0 and p[c].n_unique > 10
    ][:4]
    out.append(Option(
        "survival", False,
        "candidate: needs a follow-up time and an event flag, which no column "
        "announces itself as",
        ASSUMPTION["survival"],
        {"duration": f"maybe {durations}" if durations else "time observed",
         "event": f"maybe {events}" if events else "0/1 column"},
    ))

    continuous = [
        c for c in sorted(numeric)
        if c not in (treatment, outcome) and p[c].n_unique > 20
    ][:4]
    for lane, need in (
        ("rdd", {"running": f"maybe {continuous}" if continuous else "a continuous column",
                 "cutoff": "the threshold value"}),
        ("iv", {"instrument": "a column that moves treatment and nothing else"}),
        ("mediation", {"mediator": "a column on the causal path"}),
    ):
        out.append(Option(
            lane, False,
            "candidate: no column announces itself, so this needs your knowledge",
            ASSUMPTION[lane], need,
        ))

    return out


def available(df, p: Profile, **kw) -> list[str]:
    return [o.lane for o in options(df, p, **kw) if o.available]


def as_text(df, p: Profile, **kw) -> str:
    return "\n".join(o.describe() for o in options(df, p, **kw))
