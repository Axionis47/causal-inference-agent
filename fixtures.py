"""Every lane paired with a real dataset and, where one exists, a published number.

Nothing here is generated. Each frame is a CSV in data/, and each expected value
is either a figure from the literature or a truth computable from the data
itself. That is what stops these checks going stale: they were never recorded
from a previous run of this code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from causal import lanes

DATA = Path(__file__).parent / "data"


@dataclass
class Case:
    name: str
    run: Callable[..., object]
    frame: pd.DataFrame
    kwargs: dict
    # (label, expected, relative band) — for a single published point estimate.
    checks: list[tuple[str, float, float]] = field(default_factory=list)
    # (label, low, high) — for a published *range*, where a midpoint plus a
    # percentage would quietly accept values outside it.
    ranges: list[tuple[str, float, float]] = field(default_factory=list)
    why: str = ""


def _ihdp() -> Case:
    df = pd.read_csv(DATA / "ihdp.csv")
    truth = float((df.mu1 - df.mu0).mean())  # true potential-outcome means
    return Case(
        "observational", lanes.observational, df,
        dict(outcome="y_factual", treatment="treatment",
             covariates=tuple(c for c in df.columns if c.startswith("x"))),
        checks=[("ate", truth, 0.10)],
        why=f"IHDP ships mu0/mu1, so the true ATE is knowable: {truth:.4f}",
    )


def _lalonde() -> Case:
    df = pd.read_csv(DATA / "lalonde.csv")
    return Case(
        "matching", lanes.matching, df,
        dict(outcome="re78", treatment="treat",
             covariates=("age", "educ", "black", "hispan", "married",
                         "nodegree", "re74", "re75")),
        ranges=[("att", 1000.0, 2200.0)],
        why="NSW treated vs PSID controls: the naive difference is -635; matching "
            "that truly adjusts reaches the +1794 experimental benchmark",
    )


def _card_iv() -> Case:
    return Case(
        "iv", lanes.iv, pd.read_csv(DATA / "card.csv"),
        dict(outcome="lwage", treatment="educ", instrument="nearc4",
             covariates=("exper", "expersq", "black", "south", "smsa")),
        checks=[("late", 0.13, 0.25)],
        why="Card 1995 college-proximity IV: published 2SLS return to schooling ~0.13",
    )


def _heart_failure() -> Case:
    return Case(
        "survival", lanes.survival, pd.read_csv(DATA / "heart_failure.csv"),
        dict(treatment="high_blood_pressure", duration="time", event="DEATH_EVENT"),
        checks=[("hazard_ratio", 1.5455777143424887, 1e-9)],
        why="Pinned against the previous engine's value: two independent "
        "implementations must agree exactly",
    )


def _card_krueger() -> Case:
    return Case(
        "did", lanes.did, pd.read_csv(DATA / "card_krueger.csv"),
        dict(outcome="fte", group="state", period="period",
             treated_group="NJ", unit="store_id"),
        checks=[("att", 2.76, 0.15)],
        why="Card & Krueger 1994 NJ/PA minimum wage: published DiD ~ +2.76 FTE",
    )


def _bank_rdd() -> Case:
    return Case(
        "rdd", lanes.rdd, pd.read_csv(DATA / "bank.csv"),
        dict(outcome="actual_recovery_amount",
             running="expected_recovery_amount", cutoff=1000.0),
        why="Recovery strategy steps up at an expected amount of 1000; no "
        "published effect size, so this checks the jump is found and finite",
    )


def _student_mediation() -> Case:
    return Case(
        "mediation", lanes.mediation, pd.read_csv(DATA / "student.csv"),
        dict(outcome="G3", treatment="studytime", mediator="failures"),
        why="No published estimate for studytime -> failures -> grade, so this "
        "asserts sane output only",
    )


def _visitors_its() -> Case:
    return Case(
        "time_series", lanes.time_series, pd.read_csv(DATA / "visitors.csv"),
        dict(outcome="Unique.Visits", time="Date", intervention="2018-01-01"),
        why="No known intervention in this series. The cut date is arbitrary, so "
        "the honest expectation is an interval spanning zero",
    )


def cases() -> list[Case]:
    return [_ihdp(), _lalonde(), _card_iv(), _heart_failure(),
            _card_krueger(), _bank_rdd(), _student_mediation(), _visitors_its()]


def refusals() -> list[tuple[str, Callable[[], object], str]]:
    """Cases where a lane must refuse. Each returns (name, thunk, expected text)."""
    def noise_instrument():
        df = pd.read_csv(DATA / "card.csv").copy()
        # a column with no relationship to schooling: the first stage must fail
        df["noise"] = df["id"] % 7
        return lanes.iv(df, outcome="lwage", treatment="educ", instrument="noise")

    def thin_arm():
        df = pd.read_csv(DATA / "lalonde.csv")
        thin = pd.concat([df[df.treat == 1].head(5), df[df.treat == 0]])
        return lanes.matching(thin, outcome="re78", treatment="treat",
                              covariates=("age", "educ", "re74"))

    def cutoff_outside():
        df = pd.read_csv(DATA / "bank.csv")
        return lanes.rdd(df, outcome="actual_recovery_amount",
                         running="expected_recovery_amount", cutoff=10_000_000.0)

    return [
        ("iv rejects a dead instrument", noise_instrument, "does not move treatment"),
        ("matching rejects a 5-unit arm", thin_arm, "10+ units per arm"),
        ("rdd rejects an out-of-range cutoff", cutoff_outside, "outside the range"),
    ]
