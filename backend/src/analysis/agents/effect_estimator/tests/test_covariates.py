"""Tests for the covariate-resolution priority chain.

get_covariates_for_pair picks the adjustment set from the most-informed source
available: the DAG's backdoor set first, then confounder_discovery, then the
profiler's potential_confounders, then all numeric columns. The contract that
matters: a higher-priority source is never silently clobbered by a lower one.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd

from src.analysis.agents.effect_estimator.covariates import get_covariates_for_pair

_EIGHT = ["age", "black", "educ", "hispan", "married", "nodegree", "re74", "re75"]
_BINARY = {"black", "hispan", "married", "nodegree"}


def _df(extra: dict | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    cols: dict[str, np.ndarray] = {
        "treat": rng.integers(0, 2, n),
        "re78": rng.normal(6000, 100, n),
    }
    for c in _EIGHT:
        cols[c] = rng.integers(0, 2, n) if c in _BINARY else rng.normal(10, 2, n)
    if extra:
        cols.update(extra)
    return pd.DataFrame(cols)


def _agent(df: pd.DataFrame):
    return types.SimpleNamespace(_df=df, logger=types.SimpleNamespace(info=lambda *a, **k: None))


def _state(adjustment_set=None, edges=None, ranked=None, potential=None):
    dag = None
    if adjustment_set is not None or edges is not None:
        dag = types.SimpleNamespace(adjustment_set=adjustment_set, edges=edges or [])
    profile = types.SimpleNamespace(potential_confounders=potential) if potential is not None else None
    confounder_discovery = {"ranked_confounders": ranked} if ranked is not None else None
    return types.SimpleNamespace(
        proposed_dag=dag,
        confounder_discovery=confounder_discovery,
        data_profile=profile,
    )


def test_dag_adjustment_set_is_not_clobbered_by_profile_confounders():
    # Regression: Priority 0a resolved the DAG's 8-confounder backdoor set. A
    # non-empty profiler potential_confounders (the shorter list) must NOT
    # overwrite it. Before the fix, Priority 2's elif fired whenever Priority 1
    # (confounder_discovery) was absent and clobbered the 8 down to 3, so the
    # estimator silently ran on the profiler's stale set instead of the DAG's.
    covs = get_covariates_for_pair(
        _agent(_df()),
        _state(adjustment_set=list(_EIGHT), potential=["married", "nodegree", "re75"]),
        "treat",
        "re78",
    )
    assert set(covs) == set(_EIGHT)


def test_confounder_discovery_is_not_clobbered_by_profile_confounders():
    # Same guard one priority down: Priority 1's ranked confounders win over the
    # profiler's potential_confounders.
    covs = get_covariates_for_pair(
        _agent(_df()),
        _state(ranked=["age", "educ", "re74"], potential=["married"]),
        "treat",
        "re78",
    )
    assert set(covs) == {"age", "educ", "re74"}


def test_falls_back_to_profile_confounders_when_no_dag_or_discovery():
    covs = get_covariates_for_pair(
        _agent(_df()),
        _state(potential=["age", "educ"]),
        "treat",
        "re78",
    )
    assert set(covs) == {"age", "educ"}


def test_priority_3_uses_all_numeric_non_id_columns_when_nothing_upstream():
    df = _df(extra={"customer_id": np.arange(60)})
    covs = get_covariates_for_pair(_agent(df), _state(), "treat", "re78")
    assert set(covs) == set(_EIGHT)
    assert "customer_id" not in covs
    assert "treat" not in covs and "re78" not in covs
