"""The notebook's recompute and plot cell sources execute and bind results.

Each cell is a Python source string the notebook runs top to bottom. These
tests exec them in a namespace shaped like the one the load cell builds (df +
frozen PLAN/SPEC/DAG), so a syntax or logic error in a cell fails here, fast,
before the slow notebook_verify integration run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis_v2.agents.report.plot_cells import DAG_FIGURE, FOREST, LOVE
from src.analysis_v2.agents.report.recompute_cells import BALANCE, ESTIMATE
from src.analysis_v2.spec import (
    CausalDAG,
    CausalEdge,
    CausalNode,
    CausalSpec,
    MethodLane,
    MethodPlan,
    QuestionType,
    VariableRef,
)


def _namespace() -> dict:
    """The globals the load cell binds, for an observational run."""
    plan = MethodPlan(
        lane=MethodLane.OBSERVATIONAL, estimator="regression_adjustment",
        estimand="ate", outcome="re78", treatment="treat", covariates=["age"],
    )
    spec = CausalSpec(
        question_type=QuestionType.BINARY_TREATMENT,
        outcome=VariableRef(column="re78"), treatment=VariableRef(column="treat"),
    )
    df = pd.DataFrame(
        {
            "treat": [0, 1] * 10,
            "age": list(range(20, 40)),
            "re78": [float(i) + (5.0 if i % 2 else 0.0) for i in range(20)],
        }
    )
    return {
        "pd": pd, "np": np, "df": df,
        "PLAN": plan.model_dump(mode="json"),
        "SPEC": spec.model_dump(mode="json"),
        "DAG": None,
    }


def test_estimate_cell_recomputes_a_finite_estimate():
    ns = _namespace()
    exec(ESTIMATE, ns)
    assert np.isfinite(ns["RESULT"].primary.estimate)
    assert len(ns["effects"]) >= 1
    assert ns["RESULT"].estimator == "regression_adjustment"


def test_forest_cell_plots_without_error():
    ns = _namespace()
    exec(ESTIMATE, ns)
    exec(FOREST, ns)  # matplotlib Agg; a broken source raises here


def _dag_dict() -> dict:
    """A small confounded DAG: age -> treat, age -> re78, treat -> re78."""
    dag = CausalDAG(
        nodes=[CausalNode(name="treat"), CausalNode(name="re78"), CausalNode(name="age")],
        edges=[
            CausalEdge(source="age", target="treat", mechanism="age shapes selection"),
            CausalEdge(source="age", target="re78", mechanism="age shapes earnings"),
            CausalEdge(source="treat", target="re78", mechanism="training effect"),
        ],
        treatment="treat", outcome="re78",
    )
    return dag.model_dump(mode="json")


def test_dag_figure_cell_renders_and_finds_the_adjustment_set():
    ns = _namespace()
    ns["DAG"] = _dag_dict()
    exec(DAG_FIGURE, ns)  # builds the figure; a broken source raises here
    assert "age" in ns["adj"]  # the confounder is on the backdoor adjustment set


def test_balance_cell_computes_before_smds_for_observational():
    ns = _namespace()
    exec(ESTIMATE, ns)
    exec(BALANCE, ns)
    bal = ns["BALANCE"]
    assert "smd_before" in bal.columns
    assert "age" in list(bal["covariate"])


def test_love_plot_cell_renders_without_error():
    ns = _namespace()
    exec(ESTIMATE, ns)
    exec(BALANCE, ns)
    exec(LOVE, ns)  # a broken source raises here
