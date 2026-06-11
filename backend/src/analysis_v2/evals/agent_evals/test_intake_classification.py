"""Live intake eval: one question per taxonomy type against the real LLM.

Opt-in (RUN_AGENTIC_EVALS=1): these hit the configured provider and cost
real tokens. Assertions are structural: the detected type lands in the
case's acceptable set, every resolved column exists in the schema, and
the clear-cut cases map the right roles. Sample questions come from
representative_cases.yaml; column lists come from the committed fixture
files (or the verified column lists for download-only datasets).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.analysis_v2.agents.base import AgentCtx
from src.analysis_v2.agents.intake import IntakeAgent
from src.analysis_v2.core import AnalysisRunState, GateStatus

pytestmark = [
    pytest.mark.agentic,
    pytest.mark.skipif(
        os.environ.get("RUN_AGENTIC_EVALS") != "1",
        reason="live LLM eval; set RUN_AGENTIC_EVALS=1 to run",
    ),
]

EVALS_DIR = Path(__file__).resolve().parents[1]
DATA = EVALS_DIR / "fixtures" / "data"
MANIFEST = yaml.safe_load((EVALS_DIR / "representative_cases.yaml").read_text())
CASES = {c["case_id"]: c for c in MANIFEST["cases"]}

# Verified column lists for datasets too large to commit (fixtures README).
TELCO_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]
HILLSTROM_COLUMNS = [
    "recency", "history_segment", "history", "mens", "womens", "zip_code",
    "newbie", "channel", "segment", "visit", "conversion", "spend",
]
ROSSMANN_COLUMNS = ["Store", "DayOfWeek", "Date", "Sales", "Customers", "Open",
                    "Promo", "StateHoliday", "SchoolHoliday"]


def _frame_from_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / name, nrows=20)


def _frame_from_columns(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: ["x", "y"] for c in columns})


def _generated(case_id: str) -> pd.DataFrame:
    from src.analysis_v2.evals.fixtures.generators import GENERATORS

    return GENERATORS[case_id]().head(20)


# (case_id, frame source, acceptable types, required role -> column)
EVAL_GRID = [
    ("lalonde-nsw-experimental", lambda: _frame_from_csv("lalonde_nsw.csv"),
     {"simple_effect", "binary_treatment"}, {"outcome": "re78", "treatment": "treat"}),
    ("lalonde-observational", lambda: _frame_from_csv("lalonde.csv"),
     {"binary_treatment", "simple_effect"}, {"outcome": "re78", "treatment": "treat"}),
    ("advertising-dose-response", lambda: _frame_from_csv("advertising.csv"),
     {"dose_response", "simple_effect"}, {}),
    ("student-multi-factor", lambda: _frame_from_csv("student_data.csv"),
     {"multi_factor"}, {"outcome": "G3"}),
    ("insurance-interaction", lambda: _frame_from_csv("insurance.csv"),
     {"interaction", "heterogeneous_effects"}, {"outcome": "charges"}),
    ("synthetic-mediation", lambda: _generated("synthetic-mediation"),
     {"mediation"}, {"mediator": "weight_change"}),
    ("card-krueger-did", lambda: _frame_from_csv("employment.csv"),
     {"did", "before_after"}, {}),
    ("bank-recovery-rdd", lambda: _frame_from_csv("bank_data.csv"),
     {"rdd"}, {"running_variable": "expected_recovery_amount"}),
    ("card-proximity-iv", lambda: _frame_from_csv("card_schooling_wages.csv"),
     {"iv"}, {"instrument": "nearc4"}),
    ("website-visitors-its-step", lambda: _frame_from_csv("daily_website_visitors.csv"),
     {"time_series_intervention", "before_after"}, {}),
    ("heart-failure-survival", lambda: _frame_from_csv("heart_failure.csv"),
     {"survival"}, {}),
    ("rossmann-before-after", lambda: _frame_from_columns(ROSSMANN_COLUMNS),
     {"before_after", "time_series_intervention", "did"}, {}),
    ("hillstrom-hte", lambda: _frame_from_columns(HILLSTROM_COLUMNS),
     {"heterogeneous_effects"}, {}),
    ("telco-churn-drivers", lambda: _frame_from_columns(TELCO_COLUMNS),
     {"driver_analysis", "multi_factor"}, {"outcome": "Churn"}),
    ("hillstrom-null-subgroup", lambda: _frame_from_columns(
        [c for c in HILLSTROM_COLUMNS if c not in ("segment", "mens", "womens")]
        + ["received_email"]),
     # "did X fail to increase Y" is honestly any of these three framings
     {"no_effect", "binary_treatment", "simple_effect"}, {}),
    ("student-por-mechanism", lambda: _frame_from_csv("student_por.csv"),
     {"mechanism_search", "mediation"}, {}),
]


@pytest.mark.parametrize("case_id,frame_fn,acceptable,roles", EVAL_GRID,
                         ids=[g[0] for g in EVAL_GRID])
async def test_intake_classifies_and_maps_without_inventing_columns(
    case_id, frame_fn, acceptable, roles, tmp_path, monkeypatch
):
    monkeypatch.setenv("LOCAL_STORAGE_PATH", str(tmp_path))
    from src.config import settings as settings_mod

    settings_mod.get_settings.cache_clear()
    case = CASES[case_id]
    frame = frame_fn()
    run = AnalysisRunState(job_id=f"eval-{case_id}", causal_question=case["causal_question"])
    ctx = AgentCtx(job_id=f"eval-{case_id}", run=run, frame=frame)

    result = await IntakeAgent().execute(ctx)
    settings_mod.get_settings.cache_clear()

    assert result.gate.status == GateStatus.ADVANCE, result.gate
    spec = result.output
    got = {spec.question_type.value} | {t.value for t in spec.type_candidates}
    assert got & acceptable, (
        f"{case_id}: detected {spec.question_type.value} "
        f"(candidates {sorted(got)}), acceptable {sorted(acceptable)}"
    )

    # no invention: every resolved reference must be a real column
    columns = set(map(str, frame.columns))
    for label, ref in {
        "outcome": spec.outcome, "treatment": spec.treatment,
        "mediator": spec.mediator, "instrument": spec.instrument,
        "running_variable": spec.running_variable, "time": spec.time_column,
        "group": spec.group_column, "event": spec.event_column,
        "duration": spec.duration_column,
    }.items():
        if ref is not None and ref.column is not None:
            assert ref.column in columns, f"{case_id}: invented {label}={ref.column}"
        if ref is not None:
            for cand in ref.candidates:
                assert cand in columns, f"{case_id}: invented candidate {cand}"

    # clear-cut role mappings: resolved to the expected column, or at least
    # surfaced as a candidate / missing-info item
    for role, expected_col in roles.items():
        ref = getattr(spec, role if role != "time" else "time_column")
        ok = ref is not None and (
            ref.column == expected_col or expected_col in ref.candidates
        )
        assert ok or spec.missing_info, (
            f"{case_id}: expected {role}={expected_col}, "
            f"got {ref and (ref.column, ref.candidates)}"
        )
