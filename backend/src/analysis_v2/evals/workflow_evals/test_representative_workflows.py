"""Full-spine workflow evals over the representative fixtures.

Each case stages its dataset like the input slice would, stubs only the
LLM (a per-case intake draft from the manifest's expected fields), and
drives all ten agents through POST-run semantics: S1..S11 plus the
terminal S12, ending in a verified notebook. Assertions are structural
and truth-banded per the manifest; exact coefficients are asserted only
for synthetic ground truth. Hermetic: no Kaggle, no network.

The shared machinery lives in harness.py; the sibling
test_representative_types.py covers the taxonomy types added later.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.analysis_v2.agents.intake.schema import IntakeDraft
from src.analysis_v2.core import AnalysisStage, RunStatus
from src.analysis_v2.evals.fixtures import generators
from src.analysis_v2.spec import Confidence, MethodLane, QuestionType
from src.analysis_v2.state import JobStatus

from .harness import DATA, Case, drive, stub_llm

CASES = [
    Case(
        case_id="synthetic-did-panel",
        frame_fn=generators.did_panel,
        question="Did the program raise outcomes for the treated group after period 4?",
        draft=IntakeDraft(
            question_type=QuestionType.DID, confidence=Confidence.HIGH,
            outcome_column="outcome",
            treatment_column=None, treatment_derived=True,
            treatment_clue="treated group after period 4",
            time_column="period", group_column="group",
            reasoning_summary="A policy-style group/time comparison.",
        ),
        expected_lane=MethodLane.DID,
        estimand="att",
        truth_band=(1.4, 2.6),
    ),
    Case(
        case_id="synthetic-iv-late",
        frame_fn=generators.synthetic_iv,
        question="Does x increase y, using z as an instrument?",
        draft=IntakeDraft(
            question_type=QuestionType.IV, confidence=Confidence.HIGH,
            outcome_column="y", treatment_column="x", instrument_column="z",
            candidate_confounders=["c1", "c2"],
            reasoning_summary="An instrumented effect question.",
        ),
        expected_lane=MethodLane.IV,
        estimand="late",
        truth_band=(1.65, 2.35),
    ),
    Case(
        case_id="synthetic-mediation",
        frame_fn=generators.mediation,
        question="Does the exercise program reduce disease risk through weight change?",
        draft=IntakeDraft(
            question_type=QuestionType.MEDIATION, confidence=Confidence.HIGH,
            outcome_column="disease_risk", treatment_column="exercise_program",
            mediator_column="weight_change",
            candidate_confounders=["baseline_health"],
            reasoning_summary="A pathway question through a mediator.",
        ),
        expected_lane=MethodLane.MEDIATION,
        estimand="total",
        truth_band=(0.45, 0.75),
        extra_warning="timing",
    ),
    Case(
        case_id="heart-failure-survival",
        frame_fn=lambda: pd.read_csv(DATA / "heart_failure.csv"),
        question="Does higher ejection fraction delay death in heart failure patients?",
        draft=IntakeDraft(
            question_type=QuestionType.SURVIVAL, confidence=Confidence.HIGH,
            outcome_column="time", treatment_column="ejection_fraction",
            duration_column="time", event_column="DEATH_EVENT",
            candidate_confounders=["age", "serum_creatinine"],
            reasoning_summary="A time-to-event question with censoring.",
        ),
        expected_lane=MethodLane.SURVIVAL,
        estimand="hazard_ratio",
        truth_band=(0.5, 0.99),  # protective: hr below 1
    ),
    Case(
        case_id="advertising-dose-response",
        frame_fn=lambda: pd.read_csv(DATA / "advertising.csv").drop(columns=["Unnamed: 0"]),
        question="Does more TV advertising spend increase sales?",
        draft=IntakeDraft(
            question_type=QuestionType.DOSE_RESPONSE, confidence=Confidence.HIGH,
            outcome_column="Sales ($)", treatment_column="TV Ad Budget ($)",
            treatment_is_continuous=True,
            candidate_confounders=["Radio Ad Budget ($)", "Newspaper Ad Budget ($)"],
            reasoning_summary="A continuous dose-response question.",
        ),
        expected_lane=MethodLane.OBSERVATIONAL,
        estimand="ate",
        truth_band=(0.035, 0.06),  # textbook 0.046
    ),
    Case(
        case_id="website-visitors-its-step",
        frame_fn=lambda: generators.website_visitors_step(
            DATA / "daily_website_visitors.csv"
        ),
        question="Did the site change on 2019-04-01 increase daily visits?",
        draft=IntakeDraft(
            question_type=QuestionType.TIME_SERIES_INTERVENTION,
            confidence=Confidence.HIGH,
            outcome_column="visits", time_column="Date",
            intervention_date="2019-04-01",
            treatment_derived=True, treatment_clue="post-change period",
            reasoning_summary="A single-series intervention question.",
        ),
        expected_lane=MethodLane.TIME_SERIES,
        estimand="level_shift",
        truth_band=(200.0, 1400.0),
        extra_warning="before/after",
    ),
]


@pytest.mark.parametrize("case", CASES, ids=[c.case_id for c in CASES])
async def test_representative_case_completes_with_a_verified_notebook(
    case: Case, storage_dir, monkeypatch
):
    stub_llm(monkeypatch, case.draft)
    run, state = await drive(f"wf-{case.case_id[:12]}", case.frame_fn(), case.question)

    assert run.status == RunStatus.COMPLETED, run.error_message
    assert run.current_state == AnalysisStage.S12_JOB_COMPLETE
    assert state.status == JobStatus.COMPLETED

    # the right lane ran and the estimate lands in the truth band
    assert run.estimate_result.lane == case.expected_lane
    effect = next(
        e for e in run.estimate_result.effects if e.estimand == case.estimand
    )
    if case.truth_band is not None:
        lo, hi = case.truth_band
        assert lo <= effect.estimate <= hi, (case.case_id, effect.estimate)

    # honesty surfaces: claims bounded, limitations present
    assert run.claim_critique is not None
    assert run.claim_critique.limitations
    if case.extra_warning is not None:
        joined = " ".join(run.claim_critique.limitations).lower()
        assert case.extra_warning in joined

    # the notebook executed and the artifact files exist on disk
    assert run.notebook_verification.notebook_status.value == "verified_running"
    for artifact in run.artifact_registry.artifacts:
        assert (storage_dir / run.job_id / "analysis" / artifact.path).exists()


async def test_a_completed_job_reopens_through_the_api_with_full_tiles(
    storage_dir, monkeypatch
):
    case = CASES[0]
    stub_llm(monkeypatch, case.draft)
    run, _ = await drive("wf-reopen", case.frame_fn(), case.question)
    assert run.status == RunStatus.COMPLETED

    from fastapi.testclient import TestClient

    from src.api.main import app

    client = TestClient(app)
    view = client.get("/jobs/wf-reopen/analysis").json()
    assert view["status"] == "completed"
    assert view["current_state"] == "s12_job_complete"
    assert len(view["agents"]) == 14  # incl. the S0A assembly planner
    assert all(a["status"] in ("passed", "warning") for a in view["agents"])
    assert view["notebook"]["status"] == "verified_running"
    assert view["costs"]["total_tool_calls"] >= 0

    # artifact bytes serve through the same surface the tiles link to
    notebook_bytes = client.get(
        "/jobs/wf-reopen/analysis/artifacts/notebook/verified"
    )
    assert notebook_bytes.status_code == 200
    assert b"Causal analysis notebook" in notebook_bytes.content
