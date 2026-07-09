"""Semantic review hashes, malformed responses and replay share durable allowances."""

from copy import deepcopy
from uuid import uuid4

import pytest
import test_analysis_tasks as task_tests
from historical_analysis.checkpoints import BudgetExhausted, Harness
from model_double import model_double
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session
from test_scientific_v8 import decision

from backend import model_access
from backend.analysis import prompts
from backend.analysis.contracts import CodeReview, Decision, GateCheck
from backend.db import engine
from backend.models import AnalysisRun, AnalysisTaskAttempt

PASS = GateCheck(rule="structure", kind="deterministic", passed=True, detail="Valid structure")
run_id = task_tests.run_id


def review_harness(db, identity):
    source = db.get(AnalysisRun, identity)
    run = AnalysisRun(
        id=uuid4(), conversation_id=source.conversation_id, version_id=source.version_id,
        author_id=source.author_id, submission_id=uuid4(), question=source.question, trace_id=uuid4(),
        inputs={**source.inputs, "model": "gemini-2.5-flash", "semantic_review": prompts.SEMANTIC_REVIEW},
    )
    db.add(run)
    db.commit()
    return Harness(db, run)


@pytest.mark.parametrize("phase", ["target", "plan", "report"])
def test_changed_candidate_hashes_cannot_mint_review_allowance(run_id, monkeypatch, phase):
    calls = []

    def infer(schema, prompt, payload, model, **kwargs):
        calls.append(deepcopy(payload))
        return CodeReview(accepted=len(calls) == 2, summary="Assess the exact requested family.",
                          issues=["The candidate includes an unrequested auxiliary effect."] if len(calls) == 1 else [])

    monkeypatch.setattr(model_access, "structured", model_double(infer))
    original = Decision(**decision())
    corrected = original.model_copy(update={"summary": "Corrected candidate"})
    with Session(engine()) as db:
        h = review_harness(db, run_id)
        run_id = h.run.id
        assert not h.review(phase, original, {}, [PASS])[-1].passed
        accepted = h.review(phase, corrected, {}, [PASS])
        assert accepted[-1].passed
        assert h.review(phase, corrected, {}, [PASS]) == accepted and len(calls) == 2
        with pytest.raises(BudgetExhausted, match="allowance"):
            h.review(phase, original.model_copy(update={"summary": "Third hash"}), {}, [PASS])
        assert len(calls) == 2 and h.run.shared_calls == 2
        attempts = list(db.scalars(select(AnalysisTaskAttempt).where(AnalysisTaskAttempt.run_id == run_id)
                                  .order_by(AnalysisTaskAttempt.number)))
        assert len(attempts) == 2 and {item.metrics["obligation_key"] for item in attempts} == {"review:" + phase}
        assert all(item.metrics["obligation_limit"] == 2 for item in attempts)
        assert all(item.status == "accepted" for item in attempts), "Provider verdicts are distinct from scientific gate acceptance."
        assert attempts[0].request["payload"]["candidate"] == original.model_dump(mode="json")
        assert attempts[1].request["payload"]["candidate"] == corrected.model_dump(mode="json")


@pytest.mark.parametrize("reject_count", [1, 2])
def test_malformed_reviewer_calls_stay_charged_and_raw_is_retained(run_id, monkeypatch, reject_count):
    raw = '{"accepted":' + "x" * 143_000
    calls = []

    def infer(schema, prompt, payload, model, **kwargs):
        calls.append(deepcopy(payload))
        if len(calls) <= reject_count:
            try:
                schema.model_validate_json(raw)
            except ValidationError as error:
                raise model_access.InvalidModelOutput(raw, error) from error
        return CodeReview(accepted=True, summary="The exact candidate follows the question.")

    monkeypatch.setattr(model_access, "structured", model_double(infer))
    proposal = Decision(**decision())
    with Session(engine()) as db:
        h = review_harness(db, run_id)
        run_id = h.run.id
        gates = h.review("target", proposal, {}, [PASS])
        assert gates[-1].passed is (reject_count == 1) and h.run.shared_calls == 2
        assert len(calls) == 2 and "review_correction" in calls[1] and raw not in str(calls[1])
        assert calls[0]["candidate"] == calls[1]["candidate"] == proposal.model_dump(mode="json")
        attempts = list(db.scalars(select(AnalysisTaskAttempt).where(AnalysisTaskAttempt.run_id == run_id)
                                  .order_by(AnalysisTaskAttempt.number)))
        assert [item.status for item in attempts] == ["rejected", "accepted" if reject_count == 1 else "rejected"]
        assert all(item.raw_output == raw for item in attempts[:reject_count])
        assert h.review("target", proposal, {}, [PASS]) == gates and len(calls) == 2
        with pytest.raises(BudgetExhausted):
            h.review("target", proposal.model_copy(update={"summary": "A fresh scientific candidate"}), {}, [PASS])
        assert len(calls) == 2
