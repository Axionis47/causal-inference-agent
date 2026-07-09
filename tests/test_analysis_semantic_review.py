"""Semantic review is a bounded, immutable extra gate over the same scoped evidence."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from historical_analysis.checkpoints import Harness
from pydantic import ValidationError
from test_scientific_v8 import decision

from backend import model_access
from backend.analysis import prompts
from backend.analysis.contracts import CodeReview, Decision, GateCheck


@pytest.fixture(autouse=True)
def isolated():
    yield


PASS = GateCheck(rule="structure", kind="deterministic", passed=True, detail="Valid structure")


def harness():
    h = Harness.__new__(Harness)
    h.saved = {}
    h.run = SimpleNamespace(question="Estimate only the requested local effect with 95% family confidence.",
                            inputs={"semantic_review": prompts.SEMANTIC_REVIEW})
    return h


@pytest.mark.parametrize("phase", ["target", "plan", "report"])
@pytest.mark.parametrize("accepted,issues,passed", [(True, [], True), (False, ["Unrequested auxiliary effects"], False),
                                                   (True, ["Family error probability conflicts with requested confidence"], False),
                                                   (False, [], False)])
def test_review_uses_same_evidence_and_preserves_candidate(phase, accepted, issues, passed):
    h = harness()
    proposal = Decision(**decision())
    packet = {"question": "Caller text cannot replace the exact question", "decisions": {"target": {"summary": "Pinned target"}},
              "evidence_records": [{"key": "source", "statement": {"value": "Scoped evidence"}}],
              "_context": {"withheld": "Private material"}, "correction": {"old": "Rejected proposal"}}
    if phase == "report":
        packet.update(snapshot={"plan": {"confidence": "Prespecified family coverage"}}, execution={"valid": False})
    original, original_packet = proposal.model_dump(), deepcopy(packet)
    calls = []

    def ask(key, schema, prompt, offered):
        calls.append((key, deepcopy(offered)))
        assert schema is CodeReview and prompt == prompts.SEMANTIC_REVIEW
        return CodeReview(accepted=accepted, summary="Exact semantic assessment", issues=issues)

    h.ask = ask
    gates = h.review(phase, proposal, packet, [PASS])
    assert len(calls) == 1 and calls[0][0].startswith(f"review:{phase}:")
    offered = calls[0][1]
    assert offered["candidate"] == original and offered["question"] == h.run.question
    assert offered["evidence_records"] == packet["evidence_records"] and offered["decisions"] == packet["decisions"]
    assert "_context" not in offered and "correction" not in offered
    assert ("execution" in offered) == (phase == "report")
    assert gates[0] == PASS and gates[-1].kind == "semantic" and gates[-1].passed is passed
    assert proposal.model_dump() == original and packet == original_packet


@pytest.mark.parametrize("status,failed", [("ready", True), ("needs_input", False), ("unsupported", False)])
def test_failed_structural_checks_or_nonready_decisions_spend_no_review_call(status, failed):
    h = harness()
    value = Decision(**{**decision(), "status": status})
    gates = [PASS.model_copy(update={"passed": not failed})]
    h.ask = lambda *args: pytest.fail("A nonaccepted candidate must not spend review calls")
    assert h.review("target", value, {}, gates) is gates


def test_malformed_reviewer_gets_one_bounded_correction_without_changing_science():
    h = harness()
    proposal = Decision(**decision())
    original = proposal.model_dump(mode="json")
    raw = '{"accepted":' + "x" * 143_000
    calls = []

    def ask(key, schema, prompt, payload):
        calls.append((key, deepcopy(payload)))
        if len(calls) == 1:
            try:
                schema.model_validate_json(raw)
            except ValidationError as error:
                h.saved["model_error:" + key] = str(model_access.InvalidModelOutput(raw, error))
                raise model_access.InvalidModelOutput(raw, error) from error
        return CodeReview(accepted=True, summary="The original scientific candidate is faithful.")

    h.ask = ask
    gates = h.review("plan", proposal, {}, [PASS])
    assert gates[-1].passed and len(calls) == 2
    assert calls[1][0] == calls[0][0] + ":correction"
    assert calls[0][1]["candidate"] == calls[1][1]["candidate"] == original
    assert "review_correction" in calls[1][1] and raw not in str(calls[1][1])
    assert len(str(calls[1][1])) < 4000


def test_saved_malformed_reviews_replay_as_unavailable_without_calls_or_false_rejection():
    h = harness()
    proposal = Decision(**decision())
    attempted = []

    def invalid(key, schema, prompt, payload):
        attempted.append(key)
        try:
            schema.model_validate_json("{")
        except ValidationError as error:
            problem = model_access.InvalidModelOutput("{", error)
            h.saved["model_error:" + key] = str(problem)
            raise problem from error

    h.ask = invalid
    first = h.review("report", proposal, {}, [PASS])
    assert len(attempted) == 2 and not first[-1].passed
    assert "could not establish a valid semantic judgment" in first[-1].detail
    h.ask = lambda *args: pytest.fail("Saved malformed responses must not be redispatched")
    assert h.review("report", proposal, {}, [PASS]) == first


def test_snapshot_plan_assumptions_are_offered_to_report_reference_normalization():
    h = harness()
    proposal = Decision(**{**decision(), "citations": ["plan_premise"]})
    saved = []
    h.start = lambda *args, **kwargs: None
    h.ask = lambda *args, **kwargs: proposal
    h.remember = lambda *args, **kwargs: None
    h.finish = lambda *args, **kwargs: SimpleNamespace(status="accepted")
    packet = {"snapshot": {"frame": {}, "plan": {"assumptions": [{"id": "plan_premise"}]}}}

    def check(value):
        saved.append(value)
        return [PASS]

    assert h.decide("report", Decision, packet, check, prompt="Report") is not None
    assert saved[0].citations == ["assumption:plan_premise"]
    assert packet == {"snapshot": {"frame": {}, "plan": {"assumptions": [{"id": "plan_premise"}]}}}
