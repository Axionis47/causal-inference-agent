"""Real API, immutable revision and worker paths around the pre-analysis freeze."""

from copy import deepcopy
from uuid import UUID, uuid4

import test_analysis as analysis
import test_chat
from model_double import model_double
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from backend import model_access
from backend.analysis import delivery
from backend.chat import jobs as chat_jobs
from backend.chat.contracts import AnalysisFile, Interpretation
from backend.context.contracts import BriefDraft
from backend.db import engine
from backend.integration.readers import Analyses
from backend.models import AnalysisRun
from backend.responses import Reply

storage = analysis.storage
boundaries = analysis.boundaries


def test_launch_cannot_bypass_missing_brief_and_never_enqueues(owner, monkeypatch):

    _, _, chat = test_chat.create(owner)
    base = f"/api/conversations/{chat['id']}"
    ready = owner.get(base + "/analysis-readiness")
    assert ready.status_code == 200 and ready.json()["status"] == "needs_context"
    monkeypatch.setattr(delivery, "enqueue", lambda *args: (_ for _ in ()).throw(AssertionError("Not-ready launch enqueued")))
    body = {"submission_id": str(uuid4()), "question": "A question cannot replace missing context",
            "expected_context_revision_id": None, "expected_readiness_fingerprint": ready.json()["fingerprint"]}
    assert owner.post(base + "/analyses", json=body).status_code == 409
    body.pop("expected_readiness_fingerprint")
    assert owner.post(base + "/analyses", json=body).status_code == 422
    with Session(engine()) as db:
        assert db.scalar(select(func.count()).select_from(AnalysisRun)) == 0


def test_fingerprint_and_brief_question_are_required_before_queueing(owner, monkeypatch):

    _, _, chat = analysis.create(owner)
    base = f"/api/conversations/{chat['id']}"
    ready = owner.get(base + "/analysis-readiness").json()
    assert ready["status"] == "ready"
    monkeypatch.setattr(delivery, "enqueue", lambda *args: (_ for _ in ()).throw(AssertionError("Stale launch enqueued")))
    body = {"submission_id": str(uuid4()), "expected_context_revision_id": chat["context_revision_id"],
            "expected_readiness_fingerprint": "0" * 64}
    assert owner.post(base + "/analyses", json=body).status_code == 409
    body.update(expected_readiness_fingerprint=ready["fingerprint"], question="A different effect")
    assert owner.post(base + "/analyses", json=body).status_code == 409


def test_chat_proposal_apply_changes_readiness_but_not_the_launched_context(owner, monkeypatch):
    _, _, chat = analysis.create(owner)
    run, body = analysis.prepare(owner, chat)
    base = f"/api/conversations/{chat['id']}"
    frozen = deepcopy(run["frozen_context"])
    prior_readiness = owner.get(base + "/analysis-readiness").json()
    revised = "Estimate the prespecified group contrast in the enrolled study participants."
    draft = BriefDraft.model_validate({**frozen["brief"], "question": revised}, extra="ignore")

    def discuss(schema, system, payload, model, **kwargs):
        if schema is Interpretation:
            return Interpretation()
        return Reply(text="Proposed revised brief. Apply changes to save it.", analysis_brief=draft,
                     proposal_passage=revised)

    monkeypatch.setattr(model_access, "structured", model_double(discuss))
    response = owner.post(base + "/turns", json={"submission_id": str(uuid4()), "expected_turn": 0,
                                                 "mode": "context", "text": revised})
    assert response.status_code == 202, response.text
    turn = response.json()
    chat_jobs.process(UUID(turn["id"]), Analyses())
    saved = owner.get(base).json()
    assert saved["turns"][-1]["status"] == "succeeded", saved["turns"][-1]
    assert owner.get(base + "/analysis-readiness").json() == prior_readiness
    applied = owner.post(base + f"/turns/{turn['id']}/apply-context")
    assert applied.status_code == 200, applied.text
    assert applied.json()["context_revision_id"] != frozen["context_revision_id"]
    ready = owner.get(base + "/analysis-readiness").json()
    assert ready["status"] == "ready" and ready["question"] == revised
    assert ready["fingerprint"] != prior_readiness["fingerprint"]
    assert analysis.read(owner, run["id"])["frozen_context"] == frozen
    assert owner.post(base + "/analyses", json=body).json()["id"] == run["id"]
    assert owner.post(base + "/analyses", json={**body, "submission_id": str(uuid4())}).status_code == 409
    artifact = owner.get(f"/api/analyses/{run['id']}/files/frozen-context.json")
    assert artifact.status_code == 200 and artifact.json() == frozen


def test_chat_can_read_launch_artifact_before_an_executable_snapshot_exists(owner, monkeypatch):
    _, _, chat = analysis.create(owner)
    run, _ = analysis.prepare(owner, chat)
    base = f"/api/conversations/{chat['id']}"

    def discuss(schema, system, payload, model, **kwargs):
        if schema is Interpretation:
            assert "snapshot" not in payload["analyses"][0]
            assert payload["analyses"][0]["result_available"] is False
            assert "frozen-context.json" in payload["analyses"][0]["artifact_files"]
            return Interpretation(analysis_files=[AnalysisFile(analysis_id=UUID(run["id"]), filename="frozen-context.json")])
        assert run["frozen_context"]["context_hash"] in payload["analysis_file_evidence"][0]["content"]
        return Reply(text="This run uses its saved applied context. Later chat changes do not alter it.")

    monkeypatch.setattr(model_access, "structured", model_double(discuss))
    response = owner.post(base + "/turns", json={"submission_id": str(uuid4()), "expected_turn": 0,
                                                 "text": "Explain the saved frozen context without changing it."})
    chat_jobs.process(UUID(response.json()["id"]), Analyses())
    saved = owner.get(base).json()["turns"][-1]
    assert saved["status"] == "succeeded", saved
