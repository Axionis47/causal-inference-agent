"""Owned chat publication against real PostgreSQL/S3, real Docker and a scripted model."""

import io
import json
import shutil
import time
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from causal_agent.common.llm import set_llm
from causal_agent.explore.tests.fakes import FakeLLM
from causal_agent.explore.tests.test_agent import ANSWER, TABLE
from causal_agent.memory import artifacts as VA
from causal_agent.server.app import create_app
from causal_agent.server.conversations import schema as S
from causal_agent.server.conversations.publication import Repository, cleanup
from causal_agent.server.ingestion.routes import principal, store
from causal_agent.server.ingestion.service import Missing
from causal_agent.server.tests.conftest import STUDENTS

pytestmark = pytest.mark.ingestion


class RoutingFake(FakeLLM):
    refs = []
    bad_ref_once = False

    def with_structured_output(self, schema, include_raw=False):
        if schema.__name__ != "Interpretation":
            return super().with_structured_output(schema, include_raw)
        fake = self

        class Reply:
            def invoke(self, messages):
                refs = fake.refs
                if fake.bad_ref_once:
                    refs = ["artifact:" + ref for ref in refs]
                    fake.bad_ref_once = False
                return {
                    "raw": AIMessage(content="{}"),
                    "parsed": schema(
                        mode="discussion",
                        exploration=json.JSONDecoder().raw_decode(messages[-1][1])[0]["message"],
                        exploration_passage=json.JSONDecoder().raw_decode(messages[-1][1])[0]["message"],
                        references=[{"kind": "artifact", "id": ref} for ref in refs],
                    ),
                    "parsing_error": None,
                }

        return Reply()


def idle(client, conversation):
    for _ in range(1200):
        response = client.get(f"/api/conversations/{conversation}")
        assert response.status_code == 200, response.text
        value = response.json()
        if value["status"] != "running":
            assert value["status"] == "idle", value
            return value
        time.sleep(0.05)
    raise AssertionError("chat never became idle")


def test_chat_publishes_reloads_and_reuses_owned_artifact(storage, settings):
    owner = str(uuid4())
    original = STUDENTS.read_bytes()
    info = storage.ingest(
        owner, "chat-test", "Students", "Reading scores are observed marks, lunch labels identify groups.", "students.csv", io.BytesIO(original)
    )
    app = create_app(settings)
    app.dependency_overrides[store] = lambda: storage
    app.dependency_overrides[principal] = lambda: owner
    fake = RoutingFake([[TABLE]], [ANSWER])
    set_llm(fake)
    try:
        with TestClient(app) as client:
            from causal_agent.server.tests.review_helpers import review_body

            body = review_body(storage, owner, info, "Summarize reading scores by lunch")
            set_llm(fake)
            created = client.post(f"/api/datasets/{info['dataset_id']}/conversations", json=body)
            assert created.status_code == 201, created.text
            chat = created.json()["id"]
            idle(client, chat)
            assert (
                client.post(
                    f"/api/conversations/{chat}/commands",
                    json={
                        "submission_id": str(uuid4()),
                        "revision_id": idle(client, chat)["revision_id"],
                        "kind": "message",
                        "text": "Summarize reading scores by lunch",
                    },
                ).status_code
                == 202
            )
            result = idle(client, chat)
            artifact = client.get(f"/api/conversations/{chat}/artifacts").json()[0]
            assert artifact is not None, result
            aid = artifact["id"]
            path = f"/api/conversations/{chat}/artifacts/{aid}"
            saved = client.get(path).json()
            assert saved["owner_id"] == owner and saved["dataset_version_id"] == info["version_id"]
            assert saved["method"] and saved["findings"] and saved["limitations"]
            assert saved["facts"] == {"reading_standard": 71.65}
            assert saved["source_sha256"] == info["sha256"]
            with storage.engine.connect() as db:
                attempt = db.execute(select(S.attempts)).mappings().one()
                assert attempt["status"] == "completed"
                keys = list(db.scalars(select(S.files.c.object_key)))
                assert all(k.startswith(f"users/{owner}/conversations/{chat}/artifacts/{aid}/") for k in keys)
            shutil.rmtree(VA.folder(chat, aid))
            assert client.get(path + "/files/facts.json").json() == saved["facts"]
            # The next request resolves both the numerical file and the linked explanation from S3.
            fake.refs = [aid]
            fake.bad_ref_once = False
            code = """import json, os
from pathlib import Path
root=Path(os.environ["INPUTS_DIR"])
manifest=json.loads((root/"manifest.json").read_text())
source=next(root/n for n,v in manifest.items() if v["id"].endswith("/facts.json"))
Path("facts.json").write_bytes(source.read_bytes())
"""
            fake.rounds = [[code]]
            fake.answers = [ANSWER.model_copy(update={"table": None, "method": "Read the supplied saved mean."})]
            assert (
                client.post(
                    f"/api/conversations/{chat}/commands",
                    json={
                        "submission_id": str(uuid4()),
                        "revision_id": idle(client, chat)["revision_id"],
                        "kind": "message",
                        "text": "Explain the previous result again",
                    },
                ).status_code
                == 202
            )
            idle(client, chat)
            followup = client.get(f"/api/conversations/{chat}/artifacts").json()[0]
            followup = client.get(f"/api/conversations/{chat}/artifacts/{followup['id']}").json()
            assert followup and any(aid in ref for ref in followup["parent_refs"])
            assert followup["ask"] == "Explain the previous result again"
            # A second conversation is independent, even under the same owner and dataset.
            second = client.post(
                f"/api/datasets/{info['dataset_id']}/conversations", json=review_body(storage, owner, info, "Discuss a previous artifact")
            ).json()["id"]
            idle(client, second)
            assert client.get(f"/api/conversations/{second}/artifacts/{aid}").status_code == 404
            assert any(a["id"] == aid for a in client.get(f"/api/conversations/{second}/artifacts?earlier=true").json())
            selected = client.post(
                f"/api/conversations/{second}/commands",
                json={
                    "submission_id": str(uuid4()),
                    "revision_id": idle(client, second)["revision_id"],
                    "kind": "reuse",
                    "references": [{"kind": "artifact", "id": aid}],
                },
            )
            assert selected.status_code == 202, selected.text
            idle(client, second)
            assert client.get(f"/api/conversations/{second}/artifacts/{aid}").json()["conversation_id"] == chat
            assert client.get(f"/api/conversations/{second}/artifacts/{aid}/files/facts.json").json() == saved["facts"]
            # The database also rejects ownership/conversation mismatches, independent of HTTP guards.
            with pytest.raises(IntegrityError), storage.engine.begin() as db:
                db.execute(
                    S.files.insert().values(
                        artifact_id=aid, name="wrong.json", owner_id=owner, conversation_id=second, object_key="wrong", sha256="0" * 64, size_bytes=0
                    )
                )
            assert client.get("/api/runs/not-owned").status_code == 404
            app.dependency_overrides[principal] = lambda: str(uuid4())
            assert client.get(path).status_code == 404
            assert client.get(path + "/files/facts.json").status_code == 404
            assert client.get(f"/api/conversations/{chat}").status_code == 404
            metadata, body = storage.download(owner, info["dataset_id"])
            try:
                assert body.read() == original
            finally:
                body.close()
    finally:
        app.state.sessions.pool.shutdown(wait=True)
        set_llm(None)


def test_failed_publication_is_not_readable(storage, settings, monkeypatch):
    from causal_agent.memory.tests.test_artifacts import _artifact
    from causal_agent.server.conversations.manager import Manager
    from causal_agent.server.conversations.service import create
    from causal_agent.server.tests.review_helpers import review_body

    owner = str(uuid4())
    info = storage.ingest(owner, "failure", "Numbers", "", "data.csv", io.BytesIO(b"a,b\n1,2\n3,4\n"))
    manager = Manager(settings)
    try:
        body = review_body(storage, owner, info, "Describe these numbers")
        chat = create(storage, owner, info["dataset_id"], manager, body["review_id"], body["generation"])["id"]
        repo = Repository(storage, owner, chat)
        request, attempt, version = repo.begin(chat, "means", "", info["sha256"], [])
        a = _artifact(
            owner_id=owner,
            dataset=chat,
            conversation_id=chat,
            dataset_version_id=version,
            request_id=request,
            attempt_id=attempt,
            source_sha256=info["sha256"],
            files=["facts.json", "result.json"],
            figure=False,
            limitations=["Descriptive only."],
        )
        directory = settings.root / "publication"
        directory.mkdir()
        (directory / "facts.json").write_text(json.dumps(a.facts))
        (directory / "result.json").write_text(a.model_dump_json())
        put = storage.objects.put

        def failed(*args):
            if args[0].endswith("result.json"):
                raise OSError("storage unavailable")
            return put(*args)

        monkeypatch.setattr(storage.objects, "put", failed)
        with pytest.raises(OSError, match="storage unavailable"):
            repo.publish(a, directory)
        assert repo.list(chat) == []
        with pytest.raises(Missing):
            repo.load(chat, a.id)
        with storage.engine.begin() as db:
            db.execute(update(S.attempts).where(S.attempts.c.id == attempt).values(created_at=datetime.now(UTC) - timedelta(hours=25)))
        assert cleanup(storage) == 1
        listed = storage.objects.client.list_objects_v2(Bucket=storage.settings.bucket, Prefix=f"users/{owner}/conversations/{chat}/artifacts/")
        assert not listed.get("Contents")
        # Cleanup never touches the original dataset object.
        _, body = storage.download(owner, info["dataset_id"])
        try:
            assert body.read() == b"a,b\n1,2\n3,4\n"
        finally:
            body.close()
    finally:
        manager.pool.shutdown(wait=True)
