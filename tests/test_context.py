"""Real PostgreSQL/storage journeys with scripted model outputs, never live calls."""

from copy import deepcopy
from uuid import UUID, uuid4

import pytest
from conftest import PASSWORD
from fastapi.testclient import TestClient
from model_double import model_double
from sqlalchemy import update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session

from backend import model_access
from backend.context import jobs
from backend.context.contracts import COLUMN_FIELDS, DATASET_FIELDS, empty_content
from backend.context.extraction import Assessment, Extraction, materialize
from backend.db import engine
from backend.models import ContextPreparation, ContextRevision, SourceMaterial

SOURCE = "Each row is a pupil. Café 🧪.\r\nCourse completion was voluntary before the exam.\r\nScore is the exam mark, from 0 to 100.\r\n"


def dataset(client: TestClient) -> dict:
    response = client.post(
        "/api/datasets",
        data={"title": "Context test", "submission_id": str(uuid4())},
        files={"file": ("study.csv", b"course,score\nyes,80\nno,70\n", "text/csv")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def source(client: TestClient, base: str, content: str = SOURCE, submission: str | None = None) -> dict:
    response = client.post(
        base + "/sources",
        data={"title": "Study account", "kind": "file", "submission_id": submission or str(uuid4())},
        files={"file": ("account.md", content.encode(), "text/markdown")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def scripted(schema, system, payload, model):
    if schema is Assessment:
        return Assessment()
    cols = [c["column_id"] for c in sorted(payload["columns"], key=lambda column: column["position"])]
    sid = payload["sources"][0]["source_id"]
    raw = {
        "column_ids": cols,
        "statements": [
            {
                "id": "rows",
                "scope": "dataset",
                "field": "observations",
                "value": "Each row is a pupil.",
                "citations": [{"source_id": sid, "source_line": 1}],
            },
            {
                "id": "meaning",
                "scope": "column",
                "column_ids": [cols[0]],
                "field": "definition",
                "value": "Course completion.",
                "citations": [{"source_id": sid, "source_line": 2}],
            },
            {
                "id": "time",
                "scope": "column",
                "column_ids": [cols[0]],
                "field": "collection_time",
                "value": "Before the exam.",
                "citations": [{"source_id": sid, "source_line": 2}],
            },
            {
                "id": "event",
                "scope": "concept",
                "kind": "event",
                "value": "Exam",
                "citations": [{"source_id": sid, "source_line": 3}],
            },
            {
                "id": "relation",
                "scope": "relationship",
                "kind": "temporal_order",
                "value": "Course completion preceded the exam.",
                "participants": [
                    {"kind": "column", "id": cols[0], "role": "from"},
                    {"kind": "event", "id": "event", "role": "to"},
                ],
                "citations": [{"source_id": sid, "source_line": 2}],
                "conditions": "For these pupils.",
                "directness": "unspecified",
                "absence_scope": "unspecified",
            },
        ],
    }
    entries = raw.pop("statements")
    for name, scope in [
        ("dataset", "dataset"),
        ("columns", "column"),
        ("concepts", "concept"),
        ("relationships", "relationship"),
    ]:
        raw[name] = [entry for entry in entries if entry["scope"] == scope]
    return Extraction.model_validate(raw)


@pytest.fixture
def prepared(owner, monkeypatch):
    monkeypatch.setattr(model_access, "structured", model_double(scripted))
    data = dataset(owner)
    base = f"/api/versions/{data['version']['id']}/context"
    material = source(owner, base)
    body = {"submission_id": str(uuid4()), "expected_revision_id": None, "source_ids": [material["id"]]}
    response = owner.post(base + "/preparations", json=body)
    assert response.status_code == 202, response.text
    operation = response.json()
    jobs.process(UUID(operation["id"]))
    status = owner.get(base + "/preparations/" + operation["id"]).json()
    assert status["status"] == "succeeded", status
    revision = owner.get(base + "/revisions/" + status["result_revision_id"]).json()
    return owner, base, material, body, operation, revision


def test_source_and_keyed_revision_roundtrip(prepared, monkeypatch):
    client, base, material, body, operation, revision = prepared
    assert client.get(base + "/sources/" + material["id"] + "/file").content == SOURCE.encode()
    assert client.get(base + "/sources/" + material["id"]).json()["text"] == SOURCE
    content = revision["content"]
    assert set(content["dataset"]) == set(DATASET_FIELDS)
    assert all(set(c) == {"identity", *COLUMN_FIELDS} for c in content["columns"].values())
    statement = next(iter(content["dataset"]["observations"].values()))
    cite = statement["citations"][0]
    assert SOURCE[cite["start"] : cite["end"]] == cite["quote"] == SOURCE.splitlines(keepends=True)[0]
    monkeypatch.setattr(model_access, "structured", model_double(lambda *a: pytest.fail("Reads must never invoke a model")))
    assert client.get(base + "/revisions/" + revision["id"]).json() == revision
    assert client.post(base + "/preparations", json=body).json()["id"] == operation["id"]
    assert client.post(base + "/preparations", json={**body, "source_ids": [str(uuid4())]}).status_code == 409
    cid = next(k for k, column in content["columns"].items() if column["identity"]["position"] == 0)
    read = client.get(base + "/revisions/" + revision["id"] + "/read", params={"key": "columns." + cid})
    assert read.status_code == 200, read.text
    view = read.json()
    assert view["content"]["relationships"] and view["content"]["concepts"]
    assert view["content"]["dataset"]["observations"] and material["id"] in view["sources"]
    assert any(k.startswith("relationships.") for k in view["included_keys"])
    assert (
        client.get(
            base + "/revisions/" + revision["id"] + "/read", params={"key": f"columns.{cid}.reference_period"}
        ).status_code
        == 200
    )
    assert (
        client.get(
            base + "/revisions/" + revision["id"] + "/read", params={"key": "dataset.fake"}
        ).status_code
        == 422
    )


def test_structural_correction_is_bounded_and_recorded(owner, monkeypatch):
    data = dataset(owner)
    base = f"/api/versions/{data['version']['id']}/context"
    material = source(owner, base)
    invocation = 0

    def malformed_once(schema, system, payload, model):
        nonlocal invocation
        invocation += 1
        result = scripted(schema, system, payload, model)
        if invocation == 1:
            result.column_ids = ["invented-column"]
        return result

    monkeypatch.setattr(model_access, "structured", model_double(malformed_once))
    response = owner.post(
        base + "/preparations",
        json={
            "submission_id": str(uuid4()),
            "expected_revision_id": None,
            "source_ids": [material["id"]],
        },
    )
    operation_id = UUID(response.json()["id"])
    jobs.process(operation_id)
    with Session(engine()) as db:
        operation = db.get(ContextPreparation, operation_id)
        assert operation.status == "succeeded" and operation.calls == invocation == 3
        assert "exact column identity" in operation.steps["validation_correction"]["validation_errors"]
        assert [a["stage"] for a in operation.steps["attempts"]] == ["extract", "correct", "reassess"]
        assert operation.inputs["extract_prompt"] and operation.inputs["schema_digest"]


def test_provider_schema_preserves_scope_and_relationship_constraints():
    schema = model_access.provider_schema(Extraction)
    column = schema["properties"]["columns"]["items"]
    relation = schema["properties"]["relationships"]["items"]
    assert column["properties"]["scope"]["enum"] == ["column"]
    assert "collection_time" in column["properties"]["field"]["enum"]
    assert relation["properties"]["participants"]["minItems"] == 2
    assert "no_causal_influence" in relation["properties"]["kind"]["anyOf"][0]["enum"]
    assert "kind" not in relation["required"]


def test_shared_source_meaning_is_published_under_every_selected_column():
    version, preparation, author, source_id = uuid4(), uuid4(), uuid4(), uuid4()
    content = empty_content(version, ["math score", "reading score", "writing score"])
    columns = list(content.columns)
    extraction = Extraction.model_validate(
        {
            "column_ids": columns,
            "columns": [
                {
                    "scope": "column",
                    "id": "shared-definition",
                    "column_ids": columns,
                    "field": "definition",
                    "value": "Exam marks from 0 to 100.",
                    "citations": [{"source_id": str(source_id), "source_line": 1}],
                }
            ],
        }
    )
    saved = materialize(
        extraction,
        content,
        {str(source_id): "Math, reading and writing scores are exam marks from 0 to 100.\r\n"},
        preparation,
        author,
    )
    ids = []
    for column in saved.columns.values():
        assert len(column.definition) == 1
        identity, statement = next(iter(column.definition.items()))
        ids.append(identity)
        assert (
            statement.value == "Exam marks from 0 to 100." and statement.citations[0].source_id == source_id
        )
    assert len(set(ids)) == 3


def test_failed_delivery_is_visible_and_cannot_overwrite_a_retry(owner):
    data = dataset(owner)
    base = f"/api/versions/{data['version']['id']}/context"
    material = source(owner, base)
    response = owner.post(
        base + "/preparations",
        json={
            "submission_id": str(uuid4()),
            "expected_revision_id": None,
            "source_ids": [material["id"]],
        },
    )
    identity = UUID(response.json()["id"])
    with Session(engine()) as db:
        old_job = db.get(ContextPreparation, identity).job_id
    jobs.mark_failed_deliveries([old_job])
    assert owner.get(base + "/preparations/" + str(identity)).json()["status"] == "failed"
    assert owner.post(base + "/preparations/" + str(identity) + "/retry").status_code == 202
    jobs.mark_failed_deliveries([old_job])
    current = owner.get(base + "/preparations/" + str(identity)).json()
    assert current["status"] == "queued" and current["calls"] == 0


def test_review_replacement_history_conflicts_and_unknowns(prepared):
    client, base, _, _, _, original = prepared
    old_id = next(iter(original["content"]["dataset"]["observations"]))
    body = {
        "submission_id": str(uuid4()),
        "expected_revision_id": original["id"],
        "state": "reviewed",
        "leave_remaining_unresolved": True,
        "edits": [
            {
                "action": "replace",
                "key": f"dataset.observations.{old_id}",
                "reason": "Clarify the source account.",
                "statement": {"value": "Each row is an enrolled pupil who took the exam."},
            }
        ],
    }
    response = client.post(base + "/revisions", json=body)
    assert response.status_code == 201, response.text
    reviewed = response.json()
    assert reviewed["state"] == "reviewed"
    values = reviewed["content"]["dataset"]["observations"]
    assert values[old_id]["status"] == "withdrawn"
    new_id = next(k for k in values if k != old_id)
    assert values[new_id]["replaces"] == old_id and values[new_id]["source"]["kind"] == "review"
    assert all(r["status"] == "unresolved" for r in reviewed["content"]["relationships"].values())
    assert client.get(base + "/revisions/" + original["id"]).json() == original
    assert client.post(base + "/revisions", json=body).json()["id"] == reviewed["id"]
    assert client.post(base + "/revisions", json={**body, "submission_id": str(uuid4())}).status_code == 409
    assert client.post(base + "/revisions", json={**body, "state": "draft"}).status_code == 409
    read = client.get(
        base + "/revisions/" + reviewed["id"] + "/read", params={"key": f"dataset.observations.{new_id}"}
    )
    assert read.status_code == 200 and old_id in read.json()["content"]["dataset"]["observations"]


def test_multiple_sources_append_without_resetting_review(prepared):
    client, base, _, _, _, original = prepared
    reviewed = client.post(
        base + "/revisions",
        json={
            "submission_id": str(uuid4()),
            "expected_revision_id": original["id"],
            "state": "reviewed",
            "leave_remaining_unresolved": True,
        },
    ).json()
    extra = source(client, base, SOURCE.replace("pupil", "household"))
    run = client.post(
        base + "/preparations",
        json={
            "submission_id": str(uuid4()),
            "expected_revision_id": reviewed["id"],
            "source_ids": [extra["id"]],
        },
    ).json()
    jobs.process(UUID(run["id"]))
    status = client.get(base + "/preparations/" + run["id"]).json()
    new = client.get(base + "/revisions/" + status["result_revision_id"]).json()
    assert len(new["sources"]) == 2
    assert len(new["content"]["dataset"]["observations"]) == 2
    for sid, claim in reviewed["content"]["dataset"]["observations"].items():
        assert new["content"]["dataset"]["observations"][sid] == claim


def test_cross_owner_and_cross_version_access(prepared):
    client, base, material, _, operation, revision = prepared
    other_dataset = dataset(client)
    other_base = f"/api/versions/{other_dataset['version']['id']}/context"
    assert client.get(other_base + "/revisions/" + revision["id"]).status_code == 404
    assert client.get(other_base + "/sources/" + material["id"]).status_code == 404
    assert (
        client.post(
            other_base + "/preparations",
            json={
                "submission_id": str(uuid4()),
                "expected_revision_id": None,
                "source_ids": [material["id"]],
            },
        ).status_code
        == 404
    )
    client.post("/api/auth/login", json={"email": "other@example.com", "password": PASSWORD})
    for path in [
        "",
        "/sources/" + material["id"],
        "/sources/" + material["id"] + "/file",
        "/revisions/" + revision["id"],
        "/preparations/" + operation["id"],
    ]:
        assert client.get(base + path).status_code == 404


def test_source_retries_validation_and_storage_failure(owner, monkeypatch):
    data = dataset(owner)
    base = f"/api/versions/{data['version']['id']}/context"
    submission = str(uuid4())
    first = source(owner, base, submission=submission)
    assert source(owner, base, submission=submission)["id"] == first["id"]
    for filename, content in [
        ("bad.pdf", b"hello"),
        ("bad.txt", b"\xff"),
        ("bad.txt", b"\x00"),
        ("empty.txt", b" "),
        ("large.txt", b"x" * (1024 * 1024 + 1)),
    ]:
        response = owner.post(
            base + "/sources",
            data={"title": "Bad", "kind": "file", "submission_id": str(uuid4())},
            files={"file": (filename, content)},
        )
        assert response.status_code in [413, 422]
    from backend.storage import LocalFiles

    def fail(*a):
        raise OSError("disk offline")

    monkeypatch.setattr(LocalFiles, "put", fail)
    failed = owner.post(
        base + "/sources",
        data={"title": "Good", "kind": "file", "submission_id": str(uuid4())},
        files={"file": ("fine.md", b"Good source")},
    )
    assert failed.status_code == 503
    assert len(owner.get(base).json()["sources"]) == 1


def test_failed_call_resume_and_budget(prepared, monkeypatch):
    client, base, _, _, _, revision = prepared
    extra = source(client, base)
    response = client.post(
        base + "/preparations",
        json={
            "submission_id": str(uuid4()),
            "expected_revision_id": revision["id"],
            "source_ids": [extra["id"]],
        },
    )
    run_id = response.json()["id"]
    calls = []

    def fail_assessment(schema, system, payload, model):
        calls.append(schema.__name__)
        if schema is Assessment:
            raise RuntimeError("provider offline")
        return scripted(schema, system, payload, model)

    monkeypatch.setattr(model_access, "structured", model_double(fail_assessment))
    jobs.process(UUID(run_id))
    failed = client.get(base + "/preparations/" + run_id).json()
    assert failed["status"] == "failed" and failed["calls"] == 2
    assert client.get(base).json()["current_revision_id"] == revision["id"]
    monkeypatch.setattr(model_access, "structured", model_double(scripted))
    assert client.post(base + "/preparations/" + run_id + "/retry").status_code == 202
    jobs.process(UUID(run_id))
    done = client.get(base + "/preparations/" + run_id).json()
    assert done["status"] == "succeeded" and done["calls"] == 3
    jobs.process(UUID(run_id))
    assert client.get(base + "/preparations/" + run_id).json()["calls"] == 3


def test_stale_preparation_does_not_overwrite_edits(prepared):
    client, base, _, _, _, revision = prepared
    extra = source(client, base)
    run = client.post(
        base + "/preparations",
        json={
            "submission_id": str(uuid4()),
            "expected_revision_id": revision["id"],
            "source_ids": [extra["id"]],
        },
    ).json()
    changed = client.post(
        base + "/revisions",
        json={
            "submission_id": str(uuid4()),
            "expected_revision_id": revision["id"],
            "leave_remaining_unresolved": True,
        },
    ).json()
    jobs.process(UUID(run["id"]))
    status = client.get(base + "/preparations/" + run["id"]).json()
    assert status["status"] == "failed" and "Context changed" in status["error"]
    assert client.get(base).json()["current_revision_id"] == changed["id"]
    with Session(engine()) as db:
        assert "candidate" in db.get(ContextPreparation, UUID(run["id"])).steps


def test_database_immutability_and_forged_citations(prepared):
    client, base, material, _, operation, revision = prepared
    for model, identity, values in [
        (SourceMaterial, material["id"], {"title": "changed"}),
        (ContextRevision, revision["id"], {"state": "reviewed"}),
        (ContextPreparation, operation["id"], {"inputs": {}}),
    ]:
        with Session(engine()) as db:
            with pytest.raises(DBAPIError):
                db.execute(update(model).where(model.id == UUID(identity)).values(**values))
                db.commit()
    existing = next(iter(revision["content"]["dataset"]["observations"].values()))
    cite = deepcopy(existing["citations"][0])
    cite["quote"] = "forged"
    result = client.post(
        base + "/revisions",
        json={
            "submission_id": str(uuid4()),
            "expected_revision_id": revision["id"],
            "edits": [
                {
                    "action": "add",
                    "key": "dataset.population",
                    "statement": {"value": "Claim", "citations": [cite]},
                }
            ],
        },
    )
    assert result.status_code == 422
    assert client.get(base).json()["current_revision_id"] == revision["id"]


def test_invalid_model_citations_and_assessment_do_not_publish(owner, monkeypatch):
    data = dataset(owner)
    base = f"/api/versions/{data['version']['id']}/context"
    material = source(owner, base)

    def bad(schema, system, payload, model):
        result = scripted(schema, system, payload, model)
        result.statements[0].citations[0].source_line = 999
        return result

    monkeypatch.setattr(model_access, "structured", model_double(bad))
    run = owner.post(
        base + "/preparations",
        json={"submission_id": str(uuid4()), "expected_revision_id": None, "source_ids": [material["id"]]},
    ).json()
    jobs.process(UUID(run["id"]))
    result = owner.get(base + "/preparations/" + run["id"]).json()
    assert result["status"] == "failed" and result["stage"] == "rejected"
    assert owner.get(base).json()["current_revision_id"] is None


def test_preparation_and_queue_are_one_transaction(owner, monkeypatch):
    from sqlalchemy import select, text

    from backend.context import service
    from backend.context.contracts import PrepareInput
    from backend.main import app
    from backend.models import ContextPack, DatasetVersion, User
    from backend.storage import files

    data = dataset(owner)
    version_id = UUID(data["version"]["id"])
    base = f"/api/versions/{version_id}/context"
    material = source(owner, base)
    enqueue = jobs.enqueue

    def fail_after_queue(db, identity):
        enqueue(db, identity)
        raise RuntimeError("crash before commit")

    monkeypatch.setattr(jobs, "enqueue", fail_after_queue)
    with Session(engine()) as db:
        before = db.scalar(text("SELECT count(*) FROM procrastinate_jobs"))
        version = db.get(DatasetVersion, version_id)
        user = db.scalar(select(User).where(User.email == "owner@example.com"))
        with pytest.raises(RuntimeError):
            service.prepare(
                db,
                app.dependency_overrides[files](),
                version,
                user.id,
                PrepareInput(
                    submission_id=uuid4(), expected_revision_id=None, source_ids=[UUID(material["id"])]
                ),
            )
        db.rollback()
        assert db.scalar(text("SELECT count(*) FROM procrastinate_jobs")) == before
        pack = db.scalar(select(ContextPack).where(ContextPack.version_id == version_id))
        assert not list(db.scalars(select(ContextPreparation).where(ContextPreparation.pack_id == pack.id)))


def test_interrupted_worker_resumes_saved_step_and_enforces_budget(prepared, monkeypatch):
    from sqlalchemy import update

    client, base, _, _, _, revision = prepared
    material = source(client, base)
    run = client.post(
        base + "/preparations",
        json={
            "submission_id": str(uuid4()),
            "expected_revision_id": revision["id"],
            "source_ids": [material["id"]],
        },
    ).json()

    def interrupted(schema, system, payload, model):
        if schema is Assessment:
            raise KeyboardInterrupt("worker terminated")
        return scripted(schema, system, payload, model)

    monkeypatch.setattr(model_access, "structured", model_double(interrupted))
    with pytest.raises(KeyboardInterrupt):
        jobs.process(UUID(run["id"]))
    with Session(engine()) as db:
        record = db.get(ContextPreparation, UUID(run["id"]))
        assert record.calls == 2 and "extract" in record.steps
    monkeypatch.setattr(model_access, "structured", model_double(scripted))
    jobs.process(UUID(run["id"]))
    done = client.get(base + "/preparations/" + run["id"]).json()
    assert done["status"] == "succeeded" and done["calls"] == 3
    # A separate interrupted job with all four reservations used cannot make another call.
    material = source(client, base)
    run = client.post(
        base + "/preparations",
        json={
            "submission_id": str(uuid4()),
            "expected_revision_id": done["result_revision_id"],
            "source_ids": [material["id"]],
        },
    ).json()
    with Session(engine()) as db:
        db.execute(
            update(ContextPreparation)
            .where(ContextPreparation.id == UUID(run["id"]))
            .values(calls=4, status="running")
        )
        db.commit()
    monkeypatch.setattr(model_access, "structured", model_double(lambda *args: pytest.fail("budget exceeded")))
    jobs.process(UUID(run["id"]))
    assert client.get(base + "/preparations/" + run["id"]).json()["status"] == "failed"


def test_review_does_not_review_relationship_and_source_membership_is_frozen(prepared):
    from sqlalchemy import text

    client, base, material, _, _, revision = prepared
    cid = next(
        k for k, column in revision["content"]["columns"].items() if column["identity"]["position"] == 0
    )
    definition = next(iter(revision["content"]["columns"][cid]["definition"]))
    response = client.post(
        base + "/revisions",
        json={
            "submission_id": str(uuid4()),
            "expected_revision_id": revision["id"],
            "edits": [{"action": "review", "key": f"columns.{cid}.definition.{definition}"}],
        },
    )
    assert response.status_code == 201, response.text
    new = response.json()
    assert all(r["status"] == "proposed" for r in new["content"]["relationships"].values())
    extra = source(client, base)
    with Session(engine()) as db:
        with pytest.raises(DBAPIError):
            db.execute(
                text(
                    "INSERT INTO context_revision_sources (revision_id,source_id,version_id) VALUES (:revision,:source,:version)"
                ),
                {"revision": revision["id"], "source": extra["id"], "version": new["version_id"]},
            )
            db.commit()


def test_descriptive_multiparty_relations_and_derived_index(prepared):
    client, base, _, _, _, original = prepared
    columns = list(original["content"]["columns"])
    event = next(iter(original["content"]["concepts"]))
    participants = [{"kind": "column", "id": cid} for cid in columns]
    participants.append({"kind": "event", "id": event})
    response = client.post(
        base + "/revisions",
        json={
            "submission_id": str(uuid4()),
            "expected_revision_id": original["id"],
            "edits": [
                {
                    "action": "add",
                    "key": "relationships",
                    "statement": {
                        "value": value,
                        "participants": participants,
                        "qualifications": "Account remains uncertain.",
                        "conditions": "During the reported exam.",
                        "timing": "Same term.",
                        "directness": "unspecified",
                        "absence_scope": "direct_effect",
                    },
                }
                for value in [
                    "One account denies a direct effect, leaving other pathways unknown.",
                    "A second account denies a direct effect only under these conditions.",
                ]
            ],
        },
    )
    assert response.status_code == 201, response.text
    saved = response.json()
    assert saved["schema_version"] == "2"
    added = {
        rid: r
        for rid, r in saved["content"]["relationships"].items()
        if rid not in original["content"]["relationships"]
    }
    assert len(added) == 2
    assert all(r["kind"] is None and len(r["participants"]) == 3 for r in added.values())
    view = client.get(base + f"/revisions/{saved['id']}/read", params={"key": f"columns.{columns[0]}"}).json()
    assert set(view["related_to"][columns[0]][columns[1]]) == set(added)
    assert set(added) <= set(view["related_to"][columns[0]][event])
    for rid, claim in added.items():
        assert view["content"]["relationships"][rid] == claim
        assert f"relationships.{rid}" in view["included_keys"]
    assert client.get(base + f"/revisions/{original['id']}").json() == original
    rid = next(iter(added))
    withdrawn = client.post(
        base + "/revisions",
        json={
            "submission_id": str(uuid4()),
            "expected_revision_id": saved["id"],
            "edits": [{"action": "withdraw", "key": f"relationships.{rid}", "reason": "Account retracted."}],
        },
    ).json()
    view = client.get(
        base + f"/revisions/{withdrawn['id']}/read", params={"key": f"relationships.{rid}"}
    ).json()
    assert view["related_to"] == {}
    assert view["content"]["relationships"][rid]["status"] == "withdrawn"


def test_unchanged_saves_do_not_create_revisions(prepared):
    client, base, _, _, _, original = prepared
    body = {"submission_id": str(uuid4()), "expected_revision_id": original["id"], "state": "draft"}
    response = client.post(base + "/revisions", json=body)
    assert response.status_code == 422 and "no context" in response.text
    assert len(client.get(base).json()["revisions"]) == 1
    # A review-state transition is meaningful, even with no wording changes.
    reviewed = client.post(
        base + "/revisions", json={**body, "state": "reviewed", "leave_remaining_unresolved": True}
    )
    assert reviewed.status_code == 201, reviewed.text
    saved = reviewed.json()
    key = "dataset.observations." + next(iter(saved["content"]["dataset"]["observations"]))
    repeated = {
        "submission_id": str(uuid4()),
        "expected_revision_id": saved["id"],
        "state": "reviewed",
        "leave_remaining_unresolved": True,
        "edits": [{"action": "unresolved", "key": key}],
    }
    response = client.post(base + "/revisions", json=repeated)
    assert response.status_code == 422, response.text
    assert len(client.get(base).json()["revisions"]) == 2
    assert client.get(base + f"/revisions/{original['id']}").json() == original
