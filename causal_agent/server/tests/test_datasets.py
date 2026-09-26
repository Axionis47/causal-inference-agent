import json

import yaml

from causal_agent.common.llm import set_llm
from causal_agent.desk.tests.fakes import DeskFake
from causal_agent.server.tests.conftest import STUDENTS, create_request, upload, wait_idle


def test_profile_upload_detects_columns(client):
    prof = upload(client)
    assert prof["rows"] == 1000 and len(prof["columns"]) == 8
    by = {c["name"]: c for c in prof["columns"]}
    assert by["lunch"]["kind"] == "categorical" and set(by["lunch"]["examples"]) == {"standard", "free/reduced"}
    assert by["math score"]["kind"] == "numeric" and len(by["math score"]["examples"]) == 2
    # the preview: shape per column, the first rows, and the dataset's own facts
    assert by["math score"]["numeric"]["min"] == 0 and by["math score"]["numeric"]["max"] == 100 and by["math score"]["numeric"]["p50"] == 66
    assert by["lunch"]["top_values"][0]["value"] == "standard" and by["lunch"]["top_values"][0]["count"] == 645
    assert by["lunch"]["null_rate"] == 0 and by["lunch"]["constant"] is False and by["lunch"]["sentinels"] == [] and by["lunch"]["issues"] == []
    assert len(prof["head"]) == 8 and prof["head"][0] == ["female", "group B", "bachelor's degree", "standard", "none", "72", "72", "74"]
    assert prof["duplicate_rows"] == 0 and prof["issues"] == [] and prof["co_missing"] == []
    assert isinstance(prof["candidate_keys"], list) and (prof["grain"] is None or isinstance(prof["grain"], list))


def test_profile_preview_reports_missing_and_short_files(client):
    csv = b"id,when,flag,note\n1,2024-01-01,yes,\n2,2024-01-08,no,NA\n3,2024-01-15,yes,ok\n"
    r = client.post("/api/profile", files={"file": ("tiny.csv", csv, "text/csv")})
    assert r.status_code == 201, r.text
    prof = r.json()
    by = {c["name"]: c for c in prof["columns"]}
    assert prof["rows"] == 3 and len(prof["head"]) == 3
    assert prof["head"][0] == ["1", "2024-01-01", "yes", ""] and prof["head"][1][3] == "NA"
    assert by["when"]["kind"] == "datetime" and by["when"]["datetime"] == {"first": "2024-01-01", "last": "2024-01-15", "frequency": "weekly"}
    assert by["note"]["nulls"] == 2 and by["note"]["null_rate"] == round(2 / 3, 6)
    assert by["id"]["kind"] == "id" and ["id"] in prof["candidate_keys"] and prof["grain"] == ["id"]
    assert client.post("/api/profile", files={"file": ("notes.txt", b"hello", "text/plain")}).status_code == 400
    assert client.post("/api/profile", files={"file": ("empty.csv", b"", "text/csv")}).status_code == 400


def test_create_lists_and_deletes_a_dataset(client, settings):
    set_llm(DeskFake())
    prof = upload(client)
    req = create_request(prof)
    r = client.post("/api/datasets", json=req)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["name"] == "students_web" and body["shipped"] is False and body["rows"] == 1000
    root = settings.root
    assert (root / "data/raw/students_web" / STUDENTS.name).exists()
    assert (root / "data/profiles/students_web.json").exists()
    assert not (root / "data/context/students_web.md").exists()  # CSV only: nothing about the data is typed into a form
    entry = yaml.safe_load((root / "data/datasets.yaml").read_text())["students_web"]
    assert entry == {"csv": f"data/raw/students_web/{STUDENTS.name}", "profile": "data/profiles/students_web.json"}
    meta = json.loads((root / "data/web/students_web/meta.json").read_text())
    assert meta["question"] is None and meta["thread_id"]
    # the same name again, and a bad name
    assert client.post("/api/datasets", json=req).status_code in {409, 404}
    assert client.post("/api/datasets", json={**req, "name": "../x"}).status_code == 422
    # listed, with the session's stage
    v = wait_idle(client, "students_web")
    assert v["stage"] == "waiting"
    listed = client.get("/api/datasets").json()["datasets"]
    mine = next(d for d in listed if d["name"] == "students_web")
    assert mine["shipped"] is False and mine["session"]["stage"] == "waiting" and mine["has_claims"] is True  # the memory exists from the first turn
    # delete: every file, the entry, the meta, the caches
    assert client.delete("/api/datasets/students_web").status_code == 204
    assert not (root / "data/raw/students_web").exists()
    for rel in ("data/profiles/students_web.json", "data/memory/students_web", "data/web/students_web"):
        assert not (root / rel).exists(), rel
    assert "students_web" not in yaml.safe_load((root / "data/datasets.yaml").read_text())
    assert client.delete("/api/datasets/students_web").status_code == 404
    assert client.get("/api/sessions/students_web").status_code == 404


def test_shipped_dataset_shares_its_csv_and_survives_the_other_delete(client, settings):
    """Two entries on one file: deleting one leaves the file for the other."""
    root = settings.root
    raw = root / "data/raw/shared"
    raw.mkdir(parents=True)
    (raw / "t.csv").write_text("a,b\n1,2\n3,4\n")
    (root / "data/profiles").mkdir()
    (root / "data/context").mkdir()
    for n in ("one", "two"):
        (root / f"data/profiles/{n}.json").write_text(json.dumps({"dataset": {"rows": 2, "columns": 2}}))
        (root / f"data/context/{n}.md").write_text("# x\n")
    (root / "data/datasets.yaml").write_text(
        yaml.safe_dump({n: {"csv": "data/raw/shared/t.csv", "note": f"data/context/{n}.md", "profile": f"data/profiles/{n}.json"} for n in ("one", "two")})
    )
    listed = client.get("/api/datasets").json()["datasets"]
    assert [d["name"] for d in listed] == ["one", "two"] and all(d["shipped"] and d["rows"] == 2 for d in listed)
    assert client.delete("/api/datasets/one").status_code == 204
    assert (raw / "t.csv").exists() and not (root / "data/context/one.md").exists()
    assert [d["name"] for d in client.get("/api/datasets").json()["datasets"]] == ["two"]


def test_meta_is_written_whole_and_leaves_no_temp_file(settings):
    from causal_agent.server import datasets as DS

    DS.write_meta(settings, "x", {"name": "x", "question": None})
    DS.write_meta(settings, "x", {"name": "x", "question": "did it?"})
    assert DS.read_meta(settings, "x") == {"name": "x", "question": "did it?"}
    assert [p.name for p in DS.meta_path(settings, "x").parent.iterdir()] == ["meta.json"]
