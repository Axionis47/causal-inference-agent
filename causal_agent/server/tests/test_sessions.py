import time

from fastapi.testclient import TestClient

from causal_agent.chat import pipeline
from causal_agent.chat.contracts import AfterReply, NumberStated
from causal_agent.common.llm import set_llm
from causal_agent.server.app import create_app
from causal_agent.server.tests.conftest import CT, IT, create_request, upload, wait_idle

C = "completed_vs_none"


def _create(client, fake, name="students_web"):
    set_llm(fake)
    r = client.post("/api/datasets", json=create_request(upload(client), name=name))
    assert r.status_code == 201, r.text
    return wait_idle(client, name)


def _table(client, name):
    mgr = client.app.state.sessions
    return mgr._values(mgr.get(name))["claims"]


def _to_ready(client, fake, name="students_web"):
    v = _create(client, fake, name)
    assert v["stage"] == "waiting" and v["phase"] == "before" and v["ready"] is False
    assert v["questions"] and any(q["kind"] == "confirm" for q in v["questions"])
    assert {c["status"] for c in v["claims"]} >= {"drafted", "empty"}
    assert v["status"]["ready"] is False and v["prompt"]["text"]
    assert [t["role"] for t in v["transcript"]] == ["system", "assistant"]
    fake.script["user:turn:1"] = IT.confirm_all(_table(client, name), "user:turn:1")
    r = client.post(f"/api/sessions/{name}/messages", json={"text": "all right"})
    assert r.status_code == 202 and r.json()["stage"] == "busy"
    v = wait_idle(client, name)
    fake.script["user:turn:2"] = [IT.U("unobserved", ["user:turn:2"], exists="false"), IT.U("spillover", ["user:turn:2"], possible="false"), IT.U("exclusion", ["user:turn:2"], exists="false")]
    client.post(f"/api/sessions/{name}/messages", json={"text": "nothing hidden, no spillover, no nudge"})
    v = wait_idle(client, name)
    assert v["ready"] is True and v["status"]["ready"] is True and v["questions"] == []
    return v


def test_interview_run_answer_done_and_restart(client, settings, monkeypatch):
    calls = []

    def fake_run(dataset, question, index):
        calls.append((dataset, question, index))
        return CT.canned("dowhy", index, question, effect=5.618)

    monkeypatch.setattr(pipeline, "run", fake_run)
    after = [
        AfterReply(kind="answer", text="It raised math scores by 5.618 points.", cites=[f"estimate:{C}.value"], numbers=[NumberStated(address=f"estimate:{C}.value", value=5.618)]),
        AfterReply(kind="done", text="Bye."),
    ]
    fake = CT.Fake({"doc:context": IT.students_turn0()}, after=after)
    v = _to_ready(client, fake)
    client.post("/api/sessions/students_web/messages", json={"text": "run"})
    v = wait_idle(client, "students_web")
    assert calls == [("students_web", "Did completing the prep course raise math scores?", 1)]
    assert v["phase"] == "after" and v["stage"] == "waiting" and v["ready"] is True
    assert v["runs"][0]["effect"] == 5.618 and v["runs"][0]["family"] == "adjustment" and v["runs"][0]["checks"]
    assert f"[estimate:{C}.value]" in v["brief"] and v["prompt"]["text"] == v["brief"]
    assert v["written"] and (settings.root / "data/claims/students_web.yaml").exists()
    listed = next(d for d in client.get("/api/datasets").json()["datasets"] if d["name"] == "students_web")
    assert listed["has_claims"] is True and listed["session"] == {"stage": "waiting", "phase": "after", "runs": 1}
    n = len(v["transcript"])
    client.post("/api/sessions/students_web/messages", json={"text": "what did you find?"})
    v = wait_idle(client, "students_web")
    assert len(v["transcript"]) == n + 2 and "5.618" in v["transcript"][-1]["text"] and v["transcript"][-1]["phase"] == "after"
    client.post("/api/sessions/students_web/messages", json={"text": "done"})
    v = wait_idle(client, "students_web")
    assert v["stage"] == "ended" and v["transcript"][-1]["role"] == "system"
    assert client.post("/api/sessions/students_web/messages", json={"text": "hi"}).status_code == 409
    # a new conversation on the same dataset
    r = client.post("/api/sessions/students_web/restart")
    assert r.status_code == 202
    v = wait_idle(client, "students_web")
    assert v["stage"] == "waiting" and v["phase"] == "before" and v["runs"] == []


def test_busy_rejects_a_second_message(client, monkeypatch):
    def slow_run(dataset, question, index):
        time.sleep(0.8)
        return CT.canned("dowhy", index, question)

    monkeypatch.setattr(pipeline, "run", slow_run)
    fake = CT.Fake({"doc:context": IT.students_turn0()})
    _to_ready(client, fake)
    r = client.post("/api/sessions/students_web/messages", json={"text": "run"})
    assert r.status_code == 202 and r.json()["activity"]
    assert client.post("/api/sessions/students_web/messages", json={"text": "again"}).status_code == 409
    assert client.delete("/api/datasets/students_web").status_code == 409
    v = wait_idle(client, "students_web")
    assert v["phase"] == "after" and len(v["runs"]) == 1


def test_state_survives_a_new_server_over_the_same_checkpoints(client, settings):
    fake = CT.Fake({"doc:context": IT.students_turn0()})
    v1 = _create(client, fake)
    app2 = create_app(settings)
    with TestClient(app2) as c2:
        v2 = c2.get("/api/sessions/students_web").json()
    assert v2["stage"] == "waiting" and v2["claims"] == v1["claims"] and v2["prompt"] == v1["prompt"] and v2["questions"] == v1["questions"]
    assert [t["text"] for t in v2["transcript"]] == [t["text"] for t in v1["transcript"]]


def test_run_files_are_guarded(client, settings):
    d = settings.run_root / "students_web-abcd1234"
    d.mkdir(parents=True)
    (d / "report.md").write_text("QUESTION\n  q\n\nRESULTS\n  effect 5.6\n\nMODEL THOUGHTS (debug only)\n  secret musings\n")
    (d / "artifacts.json").write_text('{"estimates": []}')
    (d / "table.csv").write_text("a\n1\n")
    (d / "notes.txt").write_text("not served")
    files = client.get("/api/runs/students_web-abcd1234").json()["files"]
    assert [f["name"] for f in files] == ["artifacts.json", "report.md", "table.csv"]
    text = client.get("/api/runs/students_web-abcd1234/files/report.md").text
    assert "effect 5.6" in text and "secret" not in text
    assert "secret" in client.get("/api/runs/students_web-abcd1234/files/report.md?raw=1").text
    assert client.get("/api/runs/students_web-abcd1234/files/artifacts.json").json() == {"estimates": []}
    csv = client.get("/api/runs/students_web-abcd1234/files/table.csv")
    assert csv.status_code == 200 and "attachment" in csv.headers["content-disposition"]
    assert client.get("/api/runs/students_web-abcd1234/files/notes.txt").status_code == 404
    assert client.get("/api/runs/..%2F..%2Fetc/files/report.md").status_code == 404
    assert client.get("/api/runs/nope/files/report.md").status_code == 404
    assert client.get("/api/runs/nope").status_code == 404


def test_a_pending_node_with_no_worker_reads_as_stale(client, settings, monkeypatch):
    """A second server over the same checkpoints sees the first one's running step as a node left pending."""
    def slow_run(dataset, question, index):
        time.sleep(1.5)
        return CT.canned("dowhy", index, question)

    monkeypatch.setattr(pipeline, "run", slow_run)
    fake = CT.Fake({"doc:context": IT.students_turn0()})
    _to_ready(client, fake)
    client.post("/api/sessions/students_web/messages", json={"text": "run"})
    time.sleep(0.3)
    app2 = create_app(settings)
    with TestClient(app2) as c2:
        v = c2.get("/api/sessions/students_web").json()
        assert v["stage"] == "stale", v["stage"]
        assert c2.post("/api/sessions/students_web/messages", json={"text": "hello"}).status_code == 409
        wait_idle(client, "students_web")
        assert c2.get("/api/sessions/students_web").json()["stage"] == "waiting"
