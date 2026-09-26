import time

from fastapi.testclient import TestClient

from causal_agent.common.llm import set_llm
from causal_agent.desk.contracts import AfterReply, NumberStated
from causal_agent.desk.tests.fakes import DeskFake
from causal_agent.server.app import create_app
from causal_agent.server.tests.conftest import QUESTION, canned_run, create_request, infer_columns, patch_pipeline, reply_for, upload, wait_idle

C = "completed_vs_none"


def _create(client, fake, name="students_web"):
    set_llm(fake)
    r = client.post("/api/datasets", json=create_request(upload(client), name=name))
    assert r.status_code == 201, r.text
    return wait_idle(client, name)


def _say(client, name, text):
    r = client.post(f"/api/sessions/{name}/messages", json={"text": text})
    assert r.status_code == 202, r.text
    return wait_idle(client, name)


def _to_ready(client, fake, name="students_web", max_turns=20):
    v = _create(client, fake, name)
    assert v["stage"] == "waiting" and v["phase"] == "before" and v["ready"] is False
    assert v["prompt"]["kind"] == "question" and "What is the causal question" in v["prompt"]["text"] and v["questions"] == []
    assert [t["role"] for t in v["transcript"]] == ["system", "assistant"]
    v = _say(client, name, "How many students passed?")
    assert v["prompt"]["kind"] == "question" and "not yet a question" in v["prompt"]["text"]
    v = _say(client, name, QUESTION)
    assert v["question"] == QUESTION and v["prompt"]["kind"] == "ask" and len(v["questions"]) == 1
    turns = 0
    while not v["ready"]:
        assert turns < max_turns, v["prompt"]["text"]
        assert len(v["questions"]) == 1, "one question per turn"
        v = _say(client, name, reply_for(v))
        turns += 1
    assert v["status"]["ready"] is True and v["questions"] == []
    assert {c["status"] for c in v["claims"]} == {"confirmed"} and turns <= 16, [(c["key"], c["status"]) for c in v["claims"]]
    return v


def test_question_journey_run_answer_done_and_restart(client, settings, monkeypatch):
    calls = []

    def fake_run(path, n, dataset, question, decision=None, decision_record=""):
        calls.append((dataset, question, n))
        return canned_run(path, n, dataset, question, decision, decision_record)

    patch_pipeline(monkeypatch, fake_run)
    after = [
        AfterReply(
            kind="answer",
            text="It raised math scores by 5.618 points.",
            cites=[f"estimate:{C}.value"],
            numbers=[NumberStated(address=f"estimate:{C}.value", value=5.618)],
        ),
        AfterReply(kind="done", text="Bye."),
    ]
    fake = DeskFake(after=after, infer=infer_columns)
    v = _to_ready(client, fake)
    v = _say(client, "students_web", "run")
    assert calls == [("students_web", QUESTION, 1)]
    assert v["phase"] == "after" and v["stage"] == "waiting" and v["ready"] is True
    assert v["runs"][0]["effect"] == 5.618 and v["runs"][0]["family"] == "adjustment" and v["runs"][0]["checks"]
    assert f"[estimate:{C}.value]" in v["brief"] and v["prompt"]["text"] == v["brief"]
    assert (settings.root / "data/memory/students_web/fields.yaml").exists() and (settings.root / "data/memory/students_web/designs/1/handoff.json").exists()
    listed = next(d for d in client.get("/api/datasets").json()["datasets"] if d["name"] == "students_web")
    assert listed["has_claims"] is True and listed["question"] == QUESTION and listed["session"] == {"stage": "waiting", "phase": "after", "runs": 1}
    n = len(v["transcript"])
    v = _say(client, "students_web", "what did you find?")
    assert len(v["transcript"]) == n + 2 and "5.618" in v["transcript"][-1]["text"] and v["transcript"][-1]["phase"] == "after"
    v = _say(client, "students_web", "done")
    assert v["stage"] == "ended" and v["transcript"][-1]["role"] == "system"
    assert client.post("/api/sessions/students_web/messages", json={"text": "hi"}).status_code == 409
    # a new analysis on the same dataset starts with the question again; the memory stays, and so does the run it made
    r = client.post("/api/sessions/students_web/analyses")
    assert r.status_code == 202
    v = wait_idle(client, "students_web")
    assert v["stage"] == "waiting" and v["phase"] == "before" and v["prompt"]["kind"] == "question" and v["question"] is None
    assert [r["index"] for r in v["runs"]] == [1] and v["runs"][0]["question"] == QUESTION and v["runs"][0]["effect"] == 5.618
    assert any(c["key"] == "assignment" and c["status"] == "confirmed" for c in v["claims"])
    assert [t["kind"] for t in v["transcript"] if t["role"] == "system"][-2:] == [None, "divider"]  # ended, then the divider


def test_a_new_analysis_can_start_while_the_desk_waits_and_every_run_stays_listed(client, settings, monkeypatch):
    patch_pipeline(monkeypatch)
    fake = DeskFake(infer=infer_columns)
    _to_ready(client, fake)
    v = _say(client, "students_web", "run")
    assert v["phase"] == "after" and [r["index"] for r in v["runs"]] == [1]
    old_thread = client.app.state.sessions.get("students_web").thread_id
    assert client.post("/api/sessions/students_web/messages", json={"text": "x"}).status_code == 202  # still waiting after a run
    wait_idle(client, "students_web")
    # a new analysis without saying done: the thread so far ends, the run stays listed from disk
    r = client.post("/api/sessions/students_web/analyses")
    assert r.status_code == 202
    v = wait_idle(client, "students_web")
    assert v["stage"] == "waiting" and v["prompt"]["kind"] == "question" and v["transcript"][-2]["kind"] == "divider"
    assert [r["index"] for r in v["runs"]] == [1] and (settings.root / "data/memory/students_web/designs/1/record.json").exists()
    v = _say(client, "students_web", QUESTION)
    while not v["ready"]:
        v = _say(client, "students_web", reply_for(v))
    v = _say(client, "students_web", "run")
    assert [r["index"] for r in v["runs"]] == [1, 2] and v["runs"][1]["question"] == QUESTION and v["prompt"]["text"].startswith("Run 2")
    assert (settings.root / "data/memory/students_web/designs/2/record.json").exists() and (
        settings.root / "data/memory/students_web/designs/1/handoff.json"
    ).exists()
    listed = next(d for d in client.get("/api/datasets").json()["datasets"] if d["name"] == "students_web")
    assert listed["session"]["runs"] == 2
    # the same dataset in a new server lists both runs before any thread is touched; delete drops every thread's checkpoints
    mgr = client.app.state.sessions
    assert mgr.threads("students_web") == [old_thread, mgr.get("students_web").thread_id]
    assert client.delete("/api/datasets/students_web").status_code == 204
    assert mgr.saver.get_tuple({"configurable": {"thread_id": old_thread}}) is None


def test_busy_rejects_a_second_message(client, monkeypatch):
    def slow_run(path, n, dataset, question, decision=None, decision_record=""):
        time.sleep(0.8)
        return canned_run(path, n, dataset, question, decision, decision_record)

    patch_pipeline(monkeypatch, slow_run)
    fake = DeskFake(infer=infer_columns)
    _to_ready(client, fake)
    r = client.post("/api/sessions/students_web/messages", json={"text": "run"})
    assert r.status_code == 202 and r.json()["activity"]
    assert client.post("/api/sessions/students_web/messages", json={"text": "again"}).status_code == 409
    assert client.delete("/api/datasets/students_web").status_code == 409
    v = wait_idle(client, "students_web")
    assert v["phase"] == "after" and len(v["runs"]) == 1


def test_state_survives_a_new_server_over_the_same_checkpoints(client, settings):
    fake = DeskFake()
    v1 = _create(client, fake)
    v1 = _say(client, "students_web", QUESTION)
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


def test_run_view_carries_where_the_lane_disagreed_with_the_pack():
    from causal_agent.desk.contracts import RunRecord
    from causal_agent.server.sessions import run_view

    sr = {
        "status": "done",
        "declines": [
            {
                "stage": "load",
                "kind": "substituted",
                "about": "scope.target",
                "pack_value": "on_treated",
                "took": "effect_at_cutoff",
                "reason": "no average over the treated",
                "check": "target.effect_at_cutoff",
            },
            {"not": "a decline"},
        ],
    }
    v = run_view(RunRecord(index=1, dataset="d", question="q", status="done", specialist_result=sr))
    assert [d.address for d in v.declines] == ["decline:load.scope_target"] and v.declines[0].took == "effect_at_cutoff" and v.declines[0].kind == "substituted"
    assert run_view(RunRecord(index=1, dataset="d", question="q", status="done")).declines == []
