"""Result projection and guarded file delivery; reviewed chat journeys live with the storage harness."""


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
    from causal_agent.common.contracts import RunRecord
    from causal_agent.server.views import run_view

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
