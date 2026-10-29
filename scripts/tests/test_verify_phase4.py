"""The verification runner cannot report empty, skipped, missing or failed evidence as success."""

from types import SimpleNamespace

import pytest

from scripts import verify_phase4 as V


@pytest.mark.parametrize(
    "body", ["", '<testcase name="a"><skipped/></testcase>', '<testcase name="a"><error/></testcase>', '<testcase name="a"><failure/></testcase>']
)
def test_required_test_evidence_rejects_empty_or_nonpassing(tmp_path, body):
    report = tmp_path / "tests.xml"
    report.write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>")
    with pytest.raises(ValueError):
        V.junit(report)


def test_failed_check_retains_report_and_later_success_cannot_hide_it(tmp_path, monkeypatch):
    monkeypatch.setattr(V.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=2))
    checks = V.stages(tmp_path, False)
    V.execute(checks[0], tmp_path)
    checks[1].update(status="passed")
    assert V.save_report(tmp_path, checks, "now", "revision") == "failed"
    report = (tmp_path / "report.md").read_text()
    assert "python-style | failed" in report and "python-style.log" in report
    assert "not_requested" in report and "No new semantic acceptance" in report


def test_zero_exit_without_required_evidence_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(V.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    check = {"name": "missing", "argv": ["unused"], "junit": "absent.xml", "status": "pending"}
    V.execute(check, tmp_path)
    assert check["status"] == "failed"


@pytest.mark.parametrize("live,expected", [(False, "false"), (True, "true")])
def test_model_and_ordinary_stages_set_both_tracing_flags_consistently(tmp_path, monkeypatch, live, expected):
    observed = {}

    def command(*args, **kwargs):
        observed.update(kwargs["env"])
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(V.subprocess, "run", command)
    V.execute({"name": "tracing", "argv": ["unused"], "live": live}, tmp_path)
    assert observed["LANGSMITH_TRACING"] == observed["LANGCHAIN_TRACING_V2"] == expected


def test_models_are_opt_in_and_ci_retains_all_shared_controls(tmp_path):
    default = V.stages(tmp_path, False)
    live = V.stages(tmp_path, True)
    assert default[-1]["status"] == "not_requested" and live[-1]["status"] == "pending"
    backend = next(c for c in default if c["name"] == "storage-and-negative-controls")
    assert "scripts/ingestion_mutations.py" in backend["argv"] and "--phase4" in backend["argv"]
    ci = V.stages(tmp_path, False, all_controls=True)
    assert all("--phase4" not in c["argv"] for c in ci if c.get("manifest")) and ci[-1]["status"] == "not_requested"


@pytest.mark.parametrize("manifest", ['[{"verified": false}]', '[{"verified": true}]', "[]"])
def test_malformed_control_manifest_cannot_pass(tmp_path, monkeypatch, manifest):
    monkeypatch.setattr(V.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0))
    (tmp_path / "controls.json").write_text(manifest)
    check = {"name": "controls", "argv": ["unused"], "manifest": "controls.json", "status": "pending"}
    V.execute(check, tmp_path)
    assert check["status"] == "failed"
