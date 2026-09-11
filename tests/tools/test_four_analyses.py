"""Safety and delivery checks for the resumable four-dataset live runner."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from causal.shared.gateway import GatewayError, GatewayResultV1
from tests.shared.test_gateway import SCHEMA, make_envelope
from tools import four_analyses as four


def test_manifest_requires_one_dataset_per_method(tmp_path: Path) -> None:
    document = four.mq._load(four.mq.STRESS_FIXTURES)
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(document))
    assert len(four._cases(path)) == 4
    document["cases"][3]["family"] = document["cases"][0]["family"]
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="one dataset"):
        four._cases(path)


def test_tampered_cached_source_is_rejected_before_transform(tmp_path: Path) -> None:
    case = four._cases(four.mq.STRESS_FIXTURES)[0]
    (tmp_path / "source.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source_hash_mismatch"):
        four._source(case, tmp_path)
    assert not (tmp_path / "input.csv").exists()


def _passing() -> tuple[dict[str, Any], dict[str, Any]]:
    case = four._cases(four.mq.STRESS_FIXTURES)[0]
    state = {
        "case": case, "case_id": case["case_id"], "analysis_id": "analysis-1",
        "family": case["family"], "execution_history": [],
        "source_hash": "a" * 64, "input_hash": "b" * 64,
        "task_results": [{"task_kind": kind} for kind in (
            "method_design", "post_analysis_author", "post_analysis_review")],
        "hitl": [{"kind": "approval", "reviewer": "independent"}],
    }
    exported = {"payloads": {kind: {} for kind in four.REQUIRED_ARTIFACTS},
                "manifest": list(four.REQUIRED_ARTIFACTS),
                "delivery": {"export_verified": True}}
    return state, exported


def test_delivery_requires_all_stages_live_reviews_and_bound_approval() -> None:
    state, exported = _passing()
    assert four._result(state, exported, "complete_with_qualifications")["passed"]
    assert not four._result(state, exported, "blocked")["passed"]
    state["hitl"] = []
    assert not four._result(state, exported, "complete")["passed"]


def test_final_acceptance_requires_review_of_the_exact_export(tmp_path: Path) -> None:
    result = {"execution_passed": True, "analysis_id": "analysis-1", "input_hash": "a" * 64,
              "delivery": {"bundle_hash": "b" * 64}}
    assert not four._accepted(result, tmp_path)
    review = {"schema_version": "four-analysis-acceptance.v1", "reviewer": "Test reviewer",
              "analysis_id": "analysis-1", "input_hash": "a" * 64, "bundle_hash": "b" * 64,
              "numerical_verified": True, "charts_verified": True,
              "interpretation_verified": True, "evidence_paths": ["verification.json"]}
    four.mq._save_state(tmp_path / "acceptance.json", review)
    assert four._accepted(result, tmp_path)
    for change in ({"charts_verified": False}, {"bundle_hash": "c" * 64},
                   {"input_hash": "d" * 64}, {"reviewer": ""}):
        four.mq._save_state(tmp_path / "acceptance.json", review | change)
        assert not four._accepted(result, tmp_path)
    state, exported = _passing()
    state["task_results"].pop()
    assert not four._result(state, exported, "complete")["passed"]
    state, exported = _passing()
    del exported["payloads"]["PreparedFrameBundle"]
    assert not four._result(state, exported, "complete")["passed"]
    state, exported = _passing()
    exported["delivery"]["export_verified"] = False
    assert not four._result(state, exported, "complete")["passed"]


def test_pending_review_never_starts_or_approves_runtime(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = four._cases(four.mq.STRESS_FIXTURES)[0]
    state = four._initial(case)
    state["status"] = "pending_hitl"
    state["pending_request"] = {"path": "review-this.json", "cell_id": case["case_id"],
                                "request_hash": "a" * 64}
    state_path = tmp_path / case["case_id"] / "state.json"
    four.mq._save_state(state_path, state)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("an unanswered review must not invoke the runtime")

    monkeypatch.setattr(four, "build_runtime", forbidden)
    resumed = four.advance(case, tmp_path, "resume")
    assert resumed["status"] == "pending_hitl"
    assert resumed["hitl"] == []


def test_case_scoped_start_preserves_other_checkpoints(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    cases = four._cases(four.mq.STRESS_FIXTURES)

    def record(case: dict[str, Any], output: Path, mode: str) -> dict[str, Any]:
        calls.append(case["case_id"])
        return {}

    monkeypatch.setattr(four, "advance", record)
    assert four.main(["start", "--output-dir", str(tmp_path),
                      "--case-id", cases[2]["case_id"]]) == 0
    assert calls == [cases[2]["case_id"]]
    assert not (tmp_path / "summary.json").exists()


def _export_runtime(directory: Path, *, report: bool = False) -> tuple[Any, list[Any], str]:
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"summary": "Approved result."}
    raw = json.dumps(payload).encode()
    digest = four.mq._hash(raw)
    source = directory / "render.png"
    source.write_bytes(b"frozen-render")
    envelope = SimpleNamespace(
        payload_locator="objects/bundle", content_hash=digest, artifact_id="bundle-1",
        model_dump_json=lambda **_: json.dumps({"artifact_id": "bundle-1"}))
    kind = "PostAnalysisBundle" if report else "PresentationBundle"
    conn = SimpleNamespace(execute=lambda *_: SimpleNamespace(fetchall=lambda: [(kind, "bundle-1")]))
    body = {"summary": payload["summary"], "figures": [{
        "figure_id": "figure-1", "objects": {"png": str(source)},
        "object_hashes": {"png": four.mq._hash(source.read_bytes())}}]}
    if report:
        objects, hashes = {}, {}
        for key, data in {"html": b"<html>Exact report</html>", "page_001_png": b"frozen-page-png",
                          "page_001_svg": b"<svg>Exact page</svg>"}.items():
            path = directory / key
            path.write_bytes(data)
            objects[key], hashes[key] = str(path), four.mq._hash(data)
        body["delivery"] = {"objects": objects, "object_hashes": hashes,
                            "preview_keys": ["page_001_png"], "draft_hash": "d" * 64}
    calls = []

    def deliver(bundle_id: str, expected_hash: str) -> dict[str, Any]:
        calls.append((bundle_id, expected_hash))
        return body

    runtime = SimpleNamespace(deps=SimpleNamespace(conn=conn,
        products=SimpleNamespace(load_envelope=lambda _: envelope),
        objects=SimpleNamespace(get=lambda _: raw)), deliver=deliver)
    return runtime, calls, digest


def test_export_reopens_exact_bundle_and_verifies_copied_render(tmp_path: Path) -> None:
    runtime, calls, digest = _export_runtime(tmp_path / "sources")
    exported = four._export_artifacts(runtime, "analysis-1", tmp_path)
    assert calls == [("bundle-1", digest)]
    assert exported["delivery"]["export_verified"] is True
    assert "report" not in exported["delivery"]
    assert (tmp_path / "presentation" / "figure-1.png").read_bytes() == b"frozen-render"
    (tmp_path / "presentation" / "figure-1.png").write_bytes(b"tampered")
    with pytest.raises(four.DeliveryError) as caught:
        four._export_artifacts(runtime, "analysis-1", tmp_path)
    assert caught.value.code == "export_hash_mismatch"


@pytest.mark.parametrize("asset", ["html", "page_001_png", "page_001_svg", "summary"])
def test_restart_rejects_tampered_report_and_invalidates_cached_verification(
    tmp_path: Path, asset: str
) -> None:
    runtime, _, _ = _export_runtime(tmp_path / "sources", report=True)
    exported = four._export_artifacts(runtime, "analysis-1", tmp_path)
    delivery = exported["delivery"]
    assert delivery["export_verified"] is True
    report = delivery["report"]
    assert report["preview_keys"] == ["page_001_png"] and report["draft_hash"] == "d" * 64
    assert report["assets"] == {"html": "presentation/report.html",
        "page_001_png": "presentation/report.page_001.png",
        "page_001_svg": "presentation/report.page_001.svg"}
    assert four.mq._load(tmp_path / "delivery.json")["report"] == report
    copied = tmp_path / (report["assets"][asset] if asset != "summary" else "presentation/summary.txt")
    copied.write_bytes(b"tampered after the first process ended")
    restarted, calls, digest = _export_runtime(tmp_path / "sources", report=True)
    with pytest.raises(four.DeliveryError) as caught:
        four._export_artifacts(restarted, "analysis-1", tmp_path)
    assert caught.value.code == "export_hash_mismatch"
    assert calls == [("bundle-1", digest)]
    receipt = four.mq._load(tmp_path / "delivery.json")
    assert receipt["export_verified"] is False
    assert receipt["verification_error"] == "export_hash_mismatch"


def test_same_case_cannot_be_updated_by_two_processes(tmp_path: Path) -> None:
    with (four._lock(tmp_path / ".lock"), pytest.raises(RuntimeError, match="another process"),
          four._lock(tmp_path / ".lock")):
        pytest.fail("duplicate case lock was acquired")


@pytest.mark.parametrize(("case_status", "passed", "exit_code"), (
    ("terminal", False, 1), ("terminal", True, 0), ("pending_hitl", False, 0)))
def test_run_exit_status_distinguishes_failure_from_pending_review(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        case_status: str, passed: bool, exit_code: int) -> None:
    monkeypatch.setattr(four, "advance", lambda *_: {
        "status": case_status, "result": {"passed": passed}})
    case = four._cases(four.mq.STRESS_FIXTURES)[0]
    assert four.main(["start", "--output-dir", str(tmp_path),
                      "--case-id", case["case_id"]]) == exit_code


def test_status_exits_with_failure_when_four_finished_and_one_failed(tmp_path: Path) -> None:
    for index, case in enumerate(four._cases(four.mq.STRESS_FIXTURES)):
        four.mq._save_state(tmp_path / case["case_id"] / "state.json", {
            "status": "terminal", "result": {"passed": index != 0}})
    assert four.main(["status", "--output-dir", str(tmp_path)]) == 1


@pytest.mark.parametrize(("raw", "parsed"), (
    (' {"decision": "revised"} ', {"decision": "revised"}), ("invalid-json", None)))
def test_model_capture_preserves_raw_decision_schema_and_existing_audit(
        tmp_path: Path, raw: str, parsed: dict[str, object] | None) -> None:
    result = GatewayResultV1(text=raw, parsed=parsed, reasoning="private-reasoning",
                             token_usage={"total": 12}, attempts=1, seed=12, finish_reason="STOP")
    forwarded = []

    def invoke(envelope: Any, prompt: str, schema: dict[str, object]) -> GatewayResultV1:
        forwarded.append((envelope, prompt, schema))
        return result

    gateway: Any = SimpleNamespace(invoke=invoke, client_config={"secret": "not-to-be-saved"})
    capture = four._CapturedAuditGateway(gateway, "canary", {"case_id": "case"}, tmp_path)
    envelope = make_envelope("../../attempt-with-unsafe-filename")
    assert capture.invoke(envelope, "Public study context.", SCHEMA) is result
    request_path, = tmp_path.glob("*.request.json")
    response_path, = tmp_path.glob("*.response.json")
    assert len(request_path.name.removesuffix(".request.json")) == 64
    request = json.loads(request_path.read_text())
    response = json.loads(response_path.read_text())
    assert request["prompt"] == forwarded[0][1] == "Public study context."
    assert request["response_schema"] == forwarded[0][2] == SCHEMA
    assert request["task_id"] == response["task_id"] == envelope.task_id
    assert response["response"]["text"] == raw
    assert response["response"]["parsed"] == parsed
    assert capture.calls[0]["task_id"] == envelope.task_id
    assert "not-to-be-saved" not in request_path.read_text() + response_path.read_text()
    assert "private-reasoning" not in response_path.read_text()


def test_model_capture_preserves_exception_and_saves_only_safe_error_code(tmp_path: Path) -> None:
    error = GatewayError("credential or arbitrary provider message", "model_unavailable")

    def invoke(*_: Any) -> GatewayResultV1:
        raise error

    gateway: Any = SimpleNamespace(invoke=invoke)
    capture = four._CapturedAuditGateway(gateway, "canary", {"case_id": "case"}, tmp_path)
    with pytest.raises(GatewayError) as caught:
        capture.invoke(make_envelope(), "Public study context.", SCHEMA)
    assert caught.value is error
    response_path, = tmp_path.glob("*.response.json")
    response = json.loads(response_path.read_text())
    assert response["error_code"] == capture.calls[0]["error_code"] == "model_unavailable"
    assert "credential" not in response_path.read_text()


def test_recovery_requires_failed_presentation_reason_and_changed_execution_code(
        monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(four.mq, "_execution_content_hash", lambda: "new-code")
    state = {"status": "terminal", "stage_run_id": "ps:an-1:1",
             "execution_history": [{"execution_content_hash": "old-code"}],
             "result": {"terminal_outcome": "failed"}}
    request = four._recovery_request(state, "Fix description of mandatory references")
    assert request["failed_stage_run_id"] == "ps:an-1:1"
    assert request["previous_execution_content_hash"] == "old-code"
    with pytest.raises(ValueError, match="--reason"):
        four._recovery_request(state, "")
    monkeypatch.setattr(four.mq, "_execution_content_hash", lambda: "old-code")
    with pytest.raises(ValueError, match="must change"):
        four._recovery_request(state, "retry")
    state["result"]["terminal_outcome"] = "complete"
    with pytest.raises(ValueError, match="terminal failed"):
        four._recovery_request(state, "retry")
