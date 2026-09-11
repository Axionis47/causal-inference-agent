"""Run four real datasets through the live runtime, pausing for explicit design review.

Start/resume are checkpointed independently per case, so separate --case-id processes
can run concurrently. Requests in outbox use model_quality's bound response protocol;
reviewers place corresponding responses in inbox before invoking resume.
"""

from __future__ import annotations

import argparse
import fcntl
import io
import json
import os
import re
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from causal.post_analysis.presentation.delivery import DeliveryError, verify_export
from causal.post_analysis.presentation.delivery import export as _export
from causal.post_analysis.runtime import recover_presentation
from causal.runtime.composition import RuntimeConfig, build_runtime
from causal.shared.canonical import content_hash
from causal.shared.events import EventEmitter
from causal.shared.gateway import VERTEX_PROFILE_V1, GatewayResultV1, GenAiTransport, VertexGateway
from causal.shared.tracing import LangSmithTracer, TraceRedactorV1
from tools import model_quality as mq

STATE_SCHEMA = "four-analysis-case.v1"
DELIVERED = {"complete", "complete_with_qualifications"}
MAX_PRESENTATION_RECOVERIES = 2
REQUIRED_ARTIFACTS = {
    "IntakeOutcome", "CompiledDesign", "DesignApprovalDecision", "PreparedFrameBundle",
    "NumericalBundle", "AnalysisSupportingData", "PostAnalysisBundle", "PostAnalysisReview",
}


class _CapturedAuditGateway(mq._AuditGateway):
    """Keep replayable public study inputs and raw decisions outside model/review inputs."""

    def __init__(self, gateway: VertexGateway, canary: str, evaluation: dict[str, str],
                 directory: Path) -> None:
        super().__init__(gateway, canary, evaluation)
        self.directory = directory

    def invoke(self, envelope: Any, prompt: str,
               response_schema: dict[str, object], *, images: tuple[Any, ...] = ()) -> GatewayResultV1:
        identity = {"task_id": envelope.task_id, "attempt_id": envelope.attempt_id,
                    "task_kind": envelope.task_kind, "scope_ids": list(envelope.scope_ids)}
        request = {"schema_version": "four-analysis-model-request.v1", **identity,
                   "capture_id": uuid.uuid4().hex, "prompt": prompt,
                   "prompt_hash": mq._hash(prompt.encode()), "response_schema": response_schema}
        digest = content_hash(request)
        self.directory.mkdir(parents=True, exist_ok=True)
        request_path = self.directory / f"{digest}.request.json"
        response_path = self.directory / f"{digest}.response.json"
        # These files are never loaded into a runtime context or included in a review packet.
        # Deliberately exclude envelopes, environment, client configuration and credentials.
        request_path.write_text(json.dumps(request, indent=2) + "\n", encoding="utf-8")
        try:
            result = super().invoke(envelope, prompt, response_schema, **({"images": images} if images else {}))
        except Exception as error:
            code = str(getattr(error, "code", type(error).__name__))
            if re.fullmatch(r"[A-Za-z0-9_.:-]{1,120}", code) is None:
                code = type(error).__name__
            response = {"schema_version": "four-analysis-model-response.v1", **identity,
                        "request_hash": digest, "error_code": code}
            response_path.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
            raise
        response = {"schema_version": "four-analysis-model-response.v1", **identity,
                    "request_hash": digest,
                    # `text` is the exact provider answer, including invalid JSON. Preserve
                    # the parsed decision beside it; private reasoning is not a decision.
                    "response": result.model_dump(mode="json", exclude={"reasoning"})}
        response_path.write_text(json.dumps(response, indent=2) + "\n", encoding="utf-8")
        return result


def _cases(path: Path) -> list[dict[str, Any]]:
    cases = mq._load(path).get("cases")
    if not isinstance(cases, list) or len(cases) != 4:
        raise ValueError("the dataset manifest must contain exactly four cases")
    if {case.get("family") for case in cases} != set(mq.FAMILIES):
        raise ValueError("the manifest must contain one dataset for each of the four methods")
    ids = [case.get("case_id", "") for case in cases]
    if len(set(ids)) != 4 or not all(re.fullmatch(r"[a-z0-9_]+", name) for name in ids):
        raise ValueError("case IDs must be unique lowercase names with underscores")
    for case in cases:
        if not all(case.get(key) for key in ("question", "context", "file_name", "transform")):
            raise ValueError(f"{case['case_id']} has incomplete study context")
        if not re.fullmatch(r"[a-f0-9]{64}", case.get("source", {}).get("sha256", "")):
            raise ValueError(f"{case['case_id']} needs a pinned source SHA-256")
    return cases


@contextmanager
def _lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as held:
        try:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another process is already updating {path.parent}") from error
        try:
            yield
        finally:
            fcntl.flock(held, fcntl.LOCK_UN)


def _progress(case_id: str, message: str, **fields: Any) -> None:
    print(json.dumps({"case_id": case_id, "progress": message, **fields}), flush=True)


def _initial(case: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA, "run_id": f"four-{uuid.uuid4().hex}",
        "case_hash": content_hash(case), "case": case,
        "execution_content_hash_at_start": mq._execution_content_hash(),
        "execution_history": [], "created_at_utc": datetime.now(UTC).isoformat(),
        "cell_id": case["case_id"], "case_id": case["case_id"],
        "context_variant": "rich", "family": case["family"], "status": "not_started",
        "analysis_id": None, "stage_run_id": None, "database_name": None,
        "bucket": None, "task_results": [], "questions": [], "hitl": [],
        "pending_request": None, "result": None,
    }


def _source(case: dict[str, Any], directory: Path) -> tuple[bytes, bytes]:
    raw_path = directory / "source.bin"
    raw = raw_path.read_bytes() if raw_path.exists() else mq._source_bytes(case["source"])
    if mq._hash(raw) != case["source"]["sha256"]:
        raise ValueError("source_hash_mismatch")
    csv = mq._case_csv(case, raw)
    raw_path.write_bytes(raw)
    (directory / "input.csv").write_bytes(csv)
    return raw, csv


def _export_artifacts(runtime: Any, analysis_id: str, directory: Path) -> dict[str, Any]:
    """Export every committed artifact and verify all copied delivery assets on every reopen."""
    target = directory / "artifacts"
    target.mkdir(exist_ok=True)
    rows = runtime.deps.conn.execute(
        "SELECT artifact_type, artifact_id FROM causal.artifacts WHERE analysis_id=%s "
        "ORDER BY created_at_utc, artifact_id", (analysis_id,)).fetchall()
    manifest: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    diagnostics: list[dict[str, Any]] = []
    bundle = None
    for index, (kind, artifact_id) in enumerate(rows):
        envelope = runtime.deps.products.load_envelope(str(artifact_id))
        raw = runtime.deps.objects.get(envelope.payload_locator)
        if mq._hash(raw) != envelope.content_hash:
            raise RuntimeError(f"artifact_hash_mismatch:{artifact_id}")
        stem = f"{index:04d}-{kind}"
        payload_path = target / f"{stem}.json"
        payload_path.write_bytes(raw)
        (target / f"{stem}.envelope.json").write_text(
            envelope.model_dump_json(indent=2) + "\n", encoding="utf-8")
        payload = json.loads(raw)
        payloads[str(kind)] = payload
        if kind == "DiagnosticResult":
            diagnostics.append(payload)
        manifest.append({"artifact_type": kind, "artifact_id": str(artifact_id),
                         "content_hash": envelope.content_hash,
                         "path": str(payload_path.relative_to(directory))})
        if kind in {"PresentationBundle", "PostAnalysisBundle"}:
            bundle = envelope
    mq._save_state(target / "manifest.json", {"artifacts": manifest})
    delivery: dict[str, Any] | None = None
    if bundle is not None:
        delivery = runtime.deliver(bundle.artifact_id, bundle.content_hash)
        export_dir = directory / "presentation"
        if not export_dir.exists() or not any(export_dir.iterdir()):
            _export(delivery, export_dir)
        # Reopening the exact persisted bundle and verifying the actual exported bytes is
        # required even when recovering a process that already copied the delivery.
        try:
            verified = verify_export(delivery, export_dir)
        except DeliveryError as error:
            mq._save_state(directory / "delivery.json", {
                "bundle_id": bundle.artifact_id, "bundle_hash": bundle.content_hash,
                "export_verified": False, "verification_error": error.code})
            raise
        for record in [*verified["figures"], *([verified["report"]] if "report" in verified else [])]:
            record["assets"] = {key: str(Path(path).relative_to(directory.resolve()))
                                for key, path in record["assets"].items()}
        delivery = {"bundle_id": bundle.artifact_id, "bundle_hash": bundle.content_hash,
                    "summary": delivery["summary"], **verified,
                    "export_verified": True}
        mq._save_state(directory / "delivery.json", delivery)
    return {"manifest": manifest, "payloads": payloads, "delivery": delivery,
            "diagnostics": diagnostics}


def _result(state: dict[str, Any], exported: dict[str, Any], status: str) -> dict[str, Any]:
    payloads = exported["payloads"]
    missing = sorted(REQUIRED_ARTIFACTS - payloads.keys())
    calls = state["task_results"]
    kinds = {row["task_kind"] for row in calls if "error_code" not in row}
    live_review = {"method_design", "post_analysis_author", "post_analysis_review"} <= kinds
    approval = any(row["kind"] == "approval" for row in state["hitl"])
    delivery = exported["delivery"]
    execution_passed = bool(status in DELIVERED and delivery and not missing and live_review
                            and approval and delivery["export_verified"])
    return {
        "case_id": state["case_id"], "analysis_id": state["analysis_id"],
        "family": state["family"], "terminal_outcome": status,
        "passed": execution_passed, "execution_passed": execution_passed,
        "acceptance_status": "pending_review",
        "source_hash": state["source_hash"], "input_hash": state["input_hash"],
        "source": state["case"]["source"], "question": state["case"]["question"],
        "execution_history": state["execution_history"],
        "recovery_history": state.get("recovery_history", []), "missing_artifacts": missing,
        "live_design_and_post_analysis_review": live_review,
        "task_results": calls, "approval_history": state["hitl"],
        "post_analysis_review": payloads.get("PostAnalysisReview"),
        "report": payloads.get("PostAnalysisDraft"),
        "primary_result": payloads.get("PrimaryAnalysisResult"),
        "diagnostic_results": exported.get("diagnostics", []),
        "delivery": delivery, "artifact_count": len(exported["manifest"]),
    }


def _report(state: dict[str, Any], directory: Path) -> None:
    result = state.get("result") or {}
    lines = [f"# {state['case_id']}", "", state["case"]["question"], "",
             f"Status: **{result.get('terminal_outcome', state['status'])}**", "",
             f"Dataset: {state['case']['source'].get('provenance', '')}", ""]
    delivery = result.get("delivery") or {}
    if delivery:
        lines.extend([delivery["summary"], ""])
    claim = result.get("claim_judgment") or {}
    primary = result.get("primary_result") or {}
    if primary.get("primary_items"):
        lines.extend(["| Contrast | Estimate | Confidence interval | Units |",
                      "| --- | ---: | --- | --- |"])
        for row in primary["primary_items"]:
            level = f"{100 * row['confidence_level']:g}%"
            lines.append(f"| {row['estimand_label']} | {row['estimate']:.6g} | "
                         f"{level}: [{row['interval_lower']:.6g}, {row['interval_upper']:.6g}] | "
                         f"{row['estimate_units']} |")
        lines.append("")
        counts = primary["primary_items"][0].get("contributing_counts", {})
        if counts:
            lines.extend(["Contributing sample: " + ", ".join(
                f"{key}={value}" for key, value in counts.items()) + ".", ""])
        quantities = primary["primary_items"][0].get("method_quantities", {})
        if quantities.get("fitted_covariance"):
            lines.extend([f"Fitted covariance: {quantities['fitted_covariance']}.", ""])
    if claim.get("sensitivity_summary"):
        lines.extend(["## Prespecified sensitivities", "", claim["sensitivity_summary"], ""])
    for row in result.get("diagnostic_results", []):
        if row.get("diagnostic_id") == "selected_bandwidth_report":
            lines.extend(["Selected RDD bandwidth and local support: " + ", ".join(
                f"{key}={value}" for key, value in row.get("values", {}).items()) + ".", ""])
    lines.extend(["Execution checks: " + ("passed" if result.get("execution_passed",
        result.get("passed", False)) else "incomplete"),
        "Final numerical and visual acceptance: " + (
            "accepted" if _accepted(result, directory) else "pending review"), ""])
    for label, key in (("Qualifications", "qualifications"), ("Cannot conclude", "cannot_conclude")):
        if claim.get(key):
            lines.extend([f"## {label}", "", *[f"- {row}" for row in claim[key]], ""])
    for figure in delivery.get("figures", []):
        if figure["assets"].get("png"):
            lines.extend([f"![{figure.get('visual_id', figure.get('figure_id'))}]({figure['assets']['png']})", ""])
    balance = [(name, value) for row in result.get("diagnostic_results", [])
               if row.get("diagnostic_id") in {"baseline_balance", "weighted_covariate_balance"}
               for name, value in row.get("values", {}).items()
               if "standardized_difference" in name]
    if balance:
        lines.extend(["## Complete balance diagnostics", "",
                      ("All computed levels are retained below; a categorical plot may summarize "
                       "its largest absolute standardized mean difference."), "",
                      "| Covariate / level | Standardized mean difference |", "| --- | ---: |",
                      *[f"| {name.replace('|', '/')} | {value:.6g} |" for name, value in balance], ""])
    if state.get("pending_request"):
        request = state["pending_request"]
        lines.extend([f"Review required: {request['kind']}", "",
                      f"Request: `{request['path']}`", ""])
    if state.get("error"):
        lines.extend([f"Error: `{state['error']}`", ""])
    lines.extend(["[Result JSON](result.json) · [Artifact manifest](artifacts/manifest.json)", ""])
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _recovery_request(state: dict[str, Any], reason: str | None) -> dict[str, Any]:
    prior = state.get("result") or {}
    if (state["status"] != "terminal" or prior.get("terminal_outcome") not in {"failed", "incomplete"}
            or not str(state.get("stage_run_id", "")).startswith("ps:")
            or prior.get("delivery") or state.get("pending_request")):
        raise ValueError("recover requires a terminal failed presentation with no delivered bundle")
    if not reason or not reason.strip():
        raise ValueError("recover requires --reason describing the implementation fix")
    if len(state.get("recovery_history", [])) >= MAX_PRESENTATION_RECOVERIES:
        raise ValueError("the two explicit presentation recovery attempts are exhausted")
    previous = (prior.get("execution_history") or state["execution_history"])[-1][
        "execution_content_hash"]
    current = mq._execution_content_hash()
    if previous == current:
        raise ValueError("execution code must change after the failed invocation before recover")
    return {"reason": reason.strip(), "failed_stage_run_id": state["stage_run_id"],
            "previous_execution_content_hash": previous, "execution_content_hash": current,
            "requested_at_utc": datetime.now(UTC).isoformat(),
            "previous_result": json.loads(json.dumps(prior))}


def advance(case: dict[str, Any], output: Path, mode: str,
            reason: str | None = None) -> dict[str, Any]:
    directory = output / case["case_id"]
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "state.json"
    with _lock(directory / ".lock"):
        if state_path.exists():
            state = mq._load(state_path)
            if state.get("schema_version") != STATE_SCHEMA or state["case_hash"] != content_hash(case):
                raise ValueError("case input changed; use a new output directory")
            if mode == "start":
                raise ValueError(f"{case['case_id']} already exists; use resume")
        else:
            if mode in {"resume", "recover"}:
                raise ValueError(f"{case['case_id']} has not started")
            state = _initial(case)
            mq._save_state(state_path, state, create=True)
        recovery = _recovery_request(state, reason) if mode == "recover" else None
        if state["status"] == "terminal" and recovery is None:
            return state
        response = None
        if state.get("pending_request"):
            request = state["pending_request"]
            response = mq._hitl_response(
                mq._response_path(output / "inbox", request), state, state, request)
            if response is None:
                _progress(case["case_id"], "review_required", request=request["path"])
                return state
        tracer = None
        runtime = None
        audit = None
        try:
            _progress(case["case_id"], "loading_verified_source")
            if recovery is None:
                raw, csv = _source(case, directory)
            else:
                raw, csv = (directory / "source.bin").read_bytes(), (directory / "input.csv").read_bytes()
                if mq._hash(raw) != state["source_hash"] or mq._hash(csv) != state["input_hash"]:
                    raise ValueError("recovery input archive hash mismatch")
            state.update(source_hash=mq._hash(raw), input_hash=mq._hash(csv))
            state["execution_history"].append({
                "at_utc": datetime.now(UTC).isoformat(),
                "execution_content_hash": mq._execution_content_hash()})
            if not state["database_name"]:
                state["database_name"], _ = mq._database(state["run_id"], case["case_id"])
                mq._save_state(state_path, state)
            if not state["bucket"]:
                state["bucket"] = mq._bucket(state["run_id"], case["case_id"])
                mq._save_state(state_path, state)
            tracer = LangSmithTracer(
                os.environ["CAUSAL_LANGSMITH_PROJECT"], "evaluation", TraceRedactorV1())
            tracer.preflight()
            gateway = VertexGateway(
                GenAiTransport(), VERTEX_PROFILE_V1, EventEmitter(io.StringIO()),
                lambda: datetime.now(UTC), tracer)
            audit = _CapturedAuditGateway(gateway, "FOUR_ANALYSES_PRIVATE_EVALUATION_SENTINEL", {
                "run_id": state["run_id"], "case_id": case["case_id"], "mode": "four-analyses"},
                directory / "model-calls")
            runtime = build_runtime(RuntimeConfig(
                postgres_dsn=mq._matrix_dsn(state["database_name"]),
                s3_bucket=state["bucket"], s3_endpoint_url=os.environ["CAUSAL_EVAL_S3_ENDPOINT"],
                langsmith_project=os.environ["CAUSAL_LANGSMITH_PROJECT"], environment="evaluation",
                event_log=directory / "events.ndjson"),
                client_factory=lambda: mq._FixtureClient(case, csv), model=audit,
                strict_observability=True)
            if state["analysis_id"] is None:
                intake = runtime.new(mq._intake_submission(case))
                state.update(analysis_id=intake.analysis_id, stage_run_id=intake.stage_run_id,
                             status="running")
                mq._save_state(state_path, state)
            _progress(case["case_id"], "running_live_pipeline", analysis_id=state["analysis_id"])
            if recovery is not None:
                _progress(case["case_id"], "recovering_failed_presentation",
                          failed_stage_run_id=recovery["failed_stage_run_id"])
                with runtime._lock(state["analysis_id"], "recover-presentation"):
                    row = runtime.deps.conn.execute(
                        "SELECT run_record FROM presentation.runs WHERE stage_run_id=%s",
                        (recovery["failed_stage_run_id"],)).fetchone()
                    recovery["failed_presentation_record"] = None if row is None else row[0]
                    archive = directory / "recoveries" / str(len(state.get("recovery_history", [])) + 1)
                    archive.mkdir(parents=True, exist_ok=False)
                    for name in ("state.json", "result.json", "report.md"):
                        if (directory / name).exists():
                            (archive / name).write_bytes((directory / name).read_bytes())
                    state.setdefault("recovery_history", []).append(recovery)
                    mq._save_state(state_path, state)
                    latest = runtime._latest_design_run(state["analysis_id"])
                    if latest is None:
                        raise ValueError("recovery has no approved design revision")
                    outcome = recover_presentation(
                        runtime.deps, runtime.est, runtime.pres, state["analysis_id"], latest.revision,
                        failed_stage_run_id=recovery["failed_stage_run_id"])
                    recovery.update(recovered_stage_run_id=outcome.stage_run_id,
                                    terminal_outcome=outcome.status)
            elif response is not None:
                outcome = mq._apply_hitl(runtime, state, state["pending_request"], response)
                state["stage_run_id"] = outcome.stage_run_id
                mq._save_state(state_path, state)
            else:
                latest = runtime._latest_design_run(state["analysis_id"])
                expected = latest.stage_run_id if latest else state["stage_run_id"]
                outcome = runtime.run(state["analysis_id"], expected_stage_run=expected,
                                      idempotency_key=f"four:run:{uuid.uuid4().hex}")
            if outcome.status in {"approved", "changes_requested"}:
                _progress(case["case_id"], "continuing_approved_analysis")
                outcome = runtime.run(state["analysis_id"], expected_stage_run=outcome.stage_run_id,
                                      idempotency_key=f"four:analysis:{uuid.uuid4().hex}")
            state["task_results"].extend(audit.calls)
            audit.calls.clear()
            state["stage_run_id"] = outcome.stage_run_id
            exported = _export_artifacts(runtime, state["analysis_id"], directory)
            if outcome.status == "needs_user_input":
                opened, _ = mq._interrupt(runtime, outcome)
                state["pending_request"] = mq._hitl_request(
                    state["run_id"], case, state, opened, output / "outbox")
                state["status"] = "pending_hitl"
                _progress(case["case_id"], "review_required",
                          request=state["pending_request"]["path"])
            else:
                state["result"] = _result(state, exported, outcome.status)
                state["status"] = "terminal"
                mq._save_state(directory / "result.json", state["result"])
                _progress(case["case_id"], outcome.status, passed=state["result"]["passed"])
            state.pop("error", None)
        except Exception as error:
            state["error"] = str(getattr(error, "code", type(error).__name__))
            _progress(case["case_id"], "interrupted", error=state["error"],
                      detail=str(error)[:1000])
            raise
        finally:
            if audit is not None:
                state["task_results"].extend(audit.calls)
            mq._save_state(state_path, state)
            _report(state, directory)
            if runtime is not None:
                runtime.close()
            if tracer is not None:
                tracer.flush()
        return state


def _accepted(result: dict[str, Any], directory: Path) -> bool:
    """An execution result alone is not numerical and visual acceptance of its exact export."""
    path = directory / "acceptance.json"
    if not path.exists() or not result.get("execution_passed", result.get("passed", False)):
        return False
    review = mq._load(path)
    expected = {"analysis_id": result.get("analysis_id"), "input_hash": result.get("input_hash"),
                "bundle_hash": (result.get("delivery") or {}).get("bundle_hash")}
    return bool(review.get("schema_version") == "four-analysis-acceptance.v1"
                and review.get("reviewer") and all(expected.values())
                and all(review.get(key) == value for key, value in expected.items())
                and all(review.get(key) is True for key in (
                    "numerical_verified", "charts_verified", "interpretation_verified"))
                and review.get("evidence_paths"))


def status(cases: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    rows = []
    for case in cases:
        path = output / case["case_id"] / "state.json"
        state = mq._load(path) if path.exists() else {}
        result = state.get("result") or {}
        request = state.get("pending_request") or {}
        accepted = _accepted(result, output / case["case_id"])
        rows.append({"case_id": case["case_id"], "status": state.get("status", "not_started"),
                     "terminal_outcome": result.get("terminal_outcome"),
                     "passed": accepted, "accepted": accepted,
                     "execution_passed": result.get("execution_passed", result.get("passed", False)),
                     "error": state.get("error"),
                     "request": request.get("path"),
                     "report": str(output / case["case_id"] / "report.md")})
    return {"schema_version": "four-analyses-summary.v1", "cases": rows,
            "passed": all(row["passed"] for row in rows)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("start", "resume", "recover", "status"))
    parser.add_argument("--fixtures", type=Path, default=mq.STRESS_FIXTURES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--case-id", help="Run only this case; safe for parallel processes")
    parser.add_argument("--reason", help="Required for explicit recovery after an implementation fix")
    args = parser.parse_args(argv)
    if args.mode == "recover" and (not args.case_id or not args.reason or not args.reason.strip()):
        parser.error("recover requires --case-id and --reason")
    cases = _cases(args.fixtures)
    if args.case_id and args.case_id not in {case["case_id"] for case in cases}:
        parser.error("--case-id does not occur in the dataset manifest")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    failed = False
    if args.mode != "status":
        for case in cases:
            if args.case_id and case["case_id"] != args.case_id:
                continue
            try:
                state = (advance(case, output, args.mode, args.reason) if args.mode == "recover"
                         else advance(case, output, args.mode))
                if state.get("status") == "terminal" and not (state.get("result") or {}).get("passed"):
                    failed = True
            except Exception as error:  # noqa: BLE001 - one failure cannot erase other cases
                failed = True
                _progress(case["case_id"], "case_error", error=str(error)[:1000])
    summary = status(cases, output)
    # Independent case processes never share a mutable case state. A status refresh writes
    # the aggregate only after reading those atomic checkpoints.
    if args.mode == "status" or not args.case_id:
        mq._save_state(output / "summary.json", summary)
        lines = ["# Four end-to-end analyses", "", "| Dataset | Execution | Acceptance | Report |",
                 "| --- | --- | --- | --- |"]
        for row in summary["cases"]:
            lines.append(f"| {row['case_id']} | {row['terminal_outcome'] or row['status']} | "
                         f"{'accepted' if row['accepted'] else 'pending'} | "
                         f"[Open]({row['case_id']}/report.md) |")
        (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    terminal_failure = (all(row["status"] == "terminal" for row in summary["cases"])
                        and not summary["passed"])
    return 1 if failed or terminal_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
