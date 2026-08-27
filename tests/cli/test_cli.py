"""The CLI boundary: parsing, envelope validation, dispatch, rendering, and exit codes.

SYSTEM-CONTRACT §1.1 and PRD-002 §11.1; T-014 §1 and §4. The runtime is a
recording fake, so no database, object store, model, or graph is touched.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from causal.cli.main import (
    CliCommandEnvelopeV1,
    InterruptIdentity,
    StatusView,
    build_parser,
    main,
)
from causal.design.contracts import ApprovalDecision, TableSelectionDecisionV1, UserContextAnswerV1
from causal.intake.contracts import IntakeSubmissionV1

HASH = "a" * 64
ANALYSIS = "an-0123456789abcdef"
KEY = "key-1"


@dataclass(frozen=True)
class FakeIntake:
    analysis_id: str = ANALYSIS
    stage_run_id: str = "sr:an:1"
    status: str = "usable"
    outcome_artifact_id: str | None = "io-1"
    replayed: bool = False


@dataclass(frozen=True)
class FakeDesign:
    status: str = "needs_user_input"
    analysis_id: str = ANALYSIS
    stage_run_id: str = "dr:an:1"
    design_revision: int = 2
    interrupt_kind: str | None = "clarification"
    interrupt_artifact_id: str | None = "uq-1"
    interrupt_hash: str | None = HASH
    outcome_artifact_id: str | None = None
    refusal_code: str | None = None
    error_code: str | None = None


class FakeRuntime:
    """Records exactly what the CLI passed; every method satisfies `Runtime`."""

    def __init__(self, design: FakeDesign | None = None, intake: FakeIntake | None = None) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.design = design if design is not None else FakeDesign()
        self.intake = intake if intake is not None else FakeIntake()

    def new(self, submission: IntakeSubmissionV1) -> FakeIntake:
        self.calls.append(("new", (submission,)))
        return self.intake

    def status(self, analysis_id: str) -> StatusView:
        self.calls.append(("status", (analysis_id,)))
        return StatusView(analysis_id=analysis_id, stage="design", state="waiting_for_user",
                          next_command="answer-context")

    def run(self, analysis_id: str, *, expected_stage_run: str,
            idempotency_key: str) -> FakeDesign:
        self.calls.append(("run", (analysis_id, expected_stage_run, idempotency_key)))
        return self.design

    def deliver(self, bundle_id: str, expected_hash: str) -> dict[str, Any]:
        self.calls.append(("deliver", (bundle_id, expected_hash)))
        if expected_hash != HASH:
            raise ValueError("no such bundle")
        return {"summary": "one frozen sentence [a]", "figures": [], "renders": []}

    def select_table(self, analysis_id: str, decision: TableSelectionDecisionV1) -> FakeDesign:
        self.calls.append(("select_table", (analysis_id, decision)))
        return self.design

    def answer_context(self, analysis_id: str, answer: UserContextAnswerV1,
                       interrupt: InterruptIdentity, idempotency_key: str) -> FakeDesign:
        self.calls.append(("answer_context", (analysis_id, answer, interrupt, idempotency_key)))
        return self.design

    def approve_design(self, analysis_id: str, interrupt: InterruptIdentity,
                       decision: ApprovalDecision, idempotency_key: str,
                       change_requests: tuple[str, ...] = ()) -> FakeDesign:
        self.calls.append(("approve_design",
                           (analysis_id, interrupt, decision, idempotency_key, change_requests)))
        return self.design


def run_cli(runtime: FakeRuntime, *argv: str) -> tuple[int, str]:
    out = StringIO()
    code = main(list(argv), lambda: runtime, out, lambda: "cli-test")
    return code, out.getvalue().strip()


def json_cli(runtime: FakeRuntime, *argv: str) -> tuple[int, dict[str, Any]]:
    code, text = run_cli(runtime, "--format", "json", *argv)
    assert len(text.splitlines()) == 1  # exactly one document on stdout
    parsed: dict[str, Any] = json.loads(text)
    return code, parsed


def answers_file(tmp_path: Path, body: str) -> str:
    path = tmp_path / "answers.json"
    path.write_text(body, encoding="utf-8")
    return str(path)


ANSWERS = json.dumps({"packet_id": "pk-1", "answers": [
    {"question_id": "q1", "answer_kind": "value", "value": "monthly"},
    {"question_id": "q2", "answer_kind": "unknown", "value": None}]})

SELECT_ARGS = ("select-table", ANALYSIS, "--table-id", "sales.csv", "--interrupt-id", "ts-1",
               "--expected-interrupt-hash", HASH, "--expected-revision", "2",
               "--idempotency-key", KEY)
APPROVE_ARGS = ("approve-design", ANALYSIS, "--decision", "approved", "--interrupt-id", "ap-1",
                "--expected-interrupt-hash", HASH, "--expected-revision", "2",
                "--idempotency-key", KEY)
RUN_ARGS = ("run", ANALYSIS, "--expected-stage-run", "dr:an:1", "--idempotency-key", KEY)
NEW_ARGS = ("new", "--question", "Does the discount raise revenue?", "--kaggle", "owner/slug",
            "--idempotency-key", KEY)


def test_new_dispatches_a_typed_submission() -> None:
    runtime = FakeRuntime()
    code, result = json_cli(runtime, *NEW_ARGS)
    name, (submission,) = runtime.calls[0]
    assert name == "new"
    assert isinstance(submission, IntakeSubmissionV1)
    assert submission.question_text == "Does the discount raise revenue?"
    assert submission.kaggle_ref == "owner/slug"
    assert submission.context_text is None
    assert (code, result["status"], result["next_command_name"]) == (0, "accepted", "run")


def test_new_reads_the_context_file(tmp_path: Path) -> None:
    path = tmp_path / "context.txt"
    path.write_text("prior campaign notes", encoding="utf-8")
    runtime = FakeRuntime()
    run_cli(runtime, *NEW_ARGS, "--context-file", str(path))
    _, (submission,) = runtime.calls[0]
    assert isinstance(submission, IntakeSubmissionV1)
    assert submission.context_text == "prior campaign notes"


def test_new_with_a_missing_context_file_is_blocked(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    code, result = json_cli(runtime, *NEW_ARGS, "--context-file", str(tmp_path / "absent.txt"))
    assert (code, result["error_code"], runtime.calls) == (3, "invalid_payload_file", [])


def test_refused_intake_blocks() -> None:
    runtime = FakeRuntime(intake=FakeIntake(status="refused", outcome_artifact_id=None))
    code, result = json_cli(runtime, *NEW_ARGS)
    assert (code, result["status"], result["error_code"]) == (3, "blocked", "intake_refused")
    assert result["next_command_name"] is None


def test_status_reads_committed_state() -> None:
    runtime = FakeRuntime()
    code, result = json_cli(runtime, "status", ANALYSIS)
    assert runtime.calls == [("status", (ANALYSIS,))]
    assert (code, result["status"], result["next_command_name"]) == (0, "completed",
                                                                     "answer-context")
    assert result["message_args"] == {"stage": "design", "state": "waiting_for_user"}


def test_select_table_builds_the_typed_decision() -> None:
    runtime = FakeRuntime()
    code, result = json_cli(runtime, *SELECT_ARGS)
    name, (analysis_id, decision) = runtime.calls[0]
    assert (name, analysis_id) == ("select_table", ANALYSIS)
    assert isinstance(decision, TableSelectionDecisionV1)
    assert decision.selected_table == "sales.csv"
    assert decision.interrupt_id == "ts-1"
    assert decision.expected_interrupt_hash == HASH
    assert decision.expected_revision == 2
    assert decision.idempotency_key == KEY
    assert (code, result["status"], result["next_command_name"]) == (0, "accepted", "run")


def test_answer_context_parses_the_answers_file(tmp_path: Path) -> None:
    runtime = FakeRuntime()
    code, _ = json_cli(runtime, "answer-context", ANALYSIS, "--answers-file",
                       answers_file(tmp_path, ANSWERS), "--interrupt-id", "uq-1",
                       "--expected-interrupt-hash", HASH, "--expected-revision", "2",
                       "--idempotency-key", KEY)
    name, (analysis_id, answer, interrupt, key) = runtime.calls[0]
    assert (name, analysis_id, key, code) == ("answer_context", ANALYSIS, KEY, 0)
    assert isinstance(answer, UserContextAnswerV1)
    assert answer.packet_id == "pk-1"
    assert [item.question_id for item in answer.answers] == ["q1", "q2"]
    assert answer.answers[1].value is None
    assert interrupt == InterruptIdentity(interrupt_id="uq-1", expected_interrupt_hash=HASH,
                                          expected_revision=2)


@pytest.mark.parametrize("body", ["{not json", "{}", json.dumps({"packet_id": "pk-1"}),
                                  json.dumps({"packet_id": "pk-1", "answers": []})])
def test_a_malformed_answers_file_is_blocked(tmp_path: Path, body: str) -> None:
    runtime = FakeRuntime()
    code, result = json_cli(runtime, "answer-context", ANALYSIS, "--answers-file",
                            answers_file(tmp_path, body), "--interrupt-id", "uq-1",
                            "--expected-interrupt-hash", HASH, "--expected-revision", "2",
                            "--idempotency-key", KEY)
    assert (code, result["error_code"], runtime.calls) == (3, "invalid_payload_file", [])


def test_approve_design_passes_raw_decision_fields() -> None:
    runtime = FakeRuntime(design=FakeDesign(status="approved", interrupt_kind=None,
                                            interrupt_artifact_id=None, interrupt_hash=None,
                                            outcome_artifact_id="do-1"))
    code, result = json_cli(runtime, *APPROVE_ARGS)
    name, (analysis_id, interrupt, decision, key, changes) = runtime.calls[0]
    assert (name, analysis_id, key, changes) == ("approve_design", ANALYSIS, KEY, ())
    assert decision is ApprovalDecision.APPROVED
    assert interrupt.interrupt_id == "ap-1"
    assert (code, result["status"], result["message_key"]) == (0, "completed", "stage.finished")
    assert result["message_args"]["outcome_artifact_id"] == "do-1"


def test_run_reports_the_open_interrupt() -> None:
    runtime = FakeRuntime()
    code, result = json_cli(runtime, *RUN_ARGS)
    assert runtime.calls == [("run", (ANALYSIS, "dr:an:1", KEY))]
    assert code == 2
    assert result == {
        "schema_version": "cli-result.v1", "cli_invocation_id": "cli-test", "command_name": "run",
        "analysis_id": ANALYSIS, "stage_run_id": "dr:an:1", "status": "needs_user_input",
        "artifacts": [], "interrupt": {"interrupt_id": "uq-1", "expected_interrupt_hash": HASH,
                                       "expected_revision": 2, "interrupt_kind": "clarification"},
        "next_command_name": "answer-context", "error_code": None, "blocker_event_id": None,
        "message_key": "interrupt.open",
        "message_args": {"design_revision": 2, "design_status": "needs_user_input"}}


def test_presentation_opens_one_exact_bundle_and_prints_its_manifest() -> None:
    """§20: the delivery command names the bundle by exact id and hash, never a latest."""
    runtime = FakeRuntime()
    code, result = json_cli(runtime, "presentation", ANALYSIS, "--bundle-id", "pb-1",
                            "--expected-bundle-hash", HASH)
    assert (code, result["status"]) == (0, "completed")
    assert runtime.calls == [("deliver", ("pb-1", HASH))]
    assert result["artifacts"] == [{"artifact_id": "pb-1", "content_hash": HASH}]


def test_presentation_refuses_a_bundle_the_runtime_will_not_open() -> None:
    """§20: a wrong id or hash is a blocker; the authoritative bundle is untouched."""
    code, result = json_cli(FakeRuntime(), "presentation", ANALYSIS, "--bundle-id", "pb-1",
                            "--expected-bundle-hash", "0" * 64)
    assert (code, result["status"]) == (3, "blocked")


@pytest.mark.parametrize(("design_status", "exit_code", "status"), [
    ("needs_user_input", 2, "needs_user_input"), ("approved", 0, "completed"),
    ("changes_requested", 0, "completed"), ("declined", 0, "completed"),
    ("needs_context", 0, "completed"), ("refused", 3, "blocked"), ("failed", 4, "failed"),
    ("failed_observability", 5, "failed_observability"), ("something_else", 4, "failed")])
def test_exit_codes_follow_the_result_status(design_status: str, exit_code: int,
                                             status: str) -> None:
    runtime = FakeRuntime(design=FakeDesign(status=design_status, refusal_code="capacity_fail"))
    code, result = json_cli(runtime, *RUN_ARGS)
    assert (code, result["status"]) == (exit_code, status)
    assert result["error_code"] == "capacity_fail"


@pytest.mark.parametrize("argv", [
    ("teleport", ANALYSIS), ("new", "--question", "q"), ("run", ANALYSIS),
    ("run", ANALYSIS, "--expected-stage-run", "dr:an:1"),
    ("select-table", ANALYSIS, "--table-id", "t", "--interrupt-id", "i",
     "--expected-interrupt-hash", HASH, "--expected-revision", "2"),
    ("approve-design", ANALYSIS, "--decision", "maybe", "--interrupt-id", "i",
     "--expected-interrupt-hash", HASH, "--expected-revision", "2", "--idempotency-key", KEY),
    ("new", "--question", "q", "--kaggle", "owner/slug", "--idempotency-key", KEY, "--extra", "x"),
    ("status",)])
def test_usage_errors_never_reach_the_runtime(argv: tuple[str, ...]) -> None:
    runtime = FakeRuntime()
    with pytest.raises(SystemExit) as caught:
        run_cli(runtime, *argv)
    assert caught.value.code == 2
    assert runtime.calls == []


@pytest.mark.parametrize("bad_hash", ["not-a-hash", "b" * 63])
def test_payload_validation_blocks_before_the_runtime(bad_hash: str) -> None:
    runtime = FakeRuntime()
    argv = list(SELECT_ARGS)
    argv[argv.index("--expected-interrupt-hash") + 1] = bad_hash
    code, result = json_cli(runtime, *argv)
    assert (code, result["error_code"], runtime.calls) == (3, "invalid_command", [])


def test_an_invalid_kaggle_reference_blocks() -> None:
    runtime = FakeRuntime()
    code, result = json_cli(runtime, "new", "--question", "q", "--kaggle", "not a ref",
                            "--idempotency-key", KEY)
    assert (code, result["error_code"], runtime.calls) == (3, "invalid_command", [])


def test_a_typed_runtime_refusal_keeps_its_error_code() -> None:
    class Refusing(FakeRuntime):
        def run(self, analysis_id: str, *, expected_stage_run: str,
                idempotency_key: str) -> FakeDesign:
            raise _Refusal("busy", "analysis_busy")

    runtime = Refusing()
    code, result = json_cli(runtime, *RUN_ARGS)
    assert (code, result["status"], result["error_code"]) == (3, "blocked", "analysis_busy")


class _Refusal(ValueError):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


@pytest.mark.parametrize(("field", "value"), [
    ("analysis_id", ANALYSIS), ("idempotency_key", None), ("payload_type", None)])
def test_the_envelope_rejects_broken_command_shapes(field: str, value: str | None) -> None:
    base = {"cli_invocation_id": "cli-test", "command_name": "new", "analysis_id": None,
            "expected_identity": None, "idempotency_key": KEY,
            "payload_type": "intake-submission.v1", "payload_hash": HASH,
            "output_format": "human"}
    with pytest.raises(ValidationError):
        CliCommandEnvelopeV1.model_validate({**base, field: value})


def test_the_envelope_accepts_the_creation_shape() -> None:
    envelope = CliCommandEnvelopeV1(
        cli_invocation_id="cli-test", command_name="new", analysis_id=None,
        expected_identity=None, idempotency_key=KEY, payload_type="intake-submission.v1",
        payload_hash=HASH, output_format="json")
    assert envelope.schema_version == "cli-command.v1"


def test_human_output_names_the_next_command_and_hides_payloads() -> None:
    runtime = FakeRuntime()
    code, text = run_cli(runtime, *RUN_ARGS)
    assert code == 2
    assert "Next command: causal answer-context" in text
    assert f"--interrupt-id uq-1 --expected-interrupt-hash {HASH}" in text
    assert "--expected-revision 2" in text
    assert "Open interrupt: clarification uq-1" in text
    assert "Status: needs_user_input" in text


def test_human_output_never_echoes_the_question() -> None:
    runtime = FakeRuntime()
    _, text = run_cli(runtime, *NEW_ARGS)
    assert "Does the discount raise revenue?" not in text
    assert "Next command: causal run" in text
    assert f"{ANALYSIS} --expected-stage-run sr:an:1 --idempotency-key KEY" in text


def test_the_parser_declares_exactly_the_seven_commands() -> None:
    actions = [action for action in build_parser()._actions if action.choices is not None
               and action.dest == "command_name"]
    assert sorted(actions[0].choices or ()) == sorted(
        ["answer-context", "approve-design", "new", "presentation", "run", "select-table",
         "status"])


def test_format_changes_rendering_only() -> None:
    human = FakeRuntime()
    machine = FakeRuntime()
    human_code, _ = run_cli(human, *SELECT_ARGS)
    json_code, _ = json_cli(machine, *SELECT_ARGS)
    assert human.calls[0][1][1] == machine.calls[0][1][1]
    assert human_code == json_code
