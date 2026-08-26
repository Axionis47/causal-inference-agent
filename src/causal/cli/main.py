"""The seven-command `argparse` boundary, its envelope, and its result (SC §1.1).

PRD-002 §11.1 owns interrupt and resume. No model, tool, LangGraph, database, or
object-store import lives here: every effect belongs to the `Runtime` Protocol.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import IO, Annotated, Any, Final, Literal, Protocol, Self, get_args

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from causal.cli.render import render
from causal.design.contracts import ApprovalDecision, TableSelectionDecisionV1, UserContextAnswerV1
from causal.intake.contracts import IntakeSubmissionV1
from causal.shared.canonical import content_hash
from causal.shared.contracts import ArtifactRef, Identity, Sha256Hex

__all__ = [
    "COMMANDS", "EXIT_CODES", "CliCommandEnvelopeV1", "CliResultV1", "DesignOutcomeView",
    "IntakeOutcomeView", "InterruptIdentity", "OutcomeView", "Runtime", "StatusView",
    "build_parser", "main",
]

CommandName = Literal[
    "new", "status", "select-table", "answer-context", "approve-design", "run", "presentation"]
CliStatus = Literal[
    "accepted", "completed", "needs_user_input", "blocked", "failed", "failed_observability"]
MessageKey = Literal[
    "analysis.created", "status.read", "command.accepted", "interrupt.open", "stage.finished",
    "command.blocked", "command.failed"]
OutputFormat = Literal["human", "json"]
MessageArgs = dict[str, str | int | bool]

COMMANDS: Final[tuple[CommandName, ...]] = get_args(CommandName)
MUTATING: Final = ("new", "select-table", "answer-context", "approve-design", "run")
ANSWER_COMMANDS: Final = ("select-table", "answer-context", "approve-design")
BINDING_FLAGS: Final = (
    "--interrupt-id", "--expected-interrupt-hash", "--expected-revision", "--idempotency-key")
INTERRUPT_COMMANDS: Final[dict[str, CommandName]] = {
    "table_selection": "select-table", "clarification": "answer-context",
    "approval": "approve-design"}
EXIT_CODES: Final[dict[str, int]] = {
    "accepted": 0, "completed": 0, "needs_user_input": 2, "blocked": 3, "failed": 4,
    "failed_observability": 5}
# Terminal design statuses (PRD-002 §6) mapped onto the six CLI statuses.
DESIGN_STATUS: Final[dict[str, CliStatus]] = {
    "needs_user_input": "needs_user_input", "approved": "completed", "needs_context": "completed",
    "changes_requested": "completed", "declined": "completed", "refused": "blocked",
    "failed": "failed", "failed_observability": "failed_observability",
    # PRD-003 §5.2: a conflict or an unrunnable frame still completes the preparation stage.
    "prepared": "completed", "design_conflict": "completed", "not_runnable": "completed"}
MESSAGE_BY_STATUS: Final[dict[str, MessageKey]] = {
    "accepted": "command.accepted", "completed": "stage.finished",
    "needs_user_input": "interrupt.open", "blocked": "command.blocked",
    "failed": "command.failed", "failed_observability": "command.failed"}

NEEDS_USER_INPUT, INVALID_COMMAND = "needs_user_input", "invalid_command"
INVALID_PAYLOAD_FILE, INTAKE_REFUSED = "invalid_payload_file", "intake_refused"
PRESENTATION_UNAVAILABLE: Final = "presentation_unavailable"

_MODEL_CONFIG = ConfigDict(frozen=True, extra="forbid", strict=True)
_Revision = Annotated[int, Field(ge=1)]


class _CommandError(ValueError):
    """A boundary rejection carrying the stable error code the result reports."""

    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


class InterruptIdentity(BaseModel):
    """The exact open interrupt an answer command must match (PRD-002 §11.1)."""

    model_config = _MODEL_CONFIG

    interrupt_id: Identity
    expected_interrupt_hash: Sha256Hex
    expected_revision: _Revision
    interrupt_kind: Identity | None = None


class StatusView(BaseModel):
    """What `causal status` reads from committed indexes only (SC §1.1)."""

    model_config = _MODEL_CONFIG

    analysis_id: Identity
    stage: Identity
    state: Identity
    next_command: Identity | None


class CliCommandEnvelopeV1(BaseModel):
    """The sole coordinator input the CLI constructs (SC §1.1)."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["cli-command.v1"] = "cli-command.v1"
    cli_invocation_id: Identity
    command_name: CommandName
    analysis_id: Identity | None
    expected_identity: Identity | None
    idempotency_key: Identity | None
    payload_type: Identity | None
    payload_hash: Sha256Hex | None
    output_format: OutputFormat

    @model_validator(mode="after")
    def _boundary_rules(self) -> Self:
        if (self.analysis_id is None) is not (self.command_name == "new"):
            raise ValueError("analysis_id is absent exactly for the creation command")
        if (self.idempotency_key is None) is (self.command_name in MUTATING):
            raise ValueError("idempotency_key is required exactly for mutating commands")
        if self.payload_hash is not None and self.payload_type is None:
            raise ValueError("payload_hash without a registered payload_type")
        return self


class CliResultV1(BaseModel):
    """The one printed result; human output renders from this document only (SC §1.1)."""

    model_config = _MODEL_CONFIG

    schema_version: Literal["cli-result.v1"] = "cli-result.v1"
    cli_invocation_id: Identity
    command_name: CommandName
    analysis_id: Identity | None = None
    stage_run_id: Identity | None = None
    status: CliStatus
    artifacts: tuple[ArtifactRef, ...] = ()
    interrupt: InterruptIdentity | None = None
    next_command_name: CommandName | None = None
    error_code: Identity | None = None
    blocker_event_id: Identity | None = None
    message_key: MessageKey
    message_args: MessageArgs = {}


class OutcomeView(Protocol):
    """What every coordinator result must expose; read-only, so frozen results fit."""

    @property
    def analysis_id(self) -> str: ...
    @property
    def stage_run_id(self) -> str: ...
    @property
    def status(self) -> str: ...
    @property
    def outcome_artifact_id(self) -> str | None: ...


class IntakeOutcomeView(OutcomeView, Protocol):
    """The shape `causal new` renders (PRD-001 `IntakeResult`)."""

    @property
    def replayed(self) -> bool: ...


class DesignOutcomeView(OutcomeView, Protocol):
    """The shape every design command renders (PRD-002 `DesignRunResult`)."""

    @property
    def design_revision(self) -> int: ...
    @property
    def interrupt_kind(self) -> str | None: ...
    @property
    def interrupt_artifact_id(self) -> str | None: ...
    @property
    def interrupt_hash(self) -> str | None: ...
    @property
    def refusal_code(self) -> str | None: ...
    @property
    def error_code(self) -> str | None: ...


class Runtime(Protocol):
    """Everything the CLI may do; the runtime package owns every dependency behind it."""

    def new(self, submission: IntakeSubmissionV1) -> IntakeOutcomeView: ...

    def status(self, analysis_id: str) -> StatusView: ...

    def run(self, analysis_id: str, *, expected_stage_run: str,
            idempotency_key: str) -> DesignOutcomeView: ...

    def select_table(self, analysis_id: str,
                     decision: TableSelectionDecisionV1) -> DesignOutcomeView: ...

    def answer_context(self, analysis_id: str, answer: UserContextAnswerV1,
                       interrupt: InterruptIdentity,
                       idempotency_key: str) -> DesignOutcomeView: ...

    def approve_design(self, analysis_id: str, interrupt: InterruptIdentity,
                       decision: ApprovalDecision, idempotency_key: str,
                       change_requests: tuple[str, ...] = ()) -> DesignOutcomeView: ...


def build_parser() -> argparse.ArgumentParser:
    """Exactly the seven commands and their declared flags; anything else is a usage error."""
    parser = argparse.ArgumentParser(prog="causal", description="Causal-analysis V1")
    parser.add_argument("--format", dest="output_format", choices=("human", "json"),
                        default="human")
    subs = parser.add_subparsers(dest="command_name", required=True)
    made = {name: subs.add_parser(name) for name in COMMANDS}
    for name, sub in made.items():
        if name != "new":
            sub.add_argument("analysis_id", metavar="ANALYSIS_ID")
    for name in ANSWER_COMMANDS:
        for flag in BINDING_FLAGS:
            made[name].add_argument(flag, required=True,
                                    type=int if flag == "--expected-revision" else str)
    made["new"].add_argument("--question", required=True)
    made["new"].add_argument("--kaggle", required=True)
    made["new"].add_argument("--idempotency-key", required=True)
    made["new"].add_argument("--context-file", type=Path, default=None)
    made["select-table"].add_argument("--table-id", required=True)
    made["answer-context"].add_argument("--answers-file", required=True, type=Path)
    made["approve-design"].add_argument("--decision", required=True,
                                        choices=[item.value for item in ApprovalDecision])
    made["approve-design"].add_argument("--changes-file", type=Path, default=None)
    made["run"].add_argument("--expected-stage-run", required=True)
    made["run"].add_argument("--idempotency-key", required=True)
    made["presentation"].add_argument("--bundle-id", required=True)
    made["presentation"].add_argument("--expected-bundle-hash", required=True)
    made["presentation"].add_argument("--output-dir", type=Path, default=None)
    return parser


def _new_invocation_id() -> str:
    return f"cli-{uuid.uuid4().hex}"


def main(argv: list[str] | None, runtime_factory: Callable[[], Runtime],
         out: IO[str] = sys.stdout,
         invocation_id: Callable[[], str] = _new_invocation_id) -> int:
    """Parse, validate, dispatch, render, and return the contract exit code."""
    args = build_parser().parse_args(argv)
    result = _execute(args, runtime_factory, invocation_id())
    print(render(result, args.output_format), file=out)
    return EXIT_CODES[result.status]


def _execute(args: argparse.Namespace, runtime_factory: Callable[[], Runtime],
             invocation_id: str) -> CliResultV1:
    """One command, one result; no failure escapes as an untyped traceback."""
    try:
        payload = _payload(args)
        envelope = _envelope(args, payload, invocation_id)
        return _dispatch(runtime_factory(), envelope, args, payload)
    except _CommandError as error:
        return _blocked(args, invocation_id, error.code)
    except ValidationError:
        return _blocked(args, invocation_id, INVALID_COMMAND)
    except ValueError as error:  # typed coordinator refusals carry a stable code.
        return _blocked(args, invocation_id, str(getattr(error, "code", INVALID_COMMAND)))


def _payload(args: argparse.Namespace) -> BaseModel | None:
    """Build the typed payload a command carries, or None when it carries none."""
    if args.command_name == "new":
        return IntakeSubmissionV1(
            schema_version="intake-submission.v1", question_text=args.question,
            context_text=_read(args.context_file), kaggle_ref=args.kaggle,
            idempotency_key=args.idempotency_key)
    if args.command_name == "select-table":
        return TableSelectionDecisionV1(
            interrupt_id=args.interrupt_id,
            expected_interrupt_hash=args.expected_interrupt_hash,
            expected_revision=args.expected_revision, selected_table=args.table_id,
            idempotency_key=args.idempotency_key)
    if args.command_name == "answer-context":
        return _parse_file(args.answers_file)
    return None


def _read(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise _CommandError(f"unreadable context file {path}", INVALID_PAYLOAD_FILE) from error


def _parse_file(path: Path) -> UserContextAnswerV1:
    """The CLI never converts free text into a decision: the file is the typed answer."""
    try:
        return UserContextAnswerV1.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as error:
        raise _CommandError(f"invalid answers file {path}", INVALID_PAYLOAD_FILE) from error


def _expected_identity(args: argparse.Namespace) -> str | None:
    if args.command_name in ANSWER_COMMANDS:
        return str(args.expected_revision)
    if args.command_name == "run":
        return str(args.expected_stage_run)
    if args.command_name == "presentation":
        return f"{args.bundle_id}:{args.expected_bundle_hash}"
    return None


def _change_requests(args: argparse.Namespace) -> tuple[str, ...]:
    """The §22 change texts; required for changes_requested, forbidden otherwise (D-056)."""
    wanted = args.decision == "changes_requested"
    if (args.changes_file is not None) is not wanted:
        raise _CommandError("changes_requested requires --changes-file and other decisions"
                            " forbid it", INVALID_PAYLOAD_FILE)
    if not wanted:
        return ()
    items = json.loads(Path(args.changes_file).read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items or not all(isinstance(i, str) for i in items):
        raise _CommandError("--changes-file must hold a non-empty JSON list of strings",
                            INVALID_PAYLOAD_FILE)
    return tuple(items)


def _envelope(args: argparse.Namespace, payload: BaseModel | None,
              invocation_id: str) -> CliCommandEnvelopeV1:
    """The envelope validates before any runtime call happens."""
    declared: Any = getattr(payload, "schema_version", None)
    if args.command_name == "approve-design":
        # The approval payload is completed by the runtime from the open interrupt,
        # so the CLI declares the type but can hash no payload of its own.
        declared = "design-approval-decision.v1"
    return CliCommandEnvelopeV1(
        cli_invocation_id=invocation_id, command_name=args.command_name,
        analysis_id=getattr(args, "analysis_id", None),
        expected_identity=_expected_identity(args),
        idempotency_key=getattr(args, "idempotency_key", None), payload_type=declared,
        output_format=args.output_format,
        payload_hash=None if payload is None else content_hash(payload.model_dump(mode="json")))


def _interrupt(args: argparse.Namespace) -> InterruptIdentity:
    return InterruptIdentity(
        interrupt_id=args.interrupt_id, expected_interrupt_hash=args.expected_interrupt_hash,
        expected_revision=args.expected_revision)


def _dispatch(runtime: Runtime, envelope: CliCommandEnvelopeV1, args: argparse.Namespace,
              payload: BaseModel | None) -> CliResultV1:
    if isinstance(payload, IntakeSubmissionV1):
        return _from_intake(envelope, runtime.new(payload))
    analysis_id = str(envelope.analysis_id)
    if envelope.command_name == "status":
        return _from_status(envelope, runtime.status(analysis_id))
    if envelope.command_name == "presentation":
        # PRD-005 does not exist yet; the boundary refuses an unavailable action (SC §1.1).
        return _blocked(args, envelope.cli_invocation_id, PRESENTATION_UNAVAILABLE)
    key = str(envelope.idempotency_key)
    if isinstance(payload, TableSelectionDecisionV1):
        outcome = runtime.select_table(analysis_id, payload)
    elif isinstance(payload, UserContextAnswerV1):
        outcome = runtime.answer_context(analysis_id, payload, _interrupt(args), key)
    elif envelope.command_name == "approve-design":
        outcome = runtime.approve_design(
            analysis_id, _interrupt(args), ApprovalDecision(args.decision), key,
            change_requests=_change_requests(args))
    else:
        outcome = runtime.run(analysis_id, expected_stage_run=str(args.expected_stage_run),
                              idempotency_key=key)
    return _from_design(envelope, outcome)


def _from_intake(envelope: CliCommandEnvelopeV1, outcome: IntakeOutcomeView) -> CliResultV1:
    """A refused intake is a blocker; a usable or partial one may be run (PRD-001 §10)."""
    refused = outcome.status == "refused"
    return CliResultV1(
        cli_invocation_id=envelope.cli_invocation_id, command_name=envelope.command_name,
        analysis_id=outcome.analysis_id, stage_run_id=outcome.stage_run_id,
        status="blocked" if refused else "accepted",
        next_command_name=None if refused else "run",
        error_code=INTAKE_REFUSED if refused else None,
        message_key="command.blocked" if refused else "analysis.created",
        message_args=_message_args(
            outcome, {"intake_status": outcome.status, "replayed": outcome.replayed}))


def _message_args(outcome: OutcomeView, base: MessageArgs) -> MessageArgs:
    """The summary arguments plus the committed identities this outcome carries."""
    held = {"outcome_artifact_id": outcome.outcome_artifact_id,
            "handoff_id": getattr(outcome, "handoff_id", None)}
    return base | {name: value for name, value in held.items() if value is not None}


def _from_status(envelope: CliCommandEnvelopeV1, view: StatusView) -> CliResultV1:
    next_name: Any = view.next_command
    return CliResultV1(
        cli_invocation_id=envelope.cli_invocation_id, command_name=envelope.command_name,
        analysis_id=view.analysis_id, status="completed", next_command_name=next_name,
        message_key="status.read", message_args={"stage": view.stage, "state": view.state})


def _from_design(envelope: CliCommandEnvelopeV1, outcome: DesignOutcomeView) -> CliResultV1:
    """An answered interrupt is accepted and resumed by `causal run` (PRD-002 §11.1)."""
    answered = envelope.command_name in ANSWER_COMMANDS and outcome.status == NEEDS_USER_INPUT
    status: CliStatus = "accepted" if answered else DESIGN_STATUS.get(outcome.status, "failed")
    interrupt = None
    if not answered and outcome.interrupt_artifact_id and outcome.interrupt_hash:
        interrupt = InterruptIdentity(
            interrupt_id=outcome.interrupt_artifact_id, interrupt_kind=outcome.interrupt_kind,
            expected_interrupt_hash=outcome.interrupt_hash,
            expected_revision=outcome.design_revision)
    return CliResultV1(
        cli_invocation_id=envelope.cli_invocation_id, command_name=envelope.command_name,
        analysis_id=outcome.analysis_id, stage_run_id=outcome.stage_run_id, status=status,
        interrupt=interrupt, next_command_name=_next_command(status, outcome.interrupt_kind),
        error_code=outcome.refusal_code or outcome.error_code,
        message_key=MESSAGE_BY_STATUS[status],
        message_args=_message_args(outcome, {"design_revision": outcome.design_revision,
                                             "design_status": outcome.status}))


def _next_command(status: CliStatus, interrupt_kind: str | None) -> CommandName | None:
    if status == "accepted":
        return "run"
    if status == NEEDS_USER_INPUT:
        return INTERRUPT_COMMANDS.get(interrupt_kind or "")
    return None


def _blocked(args: argparse.Namespace, invocation_id: str, code: str) -> CliResultV1:
    """One typed rejection; the coordinator was either never called or changed nothing."""
    return CliResultV1(
        cli_invocation_id=invocation_id, command_name=args.command_name,
        analysis_id=getattr(args, "analysis_id", None), status="blocked", error_code=code,
        message_key="command.blocked")
