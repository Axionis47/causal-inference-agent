"""Dependency composition behind the CLI boundary (T-014 §2, §4; SC §1.1, §1.2)."""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol, TextIO, cast

import psycopg
from psycopg import Connection

from causal.analysis.integration import RESOURCE_ROOT as ANALYSIS_RESOURCES
from causal.cli.main import StatusView
from causal.design import capacity, contracts, entry, graph, packs, validators
from causal.design import compile as compiler
from causal.intake import catalog, fields, outcome
from causal.intake import contracts as intake
from causal.intake.entry import IntakeDeps, run_intake
from causal.intake.kaggle import KaggleClientProtocol
from causal.post_analysis import runtime as pr
from causal.preparation.harness import PreparationDeps
from causal.runtime import failures
from causal.runtime.kaggle_live import LiveKaggleClient
from causal.shared import events, gateway, persistence, tracing
from causal.shared.canonical import content_hash
from causal.shared.registry import ArtifactTypeRegistry, load_artifact_type_registry
from causal.shared.validation import parse_strict

__all__ = ["CausalRuntime", "CompositionError", "RuntimeConfig", "StatusView",
           "apply_migrations", "build_runtime", "build_tracer", "startup_fingerprint"]

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
ANALYSIS_BUSY, CONFIGURATION_MISSING = "analysis_busy", "configuration_missing"
DUPLICATE_KEY, FINGERPRINT_MISMATCH = "duplicate_idempotency_key", "fingerprint_mismatch"
INTERRUPT_HASH_MISMATCH, NO_OPEN_INTERRUPT = "interrupt_hash_mismatch", "no_open_interrupt"
STAGE_UNAVAILABLE, STALE_REVISION = "stage_unavailable", "stale_revision"
TRACING_UNCONFIGURED, UNKNOWN_ANALYSIS = "tracing_unconfigured", "unknown_analysis"
WRONG_INTERRUPT = "wrong_interrupt"
USABLE_INTAKE: Final = frozenset({"usable", "partial"})
APPROVED, RUN = "approved", "run"
NEXT_COMMAND: Final[dict[str, str]] = {
    "table_selection": "select-table", "clarification": "answer-context",
    "approval": "approve-design"}
# The one interrupt kind each answer command may resume; nothing else is accepted.
_KINDS: Final = {command: contracts.InterruptKind(kind)
                 for kind, command in NEXT_COMMAND.items()}
_TABLES: Final = ("CREATE TABLE IF NOT EXISTS public.causal_migrations (filename text PRIMARY"
                  " KEY, applied_at timestamptz NOT NULL DEFAULT now());"
                  "CREATE TABLE IF NOT EXISTS public.causal_command_keys (idempotency_key text"
                  " PRIMARY KEY, command_name text NOT NULL, request_hash text NOT NULL,"
                  " analysis_id text)")


class CompositionError(ValueError):
    def __init__(self, message: str, code: str) -> None:
        super().__init__(message)
        self.code = code


def _required(source: Mapping[str, str], name: str) -> str:
    if not source.get(name, "").strip():
        raise CompositionError(f"{name} is not set", CONFIGURATION_MISSING)
    return source[name].strip()


@dataclass(frozen=True)
class RuntimeConfig:
    """Every value the process needs; a credential is never a field (SC §1.1, §1.2)."""

    postgres_dsn: str
    s3_bucket: str
    s3_endpoint_url: str | None = None
    langsmith_project: str | None = None
    environment: str = "development"
    repo_root: Path = REPO_ROOT
    registries_root: Path = REPO_ROOT / "registries"
    prompts_root: Path = REPO_ROOT  # compiled prompt paths are repo-relative
    migrations_dir: Path = REPO_ROOT / "migrations"
    event_log: Path = REPO_ROOT / "logs" / "events.ndjson"

    @classmethod
    def from_env(cls, source: Mapping[str, str] | None = None) -> RuntimeConfig:
        env = os.environ if source is None else source
        root = Path(env.get("CAUSAL_REPO_ROOT") or REPO_ROOT)
        return cls(
            postgres_dsn=_required(env, "CAUSAL_POSTGRES_DSN"),
            s3_bucket=_required(env, "CAUSAL_S3_BUCKET"),
            s3_endpoint_url=env.get("CAUSAL_S3_ENDPOINT") or None,
            langsmith_project=env.get("CAUSAL_LANGSMITH_PROJECT") or None,
            environment=env.get("CAUSAL_ENV") or "development", repo_root=root,
            registries_root=root / "registries", prompts_root=root,
            migrations_dir=root / "migrations",
            event_log=Path(env.get("CAUSAL_EVENT_LOG") or root / "logs" / "events.ndjson"))


class _InterruptView(Protocol):
    @property
    def interrupt_id(self) -> str: ...
    @property
    def expected_interrupt_hash(self) -> str: ...
    @property
    def expected_revision(self) -> int: ...


def apply_migrations(conn: Connection[Any], migrations_dir: Path) -> tuple[str, ...]:
    conn.execute(_TABLES)
    applied = {str(row[0]) for row in
               conn.execute("SELECT filename FROM public.causal_migrations").fetchall()}
    # A database migrated out of band (the test fixtures do exactly that) is adopted whole,
    # never re-run: the shipped files are not individually idempotent.
    adopt = not applied and conn.execute("SELECT 1 FROM information_schema.schemata"
                                         " WHERE schema_name = 'causal'").fetchone() is not None
    executed: list[str] = []
    for path in sorted(migrations_dir.glob("*.sql")):
        if path.name in applied:
            continue
        if not adopt:  # one autocommit execute runs the whole file in one transaction
            conn.execute(path.read_text(encoding="utf-8"))
            executed.append(path.name)
        conn.execute("INSERT INTO public.causal_migrations (filename) VALUES (%s)"
                     " ON CONFLICT DO NOTHING", (path.name,))
    return tuple(executed)


def startup_fingerprint(config: RuntimeConfig) -> dict[str, str]:
    version = platform.python_version()
    if not version.startswith("3.12"):
        raise CompositionError(f"python {version} is not the pinned 3.12", FINGERPRINT_MISMATCH)
    lock = config.repo_root / "uv.lock"
    return {"python": version, "environment": config.environment,
            "dot": shutil.which("dot") or "absent",
            "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists()
            else "absent"}


def _s3_client(config: RuntimeConfig) -> persistence.S3ClientProtocol:
    import boto3  # type: ignore[import-untyped]
    from botocore.config import Config  # type: ignore[import-untyped]

    built: Any = boto3.client(  # keys and region resolve from the standard AWS chain
        "s3", endpoint_url=config.s3_endpoint_url,
        config=Config(signature_version="s3v4", retries={"max_attempts": 3}))
    return cast(persistence.S3ClientProtocol, built)


def build_tracer(config: RuntimeConfig, *, strict: bool) -> tracing.TracerProtocol | None:
    if not config.langsmith_project:
        if strict:
            raise CompositionError("CAUSAL_LANGSMITH_PROJECT is unset and tracing is required"
                                   " (SC §10.2)", TRACING_UNCONFIGURED)
        return None
    built = tracing.LangSmithTracer(config.langsmith_project, config.environment,
                                    tracing.TraceRedactorV1())
    built.preflight()
    return built


def _validate(opened: Mapping[str, Any], kind: contracts.InterruptKind,
              interrupt: _InterruptView) -> tuple[str, str] | None:
    seen = (str(opened.get("kind")), str(opened.get("interrupt_artifact_id")),
            str(opened.get("interrupt_hash")), int(opened.get("design_revision", 0)))
    wanted = (kind.value, interrupt.interrupt_id, interrupt.expected_interrupt_hash,
              interrupt.expected_revision)
    codes = (WRONG_INTERRUPT, WRONG_INTERRUPT, INTERRUPT_HASH_MISMATCH, STALE_REVISION)
    return next(((code, str(want)) for code, found, want in zip(codes, seen, wanted, strict=True)
                 if found != want), None)


class CausalRuntime:
    def __init__(self, config: RuntimeConfig, deps: graph.DesignDeps,
                 prep: PreparationDeps,
                 registry: ArtifactTypeRegistry, field_classes: fields.FieldClasses,
                 fingerprint: Mapping[str, str],
                 client_factory: Callable[[], KaggleClientProtocol] = LiveKaggleClient,
                 closing: Sequence[Any] = ()) -> None:
        self.config, self.deps, self.prep, self.registry = config, deps, prep, registry
        self.est = failures.estimation_deps(deps, config.registries_root, config.repo_root)
        self.pres = pr.presentation_deps(deps, config.registries_root, config.repo_root)
        self.field_classes, self.fingerprint = field_classes, dict(fingerprint)
        self._client_factory, self._conn, self._closing = (
            client_factory, deps.conn, tuple(closing))
        self.intake = IntakeDeps(
            client_factory, deps.committer, deps.products, catalog.CatalogStore(deps.conn),
            deps.objects, registry, field_classes, deps.emitter, deps.clock)

    def close(self) -> None:
        for item in self._closing:
            item.close()

    # -- commands ---------------------------------------------------------

    def new(self, submission: intake.IntakeSubmissionV1) -> outcome.IntakeResult:
        with self._lock(f"new:{submission.idempotency_key}", "new"):
            return run_intake(self.intake, submission)

    def status(self, analysis_id: str) -> StatusView:
        row = catalog.CatalogStore(self._conn).find_run_by_analysis(analysis_id)
        if row is None:
            raise self._blocked("status", analysis_id, UNKNOWN_ANALYSIS)
        found = self._latest_design_run(analysis_id)
        if found is None:
            return StatusView(
                analysis_id=analysis_id, stage="intake",
                state=str(row.intake_status or "running"),
                next_command=RUN if row.intake_status in USABLE_INTAKE else None)
        for stage in ("presentation", "estimation", "preparation"):  # the latest stage owns it
            started = pr.latest_stage_run(self._conn, stage, analysis_id)
            if started is not None:  # D-069b: a live stage row names `run` and nothing else
                return StatusView(analysis_id=analysis_id, stage=stage, state=started.state,
                                  next_command=pr.after(started.state, stage))
        opened = self._open_interrupt(found.thread_id) or {}
        # D-069b: an open interrupt names its answer command; a crashed revision and an
        # approved design (preparation is next) share the one command `causal run`.
        terminal = self._terminal(found)
        following = NEXT_COMMAND.get(str(opened.get("kind", ""))) or (
            RUN if found.state in failures.LIVE
            or terminal in {APPROVED, "changes_requested"} else None)
        return StatusView(analysis_id=analysis_id, stage="design", state=found.state,
                          next_command=following)

    def run(self, analysis_id: str, *, expected_stage_run: str,
            idempotency_key: str) -> graph.DesignRunResult:
        with self._lock(analysis_id, "run"):
            row = catalog.CatalogStore(self._conn).find_run_by_analysis(analysis_id)
            if row is None or row.intake_outcome_artifact_id is None:
                raise self._blocked("run", analysis_id, UNKNOWN_ANALYSIS)
            if row.intake_status not in USABLE_INTAKE:
                raise self._blocked("run", analysis_id, STAGE_UNAVAILABLE)
            intake_outcome_id = row.intake_outcome_artifact_id
            found = self._latest_design_run(analysis_id)
            current = found.stage_run_id if found else row.stage_run_id
            if expected_stage_run != current:
                raise self._blocked("run", analysis_id, STALE_REVISION, current)
            self._claim("run", idempotency_key, analysis_id, expected_stage_run)
            if found is not None:  # an open boundary is reported, never restarted
                opened = self._open_interrupt(found.thread_id) or {}
                terminal = self._terminal(found)
                if not opened and terminal == APPROVED:  # PRD-003 is next (§1.4)
                    ready = failures.prepare(self.deps, self.prep, analysis_id, found)
                    return ready if ready.status != failures.PREPARED else pr.estimate_and_present(
                        self.deps, self.est, self.pres, analysis_id, found.revision)
                if not opened and terminal == "changes_requested":
                    return failures.guard(self.deps, "run", analysis_id, lambda: graph.run_design(
                        self.deps, analysis_id=analysis_id, thread_id=f"gt:{uuid.uuid4()}",
                        intake_outcome_artifact_id=intake_outcome_id,
                        design_revision=found.revision + 1))
                return graph.DesignRunResult(
                    status=graph.NEEDS_USER_INPUT if opened else self._terminal(found),
                    analysis_id=analysis_id, stage_run_id=found.stage_run_id,
                    thread_id=found.thread_id, design_revision=found.revision,
                    interrupt_kind=opened.get("kind"), interrupt_hash=opened.get("interrupt_hash"),
                    interrupt_artifact_id=opened.get("interrupt_artifact_id"),
                    outcome_artifact_id=found.outcome_artifact_id)
            started, thread_id = row.intake_outcome_artifact_id, f"gt:{uuid.uuid4()}"
            return failures.guard(self.deps, "run", analysis_id, lambda: graph.run_design(
                self.deps, analysis_id=analysis_id, thread_id=thread_id,
                intake_outcome_artifact_id=started))

    def deliver(self, bundle_id: str, expected_hash: str) -> dict[str, Any]:
        # §20: the one completed PresentationBundle, opened by exact id and hash (SC §1.1).
        return pr.bundle_view(self.deps, bundle_id, expected_hash)

    def select_table(self, analysis_id: str,
                     decision: contracts.TableSelectionDecisionV1) -> graph.DesignRunResult:
        return self._decide(analysis_id, "select-table", decision, decision.idempotency_key,
                            lambda opened: decision.canonical_payload())

    def answer_context(self, analysis_id: str, answer: contracts.UserContextAnswerV1,
                       interrupt: _InterruptView,
                       idempotency_key: str) -> graph.DesignRunResult:
        return self._decide(analysis_id, "answer-context", interrupt, idempotency_key,
                            lambda opened: answer.canonical_payload())

    def approve_design(self, analysis_id: str, interrupt: _InterruptView,
                       decision: contracts.ApprovalDecision, idempotency_key: str,
                       change_requests: Sequence[str] = ()) -> graph.DesignRunResult:
        approved = decision is contracts.ApprovalDecision.APPROVED

        def build(opened: Mapping[str, Any]) -> Mapping[str, object]:
            return parse_strict(contracts.DesignApprovalDecisionV1, {
                "schema_version": "design-approval-decision.v1", "decision": decision.value,
                "interrupt_id": interrupt.interrupt_id,
                "expected_interrupt_hash": interrupt.expected_interrupt_hash,
                "expected_revision": interrupt.expected_revision,
                "approved_artifacts": list(opened.get("approved_artifacts") or ()) if approved
                else [], "change_requests": list(change_requests),
                "idempotency_key": idempotency_key}).canonical_payload()

        return self._decide(analysis_id, "approve-design", interrupt, idempotency_key, build)

    # -- internals --------------------------------------------------------

    def _decide(self, analysis_id: str, command: str, interrupt: _InterruptView,
                idempotency_key: str,
                build: Callable[[Mapping[str, Any]], Mapping[str, object]],
                ) -> graph.DesignRunResult:
        with self._lock(analysis_id, command):
            found = self._latest_design_run(analysis_id)
            opened = None if found is None else self._open_interrupt(found.thread_id)
            if found is None or opened is None:
                raise self._blocked(command, analysis_id,
                                    UNKNOWN_ANALYSIS if found is None else NO_OPEN_INTERRUPT)
            refused = _validate(opened, _KINDS[command], interrupt)
            if refused is not None:
                raise self._blocked(command, analysis_id, *refused)
            self._claim(command, idempotency_key, analysis_id, interrupt.interrupt_id)
            thread_id, payload = found.thread_id, build(opened)
            return failures.guard(self.deps, command, analysis_id, lambda: graph.resume_design(
                self.deps, thread_id=thread_id, resume_value=payload))

    def _latest_design_run(self, analysis_id: str) -> failures.DesignRun | None:
        return failures.latest_design_run(self._conn, analysis_id)

    def _open_interrupt(self, thread_id: str) -> dict[str, Any] | None:
        pending = tuple(graph.build_graph(self.deps).get_state(
            {"configurable": {"thread_id": thread_id}}).interrupts)
        return dict(pending[0].value) if pending else None

    def _terminal(self, found: failures.DesignRun) -> str:
        if found.outcome_artifact_id is None:
            return "failed"
        return str(failures.payload(self.deps, found.outcome_artifact_id)["status"])

    @contextmanager
    def _lock(self, key: str, command: str) -> Iterator[None]:
        row = self._conn.execute("SELECT pg_try_advisory_lock(hashtext(%s))", (key,)).fetchone()
        if not (row and row[0]):
            raise self._blocked(command, key, ANALYSIS_BUSY)
        try:
            yield
        finally:
            self._conn.execute("SELECT pg_advisory_unlock(hashtext(%s))", (key,))

    def _claim(self, command: str, key: str, analysis_id: str, request: str) -> None:
        digest = content_hash({"command": command, "analysis_id": analysis_id,
                               "request": request})
        row = self._conn.execute("SELECT request_hash FROM public.causal_command_keys"
                                 " WHERE idempotency_key = %s", (key,)).fetchone()
        if row is None:
            self._conn.execute("INSERT INTO public.causal_command_keys VALUES (%s, %s, %s, %s)",
                               (key, command, digest, analysis_id))
        elif str(row[0]) != digest:
            raise self._blocked(command, analysis_id, DUPLICATE_KEY)

    def _blocked(self, command: str, analysis_id: str, code: str,
                 expected: str = "") -> CompositionError:
        failures.emit_blocker(self.deps, command, analysis_id, code, expected)
        return CompositionError(f"{command} is blocked: {code}", code)


def build_runtime(
    config: RuntimeConfig, *,
    client_factory: Callable[[], KaggleClientProtocol] = LiveKaggleClient,
    model: graph.GatewayProtocol | None = None, strict_observability: bool = True,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> CausalRuntime:
    """Build every dependency from configuration alone and return the CLI's runtime."""
    fingerprint = startup_fingerprint(config)
    conn: Connection[Any] = psycopg.connect(config.postgres_dsn, autocommit=True)
    saver_conn: Connection[Any] = psycopg.connect(config.postgres_dsn, autocommit=True)
    apply_migrations(conn, config.migrations_dir)  # the ledger makes this a no-op when done
    root = config.registries_root
    registry = load_artifact_type_registry(root / "artifact-types.v1.json")
    tracer = build_tracer(config, strict=strict_observability)
    config.event_log.parent.mkdir(parents=True, exist_ok=True)
    sink: TextIO = config.event_log.open("a", encoding="utf-8")
    emitter = events.EventEmitter(sink)
    products = persistence.ProductStore(conn)
    objects = persistence.ObjectStore(_s3_client(config), config.s3_bucket)
    method_packs = packs.load_method_packs(root / "method-packs.v1.json")
    requirements = packs.load_requirement_templates(root / "context-requirements.v1.json")
    packs.verify_requirement_references(method_packs, requirements)
    deps = graph.DesignDeps(
        conn=conn, catalog=entry.PsycopgCatalogReader(conn), products=products, objects=objects,
        committer=persistence.ArtifactCommitter(objects, products, registry, emitter,
                                                tracer=tracer),
        registry=registry, emitter=emitter, clock=clock,
        gateway=model or gateway.VertexGateway(gateway.GenAiTransport(),
                                               gateway.VERTEX_PROFILE_V1, emitter, clock, tracer),
        checkpointer=graph.build_checkpointer(saver_conn),
        packs=method_packs, templates=requirements,
        task_table=compiler.load_task_table(root / "design-tasks.v1.json"),
        rules=validators.load_validation_rules(root / "design-validation-rules.v1.json"),
        capacity_registry=capacity.load_capacity_registry(root / "delivery-capacity.v1.json"),
        prompts_root=config.prompts_root,
        estimation_registry_path=ANALYSIS_RESOURCES / "method-pack-estimation.v1.json", tracer=tracer)
    return CausalRuntime(
        config, deps, failures.preparation_deps(deps, root, config.repo_root), registry,
        fields.load_field_classes(root / "kaggle-field-classes.v1.json"), fingerprint,
        client_factory, closing=(sink, saver_conn, conn))
