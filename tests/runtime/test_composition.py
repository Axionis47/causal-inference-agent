"""Runtime composition over dockerized PostgreSQL and MinIO (T-014 §2, §4; SC §1.1)."""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Iterator
from dataclasses import replace
from io import StringIO
from pathlib import Path
from typing import Any, ClassVar, NamedTuple

import psycopg
import pytest

from causal.cli.main import main
from causal.design.contracts import ApprovalDecision, TableSelectionDecisionV1
from causal.design.graph import DesignRunResult
from causal.estimation import judge as ej
from causal.intake.contracts import IntakeSubmissionV1
from causal.intake.outcome import IntakeResult
from causal.presentation import curate as cu
from causal.runtime import composition
from causal.shared.envelope import AgentTaskEnvelopeV1
from causal.shared.gateway import GatewayError, GatewayResultV1
from causal.shared.persistence import ArtifactCommitter
from causal.shared.tracing import ObservabilityError
from tests.conftest import MIGRATIONS, requires_docker
from tests.design.test_graph import NOW, ScriptedGateway, stub_renderer
from tests.estimation.test_coordinator_e2e import Gateway as ClaimGateway
from tests.intake.conftest import FrozenKaggleClient
from tests.presentation.test_coordinator_e2e import Curator
from tests.shared.test_tracing import FakeTracer

pytestmark = requires_docker

ROOT = Path(__file__).resolve().parents[2]
QUESTION = "Does the programme raise earnings?"
RUN_STATE = "SELECT state FROM design.design_runs WHERE analysis_id = %s"


class PipelineGateway(ScriptedGateway):
    """The design script plus the two bounded downstream receivers."""

    delegates: ClassVar[dict[str, Any]] = {
        ej.TASK_KIND: ClaimGateway(), cu.TASK_KIND: Curator()}

    def invoke(self, envelope: Any, prompt: str, schema: dict[str, object]) -> Any:
        return (self.delegates.get(envelope.task_kind) or super()).invoke(envelope, prompt, schema)


class Interrupt(NamedTuple):
    """The binding the CLI passes as `InterruptIdentity` (SC §1.1)."""

    interrupt_id: str
    expected_interrupt_hash: str
    expected_revision: int


def binding(result: DesignRunResult, **over: Any) -> Interrupt:
    return Interrupt(str(result.interrupt_artifact_id), str(result.interrupt_hash),
                     result.design_revision)._replace(**over)


def make_config(conn: Any, minio: dict[str, Any], tmp_path: Path) -> composition.RuntimeConfig:
    info = conn.info
    return composition.RuntimeConfig(
        postgres_dsn=f"postgresql://{info.user}:{info.password}@{info.host}:{info.port}"
                     f"/{info.dbname}",
        s3_bucket=minio["bucket"], s3_endpoint_url=minio["client"].meta.endpoint_url,
        repo_root=ROOT, registries_root=ROOT / "registries", prompts_root=ROOT,
        migrations_dir=MIGRATIONS, event_log=tmp_path / "events.ndjson")


@pytest.fixture()
def runtime(conn: Any, minio_s3: dict[str, Any], tmp_path: Path,
            monkeypatch: pytest.MonkeyPatch) -> Iterator[composition.CausalRuntime]:
    """One runtime over the docker stores, a frozen Kaggle client, and a scripted gateway."""
    for name, value in (("AWS_ACCESS_KEY_ID", "causal"),
                        ("AWS_SECRET_ACCESS_KEY", "causal-test"),
                        ("AWS_DEFAULT_REGION", "us-east-1")):
        monkeypatch.setenv(name, value)
    stub_renderer(monkeypatch)  # the test host needs no Graphviz binary
    built = composition.build_runtime(
        make_config(conn, minio_s3, tmp_path), client_factory=FrozenKaggleClient,
        model=PipelineGateway(), strict_observability=False, clock=lambda: NOW)
    yield built
    built.close()


def submission(key: str = "k-new") -> IntakeSubmissionV1:
    return IntakeSubmissionV1(
        schema_version="intake-submission.v1", question_text=QUESTION, context_text=None,
        kaggle_ref="lalonde/nsw", idempotency_key=key)


def start(runtime: composition.CausalRuntime) -> tuple[IntakeResult, DesignRunResult]:
    """`causal new` then `causal run`: intake to its boundary, design to its interrupt."""
    made = runtime.new(submission())
    assert made.status in composition.USABLE_INTAKE
    return made, runtime.run(made.analysis_id, expected_stage_run=made.stage_run_id,
                             idempotency_key="k-run")


def code_of(raised: pytest.ExceptionInfo[composition.CompositionError]) -> str:
    return raised.value.code


class TestMigrations:
    def test_apply_migrations_is_idempotent(self, postgres_dsn: str) -> None:
        admin = psycopg.connect(postgres_dsn, autocommit=True)
        name = f"test_{uuid.uuid4().hex[:10]}"
        admin.execute(f'CREATE DATABASE "{name}"')
        admin.close()
        fresh = psycopg.connect(postgres_dsn.rsplit("/", 1)[0] + f"/{name}", autocommit=True)
        first = composition.apply_migrations(fresh, MIGRATIONS)
        assert first == tuple(sorted(path.name for path in MIGRATIONS.glob("*.sql")))
        assert composition.apply_migrations(fresh, MIGRATIONS) == ()
        assert fresh.execute("SELECT count(*) FROM public.causal_migrations").fetchone() == (
            len(first),)
        assert fresh.execute("SELECT count(*) FROM causal.artifacts").fetchone() == (0,)
        fresh.close()

    def test_a_database_migrated_out_of_band_is_adopted_not_rerun(self, conn: Any) -> None:
        assert composition.apply_migrations(conn, MIGRATIONS) == ()
        assert composition.apply_migrations(conn, MIGRATIONS) == ()


class TestConfiguration:
    def test_from_env_reads_the_causal_variables(self) -> None:
        built = composition.RuntimeConfig.from_env({
            "CAUSAL_POSTGRES_DSN": "postgresql:///causal", "CAUSAL_S3_BUCKET": "objects",
            "CAUSAL_ENV": "production", "CAUSAL_REPO_ROOT": str(ROOT)})
        assert built.environment == "production" and built.langsmith_project is None
        assert built.registries_root == ROOT / "registries" and built.prompts_root == ROOT
        assert built.s3_endpoint_url is None

    def test_a_missing_dsn_is_refused(self) -> None:
        with pytest.raises(composition.CompositionError) as raised:
            composition.RuntimeConfig.from_env({"CAUSAL_S3_BUCKET": "objects"})
        assert code_of(raised) == "configuration_missing"

    def test_strict_observability_requires_a_langsmith_project(self) -> None:
        config = composition.RuntimeConfig(postgres_dsn="postgresql:///x", s3_bucket="objects")
        assert composition.build_tracer(config, strict=False) is None
        with pytest.raises(composition.CompositionError) as raised:
            composition.build_tracer(config, strict=True)
        assert code_of(raised) == "tracing_unconfigured"

    def test_the_startup_fingerprint_pins_python_and_records_dot(self) -> None:
        marks = composition.startup_fingerprint(
            composition.RuntimeConfig(postgres_dsn="postgresql:///x", s3_bucket="objects"))
        assert marks["python"].startswith("3.12")
        assert marks["dot"] == "absent" or Path(marks["dot"]).exists()
        assert len(marks["lock_sha256"]) == 64


class TestCommands:
    def test_new_then_run_reaches_the_approval_interrupt_and_approves(
        self, runtime: composition.CausalRuntime, conn: Any
    ) -> None:
        made, opened = start(runtime)
        assert opened.status == "needs_user_input" and opened.interrupt_kind == "approval"
        assert runtime.status(made.analysis_id) == composition.StatusView(
            analysis_id=made.analysis_id, stage="design", state="waiting_for_user",
            next_command="approve-design")
        done = runtime.approve_design(made.analysis_id, binding(opened),
                                      ApprovalDecision.APPROVED, "k-approve")
        assert done.status == "approved" and done.outcome_artifact_id is not None
        assert done.handoff_id is not None
        # D-069b: an approved design's one permitted next command is the PRD-003 run.
        assert runtime.status(made.analysis_id) == composition.StatusView(
            analysis_id=made.analysis_id, stage="design", state="completed", next_command="run")
        assert conn.execute(RUN_STATE, (made.analysis_id,)).fetchone() == ("completed",)

    def test_status_before_design_points_at_run(
        self, runtime: composition.CausalRuntime
    ) -> None:
        made = runtime.new(submission())
        assert runtime.status(made.analysis_id) == composition.StatusView(
            analysis_id=made.analysis_id, stage="intake", state=made.status, next_command="run")

    def test_a_second_run_reports_the_committed_boundary_without_restarting(
        self, runtime: composition.CausalRuntime
    ) -> None:
        made, opened = start(runtime)
        again = runtime.run(made.analysis_id, expected_stage_run=opened.stage_run_id,
                            idempotency_key="k-run-2")
        assert again.status == "needs_user_input"
        assert again.interrupt_artifact_id == opened.interrupt_artifact_id
        runtime.approve_design(made.analysis_id, binding(opened), ApprovalDecision.APPROVED,
                               "k-approve")
        # T-019 §1.4: an approved design is not re-run; the same command starts PRD-003, and
        # T-029 chains on into PRD-004 as soon as that frame reaches `prepared`.
        settled = runtime.run(made.analysis_id, expected_stage_run=opened.stage_run_id,
                              idempotency_key="k-run-3")
        assert settled.stage_run_id == f"es:{made.analysis_id}:1"
        assert (settled.status, settled.error_code) == ("invalidated", "not_reportable")
        reported = runtime.run(made.analysis_id, expected_stage_run=opened.stage_run_id,
                               idempotency_key="k-run-4")
        assert (reported.stage_run_id, reported.status) == (settled.stage_run_id, settled.status)

    def test_an_unknown_analysis_is_blocked(self, runtime: composition.CausalRuntime) -> None:
        for call in (lambda: runtime.status("an-nope"),
                     lambda: runtime.run("an-nope", expected_stage_run="sr:x",
                                         idempotency_key="k-x")):
            with pytest.raises(composition.CompositionError) as raised:
                call()
            assert code_of(raised) == "unknown_analysis"

    def test_a_blocker_event_reaches_the_configured_sink(
        self, runtime: composition.CausalRuntime
    ) -> None:
        with pytest.raises(composition.CompositionError):
            runtime.status("an-nope")
        written = runtime.config.event_log.read_text(encoding="utf-8")
        assert '"event_name":"blocker.raised"' in written
        assert '"error_code":"unknown_analysis"' in written


class TestInterruptValidation:
    def test_a_wrong_hash_kind_or_revision_never_resumes(
        self, runtime: composition.CausalRuntime
    ) -> None:
        made, opened = start(runtime)
        cases = {
            "interrupt_hash_mismatch": binding(opened, expected_interrupt_hash="0" * 64),
            "stale_revision": binding(opened, expected_revision=2),
            "wrong_interrupt": binding(opened, interrupt_id="experimentdesign:other")}
        for expected, wrong in cases.items():
            with pytest.raises(composition.CompositionError) as raised:
                runtime.approve_design(made.analysis_id, wrong, ApprovalDecision.APPROVED,
                                       f"k-{expected}")
            assert code_of(raised) == expected
        with pytest.raises(composition.CompositionError) as raised:
            runtime.select_table(made.analysis_id, TableSelectionDecisionV1(
                interrupt_id=binding(opened).interrupt_id,
                expected_interrupt_hash=binding(opened).expected_interrupt_hash,
                expected_revision=1, selected_table="nsw.csv", idempotency_key="k-table"))
        assert code_of(raised) == "wrong_interrupt"
        # No refusal consumed the interrupt: the correct decision still lands.
        assert runtime.approve_design(made.analysis_id, binding(opened),
                                      ApprovalDecision.APPROVED, "k-ok").status == "approved"

    def test_a_reused_key_with_a_different_request_is_blocked(
        self, runtime: composition.CausalRuntime
    ) -> None:
        made, opened = start(runtime)
        with pytest.raises(composition.CompositionError) as raised:
            runtime.run(made.analysis_id, expected_stage_run=opened.stage_run_id,
                        idempotency_key="k-run")
        assert code_of(raised) == "duplicate_idempotency_key"

    def test_a_stale_expected_stage_run_never_starts_design(
        self, runtime: composition.CausalRuntime
    ) -> None:
        made = runtime.new(submission())
        with pytest.raises(composition.CompositionError) as raised:
            runtime.run(made.analysis_id, expected_stage_run="sr:wrong:1",
                        idempotency_key="k-run")
        assert code_of(raised) == "stale_revision"


class DeadGateway:
    """A model call that dies terminally at the harness boundary (T-014 Amendment 3)."""

    def invoke(self, envelope: AgentTaskEnvelopeV1, prompt: str,
               response_schema: dict[str, object]) -> GatewayResultV1:
        raise GatewayError("the project has no quota left", "quota_exhausted")


class TestHarnessBoundaryFailures:
    """D-062: `cli.main` prints a typed terminal result, never a traceback (SC §1.1, §7.1)."""

    def failing_run(self, runtime: composition.CausalRuntime, **broken: Any) -> tuple[str, int]:
        """Run intake, break one dependency, then drive `causal run` through the CLI."""
        made = runtime.new(submission())
        runtime.deps = replace(runtime.deps, **broken)
        return made.analysis_id, main(
            ["run", made.analysis_id, "--expected-stage-run", made.stage_run_id,
             "--idempotency-key", "k-run"], lambda: runtime, out=StringIO())

    def test_a_terminal_gateway_error_fails_the_run(
        self, runtime: composition.CausalRuntime, conn: Any
    ) -> None:
        analysis_id, code = self.failing_run(runtime, gateway=DeadGateway())
        assert code == 4
        assert conn.execute(RUN_STATE, (analysis_id,)).fetchone() == ("failed",)
        raised = [line for line in runtime.config.event_log.read_text(encoding="utf-8").
                  splitlines() if '"event_name":"blocker.raised"' in line]
        assert len(raised) == 1 and '"error_code":"quota_exhausted"' in raised[0]

    def test_an_unacknowledged_flush_fails_observability(
        self, runtime: composition.CausalRuntime, conn: Any
    ) -> None:
        deps, lost = runtime.deps, ObservabilityError("batch lost", "flush_unacknowledged")
        analysis_id, code = self.failing_run(runtime, committer=ArtifactCommitter(
            deps.objects, deps.products, deps.registry, deps.emitter,
            tracer=FakeTracer(flush_error=lost)))
        assert code == 5
        assert conn.execute(RUN_STATE, (analysis_id,)).fetchone() == ("failed_observability",)


def test_advisory_lock_contention_reports_analysis_busy(
    runtime: composition.CausalRuntime
) -> None:
    """A second session holding the analysis lock blocks at once; nothing queues (§1.1)."""
    made, opened = start(runtime)
    holder = psycopg.connect(runtime.config.postgres_dsn, autocommit=True)
    holder.execute("SELECT pg_advisory_lock(hashtext(%s))", (made.analysis_id,))
    try:
        with pytest.raises(composition.CompositionError) as raised:
            runtime.approve_design(made.analysis_id, binding(opened),
                                   ApprovalDecision.APPROVED, "k-busy")
        assert code_of(raised) == "analysis_busy"
        assert runtime.status(made.analysis_id).state == "waiting_for_user"  # status takes none
    finally:
        holder.close()


def test_the_runtime_matches_the_cli_runtime_protocol() -> None:
    """The sibling CLI owns the Protocol; this runtime is what it calls (SC §1.1)."""
    main = pytest.importorskip("causal.cli.main", reason="the CLI boundary has not landed")
    for name in ("new", "status", "run", "select_table", "answer_context", "approve_design"):
        declared = list(inspect.signature(getattr(main.Runtime, name)).parameters)
        implemented = list(inspect.signature(getattr(composition.CausalRuntime, name)).parameters)
        assert declared == implemented[:len(declared)], name
    # One shared type since the D-056 seam fix: the runtime returns the CLI's own model.
    assert composition.StatusView is main.StatusView
