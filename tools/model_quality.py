"""Live, human-grounded quality gate for the four causal-analysis journeys."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
import uuid
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import boto3  # type: ignore[import-untyped]
import polars as pl
import psycopg
from psycopg import sql

from causal.cli.main import InterruptIdentity
from causal.design.contracts import (
    AnswerItemV1,
    AnswerKind,
    ApprovalDecision,
    TableSelectionDecisionV1,
    UserContextAnswerV1,
)
from causal.intake.contracts import IntakeSubmissionV1
from causal.runtime.composition import RuntimeConfig, build_runtime
from causal.shared.canonical import content_hash
from causal.shared.events import EventEmitter
from causal.shared.gateway import (
    VERTEX_PROFILE_V1,
    GatewayResultV1,
    GenAiTransport,
    VertexGateway,
)
from causal.shared.tracing import LangSmithTracer, TraceRedactorV1, sanitize_diagnostic_event

FIXTURES = ROOT / "evals" / "live-journeys.v1.json"
GOLD = ROOT / "evals" / "human-gold.v1.json"
APPROVED = ROOT / "evals" / "approved-gold.sha256"
MATRIX_APPROVED = ROOT / "evals" / "approved-matrix-gold.sha256"
REPORTS = ROOT / "evals" / "reports"
REVIEW = ROOT / "evals" / "review"
STRESS_FIXTURES = ROOT / "evals" / "stress-journeys.v1.json"
STRESS_GOLD = ROOT / "evals" / "stress-expectations.v1.json"
MATRIX_VARIANTS = ("rich", "sparse")
MATRIX_MODES = ("matrix-validate", "matrix-start", "matrix-resume", "matrix-finalize")
MODES = ("validate", "calibrate", "gate", "stress", *MATRIX_MODES)
LIVE_IDS = ("lalonde_nsw_rct", "groupon_aipw", "minimum_wage_did", "rd_senate_sharp")
STRESS_IDS = ("resume_audit_rct", "nhefs_aipw", "card_krueger_did", "head_start_rdd")
FAMILIES = ("randomized_experiment", "aipw", "did", "sharp_rdd")
NEUTRAL_CONTEXT = "No additional study context is available beyond the causal question and supplied CSV."
MATRIX_STATE_SCHEMA = "evaluation-matrix-state.v1"
HITL_REQUEST_SCHEMA = "evaluation-hitl-request.v1"
HITL_RESPONSE_SCHEMA = "evaluation-hitl-response.v1"
CLASSIFICATIONS = frozenset({
    "schema", "context", "evidence", "semantic_judgment", "ask_policy",
    "compiler_capability", "claim_safety", "curation", "provider", "observability",
    "rubric_compatibility"})
POST_ANALYSIS_ARTIFACTS = frozenset({
    "PostAnalysisContext", "PostAnalysisDraft", "PostAnalysisVisual", "PostAnalysisExport",
    "PostAnalysisReview", "PostAnalysisBundle"})
RUBRIC_MIGRATION_REQUIRED = "post_analysis_rubric_migration_required"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain one object")
    return value


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _gold_hash(*, stress: bool = False) -> str:
    if not stress:
        return _hash(GOLD.read_bytes())
    return str(content_hash({"release": _hash(GOLD.read_bytes()),
                             "stress": _hash(STRESS_GOLD.read_bytes())}))


def _documents(*, stress: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    fixture_path, gold_path = ((STRESS_FIXTURES, STRESS_GOLD) if stress else (FIXTURES, GOLD))
    fixtures, gold = _load(fixture_path), _load(gold_path)
    cases = fixtures.get("cases")
    labels = gold.get("cases")
    fixture_schema = "stress-journey-fixtures.v1" if stress else "live-journey-fixtures.v1"
    gold_schema = "development-expectations.v1" if stress else "human-gold.v1"
    expected_ids = STRESS_IDS if stress else LIVE_IDS
    if fixtures.get("schema_version") != fixture_schema:
        raise ValueError(f"fixture schema is not {fixture_schema}")
    if gold.get("schema_version") != gold_schema or not isinstance(labels, dict):
        raise ValueError(f"gold schema is not {gold_schema}")
    if not isinstance(cases, list) or len(cases) != len(expected_ids):
        raise ValueError("exactly four live journey fixtures are required")
    ids = [row.get("case_id") for row in cases]
    if len(set(ids)) != len(expected_ids) or set(ids) != set(labels):
        raise ValueError("fixture and gold case identities must match exactly")
    if tuple(ids) != expected_ids:
        raise ValueError("the journey order is frozen")
    encoded = json.dumps(fixtures, sort_keys=True)
    canary = str(gold.get("prompt_leak_canary", ""))
    if not canary or canary in encoded:
        raise ValueError("gold leak canary is absent or present in model fixture data")
    for row in cases:
        source = row.get("source") or {}
        if len(str(source.get("sha256", ""))) != 64 or not source.get("provenance"):
            raise ValueError(f"{row.get('case_id')} has incomplete source provenance")
        if not row.get("question") or not row.get("context"):
            raise ValueError(f"{row.get('case_id')} has incomplete model input")
        if row.get("family") not in FAMILIES:
            raise ValueError(f"{row.get('case_id')} has no supported analysis family")
        if tuple(row.get("context_variants") or ()) != MATRIX_VARIANTS:
            raise ValueError(f"{row.get('case_id')} has incomplete context variants")
        accepted = labels[row["case_id"]]
        required = {"assignment_mechanisms", "estimands", "methods", "profiles", "grains",
                    "units", "comparators", "role_bindings", "timing",
                    "required_role_relationships", "forbidden_role_relationships",
                    "required_claim_statuses", "accepted_templates", "terminal_outcomes"}
        if required - set(accepted):
            raise ValueError(f"{row['case_id']} has incomplete human gold")
        suitability = accepted.get("scientific_suitability")
        if not isinstance(suitability, dict) or suitability.get("status") not in {
                "suitable", "conditional", "unsuitable"}:
            raise ValueError(f"{row['case_id']} has no scientific suitability decision")
    return fixtures, gold


def _source_bytes(source: dict[str, Any]) -> bytes:
    if source["kind"] == "python_package":
        spec = importlib.util.find_spec(str(source["package"]))
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError(f"missing package {source['package']}")
        return (Path(next(iter(spec.submodule_search_locations))) /
                str(source["path"])).read_bytes()
    if source["kind"] == "url":
        with urlopen(str(source["url"]), timeout=30) as response:
            return bytes(response.read())
    endpoint = os.environ.get("CAUSAL_EVAL_SOURCE_S3_ENDPOINT")
    bucket = str(source["bucket"])
    client = boto3.client("s3", endpoint_url=endpoint)
    return bytes(client.get_object(Bucket=bucket, Key=str(source["key"]))["Body"].read())


def _case_csv(case: dict[str, Any], raw: bytes) -> bytes:
    transform = case["transform"]
    if transform == "identity":
        return raw
    if transform == "card_krueger_panel_v1":
        return _card_krueger_csv(raw)
    frame = pl.read_csv(io.BytesIO(raw), null_values="NA", infer_schema_length=10_000)
    if transform == "rd_senate_projection_v1":
        frame = frame.select("margin", "vote").with_columns(
            (pl.col("margin") >= 0).cast(pl.Int64).alias("treatment"))
    elif transform == "rock_the_vote_projection_v1":
        frame = frame.select(
            pl.col("rownames").cast(pl.String).alias("cable_system_id"),
            pl.col("treated").cast(pl.Int64).alias("treatment"),
            pl.col("p").cast(pl.Float64).alias("turnout_rate"),
            pl.col("strata").cast(pl.String).alias("randomization_stratum"))
    elif transform == "castle_2007_cohort_projection_v1":
        first = frame.group_by("sid").agg(
            pl.col("year").filter(pl.col("post") == 1).min().alias("first_post"))
        frame = frame.join(first, on="sid").filter(
            (pl.col("first_post").is_null() | (pl.col("first_post") == 2007))
            & (pl.col("year") != 2006))
        waves = {year: index for index, year in enumerate(
            (2000, 2001, 2002, 2003, 2004, 2005, 2007, 2008, 2009, 2010))}
        frame = frame.select(
            pl.col("sid").cast(pl.String).alias("state_id"),
            pl.col("year").replace_strict(waves).cast(pl.Int64).alias("period"),
            pl.col("first_post").is_not_null().cast(pl.Int64).alias("treated_group"),
            pl.col("post").cast(pl.Int64).alias("treatment"),
            pl.col("l_homicide").cast(pl.Float64).alias("log_homicide"),
        ).sort("state_id", "period")
    elif transform == "resume_audit_projection_v1":
        frame = frame.select(
            pl.col("rownames").cast(pl.String).alias("resume_id"),
            (pl.col("ethnicity") == "afam").cast(pl.Int64).alias("treatment"),
            (pl.col("call") == "yes").cast(pl.Int64).alias("callback"),
            pl.col("experience").cast(pl.Float64).alias("baseline_experience"))
    elif transform == "nhefs_projection_v1":
        frame = frame.select(
            pl.col("id").cast(pl.String).alias("participant_id"),
            pl.col("qsmk").alias("treatment"), pl.col("wt82_71").alias("weight_change"),
            "age", "sex", "race", "education", "smokeintensity", "smokeyrs", "exercise",
            "active", "wt71")
    elif transform == "head_start_projection_v1":
        frame = frame.select(
            pl.int_range(1, pl.len() + 1).cast(pl.String).alias("county_id"),
            pl.col("povrate60").alias("poverty_rate_1960"),
            (pl.col("povrate60") >= 59.1984).cast(pl.Int64).alias("treatment"),
            pl.col("mort_age59_related_postHS").alias("child_mortality"))
    else:
        raise ValueError(f"unknown transform {transform}")
    return frame.write_csv().encode()


def _card_krueger_csv(raw: bytes) -> bytes:
    columns = "sheet chain co_owned state southj centralj northj pa1 pa2 shore ncalls empft emppt nmgrs wage_st inctime firstinc bonus pctaff meal open hrsopen psoda pfry pentree nregs nregs11 type2 status2 date2 ncalls2 empft2 emppt2 nmgrs2 wage_st2 inctime2 firstin2 special2 meals2 open2r hrsopen2 psoda2 pfry2 pentree2 nregs2 nregs112".split()  # noqa: SIM905
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        rows = [line.split() for line in archive.read("public.dat").decode().splitlines()]
    source = pl.DataFrame(rows, schema=columns, orient="row").with_row_index("restaurant_id", 1)
    numeric = source.select(pl.exclude("restaurant_id")).with_columns(
        pl.all().replace(".", None).cast(pl.Float64, strict=False))
    source = pl.concat([source.select("restaurant_id"), numeric], how="horizontal")
    complete = source.drop_nulls(("empft", "emppt", "nmgrs", "empft2", "emppt2", "nmgrs2"))
    common = (pl.col("restaurant_id").cast(pl.Int64),
              pl.col("state").cast(pl.Int64).alias("state"))
    before = complete.select(*common, pl.lit(0, dtype=pl.Int64).alias("period"),
                             pl.lit(0, dtype=pl.Int64).alias("treatment"),
                             (pl.col("empft") + 0.5 * pl.col("emppt") + pl.col("nmgrs"))
                             .alias("fte_employment"))
    after = complete.select(*common, pl.lit(1, dtype=pl.Int64).alias("period"),
                            pl.col("state").cast(pl.Int64).alias("treatment"),
                            (pl.col("empft2") + 0.5 * pl.col("emppt2") + pl.col("nmgrs2"))
                            .alias("fte_employment"))
    return pl.concat([before, after]).sort("restaurant_id", "period").write_csv().encode()


def validate_environment(*, require_live: bool, require_source_s3: bool = False) -> list[str]:
    issues = [] if sys.version_info[:2] == (3, 12) else ["python_not_3_12"]
    if require_source_s3 and not os.environ.get("CAUSAL_EVAL_SOURCE_S3_ENDPOINT"):
        issues.append("source_s3_endpoint_missing")
    if require_live:
        required = (("CAUSAL_LANGSMITH_PROJECT", "langsmith_project_missing"),
                    ("CAUSAL_EVAL_POSTGRES_ADMIN_DSN", "postgres_admin_dsn_missing"),
                    ("CAUSAL_EVAL_S3_ENDPOINT", "evaluation_s3_endpoint_missing"),
                    ("CAUSAL_EVAL_SOURCE_S3_ENDPOINT", "source_s3_endpoint_missing"))
        issues.extend(code for name, code in required if not os.environ.get(name))
        if not any(os.environ.get(name) for name in ("LANGSMITH_API_KEY", "LANGCHAIN_API_KEY")):
            issues.insert(0, "langsmith_api_key_missing")
    return list(dict.fromkeys(issues))


def _suite_documents(stress: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    fixtures, gold = _documents()
    if not stress:
        return fixtures, gold
    extra_fixtures, extra_gold = _documents(stress=True)
    if extra_gold["prompt_leak_canary"] != gold["prompt_leak_canary"]:
        raise ValueError("release and stress gold leak canaries differ")
    return ({"cases": fixtures["cases"] + extra_fixtures["cases"]},
            {"prompt_leak_canary": gold["prompt_leak_canary"],
             "matrix_expectations": gold.get("matrix_expectations"),
             "cases": gold["cases"] | extra_gold["cases"]})


def _cell_id(case_id: str, variant: str) -> str:
    if variant not in MATRIX_VARIANTS:
        raise ValueError(f"unknown context variant {variant}")
    return f"{case_id}--{variant}"


def _matrix_documents() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fixtures, gold = _suite_documents(True)
    matrix = gold.get("matrix_expectations")
    if not isinstance(matrix, dict) or tuple(matrix) != MATRIX_VARIANTS:
        raise ValueError("combined gold has incomplete matrix expectations")
    for variant, expectation in matrix.items():
        if not isinstance(expectation, dict):
            raise TypeError(f"{variant} ask expectation must be one object")
        minimum = expectation.get("minimum_question_count")
        maximum = expectation.get("maximum_question_count")
        required = expectation.get("required_requirement_ids")
        allowed = expectation.get("allowed_requirement_ids")
        if (not isinstance(minimum, int) or not isinstance(maximum, int)
                or not isinstance(required, list) or not isinstance(allowed, list)
                or not 0 <= minimum <= maximum <= 48 or not set(required) <= set(allowed)):
            raise ValueError(f"{variant} ask expectation is invalid")
    cells = []
    for case in fixtures["cases"]:
        for variant in MATRIX_VARIANTS:
            cells.append(case | {"cell_id": _cell_id(case["case_id"], variant),
                                 "context_variant": variant})
    if len(cells) != 16 or len({row["cell_id"] for row in cells}) != 16:
        raise ValueError("the final evaluation matrix must contain exactly 16 unique cells")
    for family in FAMILIES:
        datasets = {row["case_id"] for row in cells if row["family"] == family}
        if len(datasets) != 2:
            raise ValueError(f"{family} must have exactly two datasets")
        if {row["context_variant"] for row in cells if row["family"] == family} != set(MATRIX_VARIANTS):
            raise ValueError(f"{family} has incomplete context coverage")
    return cells, gold


def _matrix_case(case: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize a context cell without changing the source dataset or causal question."""
    variant = str(case["context_variant"])
    materialized = dict(case)
    materialized["_submission_context"] = case["context"] if variant == "rich" else None
    materialized["context"] = case["context"] if variant == "rich" else NEUTRAL_CONTEXT
    return materialized


def _execution_content_hash() -> str:
    """Hash every source/config input that can affect a matrix execution, including dirty files."""
    roots = (ROOT / "src", ROOT / "prompts", ROOT / "registries", ROOT / "migrations")
    fixed = (Path(__file__).resolve(), FIXTURES, STRESS_FIXTURES, GOLD, STRESS_GOLD,
             ROOT / "tools/four_analyses.py", ROOT / "evals/four-new-journeys.v1.json",
             ROOT / "pyproject.toml", ROOT / "uv.lock")
    paths = [path for base in roots if base.exists() for path in base.rglob("*")
             if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"]
    paths.extend(path for path in fixed if path.exists())
    manifest = {str(path.relative_to(ROOT)): _hash(path.read_bytes()) for path in sorted(set(paths))}
    return str(content_hash(manifest))


def _worktree() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT,
        text=True, capture_output=True, check=True)
    status = result.stdout
    return {"dirty": bool(status), "status_hash": _hash(status.encode()),
            "execution_content_hash": _execution_content_hash()}


def validate(*, verify_sources: bool = True, stress: bool = False) -> dict[str, Any]:
    fixtures, gold = _suite_documents(stress)
    needs_source_s3 = verify_sources and any(
        case["source"]["kind"] == "s3" for case in fixtures["cases"])
    issues = validate_environment(require_live=False, require_source_s3=needs_source_s3)
    checks = []
    for case in fixtures["cases"]:
        source = case["source"]
        try:
            if verify_sources and source["kind"] == "s3" and "source_s3_endpoint_missing" in issues:
                raise RuntimeError("source_s3_endpoint_missing")
            data = _source_bytes(source) if verify_sources else b""
            actual = _hash(data) if verify_sources else source["sha256"]
            transformed = _case_csv(case, data) if verify_sources else b""
            columns = pl.read_csv(io.BytesIO(transformed), n_rows=1).columns if verify_sources else []
            expected = {name for choices in gold["cases"][case["case_id"]][
                "role_bindings"].values() for choice in choices for name in choice}
            schema_ok = not verify_sources or expected <= set(columns) | {"__row_unit_id"}
            version_ok = (source["kind"] != "python_package" or not verify_sources or
                          f"{source['package']}=={importlib.metadata.version(source['package'])}"
                          == source["version"])
            ok, detail = actual == source["sha256"] and schema_ok and version_ok, actual
        except Exception as error:  # noqa: BLE001 - validation reports inaccessible sources
            ok, detail = False, f"{type(error).__name__}:{error}"
        checks.append({"case_id": case["case_id"], "source_hash_valid": ok,
                       "source_hash": detail, "version": source["version"],
                       "provenance": source["provenance"]})
    fixture_hash = (_hash(FIXTURES.read_bytes()) if not stress else content_hash({
        "release": _hash(FIXTURES.read_bytes()), "stress": _hash(STRESS_FIXTURES.read_bytes())}))
    return {"schema_version": "evaluation-validation.v1", "suite": "stress" if stress else "release",
            "fixture_hash": fixture_hash, "gold_hash": _gold_hash(stress=stress),
            "gold_approved": APPROVED.exists() and APPROVED.read_text().strip() == _gold_hash(),
            "checks": checks, "environment_issues": issues,
            "passed": not issues and all(row["source_hash_valid"] for row in checks)}


class _FixtureClient:
    def __init__(self, case: dict[str, Any], csv: bytes) -> None:
        self.case, self.csv = case, csv

    def dataset_status(self, owner: str, slug: str) -> dict[str, object]:
        return {"status": "ready", "currentVersionNumber": self.case["source"]["version"]}

    def dataset_metadata(self, owner: str, slug: str) -> dict[str, object]:
        return {"title": self.case["case_id"], "description": self.case["context"],
                "licenseName": "evaluation-fixture"}

    def dataset_files(self, owner: str, slug: str) -> dict[str, object]:
        columns = pl.read_csv(io.BytesIO(self.csv), n_rows=10).columns
        return {"files": [{"name": self.case["file_name"], "description": self.case["context"],
                           "totalBytes": len(self.csv), "columns": [
                               {"name": name, "description": self.case["context"], "order": index}
                               for index, name in enumerate(columns)]}]}

    def download_archive(self, owner: str, slug: str, version: str) -> bytes:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(self.case["file_name"], self.csv)
            archive.writestr("README.md", self.case["context"])
        return output.getvalue()


class _AuditGateway:
    def __init__(self, gateway: VertexGateway, canary: str, evaluation: dict[str, str]) -> None:
        self.gateway, self.canary, self.evaluation = gateway, canary, evaluation
        self.calls: list[dict[str, Any]] = []

    def invoke(self, envelope: Any, prompt: str,
               response_schema: dict[str, object], *, images: tuple[Any, ...] = ()) -> GatewayResultV1:
        if self.canary in prompt:
            raise RuntimeError("gold_leak_detected")
        seed_case = self.evaluation.get("seed_case_id", self.evaluation["case_id"])
        seed_key = f"{seed_case}:{envelope.task_kind}:{','.join(envelope.scope_ids)}"
        traced = envelope.model_copy(update={
            "payload": dict(envelope.payload) | {
                "evaluation": self.evaluation | {"seed_key": seed_key}}})
        record = {"task_kind": envelope.task_kind, "task_id": envelope.task_id,
                  "attempt_id": envelope.attempt_id,
                  "prompt_hash": _hash(prompt.encode()),
                  "allowed_tool_count": len(envelope.allowed_tool_ids), **self.evaluation}
        started = time.monotonic()
        try:
            result = self.gateway.invoke(traced, prompt, response_schema, **({"images": images} if images else {}))
        except Exception as error:
            self.calls.append(record | {
                               "latency_ms": round((time.monotonic() - started) * 1000),
                               "error_code": getattr(error, "code", type(error).__name__)})
            raise
        self.calls.append(record | {
            "envelope_hash": content_hash(traced.canonical_payload()),
            "latency_ms": round((time.monotonic() - started) * 1000),
            "tokens": dict(result.token_usage), "physical_attempts": result.attempts,
            "finish_reason": result.finish_reason, "reasoning_present": bool(result.reasoning),
            "correction_attempt": envelope.attempt_id.rsplit(":", 1)[-1],
            "response_shape_issues": ([] if result.parsed is not None else
                                      [{"path": "/", "type": "json_parse_failed"}])})
        return result


def _database(run_id: str, case_id: str) -> tuple[str, str]:
    admin = os.environ["CAUSAL_EVAL_POSTGRES_ADMIN_DSN"]
    name = f"causal_eval_{run_id[-12:]}_{case_id}"[:63].replace("-", "_")
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    base = admin.rsplit("/", 1)[0]
    return name, f"{base}/{name}"


def _bucket(run_id: str, case_id: str) -> str:
    name = f"causal-eval-{run_id[-12:]}-{case_id.replace('_', '-')}"[:63]
    client = boto3.client("s3", endpoint_url=os.environ["CAUSAL_EVAL_S3_ENDPOINT"])
    client.create_bucket(Bucket=name)
    return name


def _interrupt(runtime: Any, outcome: Any) -> tuple[dict[str, Any], InterruptIdentity]:
    opened = runtime._open_interrupt(outcome.thread_id)  # evaluator reads the public interrupt
    if not opened:
        raise RuntimeError("expected interrupt is absent")
    return opened, InterruptIdentity(
        interrupt_id=opened["interrupt_artifact_id"], expected_interrupt_hash=opened["interrupt_hash"],
        expected_revision=opened["design_revision"], interrupt_kind=opened["kind"])


def _answer(case: dict[str, Any], opened: dict[str, Any]) -> UserContextAnswerV1:
    rows = []
    for question in opened["packet"]["questions"]:
        requirement = question["requirement_ids"][0]
        value = case.get("answers", {}).get(requirement)
        rows.append(AnswerItemV1(
            question_id=question["question_id"],
            answer_kind=AnswerKind.VALUE if value else AnswerKind.UNKNOWN,
            value=value))
    return UserContextAnswerV1(packet_id=opened["packet"]["packet_id"], answers=tuple(rows))


def _payloads(runtime: Any, analysis_id: str) -> dict[str, dict[str, Any]]:
    rows = runtime.deps.conn.execute(
        "SELECT artifact_type, artifact_id FROM causal.artifacts WHERE analysis_id=%s "
        "ORDER BY created_at_utc, artifact_id", (analysis_id,)).fetchall()
    found = {}
    for kind, artifact_id in rows:
        envelope = runtime.deps.products.load_envelope(str(artifact_id))
        found[str(kind)] = json.loads(runtime.deps.objects.get(envelope.payload_locator))
    return found


def _case_events(run_id: str, analysis_id: str) -> list[dict[str, Any]]:
    path = REPORTS / f"{run_id}.events.ndjson"
    return ([] if not path.exists() else [row for line in path.read_text().splitlines()
            if line and (row := json.loads(line)).get("analysis_id") == analysis_id])


def _failure_events(run_id: str, analysis_id: str) -> list[dict[str, Any]]:
    return [{key: row.get(key) for key in (
        "event_name", "task_id", "attempt_id", "attempt_number", "error_code", "safe_dimensions")}
        for row in _case_events(run_id, analysis_id) if row.get("error_code")]


def _diagnostic_events(run_id: str, analysis_id: str) -> list[dict[str, object]]:
    return [safe for row in _case_events(run_id, analysis_id)
            if (safe := sanitize_diagnostic_event(row)) is not None]


def _tool_policy(calls: list[dict[str, Any]]) -> bool:
    return all(call.get("allowed_tool_count") == (
        1 if call.get("task_kind") == "method_design" else
        6 if call.get("task_kind") == "post_analysis_author" else 0) for call in calls)


def _ask_policy(case: dict[str, Any], gold: dict[str, Any], questions: list[str]) -> bool:
    expected = gold.get("ask_expectation")
    if not isinstance(expected, dict):
        return questions == case["expected_initial_question_ids"]
    required = {f"q:{item}" for item in expected.get("required_requirement_ids", [])}
    allowed = {f"q:{item}" for item in expected.get("allowed_requirement_ids", [])}
    minimum = int(expected.get("minimum_question_count", 0))
    maximum = int(expected.get("maximum_question_count", len(questions)))
    found = set(questions)
    return minimum <= len(questions) <= maximum and required <= found and (
        not allowed or found <= allowed)


def _score(case: dict[str, Any], gold: dict[str, Any], payloads: dict[str, dict[str, Any]],
           terminal: str, questions: list[str], calls: list[dict[str, Any]],
           failure_events: list[dict[str, Any]],
           diagnostic_events: list[dict[str, object]] | None = None) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []

    def require(condition: bool, classification: str, detail: str) -> None:
        if classification not in CLASSIFICATIONS:
            raise ValueError(f"unknown failure classification {classification}")
        if not condition:
            failures.append({"classification": classification, "detail": detail})

    # HumanGoldV1/development-expectations.v1 bind removed claim/template artifacts.
    # A report or a partial authoring run cannot be graded by substituting new fields
    # or by treating old artifacts from the same analysis as the current report.
    legacy_labels = {"required_claim_statuses", "accepted_templates"} & gold.keys()
    post_analysis = bool(POST_ANALYSIS_ARTIFACTS & payloads.keys()) or any(
        call.get("task_kind") in {"post_analysis_author", "post_analysis_review"} for call in calls)
    if legacy_labels and post_analysis:
        require(False, "rubric_compatibility", RUBRIC_MIGRATION_REQUIRED)
        return failures

    if terminal in gold.get("accepted_safe_terminal_outcomes", []):
        require(_ask_policy(case, gold, questions), "ask_policy", f"questions:{questions}")
        require(all("error_code" not in call for call in calls), "provider", "model call failed")
        require(_tool_policy(calls), "evidence", "model-facing tool policy violated")
        return failures

    facts, design = payloads.get("DesignFactSet", {}), payloads.get("CompiledDesign", {})
    if not facts or not design:
        root = next((row for row in reversed(failure_events)
                     if row["event_name"] == "retry.exhausted"), failure_events[-1]
                    if failure_events else {})
        code = str(root.get("error_code") or "compiled_design_absent")
        failed_call = next((call for call in reversed(calls) if "error_code" in call), None)
        terminal_class = {"needs_context": "ask_policy", "needs_data": "semantic_judgment",
                          "unsupported": "compiler_capability"}.get(terminal)
        classification = "provider" if failed_call else terminal_class or (
            "schema" if code in {"schema_invalid", "shape_invalid"} else "context")
        detail = str(failed_call["error_code"]) if failed_call else (
            f"terminal:{terminal}" if terminal_class else
            f"{root.get('task_id', 'design')}:{code}")
        require(False, classification, detail)
        require(_tool_policy(calls), "evidence", "model-facing tool policy violated")
        return failures
    values = {row["fact_id"]: row["value"] for row in facts.get("facts", [])
              if row.get("executable")}
    require(values.get("assignment_mechanism") in gold["assignment_mechanisms"],
            "semantic_judgment", "assignment mechanism")
    require(values.get("estimand") in gold["estimands"], "semantic_judgment", "estimand")
    require(design.get("method_id") in gold["methods"], "semantic_judgment", "method")
    require(facts.get("grain") in gold["grains"], "semantic_judgment", "grain")
    normalized_unit = "".join(character for character in str(design.get("unit")).casefold()
                              if character.isalnum())
    require(any(normalized_unit == "".join(character for character in choice.casefold()
                                            if character.isalnum())
                for choice in gold["units"]), "semantic_judgment", "unit")
    require(design.get("comparator") in gold.get("comparators", [design.get("comparator")]),
            "semantic_judgment", "comparator")
    bindings = {row["role"]: row["columns"] for row in design.get("role_bindings", [])}
    for role, accepted in gold["role_bindings"].items():
        found = set(bindings.get(role, []))
        require(any(found == set(choice) for choice in accepted),
                "semantic_judgment", f"role:{role}")
    ledger = payloads.get("RoleLedger", {})
    bound_concepts = {row["role"]: row["concept_id"]
                      for row in design.get("role_bindings", [])}
    role_rows: dict[str, dict[str, Any]] = {
        role: next((row for row in ledger.get("claims", [])
                    if row["role"] == role and row["concept_id"] == concept), {})
        for role, concept in bound_concepts.items()}
    for role, accepted in gold.get("timing", {}).items():
        require(role_rows.get(role, {}).get("timing") in accepted,
                "semantic_judgment", f"timing:{role}")
    concepts = {concept: role for role, concept in bound_concepts.items()}
    concepts.update({row["concept_id"]: row["role"] for row in ledger.get("claims", [])
                     if row.get("role") in bindings
                     and set(row.get("column_refs", [])) <= set(bindings[row["role"]])})
    edges = {(concepts.get(row["source_concept_id"]), concepts.get(row["target_concept_id"]))
             for row in payloads.get("CausalContext", {}).get("edges", [])}
    require(set(map(tuple, gold["required_role_relationships"])) <= edges,
            "semantic_judgment", "required graph relationships")
    require(not (set(map(tuple, gold["forbidden_role_relationships"])) & edges),
            "semantic_judgment", "forbidden graph relationships")
    for fact_id, accepted in gold.get("structural_facts", {}).items():
        require(values.get(fact_id) in accepted, "semantic_judgment", f"fact:{fact_id}")
    require(_ask_policy(case, gold, questions), "ask_policy", f"questions:{questions}")
    if terminal not in gold["terminal_outcomes"]:
        require(False, "compiler_capability", f"terminal:{terminal}")
        require(all("error_code" not in call for call in calls), "provider", "model call failed")
        require(_tool_policy(calls), "evidence", "model-facing tool policy violated")
        return failures
    plan = payloads.get("PresentationContextManifest", {}).get("approved", {})
    require(plan.get("profile_id") in gold["profiles"], "curation", "method profile")
    claim = payloads.get("ClaimJudgment", {})
    require(claim.get("status") in gold["required_claim_statuses"], "claim_safety", "claim status")
    qualification_text = " ".join(claim.get("qualifications", [])).casefold()
    for fragment in gold.get("required_qualification_fragments", []):
        require(fragment.casefold() in qualification_text, "claim_safety",
                f"qualification:{fragment}")
    require(set(claim.get("decision", {})) <= {
        "status", "items", "finding_ids", "diagnostic_non_finding_ids", "qualification_ids",
        "alternative_explanation_ids", "cannot_conclude_ids"},
        "claim_safety", "claim decision contains compiler-owned fields")
    figure = payloads.get("FigurePlan", {})
    coverage = {evidence: row["template_id"] for row in figure.get("figures", [])
                for evidence in row["visual_evidence_ids"]}
    for evidence, accepted in gold["accepted_templates"].items():
        require(coverage.get(evidence) in accepted, "curation", f"figure:{evidence}")
    require(terminal in gold["terminal_outcomes"], "compiler_capability", f"terminal:{terminal}")
    require(all("error_code" not in call for call in calls), "provider", "model call failed")
    task_kinds = {call["task_kind"] for call in calls}
    require(task_kinds == {"intent", "semantic_batch", "role_evidence", "causal_context",
                           "role_ledger", "method_design", "claim_review", "figure_plan"},
            "context", f"decision_boundaries:{sorted(task_kinds)}")
    require(_tool_policy(calls), "evidence", "model-facing tool policy violated")
    loop_names = {row.get("event_name") for row in diagnostic_events or []}
    require({"agent.diagnostic_requested", "diagnostic.completed", "agent.design_revised"}
            <= loop_names, "semantic_judgment", "bounded diagnostic investigation absent")
    counts = {task_id: sum(row["task_id"] == task_id for row in calls)
              for task_id in {row["task_id"] for row in calls}}
    require(all(count <= 3 for count in counts.values()), "schema", f"correction_counts:{counts}")
    return failures


def _intake_submission(case: dict[str, Any]) -> IntakeSubmissionV1:
    """Build the live input without dropping the fixture's explicit study context."""
    return IntakeSubmissionV1(
        schema_version="intake-submission.v1", question_text=case["question"],
        context_text=case.get("_submission_context", case["context"]),
        kaggle_ref=f"evaluation/{case['case_id']}",
        idempotency_key=f"eval:{case.get('cell_id', case['case_id'])}")


def _run_case(case: dict[str, Any], gold: dict[str, Any], run_id: str,
              tracer: LangSmithTracer, event_sink: io.StringIO, mode: str,
              canary: str) -> dict[str, Any]:
    raw = _source_bytes(case["source"])
    if _hash(raw) != case["source"]["sha256"]:
        raise RuntimeError("source_hash_mismatch")
    csv = _case_csv(case, raw)
    _, dsn = _database(run_id, case["case_id"])
    bucket = _bucket(run_id, case["case_id"])
    base = VertexGateway(GenAiTransport(), VERTEX_PROFILE_V1, EventEmitter(event_sink),
                         lambda: datetime.now(UTC), tracer)
    evaluation = {"run_id": run_id, "case_id": case["case_id"], "mode": mode}
    audit = _AuditGateway(base, canary, evaluation)
    config = RuntimeConfig(
        postgres_dsn=dsn, s3_bucket=bucket,
        s3_endpoint_url=os.environ["CAUSAL_EVAL_S3_ENDPOINT"],
        langsmith_project=os.environ["CAUSAL_LANGSMITH_PROJECT"], environment="evaluation",
        event_log=REPORTS / f"{run_id}.events.ndjson")
    runtime = build_runtime(config, client_factory=lambda: _FixtureClient(case, csv),
                            model=audit, strict_observability=True)
    questions: list[str] = []
    try:
        intake = runtime.new(_intake_submission(case))
        outcome = runtime.run(intake.analysis_id, expected_stage_run=intake.stage_run_id,
                              idempotency_key=f"run:{case['case_id']}:design")
        for index in range(8):
            if outcome.status != "needs_user_input":
                break
            opened, identity = _interrupt(runtime, outcome)
            kind = opened["kind"]
            if kind == "table_selection":
                table = opened["candidate_tables"][0]["logical_name"]
                outcome = runtime.select_table(intake.analysis_id, TableSelectionDecisionV1(
                    interrupt_id=identity.interrupt_id,
                    expected_interrupt_hash=identity.expected_interrupt_hash,
                    expected_revision=identity.expected_revision, selected_table=table,
                    idempotency_key=f"select:{case['case_id']}"))
            elif kind == "clarification":
                questions.extend(q["question_id"] for q in opened["packet"]["questions"])
                outcome = runtime.answer_context(
                    intake.analysis_id, _answer(case, opened), identity,
                    f"answer:{case['case_id']}:{index}")
            elif kind == "approval":
                outcome = runtime.approve_design(
                    intake.analysis_id, identity, ApprovalDecision.APPROVED,
                    f"approve:{case['case_id']}")
            else:
                raise RuntimeError(f"unknown_interrupt:{kind}")
        if outcome.status == "approved":
            outcome = runtime.run(
                intake.analysis_id, expected_stage_run=outcome.stage_run_id,
                idempotency_key=f"run:{case['case_id']}:analysis")
        payloads = _payloads(runtime, intake.analysis_id)
        event_failures = _failure_events(run_id, intake.analysis_id)
        loop_events = _diagnostic_events(run_id, intake.analysis_id)
        failures = _score(case, gold, payloads, outcome.status, questions, audit.calls,
                          event_failures, loop_events)
        return {"case_id": case["case_id"], "eval_id": case["eval_id"],
                "source_hash": _hash(raw), "input_hash": _hash(csv),
                "analysis_id": intake.analysis_id, "terminal_outcome": outcome.status,
                "task_results": audit.calls, "questions": questions,
                "diagnostic_loop_events": loop_events,
                "failure_events": event_failures,
                "hard_failures": failures,
                "failure_hash": content_hash({"failures": failures}), "passed": not failures}
    finally:
        runtime.close()


def _commit(directory: Path, name: str, payload: dict[str, Any]) -> tuple[Path, str]:
    directory.mkdir(parents=True, exist_ok=True)
    digest = content_hash(payload)
    document = {"schema_version": "evaluation-report.v1", "report_hash": digest, "report": payload}
    path = directory / name
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return path, digest


def matrix_validate(*, verify_sources: bool = True) -> dict[str, Any]:
    cells, gold = _matrix_documents()
    report = validate(verify_sources=verify_sources, stress=True)
    statuses = {case_id: row["scientific_suitability"]
                for case_id, row in gold["cases"].items()}
    report.update({
        "schema_version": "evaluation-matrix-validation.v1",
        "cell_ids": [row["cell_id"] for row in cells],
        "cell_count": len(cells),
        "dataset_count": len({row["case_id"] for row in cells}),
        "family_count": len({row["family"] for row in cells}),
        "scientific_suitability": statuses,
        "scientific_positive_ready": all(
            row["status"] == "suitable" for row in statuses.values()),
        "matrix_gold_approved": (MATRIX_APPROVED.exists() and
                                   MATRIX_APPROVED.read_text().strip() == _gold_hash(stress=True)),
    })
    return report


def _save_state(path: Path, state: dict[str, Any], *, create: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = datetime.now(UTC).isoformat()
    if create:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(state, stream, indent=2, sort_keys=True)
            stream.write("\n")
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_state(path: Path) -> dict[str, Any]:
    state = _load(path)
    if state.get("schema_version") != MATRIX_STATE_SCHEMA:
        raise ValueError("matrix state has an unsupported schema")
    cells = state.get("cells")
    if not isinstance(cells, list) or len(cells) != 16:
        raise ValueError("matrix state must contain exactly 16 cells")
    expected = [row["cell_id"] for row in _matrix_documents()[0]]
    if [row.get("cell_id") for row in cells] != expected:
        raise ValueError("matrix state cell identities or order changed")
    return state


def _initial_matrix_state(run_id: str) -> dict[str, Any]:
    cells, _ = _matrix_documents()
    fixture_hash = content_hash({
        "release": _hash(FIXTURES.read_bytes()), "stress": _hash(STRESS_FIXTURES.read_bytes())})
    return {
        "schema_version": MATRIX_STATE_SCHEMA,
        "run_id": run_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "repo_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True,
            check=True).stdout.strip(),
        "worktree_at_start": _worktree(),
        "fixture_hash": str(fixture_hash),
        "gold_hash": _gold_hash(stress=True),
        "model_profile": VERTEX_PROFILE_V1.model_dump(mode="json"),
        "cells": [{
            "cell_id": row["cell_id"], "case_id": row["case_id"],
            "eval_id": row["eval_id"], "family": row["family"],
            "context_variant": row["context_variant"], "status": "not_started",
            "analysis_id": None, "stage_run_id": None, "database_name": None,
            "bucket": None, "task_results": [], "questions": [], "hitl": [],
            "pending_request": None, "result": None,
        } for row in cells],
        "final_report": None,
    }


def _matrix_dsn(database_name: str) -> str:
    return f"{os.environ['CAUSAL_EVAL_POSTGRES_ADMIN_DSN'].rsplit('/', 1)[0]}/{database_name}"


def _write_request(directory: Path, body: dict[str, Any]) -> dict[str, Any]:
    directory = directory / str(body["cell_id"])
    directory.mkdir(parents=True, exist_ok=True)
    digest = content_hash(body)
    document = body | {"request_hash": digest}
    path = directory / f"{body['cell_id']}.{digest}.request.json"
    if path.exists():
        if _load(path) != document:
            raise RuntimeError(f"existing HITL request differs: {path}")
    else:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return document | {"path": str(path)}


def _hitl_request(run_id: str, case: dict[str, Any], cell: dict[str, Any],
                  opened: dict[str, Any], directory: Path) -> dict[str, Any]:
    kind = str(opened["kind"])
    if kind == "approval":
        payload = {key: opened[key] for key in (
            "design", "review_summary", "compiled_design", "diagnostic_report", "capacity_report")}
    else:
        payload = opened[{"table_selection": "candidate_tables", "clarification": "packet"}[kind]]
    body = {
        "schema_version": HITL_REQUEST_SCHEMA, "run_id": run_id,
        "cell_id": cell["cell_id"], "case_id": cell["case_id"],
        "context_variant": cell["context_variant"], "kind": kind,
        "study_question": case["question"], "source": case["source"],
        "interrupt": {
            "interrupt_id": opened["interrupt_artifact_id"],
            "expected_interrupt_hash": opened["interrupt_hash"],
            "expected_revision": opened["design_revision"], "interrupt_kind": kind,
        },
        "payload": payload,
    }
    request = _write_request(directory, body)
    if kind == "clarification":
        cell["questions"].extend(row["question_id"] for row in opened["packet"]["questions"])
    return request


def _response_path(directory: Path, request: Mapping[str, Any]) -> Path:
    return (directory / str(request["cell_id"]) /
            f"{request['cell_id']}.{request['request_hash']}.response.json")


def _hitl_response(path: Path, state: Mapping[str, Any], cell: Mapping[str, Any],
                   request: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    response = _load(path)
    required = {"schema_version", "run_id", "cell_id", "request_hash", "reviewer",
                "isolation_attestation", "decision"}
    if set(response) != required:
        raise ValueError(f"{path} has unexpected HITL response fields")
    expected = (HITL_RESPONSE_SCHEMA, state["run_id"], cell["cell_id"],
                request["request_hash"], "cell_only")
    found = (response.get("schema_version"), response.get("run_id"), response.get("cell_id"),
             response.get("request_hash"), response.get("isolation_attestation"))
    if found != expected or not str(response.get("reviewer", "")).strip():
        raise ValueError(f"{path} does not match its isolated HITL request")
    if not isinstance(response.get("decision"), dict):
        raise TypeError(f"{path} decision must be one object")
    return response


def _apply_hitl(runtime: Any, cell: dict[str, Any], request: Mapping[str, Any],
                response: dict[str, Any]) -> Any:
    identity = InterruptIdentity.model_validate(request["interrupt"])
    decision = response["decision"]
    key = f"matrix:{cell['cell_id']}:{request['request_hash'][:16]}"
    if request["kind"] == "table_selection":
        selected = str(decision.get("selected_table", ""))
        available = {row["logical_name"] for row in request["payload"]}
        if selected not in available:
            raise ValueError("external table selection is not one of the offered tables")
        outcome = runtime.select_table(cell["analysis_id"], TableSelectionDecisionV1(
            interrupt_id=identity.interrupt_id,
            expected_interrupt_hash=identity.expected_interrupt_hash,
            expected_revision=identity.expected_revision, selected_table=selected,
            idempotency_key=key))
    elif request["kind"] == "clarification":
        answer = UserContextAnswerV1.model_validate_json(json.dumps(decision))
        outcome = runtime.answer_context(cell["analysis_id"], answer, identity, key)
    elif request["kind"] == "approval":
        change_requests = decision.get("change_requests", [])
        if not isinstance(change_requests, list) or not all(
                isinstance(item, str) for item in change_requests):
            raise ValueError("approval change_requests must be a list of strings")
        outcome = runtime.approve_design(
            cell["analysis_id"], identity, ApprovalDecision(str(decision.get("decision"))), key,
            change_requests=tuple(change_requests))
    else:
        raise ValueError(f"unknown external HITL kind {request['kind']}")
    cell["hitl"].append({
        "kind": request["kind"], "request_hash": request["request_hash"],
        "response_hash": content_hash(response), "reviewer": response["reviewer"],
        "isolation_attestation": response["isolation_attestation"],
    })
    cell["pending_request"] = None
    return outcome


def _matrix_result(case: dict[str, Any], gold: dict[str, Any], matrix_gold: dict[str, Any],
                   cell: dict[str, Any], runtime: Any, outcome: Any, run_id: str,
                   source_hash: str, input_hash: str) -> dict[str, Any]:
    payloads = _payloads(runtime, cell["analysis_id"])
    event_failures = _failure_events(run_id, cell["analysis_id"])
    loop_events = _diagnostic_events(run_id, cell["analysis_id"])
    scored_gold = gold | {"ask_expectation": matrix_gold[cell["context_variant"]]}
    failures = _score(case, scored_gold, payloads, outcome.status, cell["questions"],
                      cell["task_results"], event_failures, loop_events)
    suitability = gold["scientific_suitability"]
    execution_passed = not failures
    positive = (execution_passed and suitability["status"] == "suitable"
                and outcome.status in {"complete", "complete_with_qualifications"})
    return {
        "cell_id": cell["cell_id"], "case_id": cell["case_id"],
        "eval_id": cell["eval_id"], "family": cell["family"],
        "context_variant": cell["context_variant"], "analysis_id": cell["analysis_id"],
        "terminal_outcome": outcome.status, "source_hash": source_hash,
        "input_hash": input_hash, "question_hash": _hash(case["question"].encode()),
        "context_hash": _hash(str(case.get("_submission_context") or "").encode()),
        "task_results": cell["task_results"], "questions": cell["questions"],
        "hitl": cell["hitl"], "diagnostic_loop_events": loop_events,
        "failure_events": event_failures, "hard_failures": failures,
        "failure_hash": content_hash({"failures": failures}),
        "execution_passed": execution_passed,
        "scientific_suitability": suitability,
        "positive_analysis_passed": positive,
    }


def _failed_matrix_result(cell: dict[str, Any], error: Exception,
                          suitability: dict[str, Any]) -> dict[str, Any]:
    code = str(getattr(error, "code", type(error).__name__))
    classification = "observability" if "trace" in code or "preflight" in code else (
        "provider" if code in {"model_unavailable", "invalid_authentication", "permission_denied",
                               "quota_exhausted", "model_output_truncated"}
        else "compiler_capability")
    failures = [{"classification": classification, "detail": code}]
    return {
        "cell_id": cell["cell_id"], "case_id": cell["case_id"],
        "eval_id": cell["eval_id"], "family": cell["family"],
        "context_variant": cell["context_variant"], "analysis_id": cell["analysis_id"],
        "terminal_outcome": "failed", "task_results": cell["task_results"],
        "questions": cell["questions"], "hitl": cell["hitl"],
        "diagnostic_loop_events": [], "failure_events": [], "hard_failures": failures,
        "failure_hash": content_hash({"failures": failures}), "execution_passed": False,
        "scientific_suitability": suitability, "positive_analysis_passed": False,
    }


def _advance_matrix_cell(case: dict[str, Any], gold: dict[str, Any], matrix_gold: dict[str, Any],
                         state: dict[str, Any], cell: dict[str, Any], outbox: Path,
                         response: dict[str, Any] | None, tracer: LangSmithTracer,
                         event_sink: io.StringIO) -> None:
    materialized = _matrix_case(case)
    raw = _source_bytes(case["source"])
    if _hash(raw) != case["source"]["sha256"]:
        raise RuntimeError("source_hash_mismatch")
    csv = _case_csv(case, raw)
    if cell["status"] == "not_started":
        database_name, dsn = _database(state["run_id"], cell["cell_id"])
        bucket = _bucket(state["run_id"], cell["cell_id"])
        cell.update({"database_name": database_name, "bucket": bucket})
    else:
        dsn = _matrix_dsn(str(cell["database_name"]))
    base = VertexGateway(GenAiTransport(), VERTEX_PROFILE_V1, EventEmitter(event_sink),
                         lambda: datetime.now(UTC), tracer)
    audit = _AuditGateway(base, str(_suite_documents(True)[1]["prompt_leak_canary"]), {
        "run_id": state["run_id"], "case_id": cell["cell_id"],
        "seed_case_id": cell["case_id"], "context_variant": cell["context_variant"],
        "mode": "matrix"})
    runtime = build_runtime(RuntimeConfig(
        postgres_dsn=dsn, s3_bucket=str(cell["bucket"]),
        s3_endpoint_url=os.environ["CAUSAL_EVAL_S3_ENDPOINT"],
        langsmith_project=os.environ["CAUSAL_LANGSMITH_PROJECT"], environment="evaluation",
        event_log=REPORTS / f"{state['run_id']}.events.ndjson"),
        client_factory=lambda: _FixtureClient(materialized, csv), model=audit,
        strict_observability=True)
    try:
        if cell["status"] == "not_started":
            intake = runtime.new(_intake_submission(materialized))
            cell.update({"analysis_id": intake.analysis_id, "stage_run_id": intake.stage_run_id,
                         "status": "running"})
            outcome = runtime.run(
                intake.analysis_id, expected_stage_run=intake.stage_run_id,
                idempotency_key=f"matrix:{cell['cell_id']}:design")
        else:
            if response is None or not isinstance(cell.get("pending_request"), dict):
                return
            outcome = _apply_hitl(runtime, cell, cell["pending_request"], response)
        cell["task_results"].extend(audit.calls)
        audit.calls.clear()
        if outcome.status == "approved":
            outcome = runtime.run(
                cell["analysis_id"], expected_stage_run=outcome.stage_run_id,
                idempotency_key=f"matrix:{cell['cell_id']}:analysis")
            cell["task_results"].extend(audit.calls)
            audit.calls.clear()
        cell["stage_run_id"] = outcome.stage_run_id
        if outcome.status == "needs_user_input":
            opened, _ = _interrupt(runtime, outcome)
            cell["pending_request"] = _hitl_request(
                state["run_id"], case, cell, opened, outbox)
            cell["status"] = "pending_hitl"
            return
        cell["result"] = _matrix_result(
            materialized, gold, matrix_gold, cell, runtime, outcome, state["run_id"],
            _hash(raw), _hash(csv))
        cell["status"] = "terminal"
    finally:
        cell["task_results"].extend(audit.calls)
        runtime.close()


def _progress_matrix(
    state_file: Path, outbox: Path, responses: Path | None,
    selected_cell_id: str | None = None,
) -> dict[str, Any]:
    state = _load_state(state_file)
    if state.get("final_report") is not None:
        raise RuntimeError("matrix run is already finalized")
    if state["gold_hash"] != _gold_hash(stress=True):
        raise RuntimeError("matrix gold changed after the run started")
    if state["worktree_at_start"]["execution_content_hash"] != _execution_content_hash():
        raise RuntimeError("execution inputs changed after the matrix run started")
    cases, combined_gold = _matrix_documents()
    by_id = {row["cell_id"]: row for row in cases}
    environment = validate_environment(require_live=True, require_source_s3=True)
    if environment:
        raise RuntimeError("live evaluation unavailable: " + ", ".join(environment))
    tracer = LangSmithTracer(
        os.environ["CAUSAL_LANGSMITH_PROJECT"], "evaluation", TraceRedactorV1())
    tracer.preflight()
    event_sink = io.StringIO()
    try:
        for cell in state["cells"]:
            if selected_cell_id is not None and cell["cell_id"] != selected_cell_id:
                continue
            if cell["status"] == "terminal":
                continue
            response = None
            if cell["status"] == "pending_hitl" and responses is not None:
                request = cell["pending_request"]
                response = _hitl_response(
                    _response_path(responses, request), state, cell, request)
                if response is None:
                    continue
            try:
                case = by_id[cell["cell_id"]]
                _advance_matrix_cell(
                    case, combined_gold["cases"][cell["case_id"]],
                    combined_gold["matrix_expectations"], state, cell, outbox,
                    response, tracer, event_sink)
            except Exception as error:  # noqa: BLE001 - one cell cannot erase the other 15
                suitability = combined_gold["cases"][cell["case_id"]]["scientific_suitability"]
                cell["result"] = _failed_matrix_result(cell, error, suitability)
                cell["pending_request"] = None
                cell["status"] = "terminal"
            _save_state(state_file, state)
    finally:
        tracer.flush()
    return state


def matrix_start(
    state_file: Path, outbox: Path, selected_cell_id: str | None = None,
) -> dict[str, Any]:
    validation = matrix_validate()
    if not validation["passed"]:
        raise RuntimeError("matrix fixture validation failed")
    state = _initial_matrix_state(f"matrix-{uuid.uuid4().hex}")
    _save_state(state_file, state, create=True)
    return _progress_matrix(state_file, outbox, None, selected_cell_id)


def matrix_resume(
    state_file: Path, outbox: Path, responses: Path,
    selected_cell_id: str | None = None,
) -> dict[str, Any]:
    return _progress_matrix(state_file, outbox, responses, selected_cell_id)


def _matrix_status(state: Mapping[str, Any]) -> dict[str, Any]:
    statuses = {name: sum(row["status"] == name for row in state["cells"])
                for name in ("not_started", "running", "pending_hitl", "terminal")}
    return {"run_id": state["run_id"], "cell_count": len(state["cells"]),
            "statuses": statuses, "complete": statuses["terminal"] == 16,
            "pending_requests": [row["pending_request"]["path"] for row in state["cells"]
                                 if isinstance(row.get("pending_request"), dict)]}


def matrix_finalize(state_file: Path) -> tuple[Path, dict[str, Any]]:
    state = _load_state(state_file)
    if state.get("final_report"):
        path = Path(state["final_report"]["path"])
        document = _load(path)
        report = document.get("report")
        digest = content_hash(report) if isinstance(report, dict) else ""
        if digest != document.get("report_hash") or digest != state["final_report"]["report_hash"]:
            raise RuntimeError("final matrix report hash mismatch")
        return path, report
    if any(row["status"] != "terminal" or not isinstance(row.get("result"), dict)
           for row in state["cells"]):
        raise RuntimeError("matrix cannot finalize until exactly 16 cells are terminal")
    if state["gold_hash"] != _gold_hash(stress=True):
        raise RuntimeError("matrix gold changed after the run started")
    current_tree = _worktree()
    if state["worktree_at_start"]["execution_content_hash"] != current_tree["execution_content_hash"]:
        raise RuntimeError("execution inputs changed after the matrix run started")
    cases = [row["result"] for row in state["cells"]]
    if len(cases) != 16 or len({row["cell_id"] for row in cases}) != 16:
        raise RuntimeError("final report requires exactly 16 unique cells")
    approved = MATRIX_APPROVED.read_text().strip() if MATRIX_APPROVED.exists() else ""
    payload = {
        "report_id": state["run_id"], "mode": "matrix",
        "gating": approved == state["gold_hash"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "repo_revision": state["repo_revision"],
        "worktree_at_start": state["worktree_at_start"],
        "worktree_at_finalize": current_tree,
        "fixture_hash": state["fixture_hash"], "gold_hash": state["gold_hash"],
        "approved_gold_hash": approved or None,
        "gold_approved": approved == state["gold_hash"],
        "model_profile": state["model_profile"], "cell_count": len(cases),
        "cases": cases,
        "hard_failure_count": sum(len(row["hard_failures"]) for row in cases),
        "execution_passed": all(row["execution_passed"] for row in cases),
        "scientific_positive_ready": all(
            row["scientific_suitability"]["status"] == "suitable" for row in cases
            if isinstance(row.get("scientific_suitability"), dict)),
        "passed": (approved == state["gold_hash"]
                   and all(row["positive_analysis_passed"] for row in cases)),
        "historical_report_inputs": [],
        "diagnostics": {
            "model_call_count": sum(len(row["task_results"]) for row in cases),
            "reasoning_observed_count": sum(
                bool(call.get("reasoning_present")) for row in cases
                for call in row["task_results"]),
            "tokens": sum(call.get("tokens", {}).get("total", 0) for row in cases
                          for call in row["task_results"]),
            "thinking_tokens": sum(call.get("tokens", {}).get("thinking", 0) for row in cases
                                   for call in row["task_results"]),
            "latency_ms": sum(call.get("latency_ms", 0) for row in cases
                              for call in row["task_results"]), "cost": None,
        },
    }
    path, digest = _commit(REPORTS, f"{state['run_id']}.json", payload)
    state["final_report"] = {"path": str(path), "report_hash": digest}
    _save_state(state_file, state)
    return path, payload


def run_live(mode: str) -> tuple[Path, dict[str, Any]]:
    environment = validate_environment(require_live=True)
    if environment:
        raise RuntimeError("live evaluation unavailable: " + ", ".join(environment))
    approved = APPROVED.read_text().strip() if APPROVED.exists() else ""
    if mode == "gate" and approved != _gold_hash():
        raise RuntimeError("gate requires approved-gold.sha256 to match HumanGoldV1")
    stress = mode == "stress"
    validation = validate(stress=stress)
    if not validation["passed"]:
        raise RuntimeError("fixture validation failed")
    fixtures, gold = _suite_documents(stress)
    run_id = f"eval-{uuid.uuid4().hex}"
    tracer = LangSmithTracer(
        os.environ["CAUSAL_LANGSMITH_PROJECT"], "evaluation", TraceRedactorV1())
    tracer.preflight()
    event_sink = io.StringIO()
    cases = []
    for case in fixtures["cases"]:
        try:
            cases.append(_run_case(case, gold["cases"][case["case_id"]], run_id,
                                   tracer, event_sink, mode,
                                   str(gold["prompt_leak_canary"])))
        except Exception as error:  # noqa: BLE001 - each journey becomes a typed report row
            code = str(getattr(error, "code", type(error).__name__))
            classification = "observability" if "trace" in code or "preflight" in code else (
                "provider" if code in {"model_unavailable", "invalid_authentication",
                                      "permission_denied", "quota_exhausted",
                                      "model_output_truncated"} else "compiler_capability")
            cases.append({"case_id": case["case_id"], "eval_id": case["eval_id"],
                          "terminal_outcome": "failed", "task_results": [], "diagnostic_loop_events": [],
                          "hard_failures": [{"classification": classification, "detail": code}],
                          "failure_hash": content_hash({"failures": [{
                              "classification": classification, "detail": code}]}),
                          "passed": False})
    tracer.flush()
    repo_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True,
        check=True).stdout.strip()
    payload = {
        "report_id": run_id, "mode": mode, "created_at_utc": datetime.now(UTC).isoformat(),
        "gating": mode == "gate", "repo_revision": repo_revision,
        "fixture_hash": validation["fixture_hash"],
        "gold_hash": _gold_hash(stress=stress),
        "approved_gold_hash": approved or None,
        "model_profile": VERTEX_PROFILE_V1.model_dump(mode="json"),
        "cases": cases, "hard_failure_count": sum(len(row["hard_failures"]) for row in cases),
        "passed": all(row["passed"] for row in cases),
        "diagnostics": {
            "correction_count": sum(max(0, len(row["task_results"]) - len({
                call["task_id"] for call in row["task_results"]})) for row in cases),
            "tokens": sum(call.get("tokens", {}).get("total", 0) for row in cases
                          for call in row["task_results"]),
            "latency_ms": sum(call.get("latency_ms", 0) for row in cases
                              for call in row["task_results"]), "cost": None}}
    path, digest = _commit(REPORTS, f"{run_id}.json", payload)
    if mode == "calibrate":
        _commit(REVIEW, f"{run_id}.json", {
            "calibration_report": str(path.relative_to(ROOT)), "calibration_report_hash": digest,
            "proposed_gold_hash": _gold_hash(), "reviewer": None,
            "instruction": "Review every case, then place only the approved hash in "
                           "evals/approved-gold.sha256. The evaluator never self-approves gold."})
    return path, payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=MODES)
    parser.add_argument("--skip-source-check", action="store_true",
                        help="Validate schemas only; forbidden in live modes.")
    parser.add_argument("--state-file", type=Path,
                        help="Mutable controller state for matrix start/resume/finalize.")
    parser.add_argument("--hitl-outbox", type=Path,
                        help="Directory receiving immutable isolated HITL requests.")
    parser.add_argument("--hitl-inbox", type=Path,
                        help="Directory containing externally supplied HITL responses.")
    parser.add_argument("--cell-id",
                        help="Advance only this matrix cell; the run still contains all 16.")
    args = parser.parse_args(argv)
    try:
        if args.mode == "validate":
            report = validate(verify_sources=not args.skip_source_check)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["passed"] else 1
        if args.mode == "matrix-validate":
            report = matrix_validate(verify_sources=not args.skip_source_check)
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0 if report["passed"] else 1
        if args.mode in {"matrix-start", "matrix-resume", "matrix-finalize"}:
            if args.skip_source_check:
                raise RuntimeError("--skip-source-check is forbidden for matrix execution")
            if args.state_file is None:
                raise RuntimeError(f"{args.mode} requires --state-file")
            if args.mode == "matrix-finalize":
                path, report = matrix_finalize(args.state_file)
                print(json.dumps({"report": str(path), "passed": report["passed"]}, sort_keys=True))
                return 0 if report["passed"] else 1
            if args.hitl_outbox is None:
                raise RuntimeError(f"{args.mode} requires --hitl-outbox")
            if args.mode == "matrix-start":
                state = matrix_start(args.state_file, args.hitl_outbox, args.cell_id)
            else:
                if args.hitl_inbox is None:
                    raise RuntimeError("matrix-resume requires --hitl-inbox")
                state = matrix_resume(
                    args.state_file, args.hitl_outbox, args.hitl_inbox, args.cell_id)
            print(json.dumps(_matrix_status(state), indent=2, sort_keys=True))
            return 0
        if args.skip_source_check:
            raise RuntimeError("--skip-source-check is valid only for validate")
        path, report = run_live(args.mode)
        print(json.dumps({"report": str(path), "passed": report["passed"]}, sort_keys=True))
        return 0 if report["passed"] else 1
    except Exception as error:  # noqa: BLE001 - command boundary returns one typed exit
        print(json.dumps({"passed": False, "error": str(error)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
