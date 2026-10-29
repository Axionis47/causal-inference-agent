"""One Phase 4 verification entry point. Tests stay with their owners; evidence is run-scoped."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / ".artifacts" / "phase4"
PYTHON = sys.executable
SCOPED = ["causal_agent/memory/context", "causal_agent/server/context", "causal_agent/server/tests/context"]


def junit(path: Path) -> dict:
    cases = ET.parse(path).findall(".//testcase")
    counts = {"cases": len(cases), **{tag: sum(c.find(tag) is not None for c in cases) for tag in ("failure", "error", "skipped")}}
    if not cases or any(counts[tag] for tag in ("failure", "error", "skipped")):
        raise ValueError(f"Required cases must be nonempty and pass without skips: {counts}")
    return counts


def contracts(out: Path):
    """Generate into scratch, comparing content without rewriting checkout files."""
    schema = subprocess.check_output([PYTHON, "-m", "causal_agent.server", "--openapi"], cwd=ROOT)
    if json.loads(schema) != json.loads((ROOT / "web/openapi.json").read_text()):
        raise ValueError("OpenAPI differs from the current server; run make schema")
    generated = out / "schema.ts"
    subprocess.run(["npm", "--prefix", "web", "exec", "--", "openapi-typescript", "web/openapi.json", "-o", str(generated)], cwd=ROOT, check=True)
    subprocess.run(["npm", "--prefix", "web", "exec", "--", "prettier", "--write", str(generated)], cwd=ROOT, check=True)
    if generated.read_bytes() != (ROOT / "web/src/generated/schema.ts").read_bytes():
        raise ValueError("Generated client types differ; run make schema")


def stages(out: Path, live: bool, all_controls: bool = False) -> list[dict]:
    def stage(name, args, **extra):
        return {"name": name, "argv": args, "status": "pending", **extra}

    controls = [] if all_controls else ["--phase4"]
    checks = [
        stage("python-style", [PYTHON, "-m", "ruff", "check", *SCOPED, "causal_agent/evals/context.py", "scripts", "docker/smoke.py"]),
        stage("python-format", [PYTHON, "-m", "ruff", "format", "--check", *SCOPED, "causal_agent/evals/context.py", "scripts", "docker/smoke.py"]),
        stage("python-types", [PYTHON, "-m", "mypy"]),
        stage("ownership-contracts", [str(Path(PYTHON).parent / "lint-imports")]),
        stage(
            "runner-and-trace-policy",
            [
                PYTHON,
                "-m",
                "pytest",
                "scripts/tests",
                "causal_agent/evals/tests/test_context.py",
                "causal_agent/common/tests/test_trace_policy.py",
                "-q",
                f"--junitxml={out / 'runner.xml'}",
            ],
            junit="runner.xml",
        ),
        stage(
            "storage-and-negative-controls",
            [PYTHON, "scripts/ingestion_mutations.py", *controls, "--out", str(out / "backend")],
            manifest="backend/results.json",
            junit="backend/baseline.xml",
        ),
        stage(
            "frontend-negative-controls",
            [PYTHON, "scripts/ingestion_web_mutations.py", *controls, "--out", str(out / "frontend")],
            manifest="frontend/web-results.json",
        ),
        stage("frontend-tests", ["npm", "--prefix", "web", "test", "--", "--reporter=junit", f"--outputFile={out / 'web.xml'}"], junit="web.xml"),
        stage("frontend-style", ["npm", "--prefix", "web", "run", "lint"]),
        stage("frontend-types", ["npm", "--prefix", "web", "run", "typecheck"]),
        stage("generated-contracts", [PYTHON, "scripts/verify_phase4.py", "--contracts", str(out)]),
        stage("frontend-build", ["npm", "--prefix", "web", "run", "build"]),
        stage("browser-and-container-replacement", [PYTHON, "docker/smoke.py"], smoke=True),
    ]
    checks.append(
        stage(
            "gemini-and-trace",
            [
                PYTHON,
                "-m",
                "causal_agent.evals.context",
                "--csv",
                "data/raw/gov-transfers-uruguay/gov_transfers.csv",
                "--source",
                "data/context/gov_transfers.md",
                "--out",
                str(out / "model.json"),
                "--verify-trace",
            ],
            live=True,
        )
    )
    if not live:
        checks[-1].update(
            status="not_requested",
            detail="Real model calls are explicitly opt-in: make verify-phase4 WITH_MODEL=1. No earlier evidence is treated as this run.",
        )
    return checks


def save_report(out: Path, checks: list[dict], started: str, revision: str):
    failed = any(s["status"] in {"failed", "interrupted"} for s in checks)
    pending = any(s["status"] == "pending" for s in checks)
    status = "failed" if failed else "running" if pending else "passed"
    live = next(s for s in checks if s["name"] == "gemini-and-trace")
    report = {
        "phase": 4,
        "started_at": started,
        "updated_at": datetime.now(UTC).isoformat(),
        "revision": revision,
        "status": status,
        "checks": checks,
        "semantic_review": "Review the live result and trace for meaning, omissions and unsupported claims; code checks do not certify model understanding."
        if live["status"] == "passed"
        else "No new semantic acceptance claimed.",
        "scope": "Original dataset/column context only. Includes shared Phases 1-3 regressions; excludes chat/analysis integration and public authentication.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    lines = [
        "# Phase 4 verification",
        "",
        f"Status: **{status}**",
        f"Started: {started}",
        f"Revision: `{revision}` (working tree included; see source-state.txt)",
        "",
        report["scope"],
        "",
        "| Check | Status | Seconds | Evidence |",
        "|---|---|---:|---|",
    ]
    for s in checks:
        log = s.get("log")
        evidence = f"[log]({log})" if log else s.get("detail", "")
        if s.get("tests"):
            evidence += f"; {s['tests']['cases']} cases, {s['tests']['skipped']} skipped"
        if "controls" in s:
            evidence += f"; {s['controls']} defect controls"
        for key in ("junit", "manifest"):
            if s.get(key) and (out / s[key]).exists():
                evidence += f"; [{key}]({s[key]})"
        lines.append(f"| {s['name']} | {s['status']} | {s.get('seconds', '')} | {evidence} |")
    lines.extend(
        [
            "",
            report["semantic_review"],
            "",
            "[Machine-readable report](report.json). Logs, JUnit, mutation manifests, browser traces and optional model/trace records belong to this run.",
        ]
    )
    for s in checks:
        if s.get("detail") and s["status"] != "not_requested":
            lines.extend(["", f"{s['name']}: {s['detail']}"])
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return status


def execute(check: dict, out: Path):
    started = time.monotonic()
    log = out / (check["name"] + ".log")
    check["log"] = log.name
    tracing = "true" if check.get("live") else "false"
    env = {**os.environ, "LANGSMITH_TRACING": tracing, "LANGCHAIN_TRACING_V2": tracing}
    if check.get("smoke"):
        env["PHASE4_EVIDENCE_DIR"] = str(out / "browser")
    try:
        with log.open("w") as stream:
            # Children own their bounded waits and cleanup. Do not kill a Docker
            # harness from an outer timeout before its finally block can run.
            result = subprocess.run(check["argv"], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        check["exit_code"] = result.returncode
        if result.returncode:
            raise ValueError(f"Command exited {result.returncode}; see {log.name}")
        if check.get("junit"):
            check["tests"] = junit(out / check["junit"])
        if check.get("smoke"):
            check["browser_tests"] = {name: junit(out / "browser" / name / "results.xml") for name in check.get("browser_suites", ("ingestion", "context"))}
        if check.get("manifest"):
            rows = json.loads((out / check["manifest"]).read_text())
            if len(rows) < 2 or not all(r.get("verified") is True for r in rows):
                raise ValueError("Missing or failed negative-control evidence")
            check["controls"] = len(rows) - 1
        check["status"] = "passed"
    except KeyboardInterrupt:
        check.update(status="interrupted", detail="Interrupted; inspect the child log for cleanup status.")
        raise
    except Exception as exc:
        check.update(status="failed", detail=str(exc))
    finally:
        check["seconds"] = round(time.monotonic() - started, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-model", action="store_true", help="Send the public Uruguay study account and headers to Gemini and LangSmith; never CSV rows")
    parser.add_argument("--all-controls", action="store_true", help="Also run the existing Phases 1-3 mutation controls (CI)")
    parser.add_argument("--contracts", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.contracts:
        contracts(args.contracts)
        return
    started = datetime.now(UTC)
    out = REPORTS / (started.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    out.mkdir(parents=True)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    state = subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True)
    (out / "source-state.txt").write_text(state)
    # Record current source bytes too, since this repository may have uncommitted work.
    paths = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, text=True).splitlines()
    hashes = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths if (ROOT / p).is_file() and not (ROOT / p).is_symlink()}
    (out / "source-hashes.json").write_text(json.dumps(hashes, indent=2))
    checks = stages(out, args.with_model, args.all_controls)
    print(f"Combined report: {out / 'report.md'}", flush=True)
    try:
        for check in checks:
            save_report(out, checks, started.isoformat(), revision)
            if check["status"] != "pending":
                continue
            print(f"Running {check['name']}", flush=True)
            execute(check, out)
            print(f"{check['name']}: {check['status']}", flush=True)
    finally:
        status = save_report(out, checks, started.isoformat(), revision)
        print(f"{status.upper()}: {out / 'report.md'}", flush=True)
    raise SystemExit(1 if status != "passed" else 0)


if __name__ == "__main__":
    main()
