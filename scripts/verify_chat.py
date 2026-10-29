"""Consolidated chat verification using the existing evidence runner."""

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from scripts.verify_phase4 import PYTHON, ROOT, execute


def stages(out, live):
    def stage(name, argv, **extra):
        return {"name": name, "argv": argv, "status": "pending", **extra}

    return [
        stage(
            "python-contracts",
            [PYTHON, "-m", "pytest", "causal_agent/memory/context", "causal_agent/common/tests/test_trace_policy.py", "-q", f"--junitxml={out / 'python.xml'}"],
            junit="python.xml",
        ),
        stage(
            "storage-and-defect-controls", [PYTHON, "scripts/ingestion_mutations.py", "--chat", "--out", str(out / "storage")], manifest="storage/results.json"
        ),
        stage("python-lint", [PYTHON, "-m", "ruff", "check", "causal_agent", "scripts/verify_chat.py"]),
        stage("python-format", [PYTHON, "-m", "ruff", "format", "--check", "causal_agent", "scripts/verify_chat.py"]),
        stage("python-types", [PYTHON, "-m", "mypy"]),
        stage("layers", [str(Path(PYTHON).parent / "lint-imports")]),
        stage("web-tests", ["npm", "--prefix", "web", "test", "--", "--reporter=junit", f"--outputFile={out / 'web.xml'}"], junit="web.xml"),
        stage("web-lint", ["npm", "--prefix", "web", "run", "lint"]),
        stage("web-types", ["npm", "--prefix", "web", "run", "typecheck"]),
        stage("generated-contracts", [PYTHON, "scripts/verify_phase4.py", "--contracts", str(out)]),
        stage("web-build", ["npm", "--prefix", "web", "run", "build"]),
        stage("browser-and-container-replacement", [PYTHON, "docker/smoke.py"], smoke=True, browser_suites=["ingestion", "context", "chat"]),
        stage(
            "gemini-conversations-and-trace",
            [PYTHON, "-m", "causal_agent.evals.chat", "--out", str(out / "live")],
            live=True,
            **({} if live else {"status": "not_requested"}),
        ),
        stage(
            "real-source-lane-boundaries",
            [PYTHON, "-m", "causal_agent.evals.chat_lanes", "--out", str(out / "lanes")],
            live=True,
            **({} if live else {"status": "not_requested"}),
        ),
    ]


def report(out, checks):
    failed = any(c["status"] in {"failed", "interrupted"} for c in checks)
    status = "failed" if failed else "running" if any(c["status"] == "pending" for c in checks) else "passed"
    complete = status == "passed" and all(c["status"] == "passed" for c in checks)
    result = {
        "status": status,
        "automated_checks_complete": complete,
        "semantic_review": "required",
        "milestone_accepted": False,
        "checks": checks,
        "acceptance": "Real-model evidence requires semantic and trace readback. Scripted success alone is not milestone acceptance.",
    }
    (out / "report.json").write_text(json.dumps(result, indent=2))
    lines = [
        "# Chat verification",
        "",
        f"Checks: {status}. Semantic acceptance review remains required.",
        "",
        result["acceptance"],
        "",
        "| Stage | Status | Evidence |",
        "|---|---|---|",
    ]
    lines += [f"| {c['name']} | {c['status']} | [{c.get('log', '')}]({c.get('log', '')}) |" for c in checks]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-model", action="store_true")
    args = parser.parse_args()
    out = ROOT / ".artifacts" / "chat-verification" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    out.mkdir(parents=True)
    (out / "source-state.txt").write_text(subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True))
    checks = stages(out, args.with_model)
    print(f"Evidence: {out}", flush=True)
    for check in checks:
        report(out, checks)
        if check["status"] != "pending":
            continue
        print(f"Running {check['name']}", flush=True)
        execute(check, out)
        print(f"{check['name']}: {check['status']}", flush=True)
    raise SystemExit(0 if report(out, checks) == "passed" else 1)


if __name__ == "__main__":
    main()
