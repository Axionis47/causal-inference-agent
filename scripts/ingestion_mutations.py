"""Baseline must pass; every deliberate defect must fail its named test assertion.

Setup errors, teardown errors, skipped/empty collections and wrong tests are not kills.
Each pytest invocation provisions and destroys its own Docker resources.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = "causal_agent/server/tests/test_ingestion.py"
MUTANTS = {
    "chat_owner": "test_wrong_owner_and_dataset_cannot_read_revision",
    "chat_stale": "test_stale_edit_reports_changes_without_overwrite",
    "chat_duplicate": "test_duplicate_submission_and_active_turn_guard",
    "chat_implicit": "test_discussion_and_hypotheticals_never_write_context[discussion-What does amount represent?]",
    "semantic_owner": "test_owner_isolation",
    "semantic_version": "test_column_version_join",
    "semantic_citation": "test_invalid_citation_rejected",
    "semantic_participant": "test_invalid_participant_cannot_be_published",
    "semantic_assessment": "test_unresolved_assessment_cannot_be_published",
    "profile_owner": "test_owner_isolation",
    "profile_version": "test_column_version_join",
    "profile_numeric": "test_numeric_exception_not_discarded",
    "profile_failure": "test_check_failure_visible",
    "library_owner": "test_owner_filter",
    "library_pagination": "test_pagination",
    "library_ready": "test_only_ready",
    "cleanup_noop": "test_cleanup_removes_orphan_and_retry_publishes_once",
    "owner_fk_removed": "test_context_ownership_fk_rejects_wrong_owner",
    "context_mutable": "test_original_records_are_immutable[context]",
    "lock_disabled": "test_two_submissions_compete_through_publication",
    "verification_skipped": "test_corrupted_readback_cannot_publish",
    "owner_read_bypassed": "test_roundtrip_owner_isolation_and_replay",
    "profiler_called": "test_full_app_roundtrip_identity_and_no_runtime_calls",
}


def check_required_selection(out):
    """Prove the required integration command cannot succeed by selecting nothing or skipping."""
    with tempfile.TemporaryDirectory(prefix="ingestion-harness-controls-") as scratch:
        Path(scratch, "force_ingestion_skip.py").write_text(
            "import pytest\ndef pytest_collection_modifyitems(items):\n"
            '    for item in items:\n        item.add_marker(pytest.mark.skip(reason="deliberate harness control"))\n'
        )
        cases = [
            ("empty_integration_selection", "test_extension", [], 4),
            ("skipped_integration", "test_private_objects_refuse_anonymous_reads", ["-p", "force_ingestion_skip"], 1),
        ]
        for name, test, extra, expected in cases:
            run = subprocess.run(
                [sys.executable, "-m", "pytest", TEST + "::" + test, "--ingestion-integration", "-q", *extra],
                cwd=ROOT,
                env={**os.environ, "LANGSMITH_TRACING": "false", "PYTHONPATH": scratch},
                text=True,
                capture_output=True,
                timeout=60,
            )
            log = run.stdout + run.stderr
            (out / f"{name}.log").write_text(log)
            expected_message = "no ingestion integration cases collected" if expected == 4 else "1 skipped"
            if run.returncode != expected or expected_message not in log:
                raise RuntimeError(f"Required integration selection failed its control: {name}\n{log}")
            print(f"{name}: correctly rejected", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4", action="store_true", help="Keep the shared Phases 1-4 baseline, then run Phase 4 fault controls")
    parser.add_argument("--chat", action="store_true", help="Run reviewed-chat boundary tests and their deliberate defects")
    parser.add_argument("--out", type=Path, default=ROOT / ".artifacts" / "ingestion-tests")
    options = parser.parse_args()
    out = options.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    check_required_selection(out)
    mutants = {k: v for k, v in MUTANTS.items() if (not options.phase4 or k.startswith("semantic_")) and (not options.chat or k.startswith("chat_"))}
    results = []
    for mutation, test in [("baseline", ""), *mutants.items()]:
        xml = out / f"{mutation}.xml"
        xml.unlink(missing_ok=True)
        library_test = "causal_agent/server/tests/library/test_library.py"
        profile_test = "causal_agent/server/tests/profiling/test_api.py"
        context_test = (
            "causal_agent/server/tests/context/test_storage.py"
            if mutation in {"semantic_participant", "semantic_assessment"}
            else "causal_agent/server/tests/context/test_api.py"
        )
        target = (
            "causal_agent/server/tests/context/test_review_storage.py"
            if mutation in {"chat_owner", "chat_stale"}
            else "causal_agent/server/tests/context/test_chat_commands.py"
            if mutation.startswith("chat_")
            else context_test
            if mutation.startswith("semantic_")
            else profile_test
            if mutation.startswith("profile_")
            else library_test
            if mutation.startswith("library_")
            else TEST
        )
        targets = (
            [
                TEST,
                library_test,
                "causal_agent/server/tests/profiling",
                "causal_agent/profile/tests",
                "causal_agent/server/tests/context",
                "causal_agent/memory/context/tests",
            ]
            if not test
            else [target + "::" + test]
        )
        if mutation == "baseline" and options.chat:
            targets = [
                "causal_agent/server/tests/context/test_chat_commands.py",
                "causal_agent/server/tests/context/test_review_storage.py",
                "causal_agent/memory/context/tests/test_review.py",
            ]
        args = [sys.executable, "-m", "pytest", *targets, "--ingestion-integration", "-q", f"--junitxml={xml}"]
        if mutation != "baseline":
            args += ["--ingestion-mutation", mutation]
        print(f"Checking {mutation}: {test or 'all ingestion cases'}", flush=True)
        run = subprocess.run(
            args,
            cwd=ROOT,
            env={
                **os.environ,
                "LANGSMITH_TRACING": "false",
                "DATABASE_URL": "postgresql+psycopg://do-not-use@127.0.0.1:1/live",
                "S3_ENDPOINT": "http://127.0.0.1:1",
                "S3_BUCKET": "do-not-use-deployment",
            },
            text=True,
            capture_output=True,
            timeout=600,
        )
        (out / f"{mutation}.log").write_text(run.stdout + run.stderr)
        if not xml.exists():
            raise RuntimeError(f"{mutation}: no JUnit report; see {out}")
        cases = ET.parse(xml).findall(".//testcase")
        errors = sum(c.find("error") is not None for c in cases)
        skipped = sum(c.find("skipped") is not None for c in cases)
        failed = sum(c.find("failure") is not None for c in cases)
        if mutation == "baseline":
            passed = run.returncode == 0 and len(cases) > 0 and errors == skipped == failed == 0
        else:
            passed = run.returncode == 1 and len(cases) == 1 and cases[0].get("name") == test and failed == 1 and errors == skipped == 0
        results.append({"mutation": mutation, "test": test, "verified": passed, "failures": failed, "errors": errors, "skips": skipped})
        (out / "results.json").write_text(json.dumps(results, indent=2))
        if not passed:
            print(run.stdout + run.stderr)
            raise SystemExit(f"FAILED harness credibility check: {mutation}")
        print("baseline passed" if mutation == "baseline" else "defect detected by the named test", flush=True)
    print(f"All {len(mutants)} deliberate defects detected. Evidence: {out}")


if __name__ == "__main__":
    main()
