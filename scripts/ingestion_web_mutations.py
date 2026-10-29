"""Mutate real frontend code in scratch copies, never the working tree."""

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = "uses the real adapter to transmit exact multipart fields and bytes"
MUTANTS = {
    "context_not_sent": ("src/api.ts", "src/api.test.ts", EXPECTED, 'body.append("context", JSON.stringify(context));', "void context;"),
    "wrong_upload_route": ("src/api.ts", "src/api.test.ts", EXPECTED, 'xhr.open("POST", "/api/datasets");', 'xhr.open("POST", "/api/wrong");'),
    "wrong_submission_key": (
        "src/api.ts",
        "src/api.test.ts",
        EXPECTED,
        'body.append("idempotency_key", key);',
        'body.append("idempotency_key", "constant"); void key;',
    ),
    "context_utf16_offsets": (
        "src/components/DatasetContext.tsx",
        "src/components/DatasetContext.test.tsx",
        "loads without generation and fetches column context only on expansion",
        'Array.from(data?.source_text ?? "")',
        '(data?.source_text ?? "").split("")',
    ),
    "context_html_rendering": (
        "src/components/DatasetContext.tsx",
        "src/components/DatasetContext.test.tsx",
        "loads without generation and fetches column context only on expansion",
        '<p style={{ whiteSpace: "pre-wrap" }}>{s.value}</p>',
        "<p dangerouslySetInnerHTML={{ __html: s.value }} />",
    ),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase4", action="store_true", help="Run only context component and fault controls")
    parser.add_argument("--out", type=Path, default=ROOT / ".artifacts" / "ingestion-tests")
    options = parser.parse_args()
    output = options.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    mutants = {k: v for k, v in MUTANTS.items() if not options.phase4 or k in {"context_utf16_offsets", "context_html_rendering"}}
    results = []
    with tempfile.TemporaryDirectory(prefix="ingestion-web-mutations-") as scratch:
        project = Path(scratch) / "web"
        project.mkdir()
        shutil.copytree(ROOT / "web" / "src", project / "src")
        for name in ["package.json", "vite.config.ts", "tsconfig.json"]:
            shutil.copy2(ROOT / "web" / name, project / name)
        (project / "node_modules").symlink_to(ROOT / "web" / "node_modules", target_is_directory=True)
        originals = {path: (project / path).read_text() for path, *_ in mutants.values()}
        for name, change in [("web_baseline", None), *mutants.items()]:
            for path, original in originals.items():
                (project / path).write_text(original)
            if change:
                path, test, expected, before, after = change
                original = originals[path]
                assert original.count(before) == 1, f"Mutation location drifted: {name}"
                (project / path).write_text(original.replace(before, after))
                selected = [test]
            else:
                selected = ["src/components/DatasetContext.test.tsx"] if options.phase4 else ["src/api.test.ts", "src/components/DatasetContext.test.tsx"]
            report = output / f"{name}.xml"
            report.unlink(missing_ok=True)
            run = subprocess.run(
                ["npm", "--prefix", str(project), "test", "--", *selected, "--reporter=junit", f"--outputFile={report}"],
                cwd=ROOT,
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=60,
            )
            (output / f"{name}.log").write_text(run.stdout + run.stderr)
            if not report.exists():
                raise RuntimeError(f"Missing report for {name}")
            cases = ET.parse(report).findall(".//testcase")
            failed = [c for c in cases if c.find("failure") is not None]
            bad_harness = any(c.find("error") is not None or c.find("skipped") is not None for c in cases)
            if change:
                verified = run.returncode == 1 and len(failed) == 1 and failed[0].get("name") == expected and not bad_harness
            else:
                verified = run.returncode == 0 and len(cases) == (7 if options.phase4 else 12) and not failed and not bad_harness
            results.append({"mutation": name, "verified": verified, "failures": len(failed)})
            (output / "web-results.json").write_text(json.dumps(results, indent=2))
            print(f"{name}: {'verified' if verified else 'FAILED'}", flush=True)
            if not verified:
                print(run.stdout + run.stderr)
                raise SystemExit(1)


if __name__ == "__main__":
    main()
