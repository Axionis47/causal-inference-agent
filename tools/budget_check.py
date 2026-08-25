"""Complexity-budget checker implementing counting rule python-token-lines.v1.

SYSTEM-CONTRACT SS14.1.1: counts significant physical lines per scope, measures
module/function sizes, and emits one ImplementationBudgetReportV1 JSON object.
Standard library only.
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

COUNTING_RULE_VERSION = "python-token-lines.v1"

PY_EXTENSIONS = {".py", ".pyi"}
DECLARATIVE_EXTENSIONS = {".sql", ".json", ".yaml", ".yml", ".toml", ".txt", ".md", ".jinja"}
COMMENT_PREFIXES = {".sql": "--", ".yaml": "#", ".yml": "#", ".toml": "#"}

# Scope assignment order matters: first matching prefix wins (D-003, D-008).
SCOPE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("src/causal/shared/", "shared"),
    ("src/causal/cli/", "cli"),
    ("src/causal/runtime/", "runtime"),
    ("src/causal/intake/", "intake"),
    ("src/causal/design/", "design"),
    ("src/causal/preparation/", "preparation"),
    ("src/causal/estimation/", "estimation"),
    ("src/causal/presentation/", "presentation"),
    ("src/causal/", "shared"),
    ("tests/", "tests"),
    ("tools/", "tests"),
    ("migrations/", "declarative"),
    ("evals/", "declarative"),
    ("prompts/", "declarative"),
    ("registries/", "declarative"),
)
EXCLUDED_FILES = {"pyproject.toml", "uv.lock", ".python-version", ".gitignore", "CLAUDE.md", "README.md"}
EXCLUDED_PREFIXES = ("docs/", ".claude/")

PRODUCTION_SCOPES = ("shared", "cli", "runtime", "intake", "design", "preparation", "estimation", "presentation")
CEILINGS = {
    "shared": 2500, "cli": 500, "runtime": 800, "intake": 1200, "design": 3400,
    "preparation": 2000, "estimation": 4000, "presentation": 1500, "tests": 8000, "declarative": 3000,
}
PRODUCTION_TOTAL_CEILING = 15_000
GRAND_TOTAL_CEILING = 26_000
MAX_PRODUCTION_MODULES = 50
MAX_MODULE_LINES = 350
MAX_FUNCTION_LINES = 75
WARNING_FRACTION = 0.8

_INSIGNIFICANT_TOKENS = frozenset(
    {tokenize.ENCODING, tokenize.NEWLINE, tokenize.NL, tokenize.INDENT,
     tokenize.DEDENT, tokenize.COMMENT, tokenize.ENDMARKER}
)


def significant_python_lines(text: str) -> set[int]:
    """Physical line numbers touched by at least one significant Python token."""
    lines: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type in _INSIGNIFICANT_TOKENS:
            continue
        lines.update(range(tok.start[0], tok.end[0] + 1))
    return lines


def count_python(text: str) -> int:
    return len(significant_python_lines(text))


def count_declarative(text: str, extension: str) -> int:
    comment = COMMENT_PREFIXES.get(extension)
    total = 0
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or (comment is not None and stripped.startswith(comment)):
            continue
        total += 1
    return total


def assign_scope(path: str) -> str | None:
    """Scope name, None when excluded/uncounted, or 'UNASSIGNED' (a blocker)."""
    if path in EXCLUDED_FILES or path.startswith(EXCLUDED_PREFIXES):
        return None
    extension = Path(path).suffix.lower()
    if extension not in PY_EXTENSIONS | DECLARATIVE_EXTENSIONS:
        return None
    for prefix, scope in SCOPE_PREFIXES:
        if path.startswith(prefix):
            return scope
    return "UNASSIGNED"


def count_file(path: str, text: str) -> int:
    extension = Path(path).suffix.lower()
    if extension in PY_EXTENSIONS:
        return count_python(text)
    return count_declarative(text, extension)


def function_sizes(text: str, lines: set[int]) -> list[tuple[str, int]]:
    """(name, significant-line count) for every function/method via the AST."""
    sizes: list[tuple[str, int]] = []
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = node.end_lineno if node.end_lineno is not None else node.lineno
            sizes.append((node.name, sum(1 for line in lines if node.lineno <= line <= end)))
    return sizes


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    )
    return result.stdout


def list_files(repo: Path, revision: str | None) -> list[str]:
    if revision is None:
        output = _git(repo, "ls-files")
    else:
        output = _git(repo, "ls-tree", "-r", "--name-only", revision)
    return [line for line in output.splitlines() if line]


def read_file(repo: Path, revision: str | None, path: str) -> str:
    if revision is None:
        return (repo / path).read_text(encoding="utf-8")
    return _git(repo, "show", f"{revision}:{path}")


@dataclass
class Snapshot:
    per_file: dict[str, tuple[str, int]]
    unassigned: list[str]
    module_count: int
    largest_module: tuple[str, int] | None
    largest_function: tuple[str, str, int] | None

    def scope_totals(self) -> dict[str, int]:
        totals = {scope: 0 for scope in CEILINGS}
        for scope, count in self.per_file.values():
            totals[scope] += count
        return totals


def take_snapshot(repo: Path, revision: str | None) -> Snapshot:
    per_file: dict[str, tuple[str, int]] = {}
    unassigned: list[str] = []
    module_count = 0
    largest_module: tuple[str, int] | None = None
    largest_function: tuple[str, str, int] | None = None
    for path in list_files(repo, revision):
        scope = assign_scope(path)
        if scope is None:
            continue
        if scope == "UNASSIGNED":
            unassigned.append(path)
            continue
        text = read_file(repo, revision, path)
        count = count_file(path, text)
        per_file[path] = (scope, count)
        if scope in PRODUCTION_SCOPES and Path(path).suffix.lower() in PY_EXTENSIONS:
            module_count += 1
            if largest_module is None or count > largest_module[1]:
                largest_module = (path, count)
            lines = significant_python_lines(text)
            for name, size in function_sizes(text, lines):
                if largest_function is None or size > largest_function[2]:
                    largest_function = (path, name, size)
    return Snapshot(per_file, unassigned, module_count, largest_module, largest_function)


def find_breaches(snapshot: Snapshot) -> tuple[list[dict[str, object]], list[str]]:
    totals = snapshot.scope_totals()
    production_total = sum(totals[scope] for scope in PRODUCTION_SCOPES)
    grand_total = sum(totals.values())
    checks: list[tuple[str, int, int]] = [
        *((f"scope:{scope}", totals[scope], CEILINGS[scope]) for scope in CEILINGS),
        ("production_total", production_total, PRODUCTION_TOTAL_CEILING),
        ("grand_total", grand_total, GRAND_TOTAL_CEILING),
        ("production_modules", snapshot.module_count, MAX_PRODUCTION_MODULES),
    ]
    if snapshot.largest_module is not None:
        checks.append(("largest_module", snapshot.largest_module[1], MAX_MODULE_LINES))
    if snapshot.largest_function is not None:
        checks.append(("largest_function", snapshot.largest_function[2], MAX_FUNCTION_LINES))
    breaches = [
        {"dimension": name, "actual": actual, "limit": limit}
        for name, actual, limit in checks if actual > limit
    ]
    warnings = [
        name for name, actual, limit in checks
        if actual <= limit and actual >= limit * WARNING_FRACTION and actual > 0
    ]
    return breaches, warnings


def build_report(
    repo: Path, task_id: str, base: str, proposed: str | None,
    worktree: bool, parent_rethink_id: str | None,
) -> dict[str, object]:
    base_snapshot = take_snapshot(repo, base)
    proposed_snapshot = take_snapshot(repo, None if worktree else proposed)
    base_totals = base_snapshot.scope_totals()
    new_totals = proposed_snapshot.scope_totals()
    scopes: dict[str, dict[str, int]] = {}
    for scope in CEILINGS:
        added = deleted = 0
        paths = set(base_snapshot.per_file) | set(proposed_snapshot.per_file)
        for path in paths:
            old = base_snapshot.per_file.get(path)
            new = proposed_snapshot.per_file.get(path)
            if (old and old[0] != scope) and (new and new[0] != scope):
                continue
            old_count = old[1] if old and old[0] == scope else 0
            new_count = new[1] if new and new[0] == scope else 0
            delta = new_count - old_count
            if delta > 0:
                added += delta
            else:
                deleted -= delta
        scopes[scope] = {
            "baseline": base_totals[scope], "added": added, "deleted": deleted,
            "projected": new_totals[scope], "actual": new_totals[scope],
        }
    changed = sorted(
        path
        for path in set(base_snapshot.per_file) | set(proposed_snapshot.per_file)
        if base_snapshot.per_file.get(path) != proposed_snapshot.per_file.get(path)
    )
    breaches, warnings = find_breaches(proposed_snapshot)
    if proposed_snapshot.unassigned:
        status = "blocked_complexity_budget"
    elif breaches:
        status = "rethink_required"
    elif warnings:
        status = "warning"
    else:
        status = "within_budget"
    largest_module = proposed_snapshot.largest_module
    largest_function = proposed_snapshot.largest_function
    return {
        "counting_rule_version": COUNTING_RULE_VERSION,
        "implementation_task_id": task_id,
        "base_revision": base,
        "proposed_revision": "WORKTREE" if worktree else proposed,
        "scopes": scopes,
        "module_count": proposed_snapshot.module_count,
        "largest_module": (
            {"path": largest_module[0], "lines": largest_module[1]} if largest_module else None
        ),
        "largest_function": (
            {"path": largest_function[0], "name": largest_function[1], "lines": largest_function[2]}
            if largest_function else None
        ),
        "changed_files": changed,
        "unassigned_files": proposed_snapshot.unassigned,
        "breached_dimensions": breaches,
        "warning_dimensions": warnings,
        "status": status,
        "parent_rethink_id": parent_rethink_id,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--base", required=True, help="base git revision")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--proposed", help="proposed git revision")
    target.add_argument("--worktree", action="store_true", help="measure the working tree")
    parser.add_argument("--parent-rethink", default=None)
    parser.add_argument("--json", action="store_true", help="accepted for compatibility; output is always JSON")
    args = parser.parse_args(argv)
    repo = Path(_git(Path.cwd(), "rev-parse", "--show-toplevel").strip())
    report = build_report(
        repo, args.task_id, args.base, args.proposed, args.worktree, args.parent_rethink
    )
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0 if report["status"] in {"within_budget", "warning"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
