"""Tests for tools/budget_check.py (counting rule python-token-lines.v1)."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
import budget_check as bc


class TestPythonCounting:
    def test_docstring_lines_all_count(self) -> None:
        text = '"""line one\nline two\nline three\n"""\n'
        assert bc.count_python(text) == 4

    def test_comment_only_and_blank_lines_do_not_count(self) -> None:
        text = "# a comment\n\nx = 1\n# another\n\n"
        assert bc.count_python(text) == 1

    def test_multiline_string_span_counts_fully(self) -> None:
        text = 'x = """a\nb\nc"""\ny = 2\n'
        assert bc.count_python(text) == 4

    def test_multiline_expression_counts_each_line(self) -> None:
        text = "total = (1 +\n         2 +\n         3)\n"
        assert bc.count_python(text) == 3

    def test_empty_file_counts_zero(self) -> None:
        assert bc.count_python("") == 0


class TestDeclarativeCounting:
    def test_yaml_comments_and_blanks_excluded(self) -> None:
        text = "# header\n\nkey: value\nlist:\n  - a\n# trailing\n"
        assert bc.count_declarative(text, ".yaml") == 3

    def test_sql_double_dash_comments_excluded(self) -> None:
        text = "-- comment\nCREATE TABLE t (\n  id int\n);\n\n"
        assert bc.count_declarative(text, ".sql") == 3

    def test_json_counts_all_nonempty_lines(self) -> None:
        text = '{\n  "a": 1\n}\n\n'
        assert bc.count_declarative(text, ".json") == 3


class TestScopeAssignment:
    @pytest.mark.parametrize(
        ("path", "scope"),
        [
            ("src/causal/shared/canonical.py", "shared"),
            ("src/causal/cli/main.py", "cli"),
            ("src/causal/runtime/compose.py", "runtime"),
            ("src/causal/intake/coordinator.py", "intake"),
            ("src/causal/design/harness.py", "design"),
            ("src/causal/preparation/harness.py", "preparation"),
            ("src/causal/estimation/adapters.py", "estimation"),
            ("src/causal/presentation/coordinator.py", "presentation"),
            ("src/causal/__init__.py", "shared"),
            ("tests/shared/test_canonical.py", "tests"),
            ("tools/budget_check.py", "tests"),
            ("migrations/001_init.sql", "declarative"),
            ("evals/catalog.v1.yaml", "declarative"),
            ("prompts/intent.v1.txt", "declarative"),
        ],
    )
    def test_known_paths(self, path: str, scope: str) -> None:
        assert bc.assign_scope(path) == scope

    @pytest.mark.parametrize(
        "path",
        ["pyproject.toml", "uv.lock", "docs/LEDGER.md", ".claude/settings.json",
         "CLAUDE.md", "assets/font.ttf"],
    )
    def test_excluded_or_uncounted(self, path: str) -> None:
        assert bc.assign_scope(path) is None

    def test_stray_implementation_file_is_unassigned(self) -> None:
        assert bc.assign_scope("scripts/helper.py") == "UNASSIGNED"


class TestFunctionSizes:
    def test_function_and_method_sizes(self) -> None:
        text = textwrap.dedent(
            """\
            def small():
                return 1

            class C:
                def method(self):
                    a = 1
                    b = 2
                    return a + b
            """
        )
        sizes = dict(bc.function_sizes(text, bc.significant_python_lines(text)))
        assert sizes["small"] == 2
        assert sizes["method"] == 4


class TestEvaluation:
    def _snapshot(self, per_file: dict[str, tuple[str, int]]) -> bc.Snapshot:
        return bc.Snapshot(per_file, [], 0, None, None)

    def test_warning_at_80_percent(self) -> None:
        snapshot = self._snapshot({"src/causal/cli/a.py": ("cli", 400)})
        breaches, warnings = bc.find_breaches(snapshot)
        assert breaches == []
        assert "scope:cli" in warnings

    def test_breach_reported_with_limit_and_actual(self) -> None:
        snapshot = self._snapshot({"src/causal/cli/a.py": ("cli", 501)})
        breaches, _ = bc.find_breaches(snapshot)
        assert {"dimension": "scope:cli", "actual": 501, "limit": 500} in breaches

    def test_under_threshold_is_clean(self) -> None:
        snapshot = self._snapshot({"src/causal/cli/a.py": ("cli", 100)})
        breaches, warnings = bc.find_breaches(snapshot)
        assert breaches == [] and warnings == []


class TestEndToEnd:
    @pytest.fixture()
    def repo(self, tmp_path: Path) -> Path:
        def git(*args: str) -> None:
            subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

        git("init", "-q")
        git("config", "user.email", "t@example.com")
        git("config", "user.name", "t")
        (tmp_path / "src/causal/cli").mkdir(parents=True)
        (tmp_path / "src/causal/cli/main.py").write_text("x = 1\n")
        git("add", "-A")
        git("commit", "-qm", "base")
        (tmp_path / "src/causal/cli/main.py").write_text("x = 1\ny = 2\n")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests/test_main.py").write_text("def test_x():\n    assert True\n")
        git("add", "-A")
        return tmp_path

    def test_report_fields_and_diff(self, repo: Path) -> None:
        report = bc.build_report(repo, "T-TEST", "HEAD", None, True, None)
        for field in (
            "counting_rule_version", "implementation_task_id", "base_revision",
            "proposed_revision", "scopes", "module_count", "largest_module",
            "largest_function", "changed_files", "breached_dimensions", "status",
            "parent_rethink_id",
        ):
            assert field in report
        assert report["counting_rule_version"] == "python-token-lines.v1"
        assert report["status"] == "within_budget"
        scopes = report["scopes"]
        assert isinstance(scopes, dict)
        assert scopes["cli"] == {
            "baseline": 1, "added": 1, "deleted": 0, "projected": 2, "actual": 2,
        }
        assert scopes["tests"]["added"] == 2
        assert report["changed_files"] == [
            "src/causal/cli/main.py", "tests/test_main.py"
        ]

    def test_unassigned_file_blocks(self, repo: Path) -> None:
        (repo / "scripts").mkdir()
        (repo / "scripts/helper.py").write_text("x = 1\n")
        subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
        report = bc.build_report(repo, "T-TEST", "HEAD", None, True, None)
        assert report["status"] == "blocked_complexity_budget"
        assert report["unassigned_files"] == ["scripts/helper.py"]

    def test_cli_json_output(self, repo: Path) -> None:
        result = subprocess.run(
            [sys.executable, str(Path(bc.__file__)), "--task-id", "T-TEST",
             "--base", "HEAD", "--worktree"],
            cwd=repo, capture_output=True, text=True, check=True,
        )
        parsed = json.loads(result.stdout)
        assert parsed["implementation_task_id"] == "T-TEST"
