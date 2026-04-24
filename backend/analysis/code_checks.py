"""Bounded static verification of generated source, never target-data execution."""

import ast
import io
import os
import re
import subprocess
import sys
import tempfile
import tokenize
from importlib.metadata import version
from pathlib import Path
from typing import Any

from backend.context.service import fingerprint


class ProgramCheckError(ValueError):
    pass


def program_summary(plan: dict[str, Any]) -> str:
    """Program metadata describes frozen scope, never unobserved numerical results."""
    return "Python implementation of frozen effects: " + ", ".join(sorted(item["effect_id"] for item in plan["effects"])) + "."


def output_contract(plan: dict[str, Any]) -> dict[str, Any]:
    """Render exact field names from an already frozen plan, without choosing science."""
    effects: list[dict[str, Any]] = []
    for effect in plan["effects"]:
        identity, kind = effect["effect_id"], effect.get("result_kind", "estimate")
        prefix = identity + "_"
        fields = {name: prefix + name for name in ("estimate", "ci_low", "ci_high", "n")}
        item: dict[str, Any] = {"effect_id": identity, "result_kind": kind}
        if kind == "estimate":
            item.update(
                required_fields=list(fields.values()),
                constraints=["n is a positive integer", "ci_low <= ci_high"],
            )
        elif kind == "estimate_without_inference":
            item.update(
                required_fields=[fields["estimate"], fields["n"]],
                forbidden_fields=[fields["ci_low"], fields["ci_high"]],
                constraints=["n is a positive integer"],
            )
        elif kind == "structural_zero":
            item.update(
                required_fields=[fields["estimate"]],
                forbidden_fields=[fields["ci_low"], fields["ci_high"]],
                fixed_values={fields["estimate"]: 0},
            )
        elif kind == "confidence_set":
            item.update(
                required_fields=[prefix + "confidence_status", prefix + "region_count", fields["n"]],
                forbidden_fields=[fields["ci_low"], fields["ci_high"]],
                status_values={
                    "0": "one bounded region",
                    "1": "at least two disconnected regions",
                    "2": "empty set, no regions",
                    "3": "unresolved tail, finite searched range required",
                },
                region_fields={
                    "count_field": prefix + "region_count",
                    "index": "0 through region_count - 1",
                    "low_pattern": prefix + "region_{i}_low",
                    "high_pattern": prefix + "region_{i}_high",
                },
                conditional_fields={
                    "when_confidence_status_is_3": [prefix + "searched_low", prefix + "searched_high"]
                },
                constraints=[
                    "n is a positive integer",
                    "region_count is an integer from 0 through 100",
                    "Regions have low <= high and are ordered with each next low greater than the previous high",
                    "An unresolved-tail searched range has searched_low < searched_high and contains every returned region",
                ],
            )
        else:
            raise ValueError(f"Unsupported frozen result kind: {kind}")
        effects.append(item)
    return {
        "filename": "facts.json",
        "structure": "One flat JSON object with string keys and finite numeric values",
        "value_types": "int or float, excluding booleans",
        "maximum_fields": 500,
        "nested_effect_objects": False,
        "effects": effects,
    }


class _FactsWriters(ast.NodeVisitor):
    """Recognize imported JSON writers and literal destinations without executing code."""

    def __init__(self) -> None:
        self.imports: dict[str, str] = {}
        self.files: dict[str, str] = {}
        self.open_available = True
        self.values: list[ast.expr] = []
        self.unknown = 0
        self.other = 0

    def _bind(self, target: ast.AST) -> None:
        for item in ast.walk(target):
            if isinstance(item, ast.Name):
                self.imports.pop(item.id, None)
                self.files.pop(item.id, None)
                if item.id == "open":
                    self.open_available = False

    def _path(self, value: ast.expr) -> str | None:
        if isinstance(value, ast.Name):
            return self.files.get(value.id)
        if (
            self.open_available
            and isinstance(value, ast.Call)
            and isinstance(value.func, ast.Name)
            and value.func.id == "open"
        ):
            name = (
                value.args[0]
                if value.args
                else next((item.value for item in value.keywords if item.arg == "file"), None)
            )
            if isinstance(name, ast.Constant) and isinstance(name.value, str):
                return name.value
        return None

    def visit_Import(self, node: ast.Import) -> None:
        for item in node.names:
            name = item.asname or item.name.split(".")[0]
            self._bind(ast.Name(id=name))
            if item.name == "json":
                self.imports[name] = "module"

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for item in node.names:
            name = item.asname or item.name
            self._bind(ast.Name(id=name))
            if node.module == "json" and item.name == "dump":
                self.imports[name] = "dump"
            elif item.name == "*":
                self.imports.clear()
                self.open_available = False

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        for target in node.targets:
            self._bind(target)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value:
            self.visit(node.value)
        self._bind(node.target)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self._bind(node.target)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._bind(node.target)

    def visit_For(self, node: ast.For | ast.AsyncFor) -> None:
        self.visit(node.iter)
        self._bind(node.target)
        for statement in [*node.body, *node.orelse]:
            self.visit(statement)

    visit_AsyncFor = visit_For

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        imports, files, open_available = self.imports.copy(), self.files.copy(), self.open_available
        # A captured stream might be closed when this function eventually runs.
        self.files = {}
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        arguments += [item for item in (node.args.vararg, node.args.kwarg) if item]
        for argument in arguments:
            self._bind(ast.Name(id=argument.arg))
        for statement in node.body:
            self.visit(statement)
        self.imports, self.files, self.open_available = imports, files, open_available
        self._bind(ast.Name(id=node.name))

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_With(self, node: ast.With | ast.AsyncWith) -> None:
        before = self.files.copy()
        for item in node.items:
            self.visit(item.context_expr)
            path = self._path(item.context_expr)
            if item.optional_vars:
                self._bind(item.optional_vars)
                if path is not None and isinstance(item.optional_vars, ast.Name):
                    self.files[item.optional_vars.id] = path
        for statement in node.body:
            self.visit(statement)
        self.files = {name: path for name, path in self.files.items() if before.get(name) == path}

    visit_AsyncWith = visit_With

    def visit_Call(self, node: ast.Call) -> None:
        function = node.func
        writer = (isinstance(function, ast.Name) and self.imports.get(function.id) == "dump") or (
            isinstance(function, ast.Attribute)
            and function.attr == "dump"
            and isinstance(function.value, ast.Name)
            and self.imports.get(function.value.id) == "module"
        )
        if writer:
            value = (
                node.args[0]
                if node.args
                else next((item.value for item in node.keywords if item.arg == "obj"), None)
            )
            stream = (
                node.args[1]
                if len(node.args) > 1
                else next((item.value for item in node.keywords if item.arg == "fp"), None)
            )
            destination = self._path(stream) if stream else None
            if destination == "facts.json" and value is not None:
                self.values.append(value)
            elif destination is None:
                self.unknown += 1
            else:
                self.other += 1
        self.generic_visit(node)


def _facts_copy(code: str, tree: ast.Module) -> tuple[str, str, _FactsWriters]:
    finder = _FactsWriters()
    finder.visit(tree)
    helper = "_causal_numeric_facts_" + fingerprint(code)[:16]
    names = {item.id for item in ast.walk(tree) if isinstance(item, ast.Name)}
    while helper in names:
        helper += "_"
    encoded = code.encode("utf-8")
    offsets = [0]
    for line in code.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line.encode("utf-8")))
    insertions: list[tuple[int, bytes]] = []
    for index, value in enumerate(finder.values):
        # Python's numeric type hierarchy accepts bool as float-compatible, while
        # the facts JSON contract explicitly excludes it. Reject literal proof;
        # nonliteral boolean values still belong to runtime validation.
        if isinstance(value, ast.Dict) and any(
            isinstance(item, ast.Constant) and isinstance(item.value, bool) for item in value.values
        ):
            raise ProgramCheckError(
                f"Static verification rejected facts.json at line {value.lineno}: boolean values are not numeric facts [facts-output-contract]"
            )
        assert value.end_lineno is not None and value.end_col_offset is not None
        start = offsets[value.lineno - 1] + value.col_offset
        end = offsets[value.end_lineno - 1] + value.end_col_offset
        # A distinct checking-only assignment prevents contextual dict-item errors;
        # all contract mismatches become arg-type errors at the private helper.
        insertions.extend(((start, f"{helper}(({helper}_{index} := ".encode()), (end, b"))")))
    for position, fragment in sorted(insertions, key=lambda item: item[0], reverse=True):
        encoded = encoded[:position] + fragment + encoded[position:]
    source = encoded.decode("utf-8").splitlines(keepends=True)
    after = max(
        (
            item.end_lineno or 0
            for item in tree.body
            if isinstance(item, ast.ImportFrom) and item.module == "__future__"
        ),
        default=0,
    )
    source.insert(after, f"from _causal_output_contract import {helper}\n")
    return "".join(source), helper, finder


def verify(code: str) -> dict[str, Any]:
    ast.parse(code)
    # Generated comments must not be able to suppress application-owned verification.
    lines = code.splitlines(keepends=True)
    for token in tokenize.generate_tokens(io.StringIO(code).readline):
        if token.type == tokenize.COMMENT:
            row, col = token.start
            lines[row - 1] = lines[row - 1][:col] + "\n"
    clean = "".join(lines)
    checking_copy, helper, writers = _facts_copy(clean, ast.parse(clean))
    with tempfile.TemporaryDirectory(prefix="causal-static-") as root:
        path = Path(root)
        (path / "program.py").write_text(clean)
        (path / "facts_check.py").write_text(checking_copy)
        (path / "_causal_output_contract.pyi").write_text(
            "from typing import Mapping\n"
            f"def {helper}(value: Mapping[str, float]) -> Mapping[str, float]: ...\n"
        )
        (path / "mypy.ini").write_text("[mypy]\n")
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "mypy",
                    "--config-file=mypy.ini",
                    "--no-incremental",
                    "--cache-dir=cache",
                    "--no-site-packages",
                    "--follow-imports=skip",
                    "--ignore-missing-imports",
                    "--check-untyped-defs",
                    "--enable-error-code=possibly-undefined",
                    "--show-error-codes",
                    "--no-error-summary",
                    "--no-pretty",
                    "program.py",
                    "facts_check.py",
                ],
                cwd=root,
                env={"PATH": os.defpath, "PYTHONUTF8": "1"},
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ProgramCheckError(
                "Static verification was unavailable or exceeded its 15-second limit."
            ) from error
    if result.returncode not in {0, 1}:
        raise ProgramCheckError("Static verifier failed; program approval is blocked.")
    # Scientific packages are opaque here. These checks concern Python control flow,
    # not package stubs, numerical validity or complete semantic equivalence.
    codes = {"possibly-undefined", "used-before-def", "name-defined", "syntax"}
    issues = [
        line
        for line in result.stdout.splitlines()
        if line.startswith("program.py:")
        and (match := re.search(r"\[([a-z-]+)\]$", line))
        and match[1] in codes
    ]
    issues.extend(
        line.replace(helper, "facts.json flat numeric mapping")
        for line in result.stdout.splitlines()
        if line.startswith("facts_check.py:")
        and ((f'"{helper}"' in line and line.endswith("[arg-type]")) or line.endswith("[syntax]"))
    )
    if issues:
        raise ProgramCheckError("Static verification rejected the program:\n" + "\n".join(issues))
    if result.returncode and not re.search(r"\[[a-z-]+\]", result.stdout):
        raise ProgramCheckError("Static verifier returned an unreadable failure; approval is blocked.")
    return {
        "analyzer": "mypy",
        "version": version("mypy"),
        "source_hash": fingerprint(code),
        "rules": sorted(codes | {"facts-output-contract"}),
        "passed": True,
        "target_data_access": False,
        "facts_writers_checked": len(writers.values),
        "json_writers_unknown": writers.unknown,
        "other_json_writers": writers.other,
        "limitations": "Static name/control-flow checks and Mapping[str, float] checks for recognized facts.json writers only. Unknown paths, Any-typed values, booleans and finiteness still require runtime validation. No proof of statistical correctness or all execution paths.",
    }
