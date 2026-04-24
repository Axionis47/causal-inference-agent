"""Conservative static checks of explicit probability arguments, without data access."""

import ast
import math
import re
from collections import Counter
from typing import Any


def reject_literal_fallbacks(tree: ast.AST) -> None:
    """Catch literal substitutes for a result otherwise calculated in another branch.

    Only direct scalar dictionary writes are recognized. This does not certify
    arbitrary Python or prohibit a prespecified structural-zero result.
    """
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    writes: dict[tuple[str, str], list[tuple[ast.expr, int, bool]]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                and isinstance(target.slice, ast.Constant) and isinstance(target.slice.value, str)
                and target.slice.value.endswith(("_estimate", "_ci_low", "_ci_high"))):
            continue
        ancestor = parents.get(node)
        conditional = False
        while ancestor is not None:
            conditional |= isinstance(ancestor, ast.If | ast.ExceptHandler)
            ancestor = parents.get(ancestor)
        writes.setdefault((target.value.id, target.slice.value), []).append((node.value, node.lineno, conditional))
    def literal(value: ast.expr) -> bool:
        if isinstance(value, ast.UnaryOp):
            value = value.operand
        return isinstance(value, ast.Constant) and isinstance(value.value, int | float)
    for (_, key), assignments in writes.items():
        if any(not literal(value) for value, _, _ in assignments):
            for value, line, conditional in assignments:
                if conditional and literal(value):
                    raise ValueError(f"Line {line}: {key} substitutes a numeric literal in a branch while another branch computes that result. Do not fabricate finite fallback estimates or intervals. If the frozen inference is undefined, raise an explicit error before publishing facts.json. If a constant is scientifically prescribed, represent that prespecified calculation explicitly for independent review.")


def confidence_arguments(code: str, plan: dict[str, Any]) -> dict[str, Any]:
    """Check resolvable SciPy interval calls against one explicitly frozen coverage.

    This is a narrow regression check, not a proof of numerical correctness.
    Unresolved expressions remain the independent reviewer's responsibility.
    """
    levels = {float(match) / 100 for effect in plan["effects"]
              for match in re.findall(r"(\d+(?:\.\d+)?)\s*%", effect["inference"])}
    expected = next(iter(levels)) if len(levels) == 1 else None
    tree = ast.parse(code)
    reject_literal_fallbacks(tree)
    assigned = Counter(node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store))
    # Arguments and definition names can shadow module bindings.
    assigned.update(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))
    assigned.update(node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef))
    imports: dict[str, str] = {}
    constants: dict[str, ast.expr] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for item in node.names:
                imports[item.asname or item.name.split(".")[0]] = item.name if item.asname else item.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                imports[item.asname or item.name] = node.module + "." + item.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            constants[node.targets[0].id] = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            constants[node.target.id] = node.value

    def number(node: ast.AST | None, seen: frozenset[str] = frozenset()) -> float | None:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
            return float(node.value)
        if isinstance(node, ast.Name) and assigned[node.id] == 1 and node.id in constants and node.id not in seen:
            return number(constants[node.id], seen | {node.id})
        if isinstance(node, ast.BinOp):
            left, right = number(node.left, seen), number(node.right, seen)
            if left is not None and right is not None:
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.Div) and right:
                    return left / right
        return None

    def qualified(node: ast.AST) -> str:
        if isinstance(node, ast.Name) and not assigned[node.id]:
            return imports.get(node.id, "")
        if isinstance(node, ast.Attribute):
            parent = qualified(node.value)
            return parent + "." + node.attr if parent else ""
        return ""

    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or qualified(node.func) not in {"scipy.stats.t.interval", "scipy.stats.norm.interval"}:
            continue
        argument = node.args[0] if node.args else next((item.value for item in node.keywords if item.arg == "confidence"), None)
        coverage = number(argument)
        sites.append({"line": node.lineno, "api": qualified(node.func), "coverage": coverage, "expected": expected})
        if coverage is not None and expected is not None and not math.isclose(coverage, expected, abs_tol=1e-12):
            raise ValueError(f"Line {node.lineno}: {qualified(node.func)} takes confidence coverage, not the error probability. The argument resolves to {coverage:g}, but the frozen inference requests {expected:g}. Correct this argument without changing the frozen inference.")
    return {"sites": sites, "target_data_access": False,
            "limitations": "Checks resolvable SciPy t/norm interval coverage and direct conditional literal fallbacks for otherwise calculated estimate/interval keys. Other formulas, output writers and dynamic expressions require independent review and numerical verification."}
