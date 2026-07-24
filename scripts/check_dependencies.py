"""Check internal component boundaries, including re-exports and local imports."""

import ast
from importlib.util import resolve_name
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ("backend.chat", "backend.analysis")
CHAT_RECORDS = {"Conversation", "ChatTurn", "ChatArtifact", "ChatContextSelection"}
ANALYSIS_RECORDS = {"AnalysisRun", "AnalysisEvent", "AnalysisTask", "AnalysisTaskAttempt"}


def module_name(path: Path, root: Path) -> str:
    return ".".join(path.relative_to(root).with_suffix("").parts).removesuffix(".__init__")


def check(sources: dict[str, str]) -> list[str]:
    edges: dict[str, set[str]] = {}
    errors = []
    for module, source in sources.items():
        edges[module] = set()
        tree = ast.parse(source)
        import_functions = {"__import__", "importlib.import_module"}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "importlib":
                import_functions.update(name.asname or name.name for name in node.names if name.name == "import_module")
            if isinstance(node, ast.Import):
                import_functions.update((name.asname or name.name) + ".import_module" for name in node.names if name.name == "importlib")
        for node in ast.walk(tree):
            targets: list[str] = []
            if isinstance(node, ast.Import):
                targets = [name.name for name in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    base = ".".join(module.split(".")[:-node.level]) + ("." + base if base else "")
                targets = [base, *[base + "." + name.name for name in node.names]]
                prohibited = ANALYSIS_RECORDS if module.startswith("backend.chat.") else CHAT_RECORDS if module.startswith("backend.analysis.") else set()
                if base == "backend.models" and prohibited & {n.name for n in node.names}:
                    errors.append(f"{module}:{node.lineno}: foreign ORM record bypasses the read interface")
            elif isinstance(node, ast.Call):
                function = ast.unparse(node.func)
                if function in import_functions:
                    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                        target = node.args[0].value
                        if target.startswith("."):
                            package = next((k.value for k in node.keywords if k.arg == "package"),
                                           node.args[1] if len(node.args) > 1 else None)
                            if isinstance(package, ast.Constant) and isinstance(package.value, str):
                                target = resolve_name(target, package.value)
                            else:
                                errors.append(f"{module}:{node.lineno}: relative dynamic import requires a literal package")
                        targets = [target]
                    elif module.startswith((*FEATURES, "backend.execution", "backend.model_access")):
                        errors.append(f"{module}:{node.lineno}: nonliteral dynamic import cannot be checked")
            for target in targets:
                if module.startswith("backend.") and target.split(".")[0] in {"tests", "historical_analysis"}:
                    errors.append(f"{module}:{node.lineno}: production cannot import historical test fixtures")
                if target.startswith("backend."):
                    edges[module].add(target)
                # ORM module imports hide which owner is accessed. Require named imports.
                if module.startswith(FEATURES) and target == "backend.models" and not (isinstance(node, ast.ImportFrom) and node.module == "backend.models"):
                    errors.append(f"{module}:{node.lineno}: import ORM records by name")
                if isinstance(node, ast.ImportFrom) and node.module == "backend.models" and any(n.name == "*" for n in node.names):
                    errors.append(f"{module}:{node.lineno}: wildcard ORM imports are prohibited")
    for module in sources:
        forbidden = []
        if module.startswith("backend.chat."):
            forbidden = ["backend.analysis", "backend.integration"]
        elif module.startswith("backend.analysis."):
            forbidden = ["backend.chat", "backend.integration"]
        elif module.startswith(("backend.execution", "backend.model_access", "backend.numerical", "backend.validation")):
            forbidden = [*FEATURES, "backend.integration"]
            if module == "backend.model_access":
                forbidden.append("backend.context")
        if not forbidden:
            continue
        todo = [(module, [module])]
        seen = set()
        while todo:
            current, path = todo.pop()
            if current in seen:
                continue
            seen.add(current)
            if current != module and any(current == prefix or current.startswith(prefix + ".") for prefix in forbidden):
                errors.append("Forbidden dependency: " + " -> ".join(path))
                break
            todo.extend((target, [*path, target]) for target in edges.get(current, ()))
    return sorted(set(errors))


def main() -> None:
    sources = {module_name(p, ROOT): p.read_text() for p in (ROOT / "backend").rglob("*.py")}
    errors = check(sources)
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Python component boundaries pass ({len(sources)} modules).")


if __name__ == "__main__":
    main()
