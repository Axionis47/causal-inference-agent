"""Check manual links and source coverage without a database or provider call.

This checks documentation structure, not the results of application verification.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
CURRENT_MANUAL = (
    "README.md",
    "docs/README.md",
    "docs/application.md",
    "docs/user-guide.md",
    "docs/architecture.md",
    "docs/component-contracts.md",
    "docs/data-model.md",
    "docs/context-slice.md",
    "docs/chat-slice.md",
    "docs/analysis-slice.md",
    "docs/api-reference.md",
    "docs/operations.md",
    "docs/security.md",
    "docs/verification.md",
    "docs/glossary.md",
    "docs/tech-stack.md",
    "docs/screenshots.md",
)


def prose(text: str) -> str:
    """Exclude fenced examples from link and heading checks."""
    return re.sub(r"^```[^\n]*\n.*?^```\s*$", "", text, flags=re.M | re.S)


def anchors(text: str) -> set[str]:
    result = set(re.findall(r'<a\s+(?:id|name)="([^"]+)"', text))
    counts: dict[str, int] = {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", prose(text), re.M):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        count = counts.get(slug, 0)
        counts[slug] = count + 1
        result.add(f"{slug}-{count}" if count else slug)
    return result


def main() -> int:
    failures: list[str] = []
    documents = [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]
    link_count = 0
    for document in documents:
        text = document.read_text()
        for match in re.finditer(r"\[[^\]\n]*\]\(([^)\n]+)\)", prose(text)):
            address = match[1].strip().strip("<>")
            parsed = urlsplit(address)
            if parsed.scheme or parsed.netloc:
                continue
            target = (document.parent / unquote(parsed.path)).resolve() if parsed.path else document
            line = text[:match.start()].count("\n") + 1
            label = f"{document.relative_to(ROOT)}:{line}"
            link_count += 1
            if not target.exists():
                failures.append(f"{label}: missing local target {address}")
            elif parsed.fragment and target.suffix == ".md":
                if unquote(parsed.fragment) not in anchors(target.read_text()):
                    failures.append(f"{label}: missing heading {address}")

    for name in CURRENT_MANUAL:
        document = ROOT / name
        if not document.exists():
            failures.append(f"Missing current-manual document: {name}")
        elif "\u2014" in document.read_text():
            failures.append(f"{name}: em dash in current manual")

    # Import contracts only. OpenAPI generation does not connect to PostgreSQL.
    sys.path.insert(0, str(ROOT))
    from backend.analysis.agent_store import LIMITS
    from backend.main import app
    from backend.models import Base

    expected_operations = {
        (method.upper(), path)
        for path, methods in app.openapi()["paths"].items()
        for method in methods
        if method in {"get", "post", "put", "patch", "delete"}
    }
    api_text = (ROOT / "docs/api-reference.md").read_text()
    listed_operations = set(re.findall(r"\| `(GET|POST|PUT|PATCH|DELETE)` \| `([^`]+)` \|", api_text))
    for method, path in sorted(expected_operations - listed_operations):
        failures.append(f"API operation missing from manual: {method} {path}")
    for method, path in sorted(listed_operations - expected_operations):
        failures.append(f"Stale API operation in manual: {method} {path}")

    data_text = (ROOT / "docs/data-model.md").read_text()
    for table in Base.metadata.tables:
        if f"| `{table}` |" not in data_text:
            failures.append(f"Domain table missing from manual: {table}")
    analysis_text = (ROOT / "docs/analysis-slice.md").read_text()
    for agent in LIMITS:
        if f"| `{agent}` |" not in analysis_text:
            failures.append(f"Agent missing from manual: {agent}")

    if failures:
        print("\n".join(failures))
        return 1
    print(
        f"Documentation checks passed: {len(documents)} Markdown files, {link_count} local links, "
        f"{len(expected_operations)} API operations, {len(Base.metadata.tables)} domain tables, "
        f"{len(LIMITS)} agents."
    )
    print("Language conformity and technical correctness require separate review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
