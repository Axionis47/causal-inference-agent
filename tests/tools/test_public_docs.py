"""Public portfolio documents stay readable, complete, and locally navigable."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PUBLIC_DOCS = (
    ROOT / "README.md",
    ROOT / "docs" / "ARCHITECTURE.md",
    ROOT / "docs" / "HARNESS-AND-EVALUATIONS.md",
    ROOT / "docs" / "ENGINEERING-JUDGMENT.md",
)
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
CASE = re.compile(r"^## ([1-4])\. .+?$", re.MULTILINE)

REQUIRED_HEADINGS = {
    "README.md": (
        "Why this project matters",
        "Pipeline",
        "What the system produces",
        "Engineering highlights",
        "How the harness works",
        "Selected engineering judgment",
        "Verification",
        "Known limitations",
        "Explore the project",
        "Development note",
    ),
    "ARCHITECTURE.md": (
        "Product boundary",
        "Five-stage workflow",
        "Who decides what",
        "Artifact lifecycle",
        "State and storage",
        "External integration boundaries",
        "Reliability flow",
        "Why one modular application",
        "Deliberate limits and a distributed future",
    ),
    "HARNESS-AND-EVALUATIONS.md": (
        "What “harness” means here",
        "Complete task lifecycle",
        "Evaluation layers",
        "Claim versus proof",
        "What scripted fixtures prove",
    ),
    "ENGINEERING-JUDGMENT.md": (
        "1. External APIs contradicted assumptions",
        "2. The harness had evidence but did not give it to the model",
        "3. “The row is the unit” must be an assertion",
        "4. Delete unused infrastructure instead of advertising it",
    ),
}


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.mark.parametrize("path", PUBLIC_DOCS, ids=lambda path: path.name)
def test_public_documents_have_required_sections_and_safe_prose(path: Path) -> None:
    text = _text(path)
    assert "\N{EM DASH}" not in text
    for heading in REQUIRED_HEADINGS[path.name]:
        assert f"## {heading}" in text
    for forbidden in (
        "/Users/",
        "AWS_SECRET_ACCESS_KEY=",
        "LANGSMITH_API_KEY=",
        "X-Amz-Signature=",
        "RESULTS_PENDING",
    ):
        assert forbidden not in text
    assert re.search(r"\ban-[0-9a-f]{8,}\b", text) is None


@pytest.mark.parametrize("path", PUBLIC_DOCS, ids=lambda path: path.name)
def test_public_document_links_resolve(path: Path) -> None:
    for raw_target in LINK.findall(_text(path)):
        target = raw_target.strip().strip("<>")
        if target.startswith(("#", "https://", "http://", "mailto:")):
            continue
        local = target.split("#", 1)[0]
        assert not Path(local).is_absolute(), f"public link must be relative: {target}"
        assert (path.parent / local).resolve().exists(), f"broken link in {path}: {target}"


def test_judgment_cases_stay_focused() -> None:
    text = _text(ROOT / "docs" / "ENGINEERING-JUDGMENT.md")
    matches = list(CASE.finditer(text))
    assert len(matches) == 4
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else text.index(
            "\nReturn to the ", match.end()
        )
        words = re.findall(r"\b[\w’'-]+\b", text[match.end():end])
        assert 300 <= len(words) <= 500, (match.group(0), len(words))
