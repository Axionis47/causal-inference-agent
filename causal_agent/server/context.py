"""The description the interview reads, built from the form. The same three headings and the same column paragraphs the
hand-written notes use, so the pack loader reads it before the interview ever rewrites it."""

from __future__ import annotations

from causal_agent.intake.interview.writer import _sentence


def _para(text: str) -> str:
    return _sentence(" ".join((text or "").split()))


def render_context(title: str, about: str, changed: str, columns: list[tuple[str, str]]) -> str:
    cols = [f"**{name}** — {_para(desc) or 'Not described.'}" for name, desc in columns]
    return (
        f"# {title.strip()}\n\n"
        f"## About the dataset\n{_para(about)}\n\n"
        f"## What changed\n{_para(changed)}\n\n"
        f"## About each column\n" + "\n\n".join(cols) + "\n"
    )
